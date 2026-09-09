"""Hybrid prefix cache (APC) for hybrid-attention models such as Qwen3.5/3.8.

Opt-in (MLX_VLM_APC_HYBRID=1). Replaces the
exact-mode whole-cache snapshots with per-component storage: attention-layer
K/V is paged into 256-token blocks addressed by a chained hash and shared
between requests, while the GatedDeltaNet recurrent state is checkpointed on a
ladder of offsets from the end of the prompt, so a continued conversation only
recomputes its tail. Same store/lookup contract as the exact mode
(``store_exact_cache`` / ``lookup_exact_cache``). Bench (Qwen3.8-27B-4bit,
M5 Max, four Junie tasks): 158 s vs 191 s exact and 541 s without APC, with
11 GB on disk instead of 30 GB. The fork's stable-prefix pinning is bypassed
in this mode (the pin boundary becomes a ladder checkpoint).

Original (Russian) design notes follow.

Гибридный кэш префиксов (APC) для гибридных моделей вроде Qwen3.5/3.8.

Модель хранит два разных вида состояния:

* слои полного внимания — K/V с осью позиции: режем на блоки, адресуем по
  цепочечному хешу, делим между запросами по ссылке (пул ``APCBlock``);
* линейные слои (GatedDeltaNet) — рекуррентное состояние, функция всей
  истории: только снимок целиком на границе («чекпойнт»).

Штатный ``apc_mode()`` в mlx-vlm выбирает один режим на всю модель и для такой
раскладки уходит в ``exact`` — снимок всего кэша (~2 ГиБ на 31k токенов) на
каждый запрос. Здесь режим для ``ar.py`` остаётся ``exact`` (тот же контракт
``store_exact_cache`` / ``lookup_exact_cache``), а внутри менеджер раскладывает
кэш по компонентам:

* K/V внимания → блоки в пуле, чекпойнт держит на них ссылки;
* хвост неполного блока (< block_size токенов) и рекуррентное состояние →
  запись чекпойнта (~150 МиБ для 27B);
* чекпойнты ставятся лестницей от конца промпта (см. ``ladder_positions``),
  чтобы промах на продолжении стоил пересчёта хвоста, а не всего префикса.

Подключение: ``apc_hybrid.install()`` до старта сервера (см. ``serve.py``).
Подробности и обоснование — в HYBRID_APC.md.

Ручки (переменные окружения):
  APC_HYBRID=1                    включить (0 — штатный exact-режим)
  APC_HYBRID_LADDER=256,512,...   смещения ступеней от конца промпта
  APC_HYBRID_GRID_FROM=4096       ступени с таким смещением и больше
                                  выравниваются к кратному смещения (дедуп
                                  между соседними запросами одного диалога)
  APC_HYBRID_MAX_GB=8             бюджет памяти на блоки + чекпойнты; не меньше
                                  K/V одного контекста (64 КиБ × токенов: 64k → 4 ГиБ)
                                  плюс несколько чекпойнтов по 147 МиБ
  APC_HYBRID_CKPT_MAX_ENTRIES=32  максимум чекпойнтов в памяти
  APC_HYBRID_FINAL=1              ставить чекпойнт на всём промпте (N)
  APC_HYBRID_KEEP_GUARD=0         дополнительно ступень N − guard из exact-режима
  APC_HYBRID_SERIAL_PREFIX=4096   холодные строки с таким общим префиксом идут в
                                  префилл по одной: следующие попадают в
                                  чекпойнты первой вместо батча из копий K/V
  APC_HYBRID_PATCH_EXTEND=1       BatchKVCache.extend без промежуточных копий
                                  (одна аллокация вместо pad+pad+concat)
  APC_HYBRID_SKIP_PREFILL_LOGITS=1 не считать логиты на промежуточных чанках
                                  префилла (минус 1–2 ГБ транзиента на строку
                                  и часть времени префилла)
  APC_BLOCK_SIZE / APC_NUM_BLOCKS штатные: размер и число блоков пула
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import mlx.core as mx

import mlx_vlm.apc as _apc
from mlx_vlm.apc import (
    SEED_PARENT_HASH,
    APCBlock,
    APCManager,
    _copy_mlx_array,
    _hash_tokens,
    _sequence_hash,
    apc_trace,
    layer_kv_for_apc,
)
from mlx_vlm.models import cache as lmc

logger = logging.getLogger(__name__)

KIND_KV = "kv"
KIND_STATE = "state"

DEFAULT_LADDER = "256,512,1024,2048,4096,8192,16384"


def _env_truthy(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Не считать логиты на промежуточных чанках префилла (см. prompt_step).
SKIP_PREFILL_LOGITS = _env_truthy("APC_HYBRID_SKIP_PREFILL_LOGITS", "1")


def _parse_ladder(raw: str) -> Tuple[int, ...]:
    out = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            continue
        if v > 0:
            out.add(v)
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# Раскладка кэша и ленивые представления строки батча
# ---------------------------------------------------------------------------


def entry_kind(c: Any) -> Optional[str]:
    """Вид элемента кэша: ``kv`` (блоки), ``state`` (чекпойнт) или None."""
    if isinstance(c, lmc.ArraysCache):
        return KIND_STATE
    if type(c) is lmc.KVCache or isinstance(c, lmc.BatchKVCache):
        return KIND_KV
    return None


def layout_of(caches: Sequence[Any]) -> Optional[Tuple[str, ...]]:
    """Раскладка списка кэшей; None, если гибридная схема неприменима."""
    kinds = tuple(entry_kind(c) for c in caches)
    if not kinds or any(k is None for k in kinds):
        return None
    if KIND_KV not in kinds or KIND_STATE not in kinds:
        return None
    return kinds  # type: ignore[return-value]


class HybridRowView:
    """Ссылка на одну строку (возможно батчевого) кэша без копирования.

    Передаётся в ``store_exact_cache`` вместо клонированного списка кэшей:
    менеджер сам вырежет нужные срезы, скопирует только новые блоки, хвост и
    рекуррентное состояние.
    """

    __slots__ = ("caches", "batch_idx")

    def __init__(self, caches: Sequence[Any], batch_idx: Optional[int] = None):
        self.caches = list(caches)
        self.batch_idx = batch_idx

    def kv_views(self, layout: Sequence[str]) -> Optional[List[Tuple[mx.array, mx.array]]]:
        """Ленивые срезы ``[1, H, len, D]`` K и V по слоям внимания."""
        out: List[Tuple[mx.array, mx.array]] = []
        b = self.batch_idx
        for c, kind in zip(self.caches, layout):
            if kind != KIND_KV:
                continue
            k, v = layer_kv_for_apc(c, batch_idx=b)
            if k is None or v is None:
                return None
            if b is not None and int(k.shape[0]) != 1:
                k = k[b : b + 1]
                v = v[b : b + 1]
            out.append((k, v))
        return out

    def state_views(self, layout: Sequence[str]) -> List[List[Optional[mx.array]]]:
        """Ленивые срезы состояний ``[1, ...]`` по линейным слоям."""
        out: List[List[Optional[mx.array]]] = []
        b = self.batch_idx
        for c, kind in zip(self.caches, layout):
            if kind != KIND_STATE:
                continue
            row: List[Optional[mx.array]] = []
            for s in c.cache:
                if s is None:
                    row.append(None)
                elif b is not None and int(s.shape[0]) != 1:
                    row.append(s[b : b + 1])
                else:
                    row.append(s)
            out.append(row)
        return out


class HybridWarmRow(list):
    """Список кэшей одной строки после попадания (для stream-пути) плюс ссылка
    на чекпойнт: batch-путь собирает из его блоков батч-кэш напрямую, без
    промежуточных копий (см. ``_build_batch_from_rows``)."""

    ckpt: "HybridCheckpoint"
    block_size: int
    capacity_hint: int


@dataclass
class HybridCheckpoint:
    """Чекпойнт: рекуррентное состояние + хвост + ссылки на блоки внимания."""

    key: int
    token_ids: Tuple[int, ...]
    extra_hash: int
    layout: Tuple[str, ...]
    blocks: List[APCBlock]
    tail_keys: List[mx.array]
    tail_values: List[mx.array]
    states: List[List[Optional[mx.array]]]
    nbytes: int
    last_used: float

    @property
    def prefix_len(self) -> int:
        return len(self.token_ids)

    @property
    def aligned_len(self) -> int:
        return len(self.token_ids) - self.tail_len

    @property
    def tail_len(self) -> int:
        return int(self.tail_keys[0].shape[2]) if self.tail_keys else 0


# ---------------------------------------------------------------------------
# Менеджер
# ---------------------------------------------------------------------------


class HybridAPCManager(APCManager):
    """``APCManager`` с гибридным store/lookup для exact-контракта."""

    MODE = "hybrid"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # upstream >= 0.6.17 keeps this on the manager; older bases only have
        # the guard. Same default and env knob as upstream.
        if not hasattr(self, "exact_cache_min_tokens"):
            self.exact_cache_min_tokens = max(
                1, int(os.environ.get("APC_EXACT_MIN_TOKENS", "16"))
            )
        # Монолитный layer-major снимок вместо блоков ломает инвариант
        # «все блоки чекпойнта в пуле» — выключаем.
        self._layer_major_memory_min_tokens = 0
        self._ckpts: "OrderedDict[int, HybridCheckpoint]" = OrderedDict()
        self._ckpt_bytes = 0
        self._block_bytes = 0
        self.ladder_offsets = _parse_ladder(os.environ.get("APC_HYBRID_LADDER", DEFAULT_LADDER))
        self.ladder_grid_from = _env_int("APC_HYBRID_GRID_FROM", 4096)
        self.store_final = _env_truthy("APC_HYBRID_FINAL", "1")
        self.keep_guard = _env_truthy("APC_HYBRID_KEEP_GUARD", "0")
        self.max_bytes = int(float(os.environ.get("APC_HYBRID_MAX_GB", "8")) * (1 << 30))
        self.ckpt_max_entries = max(1, _env_int("APC_HYBRID_CKPT_MAX_ENTRIES", 32))
        # холодные строки с таким общим префиксом префиллятся по одной (0 — выкл.)
        self.serial_prefix_tokens = _env_int("APC_HYBRID_SERIAL_PREFIX", 4096)
        self.hstats = self._fresh_hstats()
        self._layout_logged = False
        self._last_ladder: List[int] = []
        logger.info(
            "APC hybrid: block_size=%d num_blocks=%d ladder=%s grid_from=%d "
            "budget=%.1f GiB ckpt_max=%d final=%s disk=%s",
            self.block_size,
            self.num_blocks,
            ",".join(str(x) for x in self.ladder_offsets),
            self.ladder_grid_from,
            self.max_bytes / (1 << 30),
            self.ckpt_max_entries,
            self.store_final,
            bool(self.disk),
        )

    @staticmethod
    def _fresh_hstats() -> dict:
        return {
            "ckpt_stores": 0,
            "ckpt_dedup": 0,
            "ckpt_hits": 0,
            "ckpt_misses": 0,
            "ckpt_evictions": 0,
            "disk_ckpt_hits": 0,
            "replay_tokens": 0,
            "fallback_stores": 0,
            "fallback_lookups": 0,
            "pool_exhausted": 0,
        }

    # ---------- лестница ----------

    def ladder_positions(self, n_tokens: int, prefix_len: int, *, safe_min: int = 0) -> List[int]:
        """Позиции чекпойнтов для промпта длиной ``n_tokens`` при восстановленном
        префиксе ``prefix_len``. Только строго между ними и не короче
        ``safe_min`` / ``exact_cache_min_tokens``. Финальная ступень N сюда не
        входит — её ставит ``generate()`` после префилла."""
        bs = self.block_size
        floor = max(prefix_len, safe_min - 1, self.exact_cache_min_tokens - 1)
        out = set()
        for off in self.ladder_offsets:
            p = n_tokens - off
            if p <= 0:
                continue
            unit = off if (self.ladder_grid_from > 0 and off >= self.ladder_grid_from) else bs
            p = (p // unit) * unit
            if p > floor and p <= n_tokens - 1:
                out.add(p)
        return sorted(out)

    # ---------- учёт памяти пула ----------

    def _evict_lru(self) -> Optional[APCBlock]:
        b = self._free_head
        if b is not None and b.block_hash is not None:
            self._block_bytes -= b.resident_bytes()
        return super()._evict_lru()

    def _reclaim_one_free_block(self) -> bool:
        """Сбросить тензоры самого старого свободного блока с содержимым."""
        b = self._free_head
        while b is not None and b.block_hash is None:
            b = b.next
        if b is None:
            return False
        self._free_remove(b)
        if self.hash_table.get(b.block_hash) is b:
            del self.hash_table[b.block_hash]
        self._block_bytes -= b.resident_bytes()
        b.release_components()
        b.block_hash = None
        b.token_ids = ()
        b.parent_hash = SEED_PARENT_HASH
        b.extra_hash = 0
        self._free_push(b)
        self.stats.evictions += 1
        return True

    def _evict_oldest_ckpt(self) -> bool:
        if not self._ckpts:
            return False
        key, ck = self._ckpts.popitem(last=False)
        self._ckpt_bytes -= ck.nbytes
        for b in ck.blocks:
            self._release_one(b)
        self.hstats["ckpt_evictions"] += 1
        apc_trace("hybrid_evict", prefix_len=ck.prefix_len, blocks=len(ck.blocks))
        return True

    def _enforce_budget(self, protect: Optional[int] = None) -> None:
        """Бюджет вытесняет старое, но никогда — самый свежий чекпойнт
        (``protect``): иначе при префиксе, чьи блоки одни занимают весь бюджет,
        в памяти не остаётся ни одной точки возврата и каждый запрос идёт на
        диск за старым чекпойнтом (наблюдалось на 64k контексте с 4 ГиБ)."""

        def can_evict() -> bool:
            if not self._ckpts:
                return False
            oldest = next(iter(self._ckpts))
            return oldest != protect or len(self._ckpts) > 1

        while len(self._ckpts) > self.ckpt_max_entries and can_evict():
            self._evict_oldest_or_next(protect)
        while self._block_bytes + self._ckpt_bytes > self.max_bytes:
            if self._reclaim_one_free_block():
                continue
            if can_evict() and self._evict_oldest_or_next(protect):
                continue
            if self._ckpts and next(iter(self._ckpts)) == protect and len(self._ckpts) == 1:
                self.hstats["budget_exceeded"] = self.hstats.get("budget_exceeded", 0) + 1
            break

    def _evict_oldest_or_next(self, protect: Optional[int]) -> bool:
        """Вытеснить самый старый чекпойнт, пропуская защищённый."""
        if not self._ckpts:
            return False
        for key in list(self._ckpts):
            if key == protect:
                continue
            ck = self._ckpts.pop(key)
            self._ckpt_bytes -= ck.nbytes
            for b in ck.blocks:
                self._release_one(b)
            self.hstats["ckpt_evictions"] += 1
            apc_trace("hybrid_evict", prefix_len=ck.prefix_len, blocks=len(ck.blocks))
            return True
        return False

    # ---------- блоки ----------

    def _acquire_chain(self, tokens: Tuple[int, ...], extra_hash: int) -> Tuple[List[APCBlock], int]:
        """Пройти цепочку по уже лежащим в пуле блокам, захватив ссылки."""
        bs = self.block_size
        parent = SEED_PARENT_HASH
        acquired: List[APCBlock] = []
        for i in range(len(tokens) // bs):
            chunk = tokens[i * bs : (i + 1) * bs]
            h = _hash_tokens(parent, chunk, extra_hash)
            b = self.hash_table.get(h)
            if b is None or b.token_ids != chunk:
                break
            acquired.append(self._acquire_existing(b))
            parent = h
        return acquired, parent

    def _materialize_blocks(
        self,
        tokens: Tuple[int, ...],
        start_block: int,
        parent: int,
        extra_hash: int,
        slab_fn: Callable[[int], Tuple[List[mx.array], List[mx.array]]],
        n_tensors_per_block: int,
    ) -> Optional[Tuple[List[APCBlock], List[APCBlock]]]:
        """Положить блоки ``start_block..`` в пул. ``slab_fn(i)`` возвращает
        ленивые срезы K и V блока ``i`` по слоям внимания. Возвращает
        (все захваченные, из них созданные) или None при нехватке пула —
        тогда ничего не остаётся захваченным."""
        bs = self.block_size
        n_blocks = len(tokens) // bs
        held: List[APCBlock] = []
        created: List[APCBlock] = []
        pending: List[Tuple[APCBlock, List[mx.array], List[mx.array]]] = []
        eval_chunk = 16

        def flush() -> None:
            targets: List[mx.array] = []
            for _, ks, vs in pending:
                targets.extend(ks)
                targets.extend(vs)
            if targets:
                mx.eval(targets)
            for b, ks, vs in pending:
                b.set_kv(ks, vs)
                self._block_bytes += b.resident_bytes()
            pending.clear()

        def rollback() -> None:
            flush()
            for b in held:
                self._release_one(b)
            self.hstats["pool_exhausted"] += 1

        for i in range(start_block, n_blocks):
            chunk = tokens[i * bs : (i + 1) * bs]
            h = _hash_tokens(parent, chunk, extra_hash)
            existing = self.hash_table.get(h)
            if existing is not None and existing.token_ids == chunk:
                held.append(self._acquire_existing(existing))
                parent = h
                continue
            if (
                self._max_pool_tensors > 0
                and (len(self.hash_table) + 1) * n_tensors_per_block > self._max_pool_tensors
            ):
                rollback()
                return None
            b = self._evict_lru()
            if b is None:
                rollback()
                return None
            ks, vs = slab_fn(i)
            b.block_hash = h
            b.parent_hash = parent
            b.token_ids = chunk
            b.extra_hash = extra_hash
            b.ref_cnt = 1
            self.hash_table[h] = b
            held.append(b)
            created.append(b)
            pending.append((b, ks, vs))
            if len(pending) >= eval_chunk:
                flush()
            self.stats.stores += 1
            self.stats.served_tokens += bs
            parent = h
        flush()
        return held, created

    # ---------- store ----------

    def store_exact_cache(
        self,
        token_ids: Sequence[int],
        prompt_cache: Any,
        *,
        extra_hash: int = 0,
        pinned: bool = False,
    ) -> bool:
        # ``pinned`` is the fork's stable-prefix flag for the exact/session
        # tier; in hybrid mode the boundary simply becomes one more
        # checkpoint on the ladder, so the flag carries no extra meaning.
        del pinned
        view = prompt_cache if isinstance(prompt_cache, HybridRowView) else HybridRowView(prompt_cache, None)
        layout = layout_of(view.caches)
        if layout is None:
            self.hstats["fallback_stores"] += 1
            if view.batch_idx is None:
                caches: Optional[List[Any]] = view.caches
            else:
                caches = _apc.snapshot_prompt_cache_row(view.caches, view.batch_idx)
            if caches is None:
                return False
            return super().store_exact_cache(token_ids, caches, extra_hash=extra_hash)
        if not self._layout_logged:
            self._layout_logged = True
            logger.info(
                "APC hybrid layout: %d entries, %d kv (pageable), %d state (checkpoint)",
                len(layout),
                layout.count(KIND_KV),
                layout.count(KIND_STATE),
            )
        try:
            return self._store_hybrid(token_ids, view, layout, int(extra_hash))
        except Exception as exc:  # не ронять генерацию из-за кэша
            logger.warning("APC hybrid store failed: %s: %s", type(exc).__name__, exc)
            return False

    def _store_hybrid(self, token_ids: Sequence[int], view: HybridRowView, layout: Tuple[str, ...], extra_hash: int) -> bool:
        tokens = tuple(int(t) for t in token_ids)
        n = len(tokens)
        if n == 0 or n < self.exact_cache_min_tokens:
            return False
        key = _sequence_hash(tokens, extra_hash, self.block_size)
        with self.lock:
            ck = self._ckpts.get(key)
            if ck is not None:
                self._ckpts.move_to_end(key)
                ck.last_used = time.time()
                self.hstats["ckpt_dedup"] += 1
                return True

        kv_views = view.kv_views(layout)
        if not kv_views:
            return False
        for k, _ in kv_views:
            if int(k.shape[2]) < n:
                logger.warning(
                    "APC hybrid store: cache row has %d tokens, prefix needs %d; skipped",
                    int(k.shape[2]),
                    n,
                )
                return False
        state_views = view.state_views(layout)
        bs = self.block_size
        aligned = (n // bs) * bs
        n_tensors = 2 * len(kv_views)

        def slab_fn(i: int) -> Tuple[List[mx.array], List[mx.array]]:
            s, e = i * bs, (i + 1) * bs
            ks = [_copy_mlx_array(k[..., s:e, :]) for k, _ in kv_views]
            vs = [_copy_mlx_array(v[..., s:e, :]) for _, v in kv_views]
            return ks, vs

        t0 = time.perf_counter()
        with self.lock:
            acquired, parent = self._acquire_chain(tokens[:aligned], extra_hash)
            result = self._materialize_blocks(tokens[:aligned], len(acquired), parent, extra_hash, slab_fn, n_tensors)
            while result is None and self._ckpts:
                # Пул исчерпан закреплёнными блоками: освободить старые чекпойнты.
                self._evict_oldest_ckpt()
                result = self._materialize_blocks(tokens[:aligned], len(acquired), parent, extra_hash, slab_fn, n_tensors)
            if result is None:
                self.release(acquired)
                self.stats.record_reject("hybrid_pool_exhausted", token_len=n)
                return False
            held, created = result
            blocks = acquired + held

            tail_k: List[mx.array] = []
            tail_v: List[mx.array] = []
            if aligned < n:
                tail_k = [_copy_mlx_array(k[..., aligned:n, :]) for k, _ in kv_views]
                tail_v = [_copy_mlx_array(v[..., aligned:n, :]) for _, v in kv_views]
            states = [[None if s is None else _copy_mlx_array(s) for s in row] for row in state_views]
            targets = tail_k + tail_v + [s for row in states for s in row if s is not None]
            if targets:
                mx.eval(targets)
            nbytes = sum(int(a.nbytes) for a in targets)
            ck = HybridCheckpoint(
                key=key,
                token_ids=tokens,
                extra_hash=extra_hash,
                layout=layout,
                blocks=blocks,
                tail_keys=tail_k,
                tail_values=tail_v,
                states=states,
                nbytes=nbytes,
                last_used=time.time(),
            )
            self._ckpts[key] = ck
            self._ckpt_bytes += nbytes
            self.hstats["ckpt_stores"] += 1
            self.stats.exact_stores += 1
            self._enforce_budget(protect=key)
            n_created = len(created)
            self.stats.pool_used = len(self.hash_table)

        if self.disk is not None:
            try:
                fresh = [b for b in created if not self.disk.has(b.block_hash)]
                if fresh and hasattr(self.disk, "save_batch"):
                    # mlx_vlm <= 0.6.17: поблочные снимки
                    self.disk.save_batch(fresh)
                elif fresh:
                    # mlx_vlm >= 0.7.0: layer-major шард прямо из K/V строки;
                    # source_block_idx — позиция блока в цепочке
                    from mlx_vlm.apc import _DiskLayerMajorBlock

                    chain_idx = {id(b): i for i, b in enumerate(blocks)}
                    lm_blocks = [
                        _DiskLayerMajorBlock(
                            block_hash=int(b.block_hash),
                            parent_hash=int(b.parent_hash),
                            extra_hash=int(b.extra_hash),
                            token_ids=tuple(int(t) for t in b.token_ids),
                            source_block_idx=chain_idx[id(b)],
                        )
                        for b in fresh
                        if id(b) in chain_idx
                    ]
                    if lm_blocks:
                        self.disk.save_layer_major_blocks(
                            lm_blocks, [k for k, _ in kv_views], [v for _, v in kv_views], bs
                        )
                self.disk.save_exact_cache(key, tokens, extra_hash, self._reduced_cache(ck))
                with self.lock:
                    self.stats.disk_writes += len(fresh) + 1
            except Exception as exc:
                logger.warning("APC hybrid disk save scheduling failed: %s", exc)

        apc_trace(
            "store",
            mode="hybrid",
            ok=True,
            token_len=n,
            blocks=len(blocks),
            new_blocks=n_created,
            tail=n - aligned,
            ckpt_mb=round(nbytes / (1 << 20), 1),
            ms=round((time.perf_counter() - t0) * 1000, 1),
            **_mem_fields(),
        )
        return True

    def _reduced_cache(self, ck: HybridCheckpoint) -> List[Any]:
        """Список кэшей для дискового exact-формата: состояние + только хвост."""
        out: List[Any] = []
        ki = si = 0
        tail_len = ck.tail_len
        for kind in ck.layout:
            if kind == KIND_KV:
                c = lmc.KVCache()
                if tail_len > 0:
                    c.keys = ck.tail_keys[ki]
                    c.values = ck.tail_values[ki]
                    c.offset = tail_len
                ki += 1
            else:
                c = lmc.ArraysCache(len(ck.states[si]))
                c.cache = list(ck.states[si])
                si += 1
            out.append(c)
        return out

    # ---------- lookup ----------

    def lookup_exact_cache(
        self,
        token_ids: Sequence[int],
        extra_hash: int = 0,
        max_prefix_tokens: Optional[int] = None,
        min_prefix_tokens: int = 0,
    ) -> Tuple[Optional[List[Any]], int]:
        tokens = tuple(int(t) for t in token_ids)
        extra_hash = int(extra_hash)
        max_len = len(tokens) - 1
        if max_prefix_tokens is not None and max_prefix_tokens > 0:
            max_len = min(max_len, int(max_prefix_tokens))
        if max_len <= min_prefix_tokens:
            return None, 0

        with self.lock:
            best = self._best_checkpoint(tokens, extra_hash, max_len, min_prefix_tokens)
            mem_len = best.prefix_len if best is not None else 0

        if self.disk is not None and mem_len < max_len and self._disk_ram_ok():
            try:
                promoted = self._lookup_disk(tokens, extra_hash, max_len, max(min_prefix_tokens, mem_len))
            except Exception as exc:
                logger.warning("APC hybrid disk lookup failed: %s: %s", type(exc).__name__, exc)
                promoted = None
            if promoted is not None:
                best = promoted
                self.hstats["disk_ckpt_hits"] += 1

        if best is None:
            self.hstats["ckpt_misses"] += 1
            if self._exact_cache:
                # Раскладка не гибридная (store ушёл в штатный exact-путь):
                # искать в его памяти. Диск не трогаем — там лежат
                # сокращённые чекпойнты, штатный загрузчик их не поймёт.
                self.hstats["fallback_lookups"] += 1
                with self.lock:
                    disk, self.disk = self.disk, None
                    try:
                        return super().lookup_exact_cache(
                            token_ids, extra_hash, max_prefix_tokens, min_prefix_tokens
                        )
                    finally:
                        self.disk = disk
            return None, 0

        t0 = time.perf_counter()
        with self.lock:
            if best.key not in self._ckpts:
                return None, 0
            self._ckpts.move_to_end(best.key)
            best.last_used = time.time()
            # запас ёмкости: промпт + генерация, чтобы батч-кэш не рос копией
            cache = self._assemble_row_cache(best, capacity_hint=len(tokens) + 1024)
            L = best.prefix_len
            self.stats.exact_hits += 1
            self.stats.hits += 1
            self.stats.matched_tokens += L
            self.hstats["ckpt_hits"] += 1
            self.hstats["replay_tokens"] += len(tokens) - L
        apc_trace(
            "lookup",
            mode="hybrid",
            prefix_len=L,
            replay=len(tokens) - L,
            blocks=len(best.blocks),
            ms=round((time.perf_counter() - t0) * 1000, 1),
            **_mem_fields(),
        )
        return cache, L

    def _best_checkpoint(self, tokens: Tuple[int, ...], extra_hash: int, max_len: int, min_len: int) -> Optional[HybridCheckpoint]:
        best: Optional[HybridCheckpoint] = None
        for ck in self._ckpts.values():
            L = ck.prefix_len
            if ck.extra_hash != extra_hash or L <= min_len or L > max_len:
                continue
            if best is not None and L <= best.prefix_len:
                continue
            if tokens[:L] != ck.token_ids:
                continue
            best = ck
        return best

    def _assemble_row_cache(self, ck: HybridCheckpoint, capacity_hint: int = 0) -> List[Any]:
        """Однострочный кэш. K/V собираются лениво (concat без eval): batch-путь
        его не материализует, а берёт блоки напрямую; stream-путь посчитает при
        первом обращении."""
        out: List[Any] = HybridWarmRow()
        out.ckpt = ck
        out.block_size = self.block_size
        out.capacity_hint = int(capacity_hint)
        ki = si = 0
        L = ck.prefix_len
        for kind in ck.layout:
            if kind == KIND_KV:
                parts_k = [b.keys[ki] for b in ck.blocks]
                parts_v = [b.values[ki] for b in ck.blocks]
                if ck.tail_keys:
                    parts_k.append(ck.tail_keys[ki])
                    parts_v.append(ck.tail_values[ki])
                c = lmc.KVCache()
                if parts_k:
                    c.keys = mx.concatenate(parts_k, axis=2)
                    c.values = mx.concatenate(parts_v, axis=2)
                    c.offset = L
                out.append(c)
                ki += 1
            else:
                c = lmc.ArraysCache(len(ck.states[si]))
                # Массивы MLX неизменяемы: модель заменяет слоты, не пишет в них.
                c.cache = list(ck.states[si])
                out.append(c)
                si += 1
        return out

    # ---------- диск ----------

    def _disk_ram_ok(self) -> bool:
        if self._disk_min_free_ram_bytes <= 0:
            return True
        free_now = _apc._free_ram_bytes()
        if free_now is not None and free_now < self._disk_min_free_ram_bytes:
            logger.info(
                "APC hybrid: skipping disk restore (free RAM %.1f GB < %.1f GB)",
                free_now / (1 << 30),
                self._disk_min_free_ram_bytes / (1 << 30),
            )
            return False
        return True

    def _lookup_disk(self, tokens: Tuple[int, ...], extra_hash: int, max_len: int, min_len: int) -> Optional[HybridCheckpoint]:
        disk = self.disk
        if disk is None:
            return None
        found = disk.find_exact_prefix(tokens, extra_hash=extra_hash, max_prefix_tokens=max_len, min_prefix_tokens=min_len)
        if found is None:
            return None
        cache_hash, L = found
        key = _sequence_hash(tokens[:L], extra_hash, self.block_size)
        with self.lock:
            if key in self._ckpts:
                return self._ckpts[key]
        t0 = time.perf_counter()
        loaded = disk.load_exact_cache(cache_hash)
        if loaded is None:
            return None
        stored_tokens, stored_extra, reduced = loaded
        if stored_extra != extra_hash or len(stored_tokens) != L or tokens[:L] != stored_tokens:
            return None
        layout = layout_of(reduced)
        if layout is None:
            return None

        tail_k: List[mx.array] = []
        tail_v: List[mx.array] = []
        states: List[List[Optional[mx.array]]] = []
        tail_len: Optional[int] = None
        for c, kind in zip(reduced, layout):
            if kind == KIND_KV:
                off = int(getattr(c, "offset", 0) or 0)
                if tail_len is None:
                    tail_len = off
                elif off != tail_len:
                    return None
                if off > 0:
                    tail_k.append(_copy_mlx_array(c.keys[..., :off, :]))
                    tail_v.append(_copy_mlx_array(c.values[..., :off, :]))
            else:
                states.append([None if s is None else _copy_mlx_array(s) for s in c.cache])
        tail_len = tail_len or 0
        aligned = L - tail_len
        bs = self.block_size
        if aligned < 0 or aligned % bs != 0:
            return None
        n_kv = layout.count(KIND_KV)
        targets = tail_k + tail_v + [s for row in states for s in row if s is not None]
        if targets:
            mx.eval(targets)
        nbytes = sum(int(a.nbytes) for a in targets)

        with self.lock:
            acquired, parent = self._acquire_chain(tokens[:aligned], extra_hash)
        n_blocks = aligned // bs
        missing_hashes: List[int] = []
        missing_chunks: List[Tuple[int, ...]] = []
        p = parent
        for i in range(len(acquired), n_blocks):
            chunk = tokens[i * bs : (i + 1) * bs]
            h = _hash_tokens(p, chunk, extra_hash)
            if not disk.has(h):
                with self.lock:
                    self.release(acquired)
                return None
            missing_hashes.append(h)
            missing_chunks.append(chunk)
            p = h

        held: List[APCBlock] = []
        if missing_hashes:
            got = disk.load_layer_major_prefix(missing_hashes)
            if got is None:
                with self.lock:
                    self.release(acquired)
                return None
            keys, values, metas = got
            if len(keys) != n_kv or len(values) != n_kv or len(metas) != len(missing_chunks):
                with self.lock:
                    self.release(acquired)
                return None
            for chunk, meta in zip(missing_chunks, metas):
                raw = meta.get("token_ids", "")
                if isinstance(raw, str):
                    stored = tuple(int(x) for x in raw.split(",") if x)
                else:
                    stored = tuple(int(x) for x in raw)
                if stored != chunk or int(meta.get("extra_hash", "0")) != extra_hash:
                    with self.lock:
                        self.release(acquired)
                    return None
            base = len(acquired)

            def slab_fn(i: int) -> Tuple[List[mx.array], List[mx.array]]:
                s, e = (i - base) * bs, (i - base + 1) * bs
                ks = [_copy_mlx_array(k[..., s:e, :]) for k in keys]
                vs = [_copy_mlx_array(v[..., s:e, :]) for v in values]
                return ks, vs

            with self.lock:
                result = self._materialize_blocks(tokens[:aligned], base, parent, extra_hash, slab_fn, 2 * n_kv)
                while result is None and self._ckpts:
                    self._evict_oldest_ckpt()
                    result = self._materialize_blocks(tokens[:aligned], base, parent, extra_hash, slab_fn, 2 * n_kv)
                if result is None:
                    self.release(acquired)
                    return None
                held, _created = result

        with self.lock:
            if key in self._ckpts:
                self.release(acquired + held)
                return self._ckpts[key]
            ck = HybridCheckpoint(
                key=key,
                token_ids=tuple(stored_tokens),
                extra_hash=extra_hash,
                layout=layout,
                blocks=acquired + held,
                tail_keys=tail_k,
                tail_values=tail_v,
                states=states,
                nbytes=nbytes,
                last_used=time.time(),
            )
            self._ckpts[key] = ck
            self._ckpt_bytes += nbytes
            self.stats.disk_hits += 1
            self._enforce_budget(protect=key)
            self.stats.pool_used = len(self.hash_table)
        apc_trace(
            "disk_restore",
            mode="hybrid",
            prefix_len=L,
            blocks_from_disk=len(missing_hashes),
            blocks_in_memory=len(acquired),
            ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return ck

    # ---------- обслуживание ----------

    def clear(self) -> None:
        with self.lock:
            super().clear()
            self._ckpts.clear()
            self._ckpt_bytes = 0
            self._block_bytes = 0
            self.hstats = self._fresh_hstats()

    def reset_stats(self) -> None:
        with self.lock:
            super().reset_stats()
            self.hstats = self._fresh_hstats()

    def stats_snapshot(self) -> dict:
        snap = super().stats_snapshot()
        with self.lock:
            snap["mode"] = self.MODE
            snap["hybrid"] = {
                **self.hstats,
                **_mem_fields(),
                "batch_direct": BATCH_STATS["direct"],
                "batch_fallback": BATCH_STATS["fallback"],
                "checkpoints": len(self._ckpts),
                "checkpoint_bytes": self._ckpt_bytes,
                "block_bytes": self._block_bytes,
                "budget_bytes": self.max_bytes,
                "ladder": list(self.ladder_offsets),
                "grid_from": self.ladder_grid_from,
                "last_ladder": list(self._last_ladder),
            }
        return snap


# ---------------------------------------------------------------------------
# Лестница чекпойнтов во время префилла (continuous batching)
# ---------------------------------------------------------------------------


def _make_prompt_batch_class():
    from mlx_vlm.generate import ar as _ar

    class HybridPromptProcessingBatch(_ar.PromptProcessingBatch):
        """``PromptProcessingBatch`` со многими чекпойнтами на строку."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not self._hybrid_active():
                return
            mgr: HybridAPCManager = self._apc_manager
            config = getattr(self.model, "config", None)
            media_ids = _apc.multimodal_token_ids_from_config(config) if config is not None else set()
            for meta in self._apc_meta:
                if meta is None:
                    continue
                ids = meta.get("full_input_ids")
                if not ids:
                    continue
                n = len(ids)
                prefix_len = int(meta.get("prefix_len") or 0)
                safe_min = _apc.media_safe_prefix_min(ids, media_ids) if media_ids else 0
                rungs = mgr.ladder_positions(n, prefix_len, safe_min=safe_min)
                guard = int(meta.get("checkpoint_len") or 0)
                if mgr.keep_guard and guard > prefix_len and guard not in rungs:
                    rungs = sorted(rungs + [guard])
                meta["checkpoint_lens"] = rungs
                meta["checkpoints_done"] = set()
                # Штатная логика одной ступени больше не нужна.
                meta["checkpoint_done"] = True
                mgr._last_ladder = rungs
                apc_trace("hybrid_ladder", n=n, prefix_len=prefix_len, rungs=rungs)

        def _hybrid_active(self) -> bool:
            return isinstance(self._apc_manager, HybridAPCManager) and self._apc_mode == "exact"

        _skip_logits_cache: dict = {}

        def _model_skips_logits(self) -> bool:
            """Модель умеет ``skip_logits`` (qwen3_5 и родня): на промежуточных
            чанках префилла LM-голова не нужна — её выход выбрасывается, а это
            2048 × словарь логитов на строку (1–2 ГБ) и заметная доля времени."""
            cls = type(self.model)
            hit = self._skip_logits_cache.get(cls)
            if hit is None:
                try:
                    import inspect

                    hit = "skip_logits" in inspect.getsource(cls.__call__)
                except Exception:  # noqa: BLE001
                    hit = False
                self._skip_logits_cache[cls] = hit
            return hit

        def _tail_mode(self) -> bool:
            return (
                SKIP_PREFILL_LOGITS
                and self._hybrid_active()
                and self._inputs_embeds is not None
                and self.prefill_step_size is not None
                and self._model_skips_logits()
            )

        def _final_tail(self) -> int:
            """Сколько колонок оставить финальному вызову в ``generate()``: по
            одному последнему реальному токену на строку. При right padding
            последний токен короткой строки стоит раньше на величину её
            паддинга, поэтому хвост = max(pad) + 1."""
            pad = max(self._right_pad_per_row) if self._right_pad_per_row else 0
            return int(pad) + 1

        def needs_processing(self):
            base = super().needs_processing()
            if base or not self._tail_mode():
                return base
            # штатно финальному чанку достаётся до prefill_step_size токенов и
            # логиты на все их позиции (2048 × словарь × строки); съедаем всё
            # промежуточными чанками без логитов
            return self._inputs_embeds.shape[1] > self._final_tail()

        def prompt_step(self) -> int:
            if not self._tail_mode():
                return super().prompt_step()
            if not self.needs_processing():
                return 0
            n = min(self.prefill_step_size, self._inputs_embeds.shape[1] - self._final_tail())
            checkpoint_col = self._next_apc_checkpoint_column()
            if checkpoint_col is not None:
                n = min(n, checkpoint_col - self._processed_prompt_columns)
            if n <= 0:
                return 0
            trace = _apc.apc_trace_enabled()
            if trace:
                active_before = mx.get_active_memory()
            self._prompt_kwargs["skip_logits"] = True
            try:
                prompt_kwargs = self._prompt_kwargs_for_step(n)
                self.model(
                    self._input_ids[:, :n],
                    cache=self.prompt_cache,
                    inputs_embeds=self._inputs_embeds[:, :n],
                    n_to_process=n,
                    **prompt_kwargs,
                )
            finally:
                self._prompt_kwargs.pop("skip_logits", None)
            mx.async_eval([c.state for c in self.prompt_cache])
            self._processed_prompt_columns += n
            self._store_apc_exact_checkpoints()
            self._inputs_embeds = self._inputs_embeds[:, n:]
            self._input_ids = self._input_ids[:, n:]
            for k in self._prompt_length_aware_keys:
                self._prompt_kwargs[k] = _ar._slice_sequence_aligned_prompt_kwarg(
                    k, self._prompt_kwargs[k], start=n
                )
            mx.clear_cache()
            if trace:
                mx.eval([c.state for c in self.prompt_cache])
                apc_trace(
                    "prefill_chunk",
                    rows=int(self._inputs_embeds.shape[0]),
                    n=n,
                    left=int(self._inputs_embeds.shape[1]),
                    active_before_gb=round(active_before / 2**30, 2),
                    active_after_gb=round(mx.get_active_memory() / 2**30, 2),
                    peak_gb=round(mx.get_peak_memory() / 2**30, 2),
                )
            return n

        def _apc_checkpoint_column_for_meta(
            self,
            batch_idx: int,
            meta: dict,
            length_key: str = "checkpoint_len",
            done_key: str = "checkpoint_done",
        ) -> Optional[int]:
            if not self._hybrid_active() or length_key != "checkpoint_len":
                # the fork's extra store specs (pinned stable prefix) keep
                # their own bookkeeping; hybrid only owns the ladder rungs
                return super()._apc_checkpoint_column_for_meta(
                    batch_idx, meta, length_key=length_key, done_key=done_key
                )
            rungs = meta.get("checkpoint_lens")
            if not rungs:
                return None
            done = meta.setdefault("checkpoints_done", set())
            prefix_len = int(meta.get("prefix_len", 0) or 0)
            for p in rungs:
                if p in done:
                    continue
                if p <= prefix_len:
                    done.add(p)
                    continue
                if self._right_pad_per_row is not None:
                    suffix_checkpoint = p - prefix_len
                    if suffix_checkpoint >= self._suffix_lens[batch_idx]:
                        return None
                    return suffix_checkpoint
                return self._left_padding_per_row[batch_idx] + p
            return None

        def _store_apc_exact_checkpoints(self) -> None:
            if not self._hybrid_active():
                return super()._store_apc_exact_checkpoints()
            for batch_idx, meta in enumerate(self._apc_meta):
                if meta is None:
                    continue
                rungs = meta.get("checkpoint_lens")
                if not rungs:
                    continue
                done = meta.setdefault("checkpoints_done", set())
                processed = self._row_real_tokens_processed(batch_idx)
                for p in rungs:
                    if p in done:
                        continue
                    if p < processed:
                        done.add(p)  # ступень проскочили — не блокировать остальные
                        continue
                    if p > processed:
                        break
                    view = HybridRowView(self.prompt_cache, batch_idx)
                    try:
                        self._apc_manager.store_exact_cache(
                            meta["full_input_ids"][:p],
                            view,
                            extra_hash=meta.get("extra_hash", 0),
                        )
                    except Exception as exc:
                        logger.warning("APC hybrid checkpoint at %d failed: %s", p, exc)
                    done.add(p)

        def _apc_prompt_cache_for_store(self, batch_idx: int):
            if not self._hybrid_active():
                return super()._apc_prompt_cache_for_store(batch_idx)
            if not self._apc_manager.store_final:
                return None
            return HybridRowView(self.prompt_cache, batch_idx)

    return HybridPromptProcessingBatch


def _mem_fields() -> dict:
    """Память MLX для трассировки: активные тензоры, кэш аллокатора, пик."""
    try:
        return {
            "active_gb": round(mx.get_active_memory() / 2**30, 2),
            "alloc_cache_gb": round(mx.get_cache_memory() / 2**30, 2),
            "peak_gb": round(mx.get_peak_memory() / 2**30, 2),
        }
    except Exception:  # noqa: BLE001
        return {}


def _round_up(x: int, step: int) -> int:
    return ((x + step - 1) // step) * step


def _build_batch_from_rows(rows: List[Any], prefix_lens: List[int]) -> Optional[Tuple[List[Any], int]]:
    """Батч-кэш из строк-попаданий (``HybridWarmRow``) и холодных строк
    (``model.make_cache()``): K/V пишутся из блоков прямо в итоговый тензор
    одной аллокацией с запасом ёмкости; состояния — concat ссылок."""
    warm = [r for r in rows if isinstance(r, HybridWarmRow)]
    if not warm:
        return None
    layout = warm[0].ckpt.layout
    n = len(layout)
    B = len(rows)
    for r, pl in zip(rows, prefix_lens):
        if len(r) != n:
            return None
        if isinstance(r, HybridWarmRow):
            if r.ckpt.layout != layout or r.ckpt.prefix_len != pl:
                return None
        else:
            if pl != 0:
                return None
            for c, kind in zip(r, layout):
                if entry_kind(c) != kind or not c.empty():
                    return None
    bs = warm[0].block_size
    max_prefix = max(prefix_lens)
    step = int(getattr(lmc.BatchKVCache, "step", 256) or 256)
    capacity = _round_up(max([max_prefix] + [r.capacity_hint for r in warm]), step)
    left_padding = [max_prefix - pl for pl in prefix_lens]

    out: List[Any] = []
    ki = si = 0
    for kind in layout:
        if kind == KIND_KV:
            sample_k = sample_v = None
            for r in warm:
                if r.ckpt.blocks:
                    sample_k, sample_v = r.ckpt.blocks[0].keys[ki], r.ckpt.blocks[0].values[ki]
                elif r.ckpt.tail_keys:
                    sample_k, sample_v = r.ckpt.tail_keys[ki], r.ckpt.tail_values[ki]
                if sample_k is not None:
                    break
            if sample_k is None:
                return None
            H, Dk, Dv = sample_k.shape[1], sample_k.shape[3], sample_v.shape[3]
            keys = mx.zeros((B, H, capacity, Dk), dtype=sample_k.dtype)
            values = mx.zeros((B, H, capacity, Dv), dtype=sample_v.dtype)
            for i, r in enumerate(rows):
                if not isinstance(r, HybridWarmRow):
                    continue
                pos = left_padding[i]
                for b in r.ckpt.blocks:
                    keys[i : i + 1, :, pos : pos + bs, :] = b.keys[ki]
                    values[i : i + 1, :, pos : pos + bs, :] = b.values[ki]
                    pos += bs
                t = r.ckpt.tail_len
                if t:
                    keys[i : i + 1, :, pos : pos + t, :] = r.ckpt.tail_keys[ki]
                    values[i : i + 1, :, pos : pos + t, :] = r.ckpt.tail_values[ki]
            mx.eval(keys, values)
            c = lmc.BatchKVCache(left_padding)
            c.keys = keys
            c.values = values
            c.offset = c.offset + max_prefix
            c._idx = max_prefix
            out.append(c)
            ki += 1
        else:
            size = len(warm[0].ckpt.states[si])
            merged: List[Optional[mx.array]] = []
            for e in range(size):
                states = [r.ckpt.states[si][e] if isinstance(r, HybridWarmRow) else None for r in rows]
                sample = next((x for x in states if x is not None), None)
                if sample is None:
                    merged.append(None)
                    continue
                parts = [
                    mx.zeros((1,) + sample.shape[1:], dtype=sample.dtype) if x is None else x
                    for x in states
                ]
                merged.append(parts[0] if B == 1 else mx.concatenate(parts, axis=0))
            c = lmc.ArraysCache(size)
            c.cache = merged
            out.append(c)
            si += 1
    mx.eval([x for c in out if isinstance(c, lmc.ArraysCache) for x in c.cache if x is not None])
    return out, max_prefix


BATCH_STATS = {"direct": 0, "fallback": 0}


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    # общий префикс монотонен: бинарный поиск по срезам
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _batch_kv_extend(self, other) -> None:
    """Замена ``BatchKVCache.extend``: та же раскладка (строки выровнены по
    правому краю к ``max(_idx)``), но итоговый тензор выделяется один раз и
    заполняется срезами вместо ``mx.pad`` каждой части и ``concatenate`` —
    одна копия K/V вместо трёх при подключении строки к генерационному батчу."""
    if self.keys is None and other.keys is None:
        self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
        self.offset = mx.concatenate([self.offset, other.offset])
        return
    max_idx = max(self._idx, other._idx)
    parts = []
    H = D = M = None
    max_size = 0
    for c in (self, other):
        k, v = c.keys, c.values
        if k is not None:
            H, D, M = k.shape[1], k.shape[3], v.shape[3]
            max_size = max(max_size, int(k.shape[2]))
    rows = 0
    for c in (self, other):
        k, v = c.keys, c.values
        B = int(c.offset.shape[0])
        if k is None:
            parts.append((B, None, None, 0, c.offset, c.left_padding + (max_idx - c._idx)))
        else:
            left = max_idx - c._idx
            keep = min(int(k.shape[2]), max_size - left)
            parts.append((B, k, v, left, c.offset, c.left_padding + left, keep))
        rows += B
    dtype = next(c.keys.dtype for c in (self, other) if c.keys is not None)
    keys = mx.zeros((rows, H, max_size, D), dtype=dtype)
    values = mx.zeros((rows, H, max_size, M), dtype=dtype)
    r = 0
    offsets, lps = [], []
    for part in parts:
        B = part[0]
        if part[1] is not None:
            _, k, v, left, off, lp, keep = part
            if keep > 0:
                keys[r : r + B, :, left : left + keep, :] = k[..., :keep, :]
                values[r : r + B, :, left : left + keep, :] = v[..., :keep, :]
        else:
            _, _, _, _, off, lp = part
        offsets.append(off)
        lps.append(lp)
        r += B
    self.keys, self.values = keys, values
    self.offset = mx.concatenate(offsets)
    self.left_padding = mx.concatenate(lps)
    self._idx = max_idx


def _wrap_build_mixed_prompt_batch(orig):
    """Тёплые строки — только по одной. Смешанный батч из нескольких строк
    с разными prefix_len собирается в mlx_vlm через right padding, и этот путь
    даёт неверный вывод короткой строке даже штатным exact-менеджером
    (см. tests: чанкованный префилл + right padding). Одиночная тёплая строка
    паддинга не требует; холодные строки батчуются штатно с left padding."""

    def _defer(self, n, deferred):
        # вызывающий код отрежет первые n элементов очереди — вернуть
        # отложенные строки сразу за ними
        self._unprocessed_sequences = (
            self._unprocessed_sequences[:n] + deferred + self._unprocessed_sequences[n:]
        )

    def _build_mixed_prompt_batch(self, sequences):
        if not isinstance(self.apc_manager, HybridAPCManager) or len(sequences) <= 1:
            return orig(self, sequences)
        n = len(sequences)
        for i, seq in enumerate(sequences):
            batch = orig(self, [seq])
            if batch is None:
                continue
            _defer(self, n, [s for j, s in enumerate(sequences) if j != i])
            return batch
        # Все холодные. Строки с длинным общим префиксом префиллим по одной:
        # вторая и дальше попадут в чекпойнты первой (тот же суммарный счёт,
        # но одна строка в батче вместо нескольких копий K/V).
        min_shared = self.apc_manager.serial_prefix_tokens
        if min_shared > 0:
            first_ids = sequences[0][1]
            for other in sequences[1:]:
                if _common_prefix_len(first_ids, other[1]) >= min_shared:
                    deferred = list(sequences[1:])
                    del sequences[1:]  # вызывающий код строит холодный батч из этого списка
                    _defer(self, n, deferred)
                    apc_trace("hybrid_serial_cold", kept=1, deferred=len(deferred))
                    break
        return None

    _build_mixed_prompt_batch.__wrapped__ = orig  # type: ignore[attr-defined]
    return _build_mixed_prompt_batch


def _wrap_make_warm_batch(orig):
    def make_warm_batch_exact_cache_multi(row_caches, prefix_lens, kv_quant_config=None):
        rows = list(row_caches)
        if kv_quant_config is None and any(isinstance(r, HybridWarmRow) for r in rows):
            try:
                built = _build_batch_from_rows(rows, [int(p) for p in prefix_lens])
            except Exception as exc:  # noqa: BLE001
                logger.warning("APC hybrid batch build failed, falling back: %s", exc)
                built = None
            if built is not None:
                BATCH_STATS["direct"] += 1
                apc_trace("hybrid_batch", rows=len(rows), max_prefix=built[1], **_mem_fields())
                return built
            BATCH_STATS["fallback"] += 1
        return orig(row_caches, prefix_lens, kv_quant_config=kv_quant_config)

    make_warm_batch_exact_cache_multi.__wrapped__ = orig  # type: ignore[attr-defined]
    return make_warm_batch_exact_cache_multi


_INSTALLED = False
_ORIGINALS: dict = {}


def install() -> None:
    """Подменить менеджер, класс префилл-батча и namespace дискового кэша."""
    global _INSTALLED
    if _INSTALLED:
        return
    import sys

    import mlx_vlm.generate  # noqa: F401  (пакет; атрибут mlx_vlm.generate — функция)

    _gen = sys.modules["mlx_vlm.generate"]
    _ORIGINALS.update(
        {
            "APCManager": _apc.APCManager,
            "PromptProcessingBatch": _gen.PromptProcessingBatch,
            "apc_disk_namespace": _apc.apc_disk_namespace,
        }
    )
    _apc.APCManager = HybridAPCManager  # from_env берёт класс из глобала модуля
    _gen.PromptProcessingBatch = _make_prompt_batch_class()

    orig_ns = _apc.apc_disk_namespace

    def apc_disk_namespace(*args, **kwargs):
        return orig_ns(*args, **kwargs) + "-hybrid"

    apc_disk_namespace.__wrapped__ = orig_ns  # type: ignore[attr-defined]
    _apc.apc_disk_namespace = apc_disk_namespace
    from mlx_vlm.generate import ar as _ar

    if _env_truthy("APC_HYBRID_PATCH_EXTEND", "1") and not getattr(lmc.BatchKVCache.extend, "_hybrid", False):
        _batch_kv_extend._hybrid = True  # type: ignore[attr-defined]
        _batch_kv_extend.__wrapped__ = lmc.BatchKVCache.extend  # type: ignore[attr-defined]
        lmc.BatchKVCache.extend = _batch_kv_extend
    if not hasattr(_ar.BatchGenerator._build_mixed_prompt_batch, "__wrapped__"):
        _ar.BatchGenerator._build_mixed_prompt_batch = _wrap_build_mixed_prompt_batch(
            _ar.BatchGenerator._build_mixed_prompt_batch
        )
    if not hasattr(_apc.make_warm_batch_exact_cache_multi, "__wrapped__"):
        _apc.make_warm_batch_exact_cache_multi = _wrap_make_warm_batch(_apc.make_warm_batch_exact_cache_multi)
    _INSTALLED = True
    logger.info("APC hybrid installed")


def uninstall() -> None:
    """Undo install(): restore the stock manager, prefill batch and wrappers
    (used by tests so the patches do not leak into the rest of a session)."""
    global _INSTALLED
    if not _INSTALLED:
        return
    import sys

    _gen = sys.modules["mlx_vlm.generate"]
    from mlx_vlm.generate import ar as _ar

    _apc.APCManager = _ORIGINALS["APCManager"]
    _gen.PromptProcessingBatch = _ORIGINALS["PromptProcessingBatch"]
    _apc.apc_disk_namespace = _ORIGINALS["apc_disk_namespace"]
    if getattr(lmc.BatchKVCache.extend, "_hybrid", False):
        lmc.BatchKVCache.extend = lmc.BatchKVCache.extend.__wrapped__
    if hasattr(_ar.BatchGenerator._build_mixed_prompt_batch, "__wrapped__"):
        _ar.BatchGenerator._build_mixed_prompt_batch = _ar.BatchGenerator._build_mixed_prompt_batch.__wrapped__
    if hasattr(_apc.make_warm_batch_exact_cache_multi, "__wrapped__"):
        _apc.make_warm_batch_exact_cache_multi = _apc.make_warm_batch_exact_cache_multi.__wrapped__
    _ORIGINALS.clear()
    _INSTALLED = False
    logger.info("APC hybrid uninstalled")

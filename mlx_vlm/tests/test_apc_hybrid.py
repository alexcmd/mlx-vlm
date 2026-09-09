"""Функциональные тесты гибридного APC на Qwen3.5-2B-4bit (та же архитектура,
что у Qwen3.8-27B: full_attention_interval=4, ArraysCache + KVCache).

Запуск:  ~/.venvs/mlx/bin/python tests/test_apc_hybrid.py   (или pytest)

Гоняет тот же путь, что и сервер: BatchGenerator → PromptProcessingBatch.
Эталон — холодная генерация без APC при greedy-декоде.
"""
from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Test-time knobs; applied by the module fixture (and by the __main__ runner)
# so they do not leak into the other APC tests of a pytest session.
TEST_ENV = {
    "APC_HYBRID_LADDER": "256,512,1024,2048,4096",
    "APC_HYBRID_GRID_FROM": "2048",
    "APC_EXACT_PREFIX_GUARD_TOKENS": "64",
}

import mlx.core as mx  # noqa: E402

import pytest  # noqa: E402

from mlx_vlm import apc_hybrid  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _hybrid_installed():
    """The hybrid cache is a set of process-wide patches; keep them and the
    test env scoped to this module so the stock APC tests in the same session
    see stock code and a stock environment."""
    with pytest.MonkeyPatch.context() as mp:
        for key, value in TEST_ENV.items():
            mp.setenv(key, value)
        apc_hybrid.install()
        try:
            yield
        finally:
            apc_hybrid.uninstall()

import mlx_vlm.apc as _apc  # noqa: E402
from mlx_vlm import load  # noqa: E402
from mlx_vlm.generate.ar import BatchGenerator  # noqa: E402

MODEL = os.environ.get("TEST_MODEL", "mlx-community/Qwen3.5-2B-4bit")
BLOCK = 64
PREFILL_STEP = int(os.environ.get("TEST_PREFILL_STEP", "512"))
MAX_NEW = 8

_model = _processor = None


def model_and_processor():
    global _model, _processor
    if _model is None:
        _model, _processor = load(MODEL)
    return _model, _processor


def tokenizer():
    _, p = model_and_processor()
    return p.tokenizer if hasattr(p, "tokenizer") else p


WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega server cache prefix block "
    "checkpoint ladder tensor matrix vector kernel memory token layer state"
).split()


def make_text(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def encode(text: str):
    return tokenizer().encode(text, add_special_tokens=False)


def make_manager(disk_dir=None, num_blocks=512):
    disk = None
    if disk_dir is not None:
        disk = _apc.DiskBlockStore(Path(disk_dir), namespace="test-hybrid", num_workers=1, max_bytes=None)
    return apc_hybrid.HybridAPCManager(num_blocks=num_blocks, block_size=BLOCK, disk=disk)


def make_gen(mgr, prefill_step=PREFILL_STEP):
    model, processor = model_and_processor()
    return BatchGenerator(
        model.language_model,
        processor,
        stop_tokens=set(),
        apc_manager=mgr,
        prefill_step_size=prefill_step,
        greedy_sampling=True,
        compute_logprobs=True,
    )


def embed_kwargs(ids):
    """То же, что делает сервер в ``_gpu_embed`` для текстового запроса."""
    model, _ = model_and_processor()
    input_ids = mx.array([list(ids)])
    embed = model.get_input_embeddings(input_ids, None, mask=None)
    kw = {k: v for k, v in embed.to_dict().items() if v is not None}
    # сервер считает семантический хеш без inputs_embeds (generation.py),
    # иначе extra_hash зависел бы от содержимого промпта
    kw["_apc_semantic_hash"] = _apc.semantic_extra_hash(
        tenant=None, image_hash=0, media={"audio": None, "video": None},
        model=model.language_model, processor=_processor,
    )
    return kw


def run(gen, prompts, max_new=MAX_NEW):
    """Прогнать список промптов (одним insert), вернуть [(tokens, logprobs)]."""
    uids = gen.insert(
        [list(p) for p in prompts],
        max_tokens=[max_new] * len(prompts),
        prompt_kwargs=[embed_kwargs(p) for p in prompts],
    )
    out = {u: ([], []) for u in uids}
    while gen.has_work:
        _, responses = gen.next()
        for r in responses:
            if r.uid in out and r.token is not None:
                out[r.uid][0].append(int(r.token))
                out[r.uid][1].append(float(r.token_logprob))
    return [out[u] for u in uids]


def cold(prompts):
    gen = make_gen(None)
    try:
        return run(gen, prompts)
    finally:
        gen.close()


def same_output(ref, got, label, steps=None):
    ref_t, ref_lp = ref
    got_t, got_lp = got
    n = min(len(ref_t), len(got_t))
    if steps is not None:
        n = min(n, steps)
    assert n >= 4, f"{label}: слишком короткий вывод {ref_t} / {got_t}"
    if ref_t[:n] != got_t[:n]:
        diff = next(i for i in range(n) if ref_t[i] != got_t[i])
        raise AssertionError(
            f"{label}: токены разошлись на шаге {diff}: {ref_t[:n]} vs {got_t[:n]}; "
            f"logprob ref={ref_lp[diff]:.3f} got={got_lp[diff]:.3f}"
        )
    # logprob в bf16 гуляет на 0.125 даже между холодными прогонами с разным
    # шагом префилла; допуск шире, а совпадение токенов — обязательное
    assert abs(ref_lp[0] - got_lp[0]) < 0.3, f"{label}: logprob первого токена {ref_lp[0]:.4f} vs {got_lp[0]:.4f}"


def checkpoint_lens(mgr):
    return sorted(ck.prefix_len for ck in mgr._ckpts.values())


# ---------------------------------------------------------------------------


def test_ladder_positions():
    mgr = make_manager()
    rungs = mgr.ladder_positions(3000, 0)
    # 256/512/1024 выравниваются к 64, 2048/4096 — к сетке 2048
    assert rungs == [1920, 2432, 2688], rungs
    assert mgr.ladder_positions(3000, 2500) == [2688]
    assert mgr.ladder_positions(3000, 0, safe_min=2600) == [2688]
    assert mgr.ladder_positions(100, 0) == []


def test_repeat_and_continue():
    base = encode(make_text(2400, seed=1))  # ~2.4k токенов общего префикса
    q1 = encode(" Question one: what is the last word?")
    q2 = encode(" Question two: count the words please.")
    p1 = base + q1
    p2 = base + q2
    cont = p1 + encode(" Answer: omega. Next question: which word appears most often?")

    ref1, ref2, ref_cont = cold([p1]) + cold([p2]) + cold([cont])

    mgr = make_manager()
    gen = make_gen(mgr)
    try:
        t0 = time.perf_counter()
        (got1,) = run(gen, [p1])
        t_cold = time.perf_counter() - t0
        same_output(ref1, got1, "холодный прогон с APC")
        st = mgr.stats_snapshot()["hybrid"]
        assert st["ckpt_stores"] >= 2, st  # ступени + финальный
        assert st["ckpt_hits"] == 0
        lens = checkpoint_lens(mgr)
        assert len(p1) in lens, lens

        # тот же промпт: попадание в ступень < N, пересчёт < 512 токенов
        t0 = time.perf_counter()
        (got1b,) = run(gen, [p1])
        t_warm = time.perf_counter() - t0
        same_output(ref1, got1b, "повтор промпта")
        st = mgr.stats_snapshot()["hybrid"]
        assert st["ckpt_hits"] == 1, st
        assert 0 < st["replay_tokens"] <= 512 + BLOCK, st
        print(f"  повтор: холодный {t_cold:.2f} с, тёплый {t_warm:.2f} с, replay {st['replay_tokens']}")

        # другой вопрос: попадание в ступень до вопроса
        (got2,) = run(gen, [p2])
        same_output(ref2, got2, "другой вопрос")
        st = mgr.stats_snapshot()["hybrid"]
        assert st["ckpt_hits"] == 2, st

        # продолжение: финальный чекпойнт p1 — префикс cont
        replay_before = st["replay_tokens"]
        (got_c,) = run(gen, [cont])
        same_output(ref_cont, got_c, "продолжение")
        st = mgr.stats_snapshot()["hybrid"]
        assert st["ckpt_hits"] == 3, st
        assert st["replay_tokens"] - replay_before == len(cont) - len(p1), st

        # дедупликация блоков: общий префикс лежит один раз
        snap = mgr.stats_snapshot()
        distinct = {len(p1), len(p2), len(cont)}
        max_blocks = (max(distinct) // BLOCK) + 2 * (len(q2) // BLOCK + 2)
        assert snap["pool_used"] <= max_blocks, (snap["pool_used"], max_blocks)
        print(f"  блоков в пуле {snap['pool_used']}, чекпойнтов {snap['hybrid']['checkpoints']}, "
              f"байт чекпойнтов {snap['hybrid']['checkpoint_bytes']/2**20:.1f} МиБ")
    finally:
        gen.close()


def _rel_err(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return float(mx.mean(mx.abs(a - b)) / (mx.mean(mx.abs(b)) + 1e-6))


def test_state_matches_cold_prefill():
    """Собранный из блоков + чекпойнта кэш совпадает с холодным префиллом."""
    from mlx_vlm.models import cache as lmc

    model, _ = model_and_processor()
    lm = model.language_model
    ids = encode(make_text(1700, seed=11))
    mgr = make_manager()
    gen = make_gen(mgr)
    try:
        run(gen, [ids])
    finally:
        gen.close()
    assert mgr._ckpts, "чекпойнтов нет"
    checked = 0
    for ck in list(mgr._ckpts.values()):
        R = ck.prefix_len
        got = mgr._assemble_row_cache(ck)
        ref = lmc.make_prompt_cache(lm)
        kw = embed_kwargs(ids[:R])
        lm(mx.array([ids[:R]]), cache=ref, **kw)
        mx.eval([c.state for c in ref])
        for j, (g, r) in enumerate(zip(got, ref)):
            if isinstance(g, lmc.KVCache):
                assert g.offset == R == r.offset, (j, g.offset, r.offset)
                ek = _rel_err(g.keys[..., :R, :], r.keys[..., :R, :])
                ev = _rel_err(g.values[..., :R, :], r.values[..., :R, :])
                assert ek < 0.05 and ev < 0.05, f"слой {j} R={R}: K {ek:.4f} V {ev:.4f}"
                # сдвиг на блок должен быть заметен: контроль чувствительности
                if R > 2 * BLOCK:
                    shifted = _rel_err(g.keys[..., BLOCK:R, :], r.keys[..., : R - BLOCK, :])
                    assert shifted > 0.2, f"слой {j}: проверка нечувствительна ({shifted:.3f})"
            else:
                for a, b in zip(g.cache, r.cache):
                    if a is None or b is None:
                        assert a is None and b is None
                        continue
                    e = _rel_err(a, b)
                    assert e < 0.05, f"слой {j} R={R}: state {e:.4f}"
        checked += 1
    print(f"  сверено чекпойнтов: {checked}, позиции {checkpoint_lens(mgr)}")


def test_divergent_tail():
    base = encode(make_text(2000, seed=2))
    tail_a = encode(make_text(300, seed=3))
    tail_b = encode(make_text(300, seed=4))
    pa, pb = base + tail_a, base + tail_b
    ref_b = cold([pb])[0]
    mgr = make_manager()
    gen = make_gen(mgr)
    try:
        run(gen, [pa])
        st0 = mgr.stats_snapshot()["hybrid"]
        (got_b,) = run(gen, [pb])
        same_output(ref_b, got_b, "расхождение хвоста")
        st = mgr.stats_snapshot()["hybrid"]
        assert st["ckpt_hits"] == 1, st
        replay = st["replay_tokens"] - st0["replay_tokens"]
        d = len(pb) - len(base)
        assert d < replay <= 2 * d + 2 * BLOCK, (replay, d)
        print(f"  расхождение на {d} от конца: пересчёт {replay} токенов")
    finally:
        gen.close()


def test_mixed_batch_right_padding():
    """Две строки с разными prefix_len в одной вставке. Тёплые строки идут в
    префилл по одной (right-padded смешанные батчи в mlx_vlm ненадёжны), затем
    сливаются в общий генерационный батч. Сверка кэша каждой строки с холодным
    префиллом одним вызовом численно: токены на промпте из случайных слов
    почти равновероятны и расходятся от шума bf16."""
    from mlx_vlm.models import cache as lmc

    model, _ = model_and_processor()
    lm = model.language_model
    base = encode(make_text(1500, seed=5))
    pa = base + encode(" first request, short tail.")
    pb = base + encode(make_text(200, seed=6))
    ca = pa + encode(" continuation of the first one with a few more words here.")
    cb = pb + encode(make_text(120, seed=7))
    mgr = make_manager()
    gen = make_gen(mgr)
    try:
        run(gen, [pa, pb])  # холодный батч из двух строк разной длины
        uids = gen.insert([list(ca), list(cb)], max_tokens=[1, 1],
                          prompt_kwargs=[embed_kwargs(ca), embed_kwargs(cb)])
        seen = {}
        while gen.has_work:
            gen.next()
            batch = gen._generation_batch
            # снять кэш строки сразу после её префилла, до decode
            for uid, ids in zip(uids, (ca, cb)):
                if uid in batch.uids and uid not in seen:
                    row = batch.uids.index(uid)
                    ref = lmc.make_prompt_cache(lm)
                    lm(mx.array([ids]), cache=ref, **embed_kwargs(ids))
                    mx.eval([c.state for c in ref])
                    worst = 0.0
                    for g, r in zip(batch.prompt_cache, ref):
                        if isinstance(g, lmc.BatchKVCache):
                            lp = int(g.left_padding[row].item())
                            L = len(ids)
                            worst = max(worst, _rel_err(g.keys[row:row + 1, :, lp:lp + L, :], r.keys[..., :L, :]),
                                        _rel_err(g.values[row:row + 1, :, lp:lp + L, :], r.values[..., :L, :]))
                        else:
                            for a, b in zip(g.cache, r.cache):
                                if a is not None:
                                    worst = max(worst, _rel_err(a[row:row + 1], b))
                    seen[uid] = worst
        assert len(seen) == 2, seen
        for uid, w in seen.items():
            assert w < 0.05, f"строка uid={uid}: отн. ошибка кэша {w:.3f}"
        st = mgr.stats_snapshot()["hybrid"]
        assert st["ckpt_hits"] == 2, st
        assert st["batch_direct"] >= 1 and st["batch_fallback"] == 0, st
        print(f"  ошибки кэша строк: {', '.join(f'{w:.3f}' for w in seen.values())}")
    finally:
        gen.close()


def test_cold_shared_prefix_serialized():
    """Два холодных промпта с общим префиксом в одной вставке: первый идёт один,
    второй попадает в его чекпойнт (батч из копий K/V не строится)."""
    base = encode(make_text(2200, seed=21))
    pa = base + encode(" question A?")
    pb = base + encode(" a rather different question B, please answer.")
    refs = cold([pa]) + cold([pb])
    mgr = make_manager()
    mgr.serial_prefix_tokens = 1024
    gen = make_gen(mgr)
    try:
        got = run(gen, [pa, pb])
        # decode строки A идёт в батче со строкой B — численно это другой
        # путь, чем одиночный эталон; на промпте из случайных слов хвост
        # расходится от шума, сверяем первые шаги
        same_output(refs[0], got[0], "холодная пара, строка A", steps=4)
        same_output(refs[1], got[1], "холодная пара, строка B", steps=4)
        st = mgr.stats_snapshot()["hybrid"]
        assert st["ckpt_hits"] == 1, st
        assert 0 < st["replay_tokens"] < 600, st
        print(f"  вторая строка: пересчёт {st['replay_tokens']} токенов")
    finally:
        gen.close()


def test_disk_restart():
    base = encode(make_text(1800, seed=8))
    p = base + encode(" disk question?")
    cont = p + encode(" answer, and a follow-up question about persistence.")
    ref = cold([cont])[0]
    tmp = tempfile.mkdtemp(prefix="apc-hybrid-test-")
    try:
        mgr1 = make_manager(disk_dir=tmp)
        gen1 = make_gen(mgr1)
        try:
            run(gen1, [p])
        finally:
            gen1.close()
        stored = checkpoint_lens(mgr1)
        mgr1.close()  # дождаться писателя
        files = list(Path(tmp).rglob("*.safetensors"))
        assert files, "на диске ничего нет"
        total_mb = sum(f.stat().st_size for f in files) / 2**20

        mgr2 = make_manager(disk_dir=tmp)
        gen2 = make_gen(mgr2)
        try:
            assert mgr2.stats_snapshot()["hybrid"]["checkpoints"] == 0
            (got,) = run(gen2, [cont])
            same_output(ref, got, "восстановление с диска")
            st = mgr2.stats_snapshot()
            assert st["hybrid"]["disk_ckpt_hits"] == 1, st["hybrid"]
            assert st["hybrid"]["ckpt_hits"] == 1, st["hybrid"]
            assert st["matched_tokens"] == len(p), (st["matched_tokens"], len(p), stored)
            print(f"  диск: {len(files)} файлов, {total_mb:.1f} МиБ; восстановлено {len(p)} токенов")
        finally:
            gen2.close()
            mgr2.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_budget_eviction():
    base = encode(make_text(1200, seed=9))
    mgr = make_manager(num_blocks=64)  # 64 блока × 64 токена = 4096 токенов пула
    mgr.max_bytes = 6 * 2**20  # крошечный бюджет: чекпойнты будут вытесняться
    mgr.ckpt_max_entries = 3
    gen = make_gen(mgr)
    try:
        for i in range(4):
            run(gen, [base + encode(make_text(100, seed=100 + i))])
        st = mgr.stats_snapshot()
        h = st["hybrid"]
        assert 1 <= h["checkpoints"] <= 3, h  # самый свежий чекпойнт бюджет не трогает
        assert h["ckpt_evictions"] > 0, h
        newest = list(mgr._ckpts.values())[-1]
        assert newest.prefix_len == len(base) + len(encode(make_text(100, seed=103)))
        # над бюджетом могут остаться только блоки и байты защищённого чекпойнта
        assert h["checkpoint_bytes"] <= newest.nbytes + 1, h
        # все живые чекпойнты держат свои блоки
        for ck in mgr._ckpts.values():
            assert all(b.ref_cnt > 0 and b.block_hash is not None for b in ck.blocks)
        print(f"  бюджет: чекпойнтов {h['checkpoints']}, вытеснений {h['ckpt_evictions']}, "
              f"блоков {st['pool_used']}")
    finally:
        gen.close()


if __name__ == "__main__":
    for _k, _v in TEST_ENV.items():
        os.environ.setdefault(_k, _v)
    apc_hybrid.install()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    if len(sys.argv) > 1:
        tests = [t for t in tests if t.__name__ in sys.argv[1:]]
    failed = 0
    for t in tests:
        print(f"== {t.__name__}", flush=True)
        t0 = time.perf_counter()
        try:
            t()
            print(f"   OK ({time.perf_counter() - t0:.1f} с)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback

            traceback.print_exc()
            print(f"   FAIL: {exc}", flush=True)
    sys.exit(1 if failed else 0)

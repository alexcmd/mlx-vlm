# Junie task benchmark

End-to-end timing of the Junie CLI (headless) on four real coding tasks
against a local model, through a logging proxy that records every model
request. It measures what a Junie user experiences: wall time per task,
time to first byte per request (cold prefill vs. warm continuation vs. the
small summarizer calls), tokens, peak server memory.

## Layout

| file | purpose |
|---|---|
| `run.sh` | orchestrates rounds × targets: proxies, engine start/stop, memory sampling |
| `real_junie.py` | runs the four tasks with `junie -p <project> "<task>"`, writes `results/*.json` |
| `oai_proxy.py` | logging reverse proxy (request bodies + timeline per request) |
| `fp_sampler.py` | per-second `phys_footprint` of the server process (+ `/v1/cache/stats`) |
| `summarize.py` | markdown tables (medians across rounds) and a full text log |
| `project/` | the sample project (small text-processing library with pytest tests, English) |
| `profiles/` | Junie model profiles to copy into `~/.junie/models/` |

## Tasks (`real_junie.py`)

1. **read+explain** – read `src/normalize.py`, explain `normalize` in three sentences (≈5 requests).
2. **project search** – list every TODO comment: file, line, gist (≈4 requests).
3. **code edit** – add `slugify` to `src/utils.py` plus a test (12–16 requests; the function already
   exists, so the agent verifies and runs pytest).
4. **architecture review** – read `AGENTS.md` and `src/`, propose three functions (≈23 requests).

Each task is a separate Junie session. The project is restored from `project/` before every task,
`AGENTS.md` is installed as `.junie/guidelines.md` (it asks for English answers), and the copy is
committed to a fresh git repo. Junie sends `stream: false`, so the proxy's time to first byte is
the full request time including generation. Requests with a response ≥ 1000 bytes are counted as
main-agent requests (tool calls), smaller ones as summarizer/title calls.

## Running

```bash
# 1. Junie CLI installed (junie on PATH), model profiles in ~/.junie/models/:
cp bench/junie-tasks/profiles/*.json ~/.junie/models/   # fill in the engine api_key
# 2. Junie Local engine benchmark (3 rounds):
TARGETS=jb REPEATS=3 bench/junie-tasks/run.sh
# 3. Any OpenAI-compatible server already running on 127.0.0.1:8080, e.g. this fork's worker:
MLX_VLM_APC_HYBRID=1 python -m mlx_vlm.server --model Qwen3.8-27B-MLX-4bit --port 8080 &
TARGETS=server SERVER_LABEL="worker hybrid" REPEATS=3 bench/junie-tasks/run.sh
# 4. Tables and log:
bench/junie-tasks/summarize.py            # medians across rounds, per-task medians
bench/junie-tasks/summarize.py --log      # console output, agent answers, TTFB of every request
```

`run.sh` gives every launch fresh request directories under `/tmp/junie-tasks/`: the proxy numbers
requests from 1 on each start and `real_junie.py` deduplicates rows by number.

Columns of the summary: total wall time of the four tasks; number of proxied requests; output
tokens (Junie's `llmUsage`); cold = TTFB of the first main request of the first task (empty cache);
warm = median TTFB of the other main requests; aux = median TTFB of the summarizer calls;
ms per output token = total / output tokens; peak RAM = max `phys_footprint` of the server (the
engine's worker) during the run.

## Caveats

* Temperature 0.6: the number of turns per task varies between rounds (the agent may loop, e.g. one
  run produced a 14k-token repetition of a test-case list until the engine's 270 s soft timeout).
  Medians across three rounds are the comparable number; single rounds are not.
* Run on an otherwise idle machine. A background build (Docker at 300 % CPU) slowed every
  configuration by 10–30 % in our runs.
* Reference numbers (M5 Max, 128 GB, Qwen3.8-27B-4bit, 4 tasks, medians of 3 clean rounds, Russian
  variant of the tasks): Junie Local engine 0.2.2 132 s (29–31 GB); mlx-vlm 0.7.0 with the hybrid
  APC + MTP + int8 prefill 102 s (27–28 GB); plain mlx-vlm 0.7.0 without a prefix cache 517 s.

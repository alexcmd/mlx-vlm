#!/usr/bin/env python3
"""Four real coding tasks through Junie CLI (headless) against one model profile.

Arguments: label, Junie model profile (custom:...), [port of the mlx-vlm
server for /v1/metrics, 0 = none].

Before every task the sample project (./project) is copied to /tmp/junie-tasks/
project, AGENTS.md is installed as .junie/guidelines.md and the copy is
committed to a fresh git repo. Per-request timing comes from the logging proxy
(oai_proxy.py): QBENCH_REQ_DIR/index.tsv with columns idx, time, status,
t_first_byte, t_first_content, t_done, bytes. The proxy numbers requests from 1
on every launch, so QBENCH_REQ_DIR must be empty when a launch starts.

Writes results/<timestamp>_<label>.json: per task the wall time, the proxy
rows, Junie's llmUsage, the first 2000 characters of the agent's answer and the
list of changed files.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
SRC = os.environ.get("JUNIE_TASKS_PROJECT", str(HERE / "project"))
CWD = os.environ.get("JUNIE_TASKS_WORKDIR", "/tmp/junie-tasks/project")
RESULTS = pathlib.Path(os.environ.get("JUNIE_TASKS_RESULTS", str(HERE / "results")))
TIMEOUT = int(os.environ.get("JUNIE_TASKS_TIMEOUT", "1200"))
REQ_DIR = pathlib.Path(os.environ.get("QBENCH_REQ_DIR", "/tmp/junie-tasks/reqs"))

TASKS = [
    ("read+explain",
     "Read src/normalize.py and explain in three sentences what the normalize function does "
     "and in which order the steps are applied."),
    ("project search",
     "Find all TODO comments in every file of the project and list them: file, line, gist."),
    ("code edit",
     "Add a function slugify(text: str) -> str to src/utils.py: it normalizes the text via "
     "src.normalize.normalize, replaces spaces with dashes and removes everything except letters, "
     "digits and dashes. Add a docstring and a test in tests/test_utils.py following the existing tests."),
    ("architecture review",
     "Read AGENTS.md and every file in src/, then propose three functions worth adding to the "
     "library to close obvious gaps. For each: name, signature, one sentence of justification."),
]


def reset_project():
    shutil.rmtree(CWD, ignore_errors=True)
    shutil.copytree(SRC, CWD)
    os.makedirs(f"{CWD}/.junie", exist_ok=True)
    shutil.copy(f"{CWD}/AGENTS.md", f"{CWD}/.junie/guidelines.md")
    subprocess.run(["git", "init", "-q", "."], cwd=CWD, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=CWD, capture_output=True)
    subprocess.run(["git", "-c", "user.email=b@b", "-c", "user.name=b", "commit", "-qm", "init"],
                   cwd=CWD, capture_output=True)


def index_rows():
    p = REQ_DIR / "index.tsv"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 7:
            rows.append({"idx": int(f[0]), "status": int(f[2]), "ttfb": float(f[3]),
                         "ttfc": float(f[4]), "done": float(f[5]), "bytes": int(f[6])})
    return rows


def metrics(port):
    if not port:
        return {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/metrics", timeout=10) as r:
            d = json.load(r)
    except Exception:
        return {}
    rows = ([d["latest"]] if d.get("latest") else []) + d.get("recent", [])
    return {r["timestamp_unix"]: r for r in rows}


def main():
    label, profile = sys.argv[1], sys.argv[2]
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    print(f"### {label} ({profile})", flush=True)
    total = 0.0
    summary = []
    for name, prompt in TASKS:
        reset_project()
        before_idx = {r["idx"] for r in index_rows()}
        before_m = set(metrics(port))
        t0 = time.time()
        timed_out = False
        try:
            p = subprocess.run(["junie", "--skip-update-check", "--model", profile, "--output-format", "json",
                                "-p", CWD, prompt], cwd=CWD, capture_output=True, text=True, timeout=TIMEOUT)
            try:
                out = json.loads(p.stdout.strip().splitlines()[-1])
            except Exception:
                out = {"raw": p.stdout[-500:], "err": p.stderr[-500:]}
        except subprocess.TimeoutExpired:
            subprocess.run(["pkill", "-f", "junie.*--model"], capture_output=True)
            timed_out = True
            out = {}
        dt = time.time() - t0
        total += dt
        rows = [r for r in index_rows() if r["idx"] not in before_idx]
        usage = (out.get("llmUsage") or [{}])[0] if isinstance(out, dict) else {}
        new_m = sorted([r for k, r in metrics(port).items() if k not in before_m], key=lambda x: x["timestamp_unix"])
        inp = sum(r["prompt_tokens"] for r in new_m)
        outp = sum(r["completion_tokens"] for r in new_m)
        ttfb = " ".join(f"{r['ttfb']:.1f}" for r in rows[:8])
        print(f"  {name:20} {dt:6.1f} s{' TIMEOUT' if timed_out else ''} | requests {len(rows):2} | "
              f"junie: in {usage.get('inputTokens', '?')} cached {usage.get('cacheInputTokens', '?')} "
              f"out {usage.get('outputTokens', '?')} calls {usage.get('calls', '?')} | "
              f"server: in {inp} out {outp} | ttfb {ttfb}", flush=True)
        summary.append({"task": name, "seconds": dt, "timeout": timed_out, "requests": rows, "usage": usage,
                        "result": (out.get("result") or "")[:2000] if isinstance(out, dict) else "",
                        "changes": out.get("changes") if isinstance(out, dict) else None})
    print(f"  TOTAL: {total:.1f} s", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    safe = label.replace(" ", "_").replace("/", "_")
    (RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}.json").write_text(
        json.dumps({"label": label, "profile": profile, "total": total, "tasks": summary}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

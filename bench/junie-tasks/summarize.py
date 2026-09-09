#!/usr/bin/env python3
"""Summary of Junie task runs: results/*.json (real_junie.py) and fp_*.log (fp_sampler.py).

Runs are grouped by label with the trailing " rN" (round number) stripped and
reported as medians across rounds. Usage:
  summarize.py [--results DIR] [--log]     # markdown tables; --log prints the
                                           # full text log (console output,
                                           # agent answers, TTFB per request)
"""
import collections
import glob
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def peak_gb(fp):
    m = 0.0
    if not os.path.exists(fp):
        return float("nan")
    for line in open(fp):
        r = re.search(r"footprint=([\d.]+)\s*([KMGT]?)", line)
        if r:
            m = max(m, float(r.group(1)) * {"K": 2**-20, "M": 2**-10, "G": 1, "T": 1024, "": 2**-30}[r.group(2)])
    return m


def load_runs(results):
    runs = collections.defaultdict(list)
    tasks = []
    for f in sorted(glob.glob(os.path.join(results, "*.json"))):
        d = json.load(open(f))
        m = re.match(r"(.*?)(?: r(\d+))?$", d["label"])
        key, rnd = m.group(1), int(m.group(2) or 1)
        reqs = [r for t in d["tasks"] for r in t["requests"]]
        main = [r for r in reqs if r["bytes"] >= 1000]
        aux = [r for r in reqs if r["bytes"] < 1000]
        warm = [r["ttfb"] for r in main[1:]]
        out_tok = sum(int((t.get("usage") or {}).get("outputTokens") or 0) for t in d["tasks"])
        tag = re.sub(r"[ /+#]", "_", d["label"])
        for t in d["tasks"]:
            if t["task"] not in tasks:
                tasks.append(t["task"])
        runs[key].append({
            "round": rnd, "file": f, "total": d["total"], "reqs": len(reqs), "out": out_tok,
            "cold": main[0]["ttfb"] if main else float("nan"),
            "warm": st.median(warm) if warm else float("nan"),
            "aux": st.median([r["ttfb"] for r in aux]) if aux else float("nan"),
            "ms_tok": 1000 * d["total"] / max(1, out_tok),
            "ram": peak_gb(os.path.join(results, f"fp_{tag}.log")),
            "timeouts": sum(1 for t in d["tasks"] if t.get("timeout")),
            "tasks": {t["task"]: t["seconds"] for t in d["tasks"]},
            "raw": d,
        })
    return runs, tasks


def med(xs):
    xs = [x for x in xs if x == x]
    return st.median(xs) if xs else float("nan")


def tables(runs, tasks):
    print("| configuration | total per round, s | total, median | requests | output tokens | cold, s | warm median, s | aux median, s | ms/output token | peak RAM, GB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, rs in runs.items():
        rs = sorted(rs, key=lambda x: x["round"])
        tot = " / ".join(f"{x['total']:.1f}" for x in rs)
        to = sum(x["timeouts"] for x in rs)
        rams = [x["ram"] for x in rs if x["ram"] == x["ram"]]
        ram = "?" if not rams else (f"{min(rams):.0f}" if min(rams) == max(rams) else f"{min(rams):.0f}-{max(rams):.0f}")
        print(f"| {key} | {tot}{' (timeouts ' + str(to) + ')' if to else ''} | **{med([x['total'] for x in rs]):.1f}** | "
              f"{med([x['reqs'] for x in rs]):.0f} | {med([x['out'] for x in rs]):.0f} | {med([x['cold'] for x in rs]):.1f} | "
              f"{med([x['warm'] for x in rs]):.1f} | {med([x['aux'] for x in rs]):.1f} | {med([x['ms_tok'] for x in rs]):.0f} | {ram} |")
    print()
    keys = list(runs)
    print("| task (median, s) | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for t in tasks:
        print(f"| {t} | " + " | ".join(f"{med([x['tasks'].get(t, float('nan')) for x in runs[k]]):.1f}" for k in keys) + " |")


def text_log(runs):
    for key, rs in runs.items():
        for x in sorted(rs, key=lambda r: r["round"]):
            d = x["raw"]
            print(f"\n##### {d['label']}  ({os.path.basename(x['file'])})  total {d['total']:.1f} s")
            for t in d["tasks"]:
                u = t.get("usage") or {}
                print(f"\n--- {t['task']}: {t['seconds']:.1f} s, requests {len(t['requests'])}, in {u.get('inputTokens', '?')} "
                      f"cached {u.get('cacheInputTokens', '?')} out {u.get('outputTokens', '?')}{' TIMEOUT' if t.get('timeout') else ''}")
                print("ttfb: " + " ".join(f"{r['ttfb']:.1f}" for r in t["requests"]))
                if t.get("changes"):
                    print("changes:", t["changes"])
                print(t.get("result", "").rstrip())


if __name__ == "__main__":
    results = os.path.join(HERE, "results")
    if "--results" in sys.argv:
        results = sys.argv[sys.argv.index("--results") + 1]
    runs, tasks = load_runs(results)
    if "--log" in sys.argv:
        text_log(runs)
    else:
        tables(runs, tasks)

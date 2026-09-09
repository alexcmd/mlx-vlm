#!/usr/bin/env python3
"""Посекундно: phys_footprint процесса (максимальный RSS среди pgrep -f PATTERN)
и, если задан порт нашего сервера, статистика кэша. Аргументы: файл, PATTERN, [порт].
"""
import json, subprocess, sys, time, urllib.request

out = open(sys.argv[1], "a")
pattern = sys.argv[2]
port = sys.argv[3] if len(sys.argv) > 3 else None


def pick_pid():
    try:
        pids = subprocess.check_output(["pgrep", "-f", pattern]).decode().split()
    except subprocess.CalledProcessError:
        return None
    best, best_rss = None, -1
    for p in pids:
        try:
            rss = int(subprocess.run(["ps", "-o", "rss=", "-p", p], capture_output=True).stdout or 0)
        except ValueError:
            rss = 0
        if rss > best_rss:
            best, best_rss = p, rss
    return best


pid = pick_pid()
n = 0
while True:
    n += 1
    if pid is None or n % 15 == 0:
        pid = pick_pid() or pid
    fp = "?"
    if pid:
        try:
            t = subprocess.check_output(["/usr/bin/footprint", "-p", pid], stderr=subprocess.DEVNULL, timeout=5).decode()
            fp = next((l.split(":")[1].strip() for l in t.splitlines() if l.strip().startswith("phys_footprint:")), "?")
        except Exception:
            fp = "?"
    st = ""
    if port:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/cache/stats", timeout=2) as r:
                d = json.load(r)
                h = d.get("hybrid") or {}
                st = (f"ckpts={h.get('checkpoints')} blocks={d.get('pool_used')} "
                      f"active={h.get('active_gb')} alloc_cache={h.get('alloc_cache_gb')} peak={h.get('peak_gb')} "
                      f"disk_gb={d.get('disk_bytes', 0) / 2**30:.1f}")
        except Exception:
            st = "stats=?"
    out.write(f"{time.strftime('%H:%M:%S')} pid={pid} footprint={fp} {st}\n")
    out.flush()
    time.sleep(1)

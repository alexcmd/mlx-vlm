#!/bin/bash
# Run the four Junie tasks against one or more targets, REPEATS rounds each,
# through the logging proxy, sampling the server's memory footprint.
#
#   TARGETS="jb server" REPEATS=3 bench/junie-tasks/run.sh
#
# Targets:
#   jb      the installed Junie Local engine (~/.local/share/junie-local,
#           serverctl.sh); started with a cleared APC disk cache before every
#           round and stopped afterwards. Junie profile: $JB_PROFILE
#           (default custom:junie-engine, see profiles/junie-engine.json).
#   server  an OpenAI-compatible server that is already running at
#           $SERVER_UPSTREAM (default 127.0.0.1:8080); its lifecycle (and its
#           cache) is the caller's business. Junie profile: $SERVER_PROFILE
#           (default custom:server-proxy). $SERVER_METRICS_PORT (default the
#           upstream port) is polled for /v1/metrics; $SERVER_PATTERN (default
#           "mlx_vlm.server") selects the process to sample memory from.
# Labels: $JB_LABEL / $SERVER_LABEL (defaults "junie-engine" / "server"),
# each suffixed with " rN". Output: results/*.json, results/fp_*.log and the
# console log; summarize with summarize.py.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python3}"
RES="$HERE/results"; mkdir -p "$RES"
TARGETS="${TARGETS:-jb}"; REPEATS="${REPEATS:-3}"
JB="${JUNIE_LOCAL_DIR:-$HOME/.local/share/junie-local}"; CTL="$JB/current/serverctl.sh"
JB_PROFILE="${JB_PROFILE:-custom:junie-engine}"; JB_LABEL="${JB_LABEL:-junie-engine}"
SERVER_UPSTREAM="${SERVER_UPSTREAM:-127.0.0.1:8080}"; SERVER_PROFILE="${SERVER_PROFILE:-custom:server-proxy}"
SERVER_LABEL="${SERVER_LABEL:-server}"; SERVER_METRICS_PORT="${SERVER_METRICS_PORT:-${SERVER_UPSTREAM##*:}}"
SERVER_PATTERN="${SERVER_PATTERN:-mlx_vlm.server}"
REQS="${QBENCH_REQ_ROOT:-/tmp/junie-tasks}/reqs_$(date +%H%M%S)"; mkdir -p "$REQS-jb" "$REQS-server"
export JUNIE_TASKS_RESULTS="$RES"

tag() { echo "$1" | tr ' /+#' '____'; }
peak() { "$PY" - "$1" <<'PY'
import re, sys
m = 0.0
for line in open(sys.argv[1]):
    r = re.search(r"footprint=([\d.]+)\s*([KMGT]?)", line)
    if r:
        m = max(m, float(r.group(1)) * {"K": 2**-20, "M": 2**-10, "G": 1, "T": 1024, "": 2**-30}[r.group(2)])
print(f"{m:.1f}")
PY
}
sample_start() { rm -f "$1"; ("$PY" "$HERE/fp_sampler.py" "$1" "$2" ${3:-} >/dev/null 2>&1 &); }
sample_stop() { for p in $(pgrep -f "[f]p_sampler.py"); do kill $p; done; sleep 1; echo "  peak RAM: $(peak "$1") GB"; }

start_jb() {
  "$CTL" stop >/dev/null 2>&1; sleep 3; for p in $(pgrep -f "[j]unie-mlx-vlm"); do kill -9 $p; done; sleep 1
  rm -rf "$JB/apc-cache"
  "$CTL" start >/dev/null 2>&1; timeout 300 "$CTL" wait >/dev/null 2>&1
  "$CTL" apply auto_unload_time=36000 >/dev/null 2>&1
  KEY=$("$PY" -c "import json,os;print(json.load(open(os.path.expanduser('$JB/server-config.json')))['api_key'])")
  curl -s -m 600 http://127.0.0.1:19239/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.8-27B-MLX-4bit","messages":[{"role":"user","content":"ok"}],"max_tokens":2}' >/dev/null
}
stop_jb() { "$CTL" stop >/dev/null 2>&1; sleep 3; }

run_jb() {
  local label="$1"; start_jb
  local fp="$RES/fp_$(tag "$label").log"; sample_start "$fp" "junie-mlx-vlm worker"
  QBENCH_REQ_DIR="$REQS-jb" "$PY" "$HERE/real_junie.py" "$label" "$JB_PROFILE" 0
  sample_stop "$fp"; echo "  disk: $(du -sh "$JB/apc-cache" 2>/dev/null | cut -f1)"; stop_jb
}
run_server() {
  local label="$1"
  curl -sf -m 5 "http://$SERVER_UPSTREAM/health" >/dev/null || { echo "server at $SERVER_UPSTREAM is not answering /health" >&2; return 1; }
  local fp="$RES/fp_$(tag "$label").log"; sample_start "$fp" "$SERVER_PATTERN" "$SERVER_METRICS_PORT"
  QBENCH_REQ_DIR="$REQS-server" "$PY" "$HERE/real_junie.py" "$label" "$SERVER_PROFILE" "$SERVER_METRICS_PORT"
  sample_stop "$fp"
}

for p in $(pgrep -f "[o]ai_proxy.py"); do kill $p; done; sleep 1
(QBENCH_REQ_DIR="$REQS-jb" nohup "$PY" "$HERE/oai_proxy.py" 8097 127.0.0.1:19239 > "$REQS-jb/proxy.log" 2>&1 &)
(QBENCH_REQ_DIR="$REQS-server" nohup "$PY" "$HERE/oai_proxy.py" 8098 "$SERVER_UPSTREAM" > "$REQS-server/proxy.log" 2>&1 &)
sleep 2
echo "junie: $(command -v junie) | requests: $REQS-{jb,server} | $(date)"
for r in $(seq 1 "$REPEATS"); do
  for t in $TARGETS; do
    case "$t" in
      jb)     run_jb "$JB_LABEL r$r" ;;
      server) run_server "$SERVER_LABEL r$r" ;;
      *)      echo "unknown target: $t" >&2 ;;
    esac
  done
done
for p in $(pgrep -f "[o]ai_proxy.py"); do kill $p; done
echo "DONE $(date)"

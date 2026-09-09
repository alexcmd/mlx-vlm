#!/usr/bin/env python3
"""Логирующий прокси перед сервером для OpenAI-трафика Junie.

Слушает 127.0.0.1:8098, проксирует на UPSTREAM (по умолчанию 127.0.0.1:8080),
стримит ответ как есть. Для каждого POST на /v1/chat/completions пишет в
QBENCH_REQ_DIR:
  req_NNN.json   тело запроса
  req_NNN.meta   таймлайн: t_first_byte, t_first_content, t_done, число чанков,
                 заголовки запроса (без Authorization), статус ответа
  index.tsv      сводка по строке на запрос

Запуск: QBENCH_REQ_DIR=/tmp/junie-bench/reqs python3 oai_proxy.py [port] [upstream]
"""
import json, os, pathlib, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8098
UP = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1:8080"
OUT = pathlib.Path(os.environ.get("QBENCH_REQ_DIR", "/tmp/junie-bench/reqs"))
OUT.mkdir(parents=True, exist_ok=True)
_n = [0]
_lock = threading.Lock()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _proxy(self, body=None, idx=None):
        t0 = time.time()
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ("host",)}
        conn = http.client.HTTPConnection(UP, timeout=3600)
        conn.request(self.command, self.path, body=body, headers=hdrs)
        r = conn.getresponse()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(k, v)
        chunked = True
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        first = None
        first_content = None
        chunks = 0
        total = 0
        tail = b""
        while True:
            data = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not data:
                break
            now = time.time()
            if first is None:
                first = now
            if first_content is None:
                tail = (tail + data)[-200000:]
                if b'"content":"' in tail or b'"tool_calls"' in tail or b'"reasoning' in tail:
                    first_content = now
            chunks += 1
            total += len(data)
            try:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()
            except BrokenPipeError:
                break
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass
        t_done = time.time()
        if idx is not None:
            meta = {
                "idx": idx, "path": self.path, "status": r.status,
                "t_start": t0, "t_first_byte": (first - t0) if first else None,
                "t_first_content": (first_content - t0) if first_content else None,
                "t_done": t_done - t0, "chunks": chunks, "bytes": total,
                "headers": {k: v for k, v in hdrs.items() if k.lower() != "authorization"},
            }
            (OUT / f"req_{idx:03d}.meta").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            with _lock, open(OUT / "index.tsv", "a") as f:
                f.write(f"{idx}\t{time.strftime('%H:%M:%S', time.localtime(t0))}\t{r.status}\t"
                        f"{meta['t_first_byte'] or 0:.2f}\t{meta['t_first_content'] or 0:.2f}\t{meta['t_done']:.2f}\t{total}\n")

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        idx = None
        if "/chat/completions" in self.path or "/messages" in self.path or "/responses" in self.path:
            with _lock:
                _n[0] += 1
                idx = _n[0]
            try:
                (OUT / f"req_{idx:03d}.json").write_bytes(body)
            except Exception:
                pass
        self._proxy(body, idx)


print(f"proxy :{PORT} -> {UP}, dir {OUT}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

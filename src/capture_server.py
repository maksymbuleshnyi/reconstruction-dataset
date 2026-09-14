#!/usr/bin/env python3
"""Receives rendered frames from the capture page and writes them to disk.

Serves fusion_assembly/ statically (so the capture page is same-origin) and
accepts:

    POST /save   {"object": id, "frames": [{"name": ..., "png": dataURL}, ...],
                  "meta": [...index rows...]}
    POST /done   {"object": id}   -> marks the object finished

Frames land in capture/<object>/<name>.png, index rows accumulate in
capture/<object>/index.json. The page drives everything; this end is dumb on
purpose so a crashed capture can simply be rerun.

  .venv/bin/python capture_server.py 8877
"""
from __future__ import annotations

import base64
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
# CAPTURE_DIR lets a preview render somewhere harmless. The server, not
# make_capture, decides where frames land -- a manual browser run against
# the default server will overwrite the live corpus.
OUT = os.environ.get("CAPTURE_DIR") or os.path.join(HERE, "capture")


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def log_message(self, *a):
        pass

    def _json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n))

    def _ok(self, obj=None):
        body = json.dumps(obj or {"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            j = self._json()
        except Exception:
            self.send_error(400)
            return
        obj = j.get("object", "unknown")
        odir = os.path.join(OUT, obj)
        os.makedirs(odir, exist_ok=True)
        if self.path == "/save":
            for f in j.get("frames", []):
                data = f["png"].split(",", 1)[1]
                with open(os.path.join(odir, f["name"] + ".png"), "wb") as fh:
                    fh.write(base64.b64decode(data))
            ip = os.path.join(odir, "index.json")
            rows = json.load(open(ip)) if os.path.exists(ip) else []
            rows += j.get("meta", [])
            json.dump(rows, open(ip, "w"))
            self._ok({"ok": True, "have": len(rows)})
        elif self.path == "/done":
            open(os.path.join(odir, "DONE"), "w").write("1")
            print(f"[capture] {obj}: done", flush=True)
            self._ok()
        else:
            self.send_error(404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8877
    os.makedirs(OUT, exist_ok=True)
    print(f"[capture] serving {HERE} on http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

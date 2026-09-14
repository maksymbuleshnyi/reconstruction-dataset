#!/usr/bin/env python3
"""Serve the demo. The browser needs a web server to fetch the scene files;
opening demo/index.html straight from disk is blocked by CORS.

    python3 serve.py            # http://localhost:8000/demo/
    python3 serve.py 9000       # a different port
"""
import http.server
import os
import socketserver
import sys
import webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    url = "http://localhost:%d/demo/" % PORT
    print("serving on %s   (ctrl-c to stop)" % url)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    httpd.serve_forever()

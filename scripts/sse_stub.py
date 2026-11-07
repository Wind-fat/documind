"""本地 SSE 桩：固定节奏的 OpenAI 兼容上游，用于度量本服务的流式链路。

为什么需要它：真网关（本机 magpie → qwen3.8-flash）是**推理模型**，正文之前要思考
10-120 秒，且波动极大（实测同一条问题 22s / 127s）。那种上游量出来的是"模型的
思考时间"，不是"DocuMind 有没有真流式"。这个桩把上游固定成 150ms/块：

    ./.venv/Scripts/python.exe scripts/sse_stub.py --port 8097 --chunk-ms 150 --chunks 8

然后把服务指过来：
    DOCUMIND_LLM_BASE_URL=http://127.0.0.1:8097/v1 DOCUMIND_RERANK_BACKEND=lexical \\
        ./.venv/Scripts/python.exe -m uvicorn documind.app:app --port 8098

再跑 scripts/measure_stream.py。桩只做两件事：流式回 N 块、非流式回整包。
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER = {
    "answer": "公司注册资本为人民币一亿元 [1]；股东按认缴比例出资 [2]。",
    "citations": [
        {"ref": 1, "quote": "公司注册资本为人民币一亿元"},
        {"ref": 2, "quote": "由股东按认缴比例出资"},
    ],
    "not_found": False,
}

CHUNK_MS = 150
CHUNKS = 8


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args, **kwargs) -> None:  # type: ignore[override]  # 安静点
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        payload = json.dumps(ANSWER, ensure_ascii=False)
        usage = {"prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240}

        if not body.get("stream"):
            # 同样的"生成耗时"：流式要等 N 块，非流式也得等 N 块，否则两边的总时长不可比
            time.sleep(CHUNK_MS * CHUNKS / 1000)
            data = json.dumps(
                {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion",
                    "model": body.get("model", "stub"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": payload}, "finish_reason": "stop"}],
                    "usage": usage,
                },
                ensure_ascii=False,
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        size = max(len(payload) // CHUNKS + 1, 1)
        for i in range(0, len(payload), size):
            time.sleep(CHUNK_MS / 1000)
            frame = {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "model": body.get("model", "stub"),
                "choices": [{"index": 0, "delta": {"content": payload[i : i + size]}, "finish_reason": None}],
            }
            self._write(f"data: {json.dumps(frame, ensure_ascii=False)}\n\n")
        time.sleep(CHUNK_MS / 1000)
        self._write(
            "data: "
            + json.dumps({"id": "chatcmpl-stub", "choices": [], "usage": usage}, ensure_ascii=False)
            + "\n\n"
        )
        self._write("data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _write(self, text: str) -> None:
        raw = text.encode()
        self.wfile.write(f"{len(raw):X}\r\n".encode() + raw + b"\r\n")
        self.wfile.flush()


def serve(port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--chunk-ms", type=int, default=150)
    ap.add_argument("--chunks", type=int, default=8)
    args = ap.parse_args()
    global CHUNK_MS, CHUNKS
    CHUNK_MS, CHUNKS = args.chunk_ms, args.chunks
    srv = serve(args.port)
    print(f"stub 在 http://127.0.0.1:{args.port}/v1/chat/completions（{args.chunks} 块 × {args.chunk_ms}ms）", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
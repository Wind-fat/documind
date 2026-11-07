"""实测 /query 与 /query/stream 的延迟：首字节（TTFT）与总时长（P1-05）。

用法（服务先起来）：
    ./.venv/Scripts/python.exe scripts/measure_stream.py --runs 3 \
        --question "公司注册资本是多少？"

判定口径（脚本会自己算，不靠嘴说）：
- `evidence_ms`  ：从发出请求到收到第一个 SSE 事件（= 首字节，检索完成即发）
- `first_token_ms`：到第一个 token 事件（模型开始吐正文）
- `answer_ms`    ：到 answer 事件（= 端到端总时长）
- 真流式的门槛：first_token_ms < answer_ms * 0.30

注意：这里量的是**服务端到客户端**的真实到达时间，不是代码里的推算。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import httpx


def _parse_sse_lines(resp: httpx.Response):
    """边读边解，收到就把 (event, payload, 到达时刻) 交出去。"""
    name, data = "", ""
    t0 = time.perf_counter()
    for line in resp.iter_lines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data += line[5:].strip()
        elif not line.strip() and name:
            yield name, json.loads(data), (time.perf_counter() - t0) * 1000
            name, data = "", ""


def measure_stream(client: httpx.Client, question: str, strategy: str, selfcheck: bool) -> dict:
    t0 = time.perf_counter()
    marks: dict[str, float] = {}
    token_times: list[float] = []
    answer = None
    with client.stream(
        "GET",
        "/query/stream",
        params={"question": question, "strategy": strategy, "selfcheck": str(selfcheck).lower()},
    ) as resp:
        resp.raise_for_status()
        for name, payload, ms in _parse_sse_lines(resp):
            marks.setdefault(name, ms)
            if name == "token":
                token_times.append(ms)
            if name == "answer":
                answer = payload
    total = (time.perf_counter() - t0) * 1000
    return {
        "evidence_ms": marks.get("evidence", -1),
        "first_token_ms": marks.get("token", -1),
        "last_token_ms": token_times[-1] if token_times else -1,
        "answer_ms": marks.get("answer", -1),
        "wall_ms": total,
        "token_events": len(token_times),
        # 0 表示所有 token 挤在最后一刻（伪流式）；越大说明越"边生成边到"
        "token_spread_ms": (token_times[-1] - token_times[0]) if len(token_times) > 1 else 0,
        "token_times_ms": [round(x, 1) for x in token_times],
        "refused": bool(answer and answer.get("refused")),
        "text_len": len(answer["text"]) if answer else 0,
        "streaming_debug": (answer or {}).get("debug", {}).get("streaming", {}),
        "chat_ms": (answer or {}).get("generation_ms"),
    }


def measure_query(client: httpx.Client, question: str, strategy: str, selfcheck: bool) -> dict:
    t0 = time.perf_counter()
    r = client.post(
        "/query",
        json={"question": question, "strategy": strategy, "selfcheck": selfcheck},
    )
    r.raise_for_status()
    body = r.json()
    return {
        "total_ms": (time.perf_counter() - t0) * 1000,
        "server_total_ms": body.get("latency_ms"),
        "refused": body.get("refused"),
        "text_len": len(body.get("text") or ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--question", default="公司注册资本是多少？")
    ap.add_argument("--strategy", default="structure")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    # trust_env=False：本机网关/服务不能被系统代理接管（Clash 会把 127.0.0.1 也接走）
    client = httpx.Client(base_url=args.base_url, timeout=180, trust_env=False, headers={"X-User": args.user})

    rows = []
    for i in range(args.runs):
        # 先流式后非流式交替，避免暖机顺序影响第一轮
        s = measure_stream(client, args.question, args.strategy, args.selfcheck)
        q = measure_query(client, args.question, args.strategy, args.selfcheck)
        rows.append({"run": i + 1, "stream": s, "query": q})
        print(
            f"run{i + 1}: stream evidence={s['evidence_ms']:.0f}ms "
            f"first_token={s['first_token_ms']:.0f}ms answer={s['answer_ms']:.0f}ms "
            f"tokens={s['token_events']} | query total={q['total_ms']:.0f}ms"
        )

    ev = statistics.median(r["stream"]["evidence_ms"] for r in rows)
    ft = statistics.median(r["stream"]["first_token_ms"] for r in rows)
    am = statistics.median(r["stream"]["answer_ms"] for r in rows)
    qt = statistics.median(r["query"]["total_ms"] for r in rows)
    sp = statistics.median(r["stream"]["token_spread_ms"] for r in rows)
    tj = statistics.median(r["stream"]["token_events"] for r in rows)
    ratio = ft / am if am else float("inf")
    dr = qt / am if am else float("inf")

    print("\n== 中位数 ==")
    print(f"stream evidence(TTFB) = {ev:.0f} ms")
    print(f"stream first_token    = {ft:.0f} ms（占端到端 {ratio:.1%}）")
    print(f"stream answer(total)  = {am:.0f} ms")
    print(f"POST /query total     = {qt:.0f} ms（= 非流式的端到端）")
    print(f"token 事件 {tj:.0f} 个，首尾跨度 {sp:.0f} ms（=0 说明攒完一次吐）")
    print(f"非流式 / 流式 总时长比 = {dr:.2f}x")
    verdict = "PASS" if ft < am * 0.30 else "FAIL"
    print(f"[{verdict}] first_token < 30% × total ?  {ft:.0f} < {am * 0.30:.0f}")
    print(f"最后一轮 token 到达时刻(ms): {rows[-1]['stream']['token_times_ms']}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "median": {"evidence_ms": ev, "first_token_ms": ft,
                                                "answer_ms": am, "query_total_ms": qt,
                                                "token_spread_ms": sp, "token_events": tj,
                                                "ttft_ratio": ratio}, "verdict": verdict}, f,
                      ensure_ascii=False, indent=2)
        print(f"明细已写入 {args.json_out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
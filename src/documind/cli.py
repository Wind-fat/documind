"""命令行入口：ingest / ask / serve / seed / eval / demo。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .db import get_session_factory, init_db, upsert_user


async def _ingest(args: argparse.Namespace) -> None:
    from .pipeline import ingest_document

    await init_db()
    async with get_session_factory()() as s:
        for p in args.paths:
            res = await ingest_document(
                s,
                p,
                acl_roles=args.acl.split(","),
                title=args.title,
            )
            print(json.dumps(res, ensure_ascii=False, indent=2))


async def _ask(args: argparse.Namespace) -> None:
    from .pipeline import ask

    await init_db()
    async with get_session_factory()() as s:
        await upsert_user(s, args.user, args.role)
        ans = await ask(
            s,
            args.question,
            role=args.role,
            conversation_id=args.conversation,
            strategy=args.strategy,
            selfcheck=not args.no_selfcheck,
        )
        print(f"\n{'='*8} 回答 {'='*8}\n{ans.text}\n")
        if ans.citations:
            print("证据：")
            for c in ans.citations:
                print(f"  [{c.ref}] 《{c.doc_title}》 第{c.page_start}页 {c.section_path}")
                print(f"      “{c.quote[:80]}”")
        else:
            print(f"（未给出引用 → 拒答: {ans.reason}）")
        print(
            f"\n延迟 {ans.latency_ms}ms · 检索 {ans.retrieval_ms}ms · 生成 {ans.generation_ms}ms "
            f"· token {ans.prompt_tokens}+{ans.completion_tokens} · 成本 ${ans.cost_usd:.6f}"
        )
        if ans.rewritten_query:
            print(f"改写后 query: {ans.rewritten_query}")


async def _seed(args: argparse.Namespace) -> None:
    await init_db()
    async with get_session_factory()() as s:
        for name, role in [("alice", "employee"), ("bob", "finance"), ("carol", "hr"), ("admin", "admin")]:
            await upsert_user(s, name, role)
    print("seeded users: alice(employee) bob(finance) carol(hr) admin(admin)")


async def _eval(args: argparse.Namespace) -> None:
    from .evals.runner import run_eval

    await init_db()
    async with get_session_factory()() as s:
        report = await run_eval(
            s, dataset=args.dataset, strategies=args.strategies, limit=args.limit, concurrency=args.concurrency
        )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        print(f"\n报告：{report['report_path']}")


async def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    await init_db()
    uvicorn.run("documind.app:app", host=args.host, port=args.port, reload=args.reload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="documind", description="DocuMind CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="入库文档")
    p.add_argument("paths", nargs="+")
    p.add_argument("--acl", default="all", help="可见角色，逗号分隔，如 all 或 finance,admin")
    p.add_argument("--title", default=None)
    p.set_defaults(func=_ingest)

    p = sub.add_parser("ask", help="提问")
    p.add_argument("question")
    p.add_argument("--user", default="alice")
    p.add_argument("--role", default="employee")
    p.add_argument("--strategy", default="structure", choices=["fixed", "semantic", "structure", "parent_child"])
    p.add_argument("--conversation", default=None)
    p.add_argument("--no-selfcheck", action="store_true")
    p.set_defaults(func=_ask)

    p = sub.add_parser("eval", help="跑评测")
    p.add_argument("--dataset", default="default")
    p.add_argument("--strategies", nargs="*", default=["fixed", "semantic", "structure", "parent_child"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.set_defaults(func=_eval)

    p = sub.add_parser("seed", help="建演示用户")
    p.set_defaults(func=_seed)

    p = sub.add_parser("serve", help="起 API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    asyncio.run(args.func(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
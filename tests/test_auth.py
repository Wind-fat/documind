"""身份鉴权测试：P0-01（X-User 可伪造）的回归网。"""

from __future__ import annotations

import httpx
import pytest

from documind.app import app
from documind.auth import sign, verify
from documind.config import get_settings
from documind.db import upsert_user


@pytest.fixture()
async def client(db):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


def test_sign_verify_roundtrip():
    tok = sign("alice", "employee", ttl_s=60)
    assert verify(tok) == ("alice", "employee")


def test_tampered_token_rejected():
    tok = sign("alice", "employee", ttl_s=60)
    name, role, exp, mac = tok.split(":")
    forged = f"{name}:admin:{exp}:{mac}"  # 改角色不改签名
    assert verify(forged) is None


def test_expired_token_rejected():
    assert verify(sign("alice", "employee", ttl_s=-1)) is None


def test_garbage_token_rejected():
    assert verify("not-a-token") is None


async def test_dev_header_still_works(client):
    assert get_settings().env == "dev"
    r = await client.get("/documents", headers={"X-User": "alice"})
    assert r.status_code in (200, 404)  # 200 空列表或已建用户


async def test_unknown_user_is_401_not_autocreated(client):
    r = await client.get("/documents", headers={"X-User": "nobody-here"})
    assert r.status_code == 401


async def test_prod_rejects_raw_header_and_accepts_token(client, db, monkeypatch):
    await upsert_user(db, "alice", "employee")
    monkeypatch.setattr(get_settings(), "env", "prod")

    r = await client.get("/documents", headers={"X-User": "admin"})
    assert r.status_code == 401, "生产环境裸头必须被拒"

    tok = sign("alice", "employee")
    r2 = await client.get("/documents", headers={"Authorization": f"Bearer {tok}"})
    assert r2.status_code == 200


async def test_forged_admin_token_is_401(client, db, monkeypatch):
    await upsert_user(db, "alice", "employee")
    monkeypatch.setattr(get_settings(), "env", "prod")
    tok = sign("alice", "employee")
    _, _, exp, _ = tok.split(":")
    r = await client.get("/documents", headers={"Authorization": f"Bearer alice:admin:{exp}:deadbeef"})
    assert r.status_code == 401


async def test_login_issues_token_and_rejects_role_escalation(client, db):
    await upsert_user(db, "alice", "employee")
    ok = await client.post("/auth/login", json={"name": "alice"})
    assert ok.status_code == 200 and verify(ok.json()["token"]) == ("alice", "employee")

    bad = await client.post("/auth/login", json={"name": "alice", "role": "admin"})
    assert bad.status_code == 403

    unknown = await client.post("/auth/login", json={"name": "ghost"})
    assert unknown.status_code == 401


async def test_users_endpoint_requires_admin(client):
    assert (await client.get("/users")).status_code == 401
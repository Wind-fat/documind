"""身份与角色：签名令牌，而不是"相信请求头"。

架构复核的 P0-01：原来 `current_user` 直接读 `X-User` 头、用户不存在还自动建号，
一条 `curl -H 'X-User: admin'` 就能拿到管理员视角——ACL 在检索阶段做得再对也没意义。
现在身份只有两种来源：

1. `Authorization: Bearer <token>`，token = `name:role:exp:hmac_sha256(name|role|exp, session_secret)`
2. 开发环境（`DOCUMIND_ENV=dev`）允许裸 `X-User`，但**不自动建号、且角色以库里为准**，
   这样本地演示不用先登录，而生产（env=prod）下裸头一律 401。

生产接入真实 IdP 时只替换 `issue()`：签名与校验逻辑不用动。
"""

from __future__ import annotations

import hashlib
import hmac
import time

from .config import get_settings


def _secret() -> bytes:
    return get_settings().session_secret.encode("utf-8")


def sign(name: str, role: str, ttl_s: int = 12 * 3600) -> str:
    exp = int(time.time()) + ttl_s
    payload = f"{name}|{role}|{exp}"
    mac = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"{name}:{role}:{exp}:{mac}"


def verify(token: str) -> tuple[str, str] | None:
    """返回 (name, role)；签名不对/过期/格式坏 → None。"""
    parts = token.split(":")
    if len(parts) != 4:
        return None
    name, role, exp_s, mac = parts
    try:
        exp = int(exp_s)
    except ValueError:
        return None
    if exp < int(time.time()):
        return None
    expect = hmac.new(_secret(), f"{name}|{role}|{exp}".encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(mac, expect):
        return None
    return name, role


def dev_header_allowed() -> bool:
    return get_settings().env != "prod"
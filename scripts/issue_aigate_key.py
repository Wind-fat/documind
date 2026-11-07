"""在项目三（aigate 网关）上为一个新项目签发 key，并写进本地文件供 DocuMind 使用。

目的：证明 DocuMind 的模型层是可替换的底座（同一个 ask 链路，只改 base_url/key）。
key 明文只落文件、不打印。
"""

import json
import sys
from pathlib import Path

import httpx

AIGATE = "http://127.0.0.1:8090"
AIGATE_ENV = Path("D:/code/aigate/.env")
OUT = Path("D:/code/documind/data/aigate.key")
OUT.parent.mkdir(parents=True, exist_ok=True)

env = {}
for line in AIGATE_ENV.read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

c = httpx.Client(trust_env=False, timeout=30)
h = {"X-Admin-Token": env["AIGATE_ADMIN_TOKEN"]}

projects = c.get(f"{AIGATE}/admin/v1/projects", headers=h)
print("projects:", projects.status_code)
items = projects.json().get("items", [])
for p in items[:5]:
    print(" -", p.get("id"), p.get("slug"), p.get("status"))
if not items:
    sys.exit("没有可用项目，先跑 scripts/seed.py")
pid = items[0]["id"]

r = c.post(
    f"{AIGATE}/admin/v1/keys",
    params={"project_id": pid, "name": "documind-demo", "environment": "dev"},
    headers=h,
)
print("issue key:", r.status_code)
body = r.json()
key = body.get("api_key") or body.get("plaintext") or ""
OUT.write_text(key, encoding="utf-8")
print("key written:", OUT, "len=", len(key), "prefix=", body.get("prefix"))
print("model route:", json.dumps({k: v for k, v in body.items() if k != "api_key"}, ensure_ascii=False)[:300])
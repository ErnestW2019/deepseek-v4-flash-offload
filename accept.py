#!/usr/bin/env python
"""服务化验收：等 ``/health`` 就绪 -> 并发 ``/v1/completions`` -> 聚合 tok/s 判定。

用法（服务已由 ``serve.sh`` 起好；仅用标准库，无需额外依赖）:

    python accept.py

环境变量:
    BASE_URL   默认 http://127.0.0.1:8000
    MODEL      默认 DeepSeek-V4-Flash
    N          并发请求数（默认 8）
    MAX_TOKENS 每请求生成长度（默认 32）
    HEALTH_TIMEOUT 等就绪上限秒数（默认 1800）
"""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "DeepSeek-V4-Flash")
N = int(os.environ.get("N", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32"))
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "Explain what a GPU is in one sentence:",
    "Write a Python function that adds two numbers:",
    "What is the boiling point of water?",
    "Name three primary colors:",
    "Translate 'hello world' into French:",
    "What year did the first man land on the moon?",
]


def _post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_health(timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit(f"[accept] /health 在 {timeout:.0f}s 内未就绪")


def one(i):
    payload = {
        "model": MODEL,
        "prompt": PROMPTS[i % len(PROMPTS)],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    t = time.time()
    out = _post("/v1/completions", payload)
    return time.time() - t, out["usage"]["completion_tokens"], out["choices"][0]["text"]


def main():
    print(f"[accept] base={BASE} model={MODEL} n={N} max_tokens={MAX_TOKENS}", flush=True)
    print(f"[accept] health OK in {wait_health(HEALTH_TIMEOUT):.1f}s", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N) as ex:
        res = list(ex.map(one, range(N)))
    wall = time.time() - t0

    ntok = sum(r[1] for r in res)
    print(f"[accept] {N} 并发、{ntok} tokens / {wall:.2f}s -> {ntok / wall:.2f} tok/s", flush=True)
    for i, (dt, nt, txt) in enumerate(res[:3]):
        print(f"[accept] #{i} {dt:.2f}s ({nt} tok): {txt!r}", flush=True)

    ok = ntok > 0 and all(r[1] > 0 for r in res)
    print(f"[{'PASS' if ok else 'FAIL'}] 服务验收", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Concurrent load test targeting a remote dev server.

Defaults to http://35.77.244.101:5002 and follows the same
flow as tests/load_test_ai_chat.py:
  1) GET  /start/ai-facilitator (sets cookie)
  2) POST /save_decision        (deterministic payload)
  3) POST /setup_chat           (deterministic payload)
  4) POST /ai_chat              (N turns; optional)

Usage examples:
  python -m tests.remote_load_test_ai_chat --concurrency 50 --iterations 50 --turns 1 --jitter-ms 500
  python -m tests.remote_load_test_ai_chat --base http://127.0.0.1:5002 --turns 0

Note: This script does not execute here; it's for external use.
"""

import argparse
import asyncio
import random
import time
from typing import List

import httpx

from common import make_user_payload, make_setup_payload, percentiles


def _err_excerpt(resp) -> str:
    try:
        j = resp.json()
        if isinstance(j, dict):
            err = j.get('error') or j.get('message') or j
            return str(err)[:200]
        return str(j)[:200]
    except Exception:
        t = getattr(resp, 'text', '')
        return (t or '')[:200]


async def one_user(base_url: str, turns: int, results: List[float], errors: List[str], user_id: int, jitter_ms: int = 0):
    timeout = httpx.Timeout(60.0, connect=15.0)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, follow_redirects=True, headers={"Accept-Language": "ja"}) as client:
        try:
            t0 = time.perf_counter()

            r = await client.get('/start/ai-facilitator')
            if r.status_code not in (200, 302):
                errors.append(f"start:{r.status_code}")
                return

            r = await client.post('/save_decision', json=make_user_payload())
            try:
                ok = (r.status_code == 200 and (r.json().get('success') is True))
            except Exception:
                ok = False
            if not ok:
                errors.append(f"save_decision:{r.status_code}:{_err_excerpt(r)}")
                return

            r = await client.post('/setup_chat', json=make_setup_payload())
            try:
                ok = (r.status_code == 200 and (r.json().get('success') is True))
            except Exception:
                ok = False
            if not ok:
                errors.append(f"setup_chat:{r.status_code}:{_err_excerpt(r)}")
                return

            # Optional chat turns
            for i in range(turns):
                if jitter_ms and jitter_ms > 0:
                    await asyncio.sleep(random.random() * (jitter_ms / 1000.0))
                rr = await client.post('/ai_chat', json={'message': f'ユーザー{user_id}のターン{i+1}メッセージです。'})
                try:
                    ok = (rr.status_code == 200 and (rr.json().get('success') is True))
                except Exception:
                    ok = False
                if not ok:
                    errors.append(f"ai_chat:{rr.status_code}:{_err_excerpt(rr)}")
                    return

            t1 = time.perf_counter()
            results.append((t1 - t0) * 1000.0)
        except Exception as e:
            errors.append(f"exc:{type(e).__name__}:{e}")


async def run_load(base: str, concurrency: int, turns: int, iterations: int, jitter_ms: int) -> None:
    results: List[float] = []
    errors: List[str] = []

    launched = 0
    while launched < iterations:
        wave = min(concurrency, iterations - launched)
        tasks = [one_user(base, turns, results, errors, launched + i + 1, jitter_ms=jitter_ms) for i in range(wave)]
        await asyncio.gather(*tasks)
        launched += wave

    ok = len(results)
    err = len(errors)
    total = ok + err
    print(f"\n===== Remote Load Test Summary =====")
    print(f"Base:        {base}")
    print(f"Total users: {total}")
    print(f"Success:     {ok}")
    print(f"Errors:      {err}")
    if err:
        from collections import Counter
        c = Counter(errors)
        print("Top errors:")
        for k, v in c.most_common(5):
            print(f"  {k}: {v}")

    if results:
        p = percentiles(results, (50, 90, 95, 99))
        print("Latency (ms) for full flow (start -> setup -> turns):")
        print(f"  p50: {p[50]:.1f}  p90: {p[90]:.1f}  p95: {p[95]:.1f}  p99: {p[99]:.1f}")
        print(f"  min: {min(results):.1f}  max: {max(results):.1f}  avg: {sum(results)/len(results):.1f}")
    print("===================================\n")


def main():
    ap = argparse.ArgumentParser(description='Concurrent remote load test for /ai_chat flow')
    ap.add_argument('--base', default='http://35.77.244.101:5002', help='Base URL (e.g., http://IP:PORT)')
    ap.add_argument('--concurrency', type=int, default=50, help='Concurrent users in a wave')
    ap.add_argument('--iterations', type=int, default=50, help='Total user flows to run')
    ap.add_argument('--turns', type=int, default=0, help='Chat turns per user (0 to skip /ai_chat)')
    ap.add_argument('--jitter-ms', type=int, default=0, help='Random delay up to N ms before each /ai_chat call')
    args = ap.parse_args()

    asyncio.run(run_load(args.base, args.concurrency, args.turns, args.iterations, args.jitter_ms))


if __name__ == '__main__':
    main()


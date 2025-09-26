#!/usr/bin/env python3
import argparse
import asyncio
import json
import time
import random
from typing import List, Tuple

import httpx

from common import make_user_payload, make_setup_payload, percentiles


def _err_excerpt(resp) -> str:
    try:
        j = resp.json()
        # common error envelope from our API
        if isinstance(j, dict):
            err = j.get('error') or j.get('message') or j
            return str(err)[:200]
        return str(j)[:200]
    except Exception:
        t = getattr(resp, 'text', '')
        return (t or '')[:200]


async def one_user(base_url: str, turns: int, results: List[float], errors: List[str], user_id: int, jitter_ms: int = 0):
    timeout = httpx.Timeout(60.0, connect=30.0)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, follow_redirects=True) as client:
        try:
            t0 = time.perf_counter()
            # 1) Start session (sets cookie)
            r = await client.get('/start/ai-facilitator')
            if r.status_code not in (200, 302):
                errors.append(f"start:{r.status_code}")
                return

            # 2) Save initial decision
            r = await client.post('/save_decision', json=make_user_payload())
            ok = False
            try:
                ok = (r.status_code == 200 and (r.json().get('success') is True))
            except Exception:
                ok = False
            if not ok:
                errors.append(f"save_decision:{r.status_code}:{_err_excerpt(r)}")
                return

            # 3) Setup chat
            r = await client.post('/setup_chat', json=make_setup_payload())
            ok = False
            try:
                ok = (r.status_code == 200 and (r.json().get('success') is True))
            except Exception:
                ok = False
            if not ok:
                errors.append(f"setup_chat:{r.status_code}:{_err_excerpt(r)}")
                return

            # 4) Optional: do N chat turns
            for i in range(turns):
                if jitter_ms and jitter_ms > 0:
                    # Spread requests to avoid spikes at the LLM endpoint
                    await asyncio.sleep(random.random() * (jitter_ms / 1000.0))
                rr = await client.post('/ai_chat', json={'message': f'ユーザー{user_id}のターン{i+1}メッセージです。'} )
                ok = False
                try:
                    ok = (rr.status_code == 200 and (rr.json().get('success') is True))
                except Exception:
                    ok = False
                if not ok:
                    errors.append(f"ai_chat:{rr.status_code}:{_err_excerpt(rr)}")
                    return

            t1 = time.perf_counter()
            results.append((t1 - t0) * 1000.0)  # ms
        except Exception as e:
            errors.append(f"exc:{type(e).__name__}:{e}")


async def run_load(base: str, concurrency: int, turns: int, iterations: int, jitter_ms: int) -> None:
    results: List[float] = []
    errors: List[str] = []

    tasks = []
    # Spread iterations across waves of concurrency
    users = max(iterations, concurrency)
    launched = 0
    while launched < iterations:
        wave = min(concurrency, iterations - launched)
        tasks = [one_user(base, turns, results, errors, launched + i + 1, jitter_ms=jitter_ms) for i in range(wave)]
        await asyncio.gather(*tasks)
        launched += wave

    ok = len(results)
    err = len(errors)
    total = ok + err
    print(f"\n===== Load Test Summary =====")
    print(f"Total users: {total}")
    print(f"Success:     {ok}")
    print(f"Errors:      {err}")
    if err:
        # Show top few error types
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
    print("============================\n")


def main():
    ap = argparse.ArgumentParser(description='Concurrent load test for /ai_chat flow')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', default='5003')
    ap.add_argument('--concurrency', type=int, default=50, help='Concurrent users in a wave')
    ap.add_argument('--iterations', type=int, default=50, help='Total user flows to run')
    ap.add_argument('--turns', type=int, default=0, help='Chat turns per user (0 to skip /ai_chat)')
    ap.add_argument('--jitter-ms', type=int, default=0, help='Random delay up to N ms before each /ai_chat call')
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    asyncio.run(run_load(base, args.concurrency, args.turns, args.iterations, args.jitter_ms))


if __name__ == '__main__':
    main()

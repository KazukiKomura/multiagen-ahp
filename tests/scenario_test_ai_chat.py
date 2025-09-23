#!/usr/bin/env python3
import argparse
import sys
import time
import json
import requests
from urllib.parse import urljoin

from common import make_user_payload, make_setup_payload


def run_scenario(base: str, skip_llm: bool = False) -> int:
    s = requests.Session()

    # 1) Initialize session
    r = s.get(urljoin(base, '/start/ai-facilitator'), allow_redirects=True, timeout=10)
    if r.status_code not in (200, 302):
        print(f"[ERROR] start failed: {r.status_code}")
        return 1

    # 2) Save initial decision (primes participant opinions)
    payload = make_user_payload()
    r = s.post(urljoin(base, '/save_decision'), json=payload, timeout=10)
    if r.status_code != 200 or not r.json().get('success'):
        print(f"[ERROR] save_decision failed: {r.status_code} {r.text}")
        return 1

    # 3) Setup chat (server creates initial bubbles)
    setup = make_setup_payload()
    r = s.post(urljoin(base, '/setup_chat'), json=setup, timeout=10)
    if r.status_code != 200:
        print(f"[ERROR] setup_chat http: {r.status_code} {r.text}")
        return 1
    jr = r.json()
    if not jr.get('success'):
        print(f"[ERROR] setup_chat api: {jr}")
        return 1
    messages = jr.get('messages', [])
    print(f"[INFO] initial bubbles: {len(messages)}")

    # 4) Optional: one chat turn (requires OpenAI access in the app)
    if not skip_llm:
        r = s.post(urljoin(base, '/ai_chat'), json={'message': 'テストです。要点を1つ質問してください。'}, timeout=30)
        if r.status_code != 200:
            print(f"[ERROR] ai_chat http: {r.status_code} {r.text}")
            return 1
        jr = r.json()
        if not jr.get('success'):
            print(f"[ERROR] ai_chat api: {jr}")
            return 1
        print(f"[INFO] ai turn: {jr.get('turn')}, len={len(jr.get('message',''))}")

    print("[OK] scenario completed")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description='Scenario test for AI chat endpoints')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', default='5002')
    p.add_argument('--skip-llm', action='store_true', help='Skip /ai_chat turn (no OpenAI required)')
    args = p.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    return run_scenario(base, skip_llm=args.skip_llm)


if __name__ == '__main__':
    sys.exit(main())


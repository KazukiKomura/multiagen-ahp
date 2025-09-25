#!/usr/bin/env python3
"""
簡単な負荷テストシミュレーター
50人のユーザーが3回のチャットやり取りを行う場合の応答時間をシミュレート
"""

import asyncio
import httpx
import time
import random
import statistics
from typing import List, Dict, Any
import argparse


async def simulate_chat_session(client: httpx.AsyncClient, user_id: int, base_url: str) -> Dict[str, Any]:
    """単一ユーザーのチャットセッションをシミュレート"""
    start_time = time.time()
    
    try:
        # 1. セッション開始
        r = await client.get("/start/ai-facilitator")
        if r.status_code not in (200, 302):
            return {"success": False, "error": f"Start failed: {r.status_code}", "duration": 0}
        
        # 2. ユーザー情報保存
        user_payload = {
            "decision": "一次通過",
            "weights": {
                "学業成績": 20,
                "基礎能力テスト": 20,
                "実践経験": 20,
                "推薦・評価": 20,
                "志望動機・フィット": 20
            },
            "reasoning": f"ユーザー{user_id}の判断理由です。",
            "timestamp": int(time.time() * 1000)
        }
        
        r = await client.post("/save_decision", json=user_payload)
        if r.status_code != 200:
            return {"success": False, "error": f"Save decision failed: {r.status_code}", "duration": 0}
        
        # 3. チャット設定
        setup_payload = {
            "decision": "一次通過",
            "weights": {
                "学業成績": 20,
                "基礎能力テスト": 20,
                "実践経験": 20,
                "推薦・評価": 20,
                "志望動機・フィット": 20
            }
        }
        
        r = await client.post("/setup_chat", json=setup_payload)
        if r.status_code != 200:
            return {"success": False, "error": f"Setup chat failed: {r.status_code}", "duration": 0}
        
        # 4. 3回のチャットターン
        for turn in range(3):
            # リアルなユーザー行動をシミュレート（入力時間）
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            chat_payload = {
                "message": f"ユーザー{user_id}のターン{turn + 1}メッセージです。どのような判断をすべきでしょうか。"
            }
            
            r = await client.post("/ai_chat", json=chat_payload)
            if r.status_code != 200:
                return {"success": False, "error": f"Turn {turn + 1} failed: {r.status_code}", "duration": 0}
            
            # レスポンス確認
            try:
                result = r.json()
                if not result.get("success"):
                    return {"success": False, "error": f"Turn {turn + 1} API error: {result}", "duration": 0}
            except Exception as e:
                return {"success": False, "error": f"Turn {turn + 1} JSON error: {str(e)}", "duration": 0}
        
        duration = (time.time() - start_time) * 1000  # ミリ秒
        return {"success": True, "duration": duration, "user_id": user_id}
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        return {"success": False, "error": str(e), "duration": duration, "user_id": user_id}


async def run_load_test(concurrent_users: int = 50, base_url: str = "http://35.77.244.101:5002"):
    """負荷テストを実行"""
    print(f"🚀 負荷テスト開始: {concurrent_users}人の同時ユーザー")
    print(f"📍 対象サーバー: {base_url}")
    print(f"📝 各ユーザーは3回のチャットターンを実行\n")
    
    start_time = time.time()
    
    # HTTPクライアントを作成（接続プールを使用）
    timeout = httpx.Timeout(60.0, connect=15.0)
    
    async with httpx.AsyncClient(
        base_url=base_url, 
        timeout=timeout, 
        follow_redirects=True,
        headers={"Accept-Language": "ja"}
    ) as client:
        # 全ユーザーのタスクを並行実行
        tasks = [
            simulate_chat_session(client, user_id, base_url) 
            for user_id in range(concurrent_users)
        ]
        
        print("⏳ テスト実行中...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 結果分析
    total_time = time.time() - start_time
    successes = [r for r in results if isinstance(r, dict) and r.get("success", False)]
    failures = [r for r in results if isinstance(r, dict) and not r.get("success", False)]
    exceptions = [r for r in results if isinstance(r, Exception)]
    
    success_durations = [r["duration"] for r in successes]
    
    print("\n" + "="*60)
    print("📊 負荷テスト結果")
    print("="*60)
    print(f"🎯 対象サーバー: {base_url}")
    print(f"👥 同時ユーザー数: {concurrent_users}")
    print(f"⏱️  総テスト時間: {total_time:.1f}秒")
    print(f"✅ 成功: {len(successes)}")
    print(f"❌ 失敗: {len(failures)}")
    print(f"💥 例外: {len(exceptions)}")
    
    if success_durations:
        print(f"\n📈 応答時間統計 (ミリ秒):")
        print(f"   平均: {statistics.mean(success_durations):.1f}ms")
        print(f"   中央値: {statistics.median(success_durations):.1f}ms")
        print(f"   最小: {min(success_durations):.1f}ms")
        print(f"   最大: {max(success_durations):.1f}ms")
        
        if len(success_durations) >= 10:
            sorted_durations = sorted(success_durations)
            p90 = sorted_durations[int(len(sorted_durations) * 0.9)]
            p95 = sorted_durations[int(len(sorted_durations) * 0.95)]
            print(f"   P90: {p90:.1f}ms")
            print(f"   P95: {p95:.1f}ms")
    
    # エラー詳細
    if failures or exceptions:
        print(f"\n🔍 エラー詳細:")
        for i, failure in enumerate(failures[:5]):  # 最初の5つのエラーを表示
            print(f"   {i+1}. ユーザー{failure.get('user_id', '?')}: {failure.get('error', 'Unknown')}")
        
        for i, exc in enumerate(exceptions[:3]):  # 最初の3つの例外を表示
            print(f"   例外{i+1}: {str(exc)}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="簡単な負荷テストシミュレーター")
    parser.add_argument("--users", "-u", type=int, default=50, 
                       help="同時ユーザー数 (デフォルト: 50)")
    parser.add_argument("--url", type=str, default="http://35.77.244.101:5002",
                       help="対象サーバーURL")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="クイックテスト (10ユーザー)")
    
    args = parser.parse_args()
    
    if args.quick:
        users = 10
        print("🏃 クイックテストモード")
    else:
        users = args.users
    
    try:
        asyncio.run(run_load_test(users, args.url))
    except KeyboardInterrupt:
        print("\n⏹️  テストが中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
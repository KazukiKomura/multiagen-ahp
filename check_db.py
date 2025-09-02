#!/usr/bin/env python3
"""
データベース内容確認スクリプト
Database Content Inspection Script for Multi-Agent AHP System
"""

import sys
import os
sys.path.append('src')

import sqlite3
import json
from datetime import datetime
from typing import Dict, Any

def connect_db(db_path: str = 'data/sessions.db'):
    """データベースに接続"""
    if not os.path.exists(db_path):
        print(f"❌ データベースファイルが見つかりません: {db_path}")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except Exception as e:
        print(f"❌ データベース接続エラー: {e}")
        return None

def show_table_info(conn):
    """テーブル情報を表示"""
    print("📊 テーブル情報")
    print("=" * 50)
    
    cursor = conn.cursor()
    
    # テーブル一覧
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        print(f"\n🗂️  テーブル: {table_name}")
        
        # テーブル構造
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        print("   カラム:")
        for col in columns:
            col_name, col_type, not_null, default, pk = col[1], col[2], col[3], col[4], col[5]
            pk_str = " (PRIMARY KEY)" if pk else ""
            null_str = " NOT NULL" if not_null else ""
            default_str = f" DEFAULT {default}" if default else ""
            print(f"     - {col_name}: {col_type}{pk_str}{null_str}{default_str}")
        
        # レコード数
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"   レコード数: {count}")

def show_sessions(conn):
    """セッションデータを表示"""
    print("\n📋 セッションデータ")
    print("=" * 50)
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT session_id, condition, trial, phase, created_at, updated_at
        FROM sessions 
        ORDER BY created_at DESC
    """)
    
    sessions = cursor.fetchall()
    
    if not sessions:
        print("   セッションデータがありません")
        return
    
    for i, session in enumerate(sessions, 1):
        session_id, condition, trial, phase, created_at, updated_at = session
        print(f"\n🔹 セッション {i}")
        print(f"   ID: {session_id}")
        print(f"   条件: {condition}")
        print(f"   試行: {trial}")
        print(f"   フェーズ: {phase}")
        print(f"   作成日時: {created_at}")
        print(f"   更新日時: {updated_at}")

def show_session_details(conn, session_id: str):
    """特定セッションの詳細を表示"""
    print(f"\n🔍 セッション詳細: {session_id}")
    print("=" * 50)
    
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    session = cursor.fetchone()
    
    if not session:
        print("   セッションが見つかりません")
        return
    
    columns = ['session_id', 'condition', 'trial', 'phase', 'student_data', 
               'questionnaire_data', 'decision_data', 'ai_chat_data', 
               'created_at', 'updated_at']
    
    session_dict = dict(zip(columns, session))
    
    # 基本情報
    print("📝 基本情報:")
    for key in ['session_id', 'condition', 'trial', 'phase', 'created_at', 'updated_at']:
        print(f"   {key}: {session_dict[key]}")
    
    # JSON データ
    json_fields = ['student_data', 'questionnaire_data', 'decision_data', 'ai_chat_data']
    
    for field in json_fields:
        print(f"\n📄 {field}:")
        data = session_dict[field]
        if data:
            try:
                parsed = json.loads(data)
                if parsed:
                    print(json.dumps(parsed, indent=4, ensure_ascii=False))
                else:
                    print("   データなし")
            except json.JSONDecodeError:
                print(f"   JSON解析エラー: {data}")
        else:
            print("   データなし")

def show_ai_chat_logs(conn, session_id: str = None):
    """AIチャットログを表示"""
    print(f"\n💬 AIチャットログ" + (f" (セッション: {session_id})" if session_id else ""))
    print("=" * 50)
    
    cursor = conn.cursor()
    
    if session_id:
        cursor.execute("""
            SELECT session_id, turn, user_message, ai_response, satisfaction_scores, timestamp
            FROM ai_chat_logs 
            WHERE session_id = ?
            ORDER BY turn
        """, (session_id,))
    else:
        cursor.execute("""
            SELECT session_id, turn, user_message, ai_response, satisfaction_scores, timestamp
            FROM ai_chat_logs 
            ORDER BY session_id, turn
        """)
    
    logs = cursor.fetchall()
    
    if not logs:
        print("   チャットログがありません")
        return
    
    for log in logs:
        sess_id, turn, user_msg, ai_resp, scores, timestamp = log
        print(f"\n🔹 セッション: {sess_id}, ターン: {turn}")
        print(f"   時刻: {timestamp}")
        print(f"   👤 ユーザー: {user_msg}")
        print(f"   🤖 AI: {ai_resp}")
        
        if scores:
            try:
                score_data = json.loads(scores)
                print(f"   📊 満足度スコア: {score_data}")
            except json.JSONDecodeError:
                print(f"   📊 満足度スコア (解析エラー): {scores}")

def main():
    """メイン実行関数"""
    print("🗄️  Multi-Agent AHP Database Inspector")
    print("=" * 50)
    
    # データベース接続
    conn = connect_db()
    if not conn:
        return
    
    try:
        # コマンドライン引数によって動作を変更
        if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == "info":
                show_table_info(conn)
                
            elif command == "sessions":
                show_sessions(conn)
                
            elif command == "session" and len(sys.argv) > 2:
                session_id = sys.argv[2]
                show_session_details(conn, session_id)
                
            elif command == "chat":
                session_id = sys.argv[2] if len(sys.argv) > 2 else None
                show_ai_chat_logs(conn, session_id)
                
            elif command == "all":
                show_table_info(conn)
                show_sessions(conn)
                show_ai_chat_logs(conn)
                
            else:
                print("❌ 不正なコマンドです")
                print_usage()
        else:
            # デフォルト: 基本情報を表示
            show_table_info(conn)
            show_sessions(conn)
            
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        
    finally:
        conn.close()

def print_usage():
    """使用方法を表示"""
    print("""
🚀 使用方法:
   python check_db.py                    # 基本情報とセッション一覧
   python check_db.py info               # テーブル構造情報
   python check_db.py sessions           # セッション一覧
   python check_db.py session <id>       # 特定セッションの詳細
   python check_db.py chat [session_id]  # チャットログ (全体またはセッション別)
   python check_db.py all                # 全情報を表示

📝 例:
   python check_db.py session abc-123-def
   python check_db.py chat abc-123-def
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print_usage()
    else:
        main()
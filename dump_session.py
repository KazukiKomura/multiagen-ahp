#!/usr/bin/env python3
"""
Dump a specific session's data from data/sessions.db into a human-readable text file.

Usage examples:
  python dump_session.py --latest
  python dump_session.py --session-id <UUID>
  python dump_session.py --session-id <UUID> --out data/custom_dump.txt
"""

import os
import sys
import sqlite3
import json
import argparse
from typing import Optional, Dict, Any

DB_PATH = os.path.join('data', 'sessions.db')


def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        sys.exit(1)
    return sqlite3.connect(db_path)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None


def _get_latest_session_id(conn: sqlite3.Connection) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT session_id FROM sessions ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row else None


def _pretty_json(data: Optional[str]) -> str:
    if not data:
        return "(none)"
    try:
        parsed = json.loads(data)
        if parsed is None or parsed == {} or parsed == []:
            return "(empty)"
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def dump_session(conn: sqlite3.Connection, session_id: str) -> str:
    cur = conn.cursor()

    # Fetch main session row
    cur.execute(
        "SELECT session_id, condition, trial, phase, student_data, questionnaire_data, "
        "decision_data, ai_chat_data, created_at, updated_at "
        "FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    row = cur.fetchone()
    if not row:
        raise SystemExit(f"[ERROR] Session not found: {session_id}")

    (
        sid,
        condition,
        trial,
        phase,
        student_data,
        questionnaire_data,
        decision_data,
        ai_chat_data,
        created_at,
        updated_at,
    ) = row

    lines = []
    lines.append("===== SESSION DETAILS =====")
    lines.append(f"session_id: {sid}")
    lines.append(f"condition: {condition}")
    lines.append(f"trial: {trial}")
    lines.append(f"phase: {phase}")
    lines.append(f"created_at: {created_at}")
    lines.append(f"updated_at: {updated_at}")
    lines.append("")

    lines.append("--- student_data ---")
    lines.append(_pretty_json(student_data))
    lines.append("")

    lines.append("--- questionnaire_data ---")
    lines.append(_pretty_json(questionnaire_data))
    lines.append("")

    lines.append("--- decision_data ---")
    lines.append(_pretty_json(decision_data))
    lines.append("")

    lines.append("--- ai_chat_data ---")
    lines.append(_pretty_json(ai_chat_data))
    lines.append("")

    # AI chat logs
    lines.append("===== AI CHAT LOGS =====")
    if _table_exists(conn, 'ai_chat_logs'):
        cur.execute(
            "SELECT turn, user_message, ai_response, satisfaction_scores, pj_state, timestamp "
            "FROM ai_chat_logs WHERE session_id = ? ORDER BY turn",
            (session_id,),
        )
        rows = cur.fetchall()
        if not rows:
            lines.append("(no chat logs)")
        else:
            for (turn, user_msg, ai_resp, scores, pj_state, ts) in rows:
                lines.append("")
                lines.append(f"[Turn {turn}] @ {ts}")
                lines.append("User:")
                lines.append(user_msg or "")
                lines.append("")
                lines.append("Assistant:")
                lines.append(ai_resp or "")
                try:
                    if scores:
                        sd = json.loads(scores)
                        lines.append("Scores: " + json.dumps(sd, ensure_ascii=False))
                    if pj_state:
                        pj = json.loads(pj_state)
                        lines.append("PJ State: " + json.dumps(pj, ensure_ascii=False))
                except Exception:
                    pass
    else:
        lines.append("(ai_chat_logs table not found)")

    # Optional: decisions table (if exists)
    lines.append("")
    lines.append("===== DECISIONS TABLE (optional) =====")
    if _table_exists(conn, 'decisions'):
        cur.execute(
            "SELECT id, phase, trial, decision_data, timestamp FROM decisions WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        rows = cur.fetchall()
        if not rows:
            lines.append("(no rows)")
        else:
            for (rid, ph, tr, ddata, ts) in rows:
                lines.append("")
                lines.append(f"[Row {rid}] phase={ph} trial={tr} @ {ts}")
                lines.append(_pretty_json(ddata))
    else:
        lines.append("(decisions table not found)")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Dump a session to a text file")
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--session-id', help='Target session_id (UUID)')
    g.add_argument('--latest', action='store_true', help='Dump the most recently created session')
    ap.add_argument('--out', help='Output file path (default: data/session_<id>_dump.txt)')
    args = ap.parse_args()

    conn = _connect(DB_PATH)
    try:
        sid = args.session_id
        if not sid:
            if args.latest:
                sid = _get_latest_session_id(conn)
                if not sid:
                    print('[ERROR] No sessions found')
                    sys.exit(1)
            else:
                print('[ERROR] Please provide --session-id or --latest')
                sys.exit(1)

        dump_text = dump_session(conn, sid)
        out_path = args.out or os.path.join('data', f'session_{sid}_dump.txt')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(dump_text)
        print(f'[OK] Wrote: {out_path}')
    finally:
        conn.close()


if __name__ == '__main__':
    main()


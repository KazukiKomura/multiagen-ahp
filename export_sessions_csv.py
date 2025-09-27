#!/usr/bin/env python3
"""セッションデータCSVエクスポートスクリプト

`dump_session.py` が出力する各項目（セッション概要 / AIチャットログ / decisions テーブル）を
CSV 形式で書き出すユーティリティ。

Usage examples:
  python export_sessions_csv.py                     # デフォルト: 本日7:30以降のセッション
  python export_sessions_csv.py --since "2025-09-24"  # 指定日時以降
  python export_sessions_csv.py --session-id <UUID>   # 特定セッションのみ
  python export_sessions_csv.py --latest              # 直近1件
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from datetime import datetime, time
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_DB_PATH = os.path.join('data', 'sessions.db')
DEFAULT_EXPORT_DIR = os.path.join('data', 'exports')


def ensure_text(value: Any) -> str:
    """Best-effort conversion to str with UTF-8 decoding."""

    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return str(value)


def extract_json_columns(raw_value: Any, prefix: str) -> Dict[str, Any]:
    """Parse a JSON field and map top-level entries to prefixed columns.

    Nested dict/list values are serialized back to JSON (UTF-8, no ASCII escape)
    so that multi-byte characters remain readable inside spreadsheet software.
    When parsing fails, the column `<prefix>raw` keeps the original string.
    """

    columns: Dict[str, Any] = {}
    text = ensure_text(raw_value).strip()
    if not text:
        return columns

    try:
        parsed = json.loads(text)
    except Exception:
        columns[f"{prefix}raw"] = text
        return columns

    if isinstance(parsed, dict):
        for key, value in parsed.items():
            column = f"{prefix}{key}"
            if isinstance(value, (dict, list)):
                columns[column] = json.dumps(value, ensure_ascii=False)
            else:
                columns[column] = value
    else:
        column = prefix[:-1] if prefix.endswith('_') else prefix
        if isinstance(parsed, (dict, list)):
            columns[column] = json.dumps(parsed, ensure_ascii=False)
        else:
            columns[column] = parsed

    return columns


def connect_db(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise SystemExit(f"❌ データベースファイルが見つかりません: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:  # pragma: no cover - defensive default
        raise SystemExit(f"❌ データベース接続エラー: {exc}") from exc


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


def fetch_sessions(
    conn: sqlite3.Connection,
    cutoff: datetime | None,
    session_id: str | None,
    latest: bool,
) -> List[sqlite3.Row]:
    cur = conn.cursor()
    base_query = (
        "SELECT session_id, condition, trial, phase, "
        "student_data, questionnaire_data, decision_data, ai_chat_data, "
        "created_at, updated_at "
        "FROM sessions "
    )

    if session_id:
        cur.execute(base_query + "WHERE session_id = ?", (session_id,))
        rows = cur.fetchall()
    elif latest:
        cur.execute(base_query + "ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
        rows = [row] if row else []
    else:
        if cutoff is None:
            raise ValueError('cutoff must be provided when neither session_id nor latest is specified')
        cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')
        cur.execute(base_query + "WHERE created_at >= ? ORDER BY created_at", (cutoff_str,))
        rows = cur.fetchall()

    return rows


def fetch_ai_chat_logs(conn: sqlite3.Connection, session_ids: Sequence[str]) -> List[sqlite3.Row]:
    if not session_ids or not table_exists(conn, 'ai_chat_logs'):
        return []

    placeholders = ','.join('?' for _ in session_ids)
    query = (
        "SELECT session_id, turn, user_message, ai_response, satisfaction_scores, pj_state, timestamp "
        "FROM ai_chat_logs WHERE session_id IN (" + placeholders + ") ORDER BY session_id, turn"
    )
    cur = conn.cursor()
    cur.execute(query, tuple(session_ids))
    return cur.fetchall()


def fetch_decisions(conn: sqlite3.Connection, session_ids: Sequence[str]) -> List[sqlite3.Row]:
    if not session_ids or not table_exists(conn, 'decisions'):
        return []

    placeholders = ','.join('?' for _ in session_ids)
    query = (
        "SELECT id, session_id, phase, trial, decision_data, timestamp "
        "FROM decisions WHERE session_id IN (" + placeholders + ") ORDER BY session_id, id"
    )
    cur = conn.cursor()
    cur.execute(query, tuple(session_ids))
    return cur.fetchall()


def build_session_rows(rows: Iterable[sqlite3.Row]) -> Tuple[List[str], List[Dict[str, Any]]]:
    processed: List[Dict[str, Any]] = []
    fieldnames = {
        'session_id',
        'condition',
        'trial',
        'phase',
        'created_at',
        'updated_at',
    }

    for row in rows:
        record: Dict[str, Any] = {
            'session_id': row['session_id'],
            'condition': row['condition'],
            'trial': row['trial'],
            'phase': row['phase'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
        }

        record.update(extract_json_columns(row['student_data'], 'student_'))
        record.update(extract_json_columns(row['questionnaire_data'], 'questionnaire_'))
        record.update(extract_json_columns(row['decision_data'], 'decision_'))
        record.update(extract_json_columns(row['ai_chat_data'], 'ai_chat_'))

        fieldnames.update(record.keys())
        processed.append(record)

    ordered_fields = [
        'session_id',
        'condition',
        'trial',
        'phase',
        'created_at',
        'updated_at',
    ]
    ordered_fields.extend(sorted(fieldnames - set(ordered_fields)))
    return ordered_fields, processed


def build_chat_rows(rows: Iterable[sqlite3.Row]) -> Tuple[List[str], List[Dict[str, Any]]]:
    processed: List[Dict[str, Any]] = []
    fieldnames = {
        'session_id',
        'turn',
        'timestamp',
        'user_message',
        'ai_response',
    }

    for row in rows:
        record: Dict[str, Any] = {
            'session_id': row['session_id'],
            'turn': row['turn'],
            'timestamp': row['timestamp'],
            'user_message': ensure_text(row['user_message']),
            'ai_response': ensure_text(row['ai_response']),
        }

        record.update(extract_json_columns(row['satisfaction_scores'], 'satisfaction_'))
        record.update(extract_json_columns(row['pj_state'], 'pj_state_'))

        fieldnames.update(record.keys())
        processed.append(record)

    ordered_fields = [
        'session_id',
        'turn',
        'timestamp',
        'user_message',
        'ai_response',
    ]
    ordered_fields.extend(sorted(fieldnames - set(ordered_fields)))
    return ordered_fields, processed


def build_decision_rows(rows: Iterable[sqlite3.Row]) -> Tuple[List[str], List[Dict[str, Any]]]:
    processed: List[Dict[str, Any]] = []
    fieldnames = {
        'decision_id',
        'session_id',
        'phase',
        'trial',
        'timestamp',
    }

    for row in rows:
        record: Dict[str, Any] = {
            'decision_id': row['id'],
            'session_id': row['session_id'],
            'phase': row['phase'],
            'trial': row['trial'],
            'timestamp': row['timestamp'],
        }

        record.update(extract_json_columns(row['decision_data'], 'decision_'))

        fieldnames.update(record.keys())
        processed.append(record)

    ordered_fields = [
        'decision_id',
        'session_id',
        'phase',
        'trial',
        'timestamp',
    ]
    ordered_fields.extend(sorted(fieldnames - set(ordered_fields)))
    return ordered_fields, processed


def normalize_row(fieldnames: Sequence[str], record: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for field in fieldnames:
        value = record.get(field, '')
        if value is None:
            normalized[field] = ''
        elif isinstance(value, (int, float)):
            normalized[field] = value
        elif isinstance(value, str):
            normalized[field] = value
        else:
            normalized[field] = json.dumps(value, ensure_ascii=False)
    return normalized


def write_csv(path: str, fieldnames: Sequence[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for record in rows:
            writer.writerow(normalize_row(fieldnames, record))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='セッションデータをCSVへエクスポート')
    parser.add_argument('--db', default=DEFAULT_DB_PATH, help='SQLite DB ファイルパス')
    parser.add_argument('--outdir', default=DEFAULT_EXPORT_DIR, help='CSV出力先ディレクトリ')
    parser.add_argument('--since', help='作成日時の下限 (YYYY-MM-DD または YYYY-MM-DD HH:MM[:SS])')
    parser.add_argument('--session-id', help='対象セッションID (UUID)')
    parser.add_argument('--latest', action='store_true', help='直近1件のセッションのみを出力')
    return parser.parse_args()


def resolve_cutoff(args: argparse.Namespace) -> datetime | None:
    if args.session_id or args.latest:
        return None

    if args.since:
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
            try:
                parsed = datetime.strptime(args.since, fmt)
                if fmt == '%Y-%m-%d':
                    return datetime.combine(parsed.date(), time(0, 0))
                return parsed
            except ValueError:
                continue
        raise SystemExit('❌ --since は YYYY-MM-DD もしくは YYYY-MM-DD HH:MM[:SS] 形式で指定してください')

    today = datetime.now().date()
    return datetime.combine(today, time(7, 30))


def main() -> None:
    args = parse_args()
    cutoff = resolve_cutoff(args)

    conn = connect_db(args.db)
    try:
        sessions = fetch_sessions(conn, cutoff, args.session_id, args.latest)
        if not sessions:
            if args.session_id:
                print(f"⚠️ 指定された session_id のデータが見つかりません: {args.session_id}")
            elif args.latest:
                print('⚠️ セッションデータが存在しません')
            else:
                cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S') if cutoff else 'N/A'
                print(f"⚠️ {cutoff_str} 以降のセッションはありません")
            return

        session_ids = [row['session_id'] for row in sessions]
        session_fields, session_records = build_session_rows(sessions)

        chat_rows = fetch_ai_chat_logs(conn, session_ids)
        chat_fields, chat_records = build_chat_rows(chat_rows) if chat_rows else ([], [])

        decision_rows = fetch_decisions(conn, session_ids)
        decision_fields, decision_records = build_decision_rows(decision_rows) if decision_rows else ([], [])

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        outputs: List[Tuple[str, Sequence[str], List[Dict[str, Any]]]] = []

        summary_path = os.path.join(args.outdir, f'sessions_summary_{timestamp}.csv')
        outputs.append((summary_path, session_fields, session_records))

        if chat_records:
            chat_path = os.path.join(args.outdir, f'sessions_ai_chat_logs_{timestamp}.csv')
            outputs.append((chat_path, chat_fields, chat_records))
        else:
            print('ℹ️ 対象セッションの AI チャットログはありません（またはテーブルが存在しません）')

        if decision_records:
            decision_path = os.path.join(args.outdir, f'sessions_decisions_{timestamp}.csv')
            outputs.append((decision_path, decision_fields, decision_records))
        else:
            print('ℹ️ 対象セッションの decisions レコードはありません（またはテーブルが存在しません）')

        for path, fields, records in outputs:
            write_csv(path, fields, records)
            print(f"✅ {len(records)} 件をエクスポート: {path}")

    finally:
        conn.close()


if __name__ == '__main__':
    main()


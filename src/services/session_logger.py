#!/usr/bin/env python3
"""
Session Logger for Procedural Justice System
ユーザーとAIの入出力をJSONファイルに構造化して保存
"""

import json
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path

class SessionLogger:
    """手続的公正システムのセッションログ管理クラス"""
    
    def __init__(self, session_id: Optional[str] = None, output_dir: str = "session_logs"):
        """
        Args:
            session_id: セッション識別子（指定なしの場合は自動生成）
            output_dir: ログファイル出力ディレクトリ
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.session_start_time = datetime.now(timezone.utc)
        self.session_data = {
            "session_id": self.session_id,
            "start_timestamp": self.session_start_time.isoformat(),
            "end_timestamp": None,
            "session_metadata": {},
            "turns": [],
            "summary": {}
        }
        
        print(f"[SessionLogger] Session {self.session_id} started")
    
    def set_session_metadata(self, metadata: Dict[str, Any]):
        """セッション全体のメタデータを設定"""
        self.session_data["session_metadata"].update({
            "user_decision": metadata.get("user_decision"),
            "user_weights": metadata.get("user_weights", {}),
            "profile_facts": self._sanitize_profile_facts(metadata.get("profile_facts", {})),
            "rule_summary": metadata.get("rule_summary", {}),
            "threshold": metadata.get("threshold"),
            "scenario_type": metadata.get("scenario_type", "unknown"),
            "test_mode": metadata.get("test_mode", False)
        })
    
    def log_turn(self, 
                 turn_number: int,
                 action: str,
                 user_input: str,
                 user_context: Dict[str, Any],
                 ai_candidates: List[str],
                 selected_response: str,
                 judge_evaluation: Dict[str, Any],
                 watchdog_evaluation: Dict[str, Any],
                 validation_results: Optional[Dict[str, Any]] = None,
                 processing_time_ms: Optional[float] = None):
        """1ターンのログ（最小限の情報）を記録"""

        # ユーザーの入力とAIの応答、状態コンテキスト、主要メトリクスのみ
        turn_data = {
            "turn_number": turn_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "user_input": {
                "message": user_input,
                "length": len(user_input),
                "context": user_context
            },
            "ai_response": {
                "text": selected_response,
                "length": len(selected_response)
            },
            "metrics": {
                "processing_time_ms": processing_time_ms,
                "validation_score": validation_results.get("overall_score", 0) if validation_results else 0,
                "watchdog_overall": watchdog_evaluation.get("overall", 0)
            }
        }
        
        self.session_data["turns"].append(turn_data)
        print(f"[SessionLogger] Turn {turn_number} logged: {action}")
    
    def set_session_summary(self, summary: Dict[str, Any]):
        """セッション終了時の総括情報を設定"""
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.session_start_time).total_seconds()
        
        self.session_data["end_timestamp"] = end_time.isoformat()
        self.session_data["summary"] = {
            "total_turns": len(self.session_data["turns"]),
            "session_duration_seconds": duration,
            "success_rate": summary.get("success_rate", 0),
            "successful_turns": summary.get("successful_turns", 0),
            "average_validation_score": summary.get("average_validation_score", 0),
            "average_watchdog_score": summary.get("average_watchdog_score", 0),
            "final_watchdog_scores": summary.get("final_scores", {}),
            "scenario_pass": summary.get("scenario_pass", False),
            "completion_status": summary.get("completion_status", "unknown")
        }
    
    def save_to_json(self, filename: Optional[str] = None) -> str:
        """セッションデータをJSONファイルに保存"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"session_{self.session_id[:8]}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, ensure_ascii=False, indent=2)
            
            print(f"[SessionLogger] Session saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"[SessionLogger] Error saving session: {e}")
            return ""
    
    def get_session_data(self) -> Dict[str, Any]:
        """現在のセッションデータを取得"""
        return self.session_data.copy()
    
    def _sanitize_profile_facts(self, profile_facts: Dict[str, Any]) -> Dict[str, Any]:
        """個人情報を含む可能性のあるプロファイル情報をサニタイズ"""
        # プライバシー保護のため、スコアと評価のみ保持
        sanitized = {}
        
        if isinstance(profile_facts, dict):
            for key, value in profile_facts.items():
                if key == "detailed_scores" and isinstance(value, dict):
                    # スコア情報は保持（数値データのため）
                    sanitized[key] = value
                elif key in ["student_info", "personal_data"]:
                    # 個人特定可能情報は除外
                    sanitized[key] = "[REDACTED_FOR_PRIVACY]"
                else:
                    sanitized[key] = value
        
        return sanitized
    
    def generate_analytics_summary(self) -> Dict[str, Any]:
        """分析用の統計サマリーを生成"""
        if not self.session_data["turns"]:
            return {}
        
        turns = self.session_data["turns"]
        
        # アクション別の統計
        action_stats = {}
        for turn in turns:
            action = turn["action"]
            if action not in action_stats:
                action_stats[action] = {
                    "count": 0,
                    "avg_validation_score": 0,
                    "avg_processing_time": 0,
                    "success_count": 0
                }
            
            stats = action_stats[action]
            stats["count"] += 1
            stats["avg_validation_score"] += turn["metrics"]["validation_score"]
            if turn["metrics"]["processing_time_ms"]:
                stats["avg_processing_time"] += turn["metrics"]["processing_time_ms"]
            if turn["metrics"]["validation_score"] > 0.5:  # 成功の閾値
                stats["success_count"] += 1
        
        # 平均値を計算
        for stats in action_stats.values():
            if stats["count"] > 0:
                stats["avg_validation_score"] /= stats["count"]
                stats["avg_processing_time"] /= stats["count"]
                stats["success_rate"] = stats["success_count"] / stats["count"]
        
        return {
            "session_id": self.session_id,
            "total_turns": len(turns),
            "unique_actions": len(action_stats),
            "action_statistics": action_stats,
            "overall_metrics": {
                "avg_validation_score": sum(t["metrics"]["validation_score"] for t in turns) / len(turns),
                "avg_watchdog_score": sum(t["metrics"]["watchdog_overall"] for t in turns) / len(turns),
                "success_rate": sum(1 for t in turns if t["metrics"]["validation_score"] > 0.5) / len(turns),
                "total_processing_time": sum(t["metrics"]["processing_time_ms"] or 0 for t in turns),
            }
        }

    def attach_validation_to_last_turn(self, validation_results: Dict[str, Any]):
        """直近のターンに検証結果を付与し、関連メトリクスも更新"""
        if not self.session_data["turns"]:
            return
        last = self.session_data["turns"][ -1 ]
        last["validation_results"] = validation_results
        # メトリクスも更新
        try:
            if "metrics" not in last:
                last["metrics"] = {}
            last["metrics"]["validation_score"] = validation_results.get("overall_score", 0)
        except Exception:
            pass

class SessionLoggerManager:
    """複数セッションの管理とバッチ処理"""
    
    def __init__(self, output_dir: str = "session_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.active_sessions = {}
    
    def create_session(self, session_id: Optional[str] = None) -> SessionLogger:
        """新しいセッションロガーを作成"""
        logger = SessionLogger(session_id, str(self.output_dir))
        self.active_sessions[logger.session_id] = logger
        return logger
    
    def close_session(self, session_id: str, summary: Dict[str, Any]) -> str:
        """セッションを終了してファイルに保存"""
        if session_id in self.active_sessions:
            logger = self.active_sessions[session_id]
            logger.set_session_summary(summary)
            filepath = logger.save_to_json()
            del self.active_sessions[session_id]
            return filepath
        return ""
    
    def generate_batch_analytics(self, pattern: str = "session_*.json") -> Dict[str, Any]:
        """複数セッションファイルの一括分析"""
        session_files = list(self.output_dir.glob(pattern))
        
        if not session_files:
            return {"error": "No session files found"}
        
        all_sessions = []
        for file_path in session_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    all_sessions.append(session_data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not all_sessions:
            return {"error": "No valid session data found"}
        
        # 集計分析
        total_sessions = len(all_sessions)
        total_turns = sum(len(s.get("turns", [])) for s in all_sessions)
        
        # 成功率の計算
        successful_sessions = sum(1 for s in all_sessions 
                                if s.get("summary", {}).get("scenario_pass", False))
        
        # 平均スコアの計算
        all_validation_scores = []
        all_watchdog_scores = []
        
        for session in all_sessions:
            for turn in session.get("turns", []):
                metrics = turn.get("metrics", {})
                all_validation_scores.append(metrics.get("validation_score", 0))
                all_watchdog_scores.append(metrics.get("watchdog_overall", 0))
        
        return {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_info": {
                "total_sessions": total_sessions,
                "total_turns": total_turns,
                "files_analyzed": len(session_files)
            },
            "performance_metrics": {
                "session_success_rate": successful_sessions / total_sessions if total_sessions > 0 else 0,
                "average_validation_score": sum(all_validation_scores) / len(all_validation_scores) if all_validation_scores else 0,
                "average_watchdog_score": sum(all_watchdog_scores) / len(all_watchdog_scores) if all_watchdog_scores else 0,
                "validation_score_range": {
                    "min": min(all_validation_scores) if all_validation_scores else 0,
                    "max": max(all_validation_scores) if all_validation_scores else 0
                }
            },
            "session_summaries": [
                {
                    "session_id": s["session_id"],
                    "turns": len(s.get("turns", [])),
                    "success": s.get("summary", {}).get("scenario_pass", False),
                    "avg_validation": s.get("summary", {}).get("average_validation_score", 0)
                }
                for s in all_sessions
            ]
        }

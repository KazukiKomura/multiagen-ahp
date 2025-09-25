"""
Session Repository for the multi-agent AHP experiment system.
Implements Repository pattern for session data storage and retrieval.
"""

import sqlite3
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    JST = ZoneInfo("Asia/Tokyo")
except Exception:
    JST = None


class SessionRepository:
    """Session data repository with SQLite backend"""
    
    def __init__(self, db_path: str = 'data/sessions.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with required tables (non-destructive)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create sessions table only if it does not exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                condition TEXT,
                trial INTEGER,
                phase TEXT,
                student_data TEXT,
                questionnaire_data TEXT,
                decision_data TEXT,
                ai_chat_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create ai_chat_logs table only if it does not exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS ai_chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                turn INTEGER,
                user_message TEXT,
                ai_response TEXT,
                satisfaction_scores TEXT,
                pj_state TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def _now_jst_str(self) -> str:
        """Return current timestamp string in Japan Standard Time (YYYY-MM-DD HH:MM:SS)."""
        if JST is not None:
            return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        # Fallback: naive localtime string
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def create_session(self, session_id: str, condition: str) -> bool:
        """
        Create a new session record
        
        Args:
            session_id: Unique session identifier
            condition: Experimental condition
            
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            now_ts = self._now_jst_str()
            c.execute('''
                INSERT INTO sessions (session_id, condition, phase, trial, created_at, updated_at)
                VALUES (?, ?, 'pre_questionnaire', 1, ?, ?)
            ''', (session_id, condition, now_ts, now_ts))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error creating session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict: Session data or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
            row = c.fetchone()
            conn.close()
            
            if row:
                columns = ['session_id', 'condition', 'trial', 'phase', 'student_data', 
                          'questionnaire_data', 'decision_data', 'ai_chat_data', 
                          'created_at', 'updated_at']
                
                session_data = dict(zip(columns, row))
                
                # Parse JSON fields
                for field in ['student_data', 'questionnaire_data', 'decision_data', 'ai_chat_data']:
                    if session_data[field]:
                        try:
                            session_data[field] = json.loads(session_data[field])
                        except json.JSONDecodeError:
                            session_data[field] = {}
                    else:
                        session_data[field] = {}
                
                return session_data
            
            return None
        except Exception as e:
            print(f"Error retrieving session: {e}")
            return None
    
    def update_session(self, session_id: str, **kwargs) -> bool:
        """
        Update session data
        
        Args:
            session_id: Session identifier
            **kwargs: Fields to update
            
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            # Build update query dynamically
            set_clauses = []
            values = []

            # If decision_data is being updated, augment it with server-side timing info
            if 'decision_data' in kwargs and isinstance(kwargs.get('decision_data'), dict):
                try:
                    # Load current decision_data to detect transitions
                    current = self.get_session(session_id) or {}
                    prev_decision = current.get('decision_data', {}) or {}
                    new_decision = dict(kwargs['decision_data'])  # shallow copy

                    # Determine target trial number (prefer explicit in decision_data)
                    trial_num = new_decision.get('trial') or prev_decision.get('trial')
                    if trial_num is not None:
                        trial_key = str(trial_num)
                        timings = new_decision.get('trial_timings') or {}
                        trial_timing = timings.get(trial_key) or {}

                        now_ts = self._now_jst_str()

                        # Record initial timestamp when user_decision first appears for the trial
                        if new_decision.get('user_decision'):
                            if not trial_timing.get('initial_saved_at_server'):
                                trial_timing['initial_saved_at_server'] = now_ts

                        # Record final timestamp when final_decision is present
                        if new_decision.get('final_decision'):
                            if not trial_timing.get('final_submitted_at_server'):
                                trial_timing['final_submitted_at_server'] = now_ts

                            # If both times exist, compute elapsed seconds
                            start_str = trial_timing.get('initial_saved_at_server')
                            end_str = trial_timing.get('final_submitted_at_server')
                            if start_str and end_str:
                                try:
                                    fmt = "%Y-%m-%d %H:%M:%S"
                                    start_dt = datetime.strptime(start_str, fmt)
                                    end_dt = datetime.strptime(end_str, fmt)
                                    elapsed = (end_dt - start_dt).total_seconds()
                                    trial_timing['elapsed_seconds'] = elapsed
                                except Exception:
                                    # If parsing fails, skip elapsed computation gracefully
                                    pass

                        timings[trial_key] = trial_timing
                        new_decision['trial_timings'] = timings

                        # Build per-trial snapshot (non-destructive, keep top-level as-is)
                        per_trial = new_decision.get('per_trial') or {}
                        snapshot_keys = [
                            'user_decision', 'user_weights', 'timestamp',
                            'participant_opinions', 'participant_decisions',
                            'final_decision', 'final_weights', 'change_reasoning',
                            'confidence', 'final_timestamp', 'group_outcome'
                        ]
                        snapshot = {k: new_decision.get(k) for k in snapshot_keys if k in new_decision}
                        # Only store if we have at least one meaningful key
                        if snapshot:
                            per_trial[trial_key] = snapshot
                            new_decision['per_trial'] = per_trial

                        # Map student_id by trial if available
                        try:
                            student_json = current.get('student_data') or {}
                            student_id = student_json.get('student_id')
                            if student_id:
                                sid_map = new_decision.get('student_id_by_trial') or {}
                                sid_map[trial_key] = student_id
                                new_decision['student_id_by_trial'] = sid_map
                        except Exception:
                            pass

                    # Replace kwargs payload with augmented decision_data
                    kwargs['decision_data'] = new_decision
                except Exception:
                    # On any failure, fall back to original payload without blocking the update
                    pass

            for key, value in kwargs.items():
                if key in ['student_data', 'questionnaire_data', 'decision_data', 'ai_chat_data']:
                    # JSON fields
                    set_clauses.append(f"{key} = ?")
                    values.append(json.dumps(value) if value else None)
                elif key in ['condition', 'trial', 'phase']:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
            
            if set_clauses:
                set_clauses.append("updated_at = ?")
                query = f"UPDATE sessions SET {', '.join(set_clauses)} WHERE session_id = ?"
                values.append(self._now_jst_str())
                values.append(session_id)
                
                c.execute(query, values)
                conn.commit()
            
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating session: {e}")
            return False
    
    def save_ai_chat_turn(self, session_id: str, turn: int, user_message: str, 
                         ai_response: str, satisfaction_scores: Dict[str, Any], 
                         pj_state: Dict[str, Any]) -> bool:
        """
        Save AI chat turn data
        
        Args:
            session_id: Session identifier
            turn: Turn number
            user_message: User's message
            ai_response: AI's response
            satisfaction_scores: Satisfaction metrics
            pj_state: Procedural justice state
            
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                INSERT INTO ai_chat_logs 
                (session_id, turn, user_message, ai_response, satisfaction_scores, pj_state, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, 
                turn, 
                user_message, 
                ai_response,
                json.dumps(satisfaction_scores),
                json.dumps(pj_state),
                self._now_jst_str()
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving chat turn: {e}")
            return False
    
    def get_ai_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve AI chat history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List[Dict]: Chat turn data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                SELECT turn, user_message, ai_response, satisfaction_scores, pj_state, timestamp
                FROM ai_chat_logs 
                WHERE session_id = ? 
                ORDER BY turn
            ''', (session_id,))
            
            rows = c.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                turn_data = {
                    'turn': row[0],
                    'user_message': row[1],
                    'ai_response': row[2],
                    'satisfaction_scores': json.loads(row[3]) if row[3] else {},
                    'pj_state': json.loads(row[4]) if row[4] else {},
                    'timestamp': row[5]
                }
                history.append(turn_data)
            
            return history
        except Exception as e:
            print(f"Error retrieving chat history: {e}")
            return []
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Retrieve all session data for admin purposes
        
        Returns:
            List[Dict]: All session records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('SELECT * FROM sessions ORDER BY created_at DESC')
            rows = c.fetchall()
            conn.close()
            
            columns = ['session_id', 'condition', 'trial', 'phase', 'student_data', 
                      'questionnaire_data', 'decision_data', 'ai_chat_data', 
                      'created_at', 'updated_at']
            
            sessions = []
            for row in rows:
                session_data = dict(zip(columns, row))
                
                # Parse JSON fields for display
                for field in ['student_data', 'questionnaire_data', 'decision_data', 'ai_chat_data']:
                    if session_data[field]:
                        try:
                            session_data[field] = json.loads(session_data[field])
                        except json.JSONDecodeError:
                            session_data[field] = {}
                    else:
                        session_data[field] = {}
                
                sessions.append(session_data)
            
            return sessions
        except Exception as e:
            print(f"Error retrieving all sessions: {e}")
            return []


# Global instance for easy import
session_repository = SessionRepository()

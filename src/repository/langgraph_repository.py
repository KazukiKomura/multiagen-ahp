"""
LangGraph Repository - Logging and state management for LangGraph flows.

Handles session logging, turn tracking, and data persistence specific to LangGraph conversations.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

try:
    from zoneinfo import ZoneInfo
    JST = ZoneInfo("Asia/Tokyo")
except ImportError:
    JST = None


class LangGraphLogger:
    """Logger for LangGraph conversation sessions."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.session_metadata = {}
        self.conversation_turns = []
        self.session_summary = {}
        self.start_time = datetime.now(JST) if JST else datetime.now()
        
    def set_session_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set session-level metadata."""
        self.session_metadata = metadata
        
    def log_turn(
        self, 
        turn_number: int, 
        action: str, 
        user_input: str, 
        user_context: Dict[str, Any],
        ai_response: str,
        processing_time_ms: float
    ) -> None:
        """Log a conversation turn."""
        turn_data = {
            'turn_number': turn_number,
            'action': action,
            'user_input': user_input,
            'user_context': user_context,
            'ai_response': ai_response,
            'processing_time_ms': processing_time_ms,
            'timestamp': (datetime.now(JST) if JST else datetime.now()).isoformat()
        }
        self.conversation_turns.append(turn_data)
        
    def set_session_summary(self, summary: Dict[str, Any]) -> None:
        """Set session summary data."""
        self.session_summary = summary
        
    def save_to_json(self) -> str:
        """Save session data to JSON file."""
        output_dir = Path('logs/langgraph')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"langgraph_session_{self.session_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename
        
        session_data = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': (datetime.now(JST) if JST else datetime.now()).isoformat(),
            'session_metadata': self.session_metadata,
            'conversation_turns': self.conversation_turns,
            'session_summary': self.session_summary,
            'total_turns': len(self.conversation_turns)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
            
        return str(filepath)


class LangGraphRepository:
    """Database repository for LangGraph sessions."""
    
    def __init__(self, db_path: str = 'data/langgraph.db'):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self) -> None:
        """Initialize database with required tables."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # LangGraph sessions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS langgraph_sessions (
                session_id TEXT PRIMARY KEY,
                flow_type TEXT,
                user_decision TEXT,
                user_weights TEXT,
                student_info TEXT,
                rule_summary TEXT,
                threshold REAL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_turns INTEGER,
                final_stage REAL,
                appeal_made BOOLEAN,
                session_metadata TEXT,
                session_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Conversation turns table
        c.execute('''
            CREATE TABLE IF NOT EXISTS langgraph_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                turn_number INTEGER,
                stage REAL,
                route TEXT,
                action TEXT,
                user_input TEXT,
                ai_response TEXT,
                user_context TEXT,
                llm_metadata TEXT,
                processing_time_ms REAL,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES langgraph_sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def save_session(
        self, 
        session_id: str, 
        session_data: Dict[str, Any], 
        summary: Dict[str, Any]
    ) -> bool:
        """Save completed session data."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                INSERT OR REPLACE INTO langgraph_sessions 
                (session_id, flow_type, user_decision, user_weights, student_info, 
                 rule_summary, threshold, start_time, end_time, total_turns, 
                 final_stage, appeal_made, session_metadata, session_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                session_data.get('flow_type', 'procedural_justice'),
                session_data.get('user_decision', ''),
                json.dumps(session_data.get('user_weights', {})),
                json.dumps(session_data.get('student_info', {})),
                json.dumps(session_data.get('rule_summary', {})),
                session_data.get('threshold', 2.5),
                session_data.get('start_time'),
                session_data.get('end_time'),
                session_data.get('total_turns', 0),
                session_data.get('final_stage', 0.0),
                session_data.get('appeal_made', False),
                json.dumps(session_data.get('session_metadata', {})),
                json.dumps(summary)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving LangGraph session: {e}")
            return False
            
    def save_turn(self, session_id: str, turn_data: Dict[str, Any]) -> bool:
        """Save individual conversation turn."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                INSERT INTO langgraph_turns 
                (session_id, turn_number, stage, route, action, user_input, 
                 ai_response, user_context, llm_metadata, processing_time_ms, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                turn_data.get('turn_number', 0),
                turn_data.get('stage', 0.0),
                turn_data.get('route', ''),
                turn_data.get('action', ''),
                turn_data.get('user_input', ''),
                turn_data.get('ai_response', ''),
                json.dumps(turn_data.get('user_context', {})),
                json.dumps(turn_data.get('llm_metadata', {})),
                turn_data.get('processing_time_ms', 0),
                turn_data.get('timestamp', datetime.now().isoformat())
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving LangGraph turn: {e}")
            return False
            
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('''
                SELECT * FROM langgraph_turns 
                WHERE session_id = ? 
                ORDER BY turn_number ASC
            ''', (session_id,))
            
            rows = c.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                turn_data = dict(row)
                # Parse JSON fields
                turn_data['user_context'] = json.loads(turn_data['user_context'] or '{}')
                turn_data['llm_metadata'] = json.loads(turn_data['llm_metadata'] or '{}')
                history.append(turn_data)
                
            return history
            
        except Exception as e:
            print(f"Error retrieving session history: {e}")
            return []
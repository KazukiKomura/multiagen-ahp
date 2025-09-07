"""
LangGraph Service - Frontend integration layer.

Provides compatible interface for routes/ai_chat.py integration.
Replaces the old ProceduralJusticeSystem with LangGraph-based implementation.
"""

from typing import Any, Dict, Optional
import time
from datetime import datetime

from .langgraph.flow_engine import LGFlow
from .langgraph.models import FlowState, initial_flow_state, SessionData


class LangGraphService:
    """
    LangGraph-based procedural justice service.
    
    Compatible with existing routes/ai_chat.py interface.
    """
    
    def __init__(self, enable_logging: bool = True, session_id: Optional[str] = None):
        self.enable_logging = enable_logging
        self.session_id = session_id
        self.flow = LGFlow(enable_logging=enable_logging, session_id=session_id)
        self.app = self.flow.compile()
        
    def execute_turn(
        self, 
        message: str, 
        decision: str, 
        state: Dict[str, Any], 
        session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a conversation turn using LangGraph flow.
        
        Compatible with ProceduralJusticeSystem.execute_turn interface.
        
        Args:
            message: User input message
            decision: User decision (合格/不合格)
            state: Current conversation state from Flask session
            session_data: Session context data
            
        Returns:
            Dict with response, satisfaction_scores, action, state, turn
        """
        try:
            # Convert Flask session state to LangGraph FlowState
            flow_state = self._convert_to_flow_state(state, message, session_data)
            
            # Execute LangGraph flow
            start_time = time.time()
            result_state = self.app.invoke(flow_state)
            processing_time = (time.time() - start_time) * 1000
            
            # Extract response
            ai_response = result_state.get('ai_response', '')
            action = self._determine_action(result_state)
            
            # Generate satisfaction scores (simplified for now)
            satisfaction_scores = self._generate_satisfaction_scores(result_state, action)
            
            # Update Flask session state
            updated_state = self._convert_to_session_state(result_state, state)
            
            return {
                'response': ai_response,
                'satisfaction_scores': satisfaction_scores,
                'action': action,
                'state': updated_state,
                'turn': updated_state.get('turn', 0),  # Add turn to main result
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            # Fallback response on error
            return {
                'response': f'申し訳ございません。システムエラーが発生しました: {str(e)}',
                'satisfaction_scores': {'Voice': 0, 'Neutrality': 0, 'Transparency': 0, 'Respect': 0, 'Consistency': 0},
                'action': 'error',
                'state': state,
                'turn': state.get('turn', 0),  # Add turn to error response too
                'processing_time_ms': 0
            }
    
    def _convert_to_flow_state(self, session_state: Dict[str, Any], message: str, session_data: Dict[str, Any]) -> FlowState:
        """Convert Flask session state to LangGraph FlowState."""
        # Start with initial state if not present
        flow_state = initial_flow_state()
        
        # Update with session state
        flow_state['turn'] = session_state.get('turn', 0) + 1
        flow_state['last_user_input'] = message

        # ルートは基本的にノードで制御し、ここでは保持のみ（自動前進しない）
        flow_state['route'] = self._determine_route(session_state)
        # ステージはルートから導出（整合のため）
        flow_state['stage'] = self._determine_stage(flow_state['route'])
        
        # Inject session context
        flow_state['user_weights'] = session_data.get('decision_data', {}).get('user_weights', {})
        flow_state['student_info'] = session_data.get('student_info', {})
        flow_state['rule_summary'] = session_data.get('rule_summary', {})
        flow_state['threshold'] = session_data.get('threshold', 2.5)
        flow_state['bot_evaluators'] = session_data.get('decision_data', {}).get('bot_evaluators', {})
        
        # Transfer previous state
        flow_state['appeal_made'] = session_state.get('appeal_made', False)
        flow_state['rules_shown'] = session_state.get('rules_shown', False)
        flow_state['questions'] = session_state.get('questions', [])
        
        return flow_state
    
    def _convert_to_session_state(self, flow_state: FlowState, old_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LangGraph FlowState back to Flask session state."""
        new_state = old_state.copy()
        new_state.update({
            'turn': flow_state.get('turn', 0),
            'stage': flow_state.get('stage', 0.0),
            'appeal_made': flow_state.get('appeal_made', False),
            'rules_shown': flow_state.get('rules_shown', False),
            'questions': flow_state.get('questions', []),
            'route': flow_state.get('route', 'Rules'),
            'prefs_ok': flow_state.get('prefs_ok', new_state.get('prefs_ok', False)),
        })
        return new_state
    
    def _determine_stage(self, route: str) -> float:
        """ルートから現在のステージを導出（自動前進しない）。"""
        stage_to_route = {
            'Rules': 0.0,
            'Prefs': 1.0,
            'Analysis': 2.0,
            'RequestFeedback': 2.5,
            'WrapUp': 3.0
        }
        return stage_to_route.get(route or 'Rules', 0.0)
    
    def _determine_route(self, session_state: Dict[str, Any]) -> str:
        """現在のルートを維持（ノードでのみ遷移）。"""
        return session_state.get('route', 'Rules')
    
    def _determine_action(self, flow_state: FlowState) -> str:
        """Determine action type from flow state."""
        stage = flow_state.get('stage', 0.0)
        route = flow_state.get('route', 'Rules')
        
        action_map = {
            ('Rules', 0.0): 'Stage0_Rules',
            ('Prefs', 1.0): 'Stage1_Preferences',
            ('Analysis', 2.0): 'Stage2_Analysis',
            ('RequestFeedback', 2.5): 'Stage2_5_RequestFeedback',
            ('WrapUp', 3.0): 'Stage3_WrapUp'
        }
        
        return action_map.get((route, stage), f'Stage{stage}_{route}')
    
    def _generate_satisfaction_scores(self, flow_state: FlowState, action: str) -> Dict[str, float]:
        """Generate procedural justice satisfaction scores."""
        # Simplified scoring based on stage and completion
        base_scores = {'Voice': 3.0, 'Neutrality': 4.0, 'Transparency': 4.0, 'Respect': 4.5, 'Consistency': 4.0}
        
        stage = flow_state.get('stage', 0.0)
        
        # Increase scores as conversation progresses
        if stage >= 1.0:  # Preferences confirmed
            base_scores['Voice'] = 4.0
        if stage >= 2.0:  # Analysis presented
            base_scores['Transparency'] = 4.5
        if flow_state.get('appeal_made', False):  # Appeal handled
            base_scores['Voice'] = 4.5
            base_scores['Neutrality'] = 4.5
        if stage >= 3.0:  # Wrap-up completed
            base_scores['Consistency'] = 4.5
            
        return base_scores

"""
Data models and types for LangGraph procedural justice flow.

Based on pj_simulation.py session structure and existing FlowState.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict, List
from dataclasses import dataclass


class FlowState(TypedDict, total=False):
    """State container for LangGraph flow execution."""
    stage: float  # 0,1,2,2.5,3
    turn: int
    last_user_input: str
    ai_response: str
    rules_shown: bool
    user_weights: Dict[str, float]
    student_info: Dict[str, Any]
    rule_summary: Dict[str, Any]
    threshold: float
    appeal_made: bool
    questions: List[str]
    route: str
    bot_evaluators: Dict[str, Any]


@dataclass
class StudentInfo:
    """Student evaluation data structure."""
    detailed_scores: Dict[str, Dict[str, Any]]
    
    @classmethod
    def from_simulation_data(cls, data: Dict[str, Any]) -> 'StudentInfo':
        """Create from pj_simulation.py format."""
        return cls(detailed_scores=data.get('detailed_scores', {}))


@dataclass
class DecisionData:
    """User decision and evaluation data."""
    user_decision: str
    user_weights: Dict[str, float]
    reasoning: str
    trial: int
    bot_evaluators: Dict[str, Dict[str, str]]
    
    @classmethod
    def from_simulation_data(cls, data: Dict[str, Any]) -> 'DecisionData':
        """Create from pj_simulation.py format."""
        return cls(
            user_decision=data.get('user_decision', ''),
            user_weights=data.get('user_weights', {}),
            reasoning=data.get('reasoning', ''),
            trial=data.get('trial', 1),
            bot_evaluators=data.get('bot_evaluators', {})
        )


@dataclass  
class SessionData:
    """Complete session data structure."""
    student_info: StudentInfo
    decision_data: DecisionData
    rule_summary: Dict[str, Any]
    threshold: float
    last_user_text: str = ''
    
    @classmethod
    def from_simulation_format(cls, sim_data: Dict[str, Any]) -> 'SessionData':
        """Convert from pj_simulation.py build_basic_session format."""
        return cls(
            student_info=StudentInfo.from_simulation_data(sim_data['student_info']),
            decision_data=DecisionData.from_simulation_data(sim_data['decision_data']),
            rule_summary=sim_data.get('rule_summary', {}),
            threshold=sim_data.get('threshold', 2.5),
            last_user_text=sim_data.get('last_user_text', '')
        )


def build_default_session() -> Dict[str, Any]:
    """Build default session data similar to pj_simulation.py."""
    return {
        'student_info': {
            'detailed_scores': {
                '学業成績': {'main_score': 4.0, 'subscores': ['GPA: 4.0', '機関: 高ランク', '成績: 優秀']},
                '試験スコア': {'main_score': 168, 'subscores': ['定量: 168', '定性: 155', '総合: 上位']},
                '研究能力': {'main_score': 1.0, 'subscores': ['計画書: 1点', '経験: 限定的', '独創性: 不足']},
                '推薦状': {'main_score': 3.0, 'subscores': ['評価: 混在', '懸念: 一部あり', '信頼性: 中程度']},
                '多様性': {'main_score': 3.5, 'subscores': ['背景: 一般的', 'スコア: 3.5', '特色: 標準的']}
            }
        },
        'decision_data': {
            'user_decision': '合格',
            'user_weights': {'学業成績': 20, '試験スコア': 20, '研究能力': 20, '推薦状': 20, '多様性': 20},
            'reasoning': '研究能力を重視した総合判断',
            'trial': 1,
            'bot_evaluators': {
                'evaluator_A': {'decision': '不合格', 'reasoning': '研究能力1.0点は致命的'},
                'evaluator_B': {'decision': '不合格', 'reasoning': '基礎力はあるが研究能力が低すぎる'}
            }
        },
        'rule_summary': {
            'criteria': '5項目の加重平均による総合評価',
            'threshold_description': '各項目2.5点以上が基準',
            'evaluation_method': '重み配分に基づく加重平均と多数決'
        },
        'threshold': 2.5,
        'last_user_text': ''
    }


def initial_flow_state() -> FlowState:
    """Create initial flow state."""
    return {
        'turn': 0,
        'stage': 0.0,
        'last_user_input': '',
        'ai_response': '',
        'appeal_made': False,
        'questions': [],
        'route': 'Rules'
    }
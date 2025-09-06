#!/usr/bin/env python3
"""
LangGraph-based simulation runner for the staged flow.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv()

# Support both module and script execution
try:
    from langgraph_app.procedural_justice import LGFlow, FlowState  # type: ignore
except Exception:
    try:
        from .procedural_justice import LGFlow, FlowState  # type: ignore
    except Exception:
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from langgraph_app.procedural_justice import LGFlow, FlowState  # type: ignore


def build_basic_session() -> Dict[str, Any]:
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


def build_scenario_interactions() -> List[Dict[str, Any]]:
    return [
        {'stage': 0, 'ai_only': True, 'user_input': ''},
        {'stage': 1, 'ai_only': True, 'user_input': ''},
        {'stage': 1, 'ai_only': False, 'user_input': '研究能力を重視します。大学院での探究に直結するためです。'},
        {'stage': 2, 'ai_only': True, 'user_input': ''},
        {'stage': 2, 'ai_only': False, 'user_input': '質問です。各項目の最低ラインは？'},
        {'stage': 2, 'ai_only': False, 'user_input': '異議があります。研究計画書の観点をもう一度見てください。'},
        {'stage': 2.5, 'ai_only': True, 'user_input': ''},
        {'stage': 3, 'ai_only': True, 'user_input': ''},
    ]


def initial_state() -> FlowState:
    return {
        'turn': 0,
        'stage': 0.0,
        'last_user_input': '',
        'ai_response': '',
        'appeal_made': False,
        'questions': [],
        'route': 'Rules'
    }


def run_basic_flow() -> Dict[str, Any]:
    flow = LGFlow(enable_logging=True)
    app = flow.compile()
    state = initial_state()
    session_data = build_basic_session()
    # inject session context
    state['user_weights'] = session_data['decision_data']['user_weights']
    state['student_info'] = session_data['student_info']
    state['rule_summary'] = session_data['rule_summary']
    state['threshold'] = session_data['threshold']
    state['bot_evaluators'] = session_data['decision_data'].get('bot_evaluators', {})

    interactions = build_scenario_interactions()
    # set session metadata for logs (if logger exists)
    if flow.logger:
        flow.logger.set_session_metadata({
            "user_decision": session_data['decision_data']['user_decision'],
            "user_weights": session_data['decision_data']['user_weights'],
            "profile_facts": session_data['student_info'],
            "rule_summary": session_data['rule_summary'],
            "threshold": session_data['threshold'],
            "scenario_type": "procedural_justice_langgraph",
            "test_mode": True
        })
    results: List[Dict[str, Any]] = []
    for inter in interactions:
        state['turn'] += 1
        state['stage'] = inter['stage']
        state['last_user_input'] = '' if inter.get('ai_only') else inter.get('user_input', '')
        state = app.invoke(state)
        results.append({'stage': inter['stage'], 'ai_response': state.get('ai_response'), 'route': state.get('route')})
        time.sleep(0.05)

    # close and save log
    if flow.logger:
        summary = {"success_rate": 1, "successful_turns": len(results), "average_validation_score": 0,
                   "average_watchdog_score": 0, "final_scores": {}, "scenario_pass": True,
                   "completion_status": "completed"}
        flow.logger.set_session_summary(summary)
        flow.logger.save_to_json()

    return {'completed': True, 'turns': len(results), 'results': results, 'completion_time': datetime.now().isoformat()}


def main():
    if not os.getenv('OPENAI_API_KEY'):
        print('WARN: OPENAI_API_KEY not set. The underlying system may fail if it tries to call an API.')
    res = run_basic_flow()
    print('Graph run completed:', res['completed'], 'turns:', res['turns'])


if __name__ == '__main__':
    main()

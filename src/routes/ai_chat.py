"""
AI Chat routes for the multi-agent AHP experiment system.
Handles AI-facilitated dialogue using the LangGraph-based Procedural Justice System.
"""

from flask import Blueprint, request, jsonify, session
from ..services.langgraph_service import LangGraphService
from ..repository.session_repository import session_repository
from ..services.simple_llm import SimpleLLMResponder

ai_chat_bp = Blueprint('ai_chat', __name__)


def generate_ai_response(user_message, user_decision=None):
    """
    LangGraph手続き的公正システムによる応答生成
    
    Args:
        user_message: User's message
        user_decision: User's decision (合格/不合格)
        
    Returns:
        Dict: AI response with procedural justice metrics
    """
    
    # セッション状態の取得・初期化
    if 'lg_state' not in session:
        session['lg_state'] = {
            'turn': 0,
            'stage': 0.0,
            'route': 'Rules',
            'appeal_made': False,
            'rules_shown': False,
            'questions': []
        }
    
    # セッションデータ準備（LangGraph入力形式）
    session_record = session_repository.get_session(session['session_id']) or {}
    session_data = {
        'student_info': session.get('student_info', {}),
        'decision_data': session_record.get('decision_data', {}),
        'rule_summary': {
            'criteria': '5項目の加重平均による総合評価',
            'threshold_description': '各項目2.5点以上が基準',
            'evaluation_method': '重み配分に基づく加重平均と多数決'
        },
        'threshold': session_record.get('threshold', 2.5),
        'last_user_text': user_message,
    }
    
    # LangGraphシステム実行
    lg_service = LangGraphService(enable_logging=True, session_id=session.get('session_id'))
    result = lg_service.execute_turn(
        message=user_message,
        decision=user_decision or '未定',
        state=session['lg_state'],
        session_data=session_data
    )
    
    # セッション状態更新
    session['lg_state'] = result['state']
    
    return {
        'message': result['response'],
        'satisfaction_scores': result['satisfaction_scores'],
        'action': result['action'],
        'turn': session['lg_state']['turn']
    }


@ai_chat_bp.route('/ai_chat', methods=['POST'])
def ai_chat():
    """AI chat endpoint with procedural justice system"""
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400
    
    user_message = request.json.get('message', '')
    user_decision = request.json.get('decision')
    
    if not user_message.strip():
        return jsonify({'error': 'Empty message'}), 400
    
    # PJシステムによるAI応答生成
    conversation_count = session.get('conversation_count', 0)
    try:
        ai_result = generate_ai_response(user_message, user_decision)
    except Exception as e:
        # LLM呼び出し・整形の失敗をクライアントにJSONで返却
        return jsonify({'success': False, 'error': str(e)}), 500
    
    # 会話カウント更新
    session['conversation_count'] = conversation_count + 1
    
    # データベースに保存
    session_id = session['session_id']
    success = session_repository.save_ai_chat_turn(
        session_id=session_id,
        turn=ai_result['turn'],
        user_message=user_message,
        ai_response=ai_result['message'],
        satisfaction_scores=ai_result['satisfaction_scores'],
        pj_state=session['lg_state']
    )
    
    if not success:
        print(f"Failed to save chat turn for session {session_id}")
    
    # チャット履歴をセッションに更新
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    session['chat_history'].append({
        'user': user_message,
        'ai': ai_result['message'],
        'turn': ai_result['turn'],
        'action': ai_result['action']
    })
    
    return jsonify({
        'success': True,
        'message': ai_result['message'],
        'satisfaction_scores': ai_result['satisfaction_scores'],
        'turn': ai_result['turn'],
        'action': ai_result['action']
    })


@ai_chat_bp.route('/ai_chat_simple', methods=['POST'])
def ai_chat_simple():
    """軽量LLM: 合否と重みだけで深掘りする最小実装"""
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400

    user_message = request.json.get('message', '')
    user_decision = request.json.get('decision')
    if not user_message.strip():
        return jsonify({'error': 'Empty message'}), 400

    # 事前入力の取得（DBのdecision_data優先）
    session_id = session['session_id']
    record = session_repository.get_session(session_id) or {}
    decision_data = record.get('decision_data', {}) if record else {}

    # ターン管理（軽量系は独立カウンタ）
    session['simple_turn'] = session.get('simple_turn', 0) + 1
    turn = session['simple_turn']

    try:
        responder = SimpleLLMResponder()
        message_text = responder.generate(user_message, decision_data, fallback_decision=user_decision)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    # 既存のログテーブルを流用（満足度・状態は空で保存）
    session_repository.save_ai_chat_turn(
        session_id=session_id,
        turn=turn,
        user_message=user_message,
        ai_response=message_text,
        satisfaction_scores={},
        pj_state={}
    )

    return jsonify({
        'success': True,
        'message': message_text,
        'turn': turn
    })


@ai_chat_bp.route('/chat_history')
def get_chat_history():
    """Retrieve chat history for current session"""
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400
    
    session_id = session['session_id']
    history = session_repository.get_ai_chat_history(session_id)
    
    return jsonify({
        'success': True,
        'history': history
    })


@ai_chat_bp.route('/pj_state')
def get_pj_state():
    """Get current LangGraph state"""
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400
    
    lg_state = session.get('lg_state', {})
    
    return jsonify({
        'success': True,
        'state': lg_state
    })


@ai_chat_bp.route('/reset_chat', methods=['POST'])
def reset_chat():
    """Reset chat session for new trial"""
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400
    
    # Reset LangGraph state
    session['lg_state'] = {
        'turn': 0,
        'stage': 0.0,
        'route': 'Rules',
        'appeal_made': False,
        'rules_shown': False,
        'questions': []
    }
    
    # Reset chat history
    session['chat_history'] = []
    session['conversation_count'] = 0
    
    return jsonify({'success': True, 'message': 'Chat reset successfully'})

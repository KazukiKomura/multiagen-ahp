"""
AI Chat routes for the multi-agent AHP experiment system.
Handles AI-facilitated dialogue using the Procedural Justice System.
"""

from flask import Blueprint, request, jsonify, session
from ..services.procedural_justice import ProceduralJusticeSystem
from ..repository.session_repository import session_repository

ai_chat_bp = Blueprint('ai_chat', __name__)


def generate_ai_response(user_message, user_decision=None):
    """
    手続き的公正システムによる応答生成
    
    Args:
        user_message: User's message
        user_decision: User's decision (合格/不合格)
        
    Returns:
        Dict: AI response with procedural justice metrics
    """
    
    # セッション状態の取得・初期化
    if 'pj_state' not in session:
        session['pj_state'] = {
            'turn': 0,
            'invariants': {
                'Voice': False,
                'Neutrality': False,
                'Transparency': False,
                'Respect': True,  # デフォルト True
                'Consistency': True  # デフォルト True
            },
            'voice_summary_ack': False,
            'appeal_offered': False,
            'user_priority': None
        }
    
    # ターン数更新
    session['pj_state']['turn'] += 1
    
    # セッションデータ準備
    session_data = {
        'student_info': session.get('student_info', {}),
        'criteria': ['学業成績', '試験スコア', '研究能力', '推薦状', '多様性'],
        'user_decision': user_decision
    }
    
    # PJシステム実行
    pj_system = ProceduralJusticeSystem()
    result = pj_system.execute_turn(
        message=user_message,
        decision=user_decision or '未定',
        state=session['pj_state'],
        session_data=session_data
    )
    
    # セッション状態更新
    session['pj_state'] = result['state']
    
    return {
        'message': result['response'],
        'satisfaction_scores': result['satisfaction_scores'],
        'action': result['action'],
        'turn': session['pj_state']['turn']
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
    ai_result = generate_ai_response(user_message, user_decision)
    
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
        pj_state=session['pj_state']
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
    """Get current procedural justice state"""
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400
    
    pj_state = session.get('pj_state', {})
    
    return jsonify({
        'success': True,
        'state': pj_state
    })


@ai_chat_bp.route('/reset_chat', methods=['POST'])
def reset_chat():
    """Reset chat session for new trial"""
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400
    
    # Reset PJ state
    session['pj_state'] = {
        'turn': 0,
        'invariants': {
            'Voice': False,
            'Neutrality': False,
            'Transparency': False,
            'Respect': True,
            'Consistency': True
        },
        'voice_summary_ack': False,
        'appeal_offered': False,
        'user_priority': None
    }
    
    # Reset chat history
    session['chat_history'] = []
    session['conversation_count'] = 0
    
    return jsonify({'success': True, 'message': 'Chat reset successfully'})
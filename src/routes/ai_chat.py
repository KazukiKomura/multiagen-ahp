"""
AI Chat routes (LangGraph-free version).
Streamlitの `streamlit_simple_chat.py` と同等の吹き出し・やり取りに統一。
初期2メッセージは静的アシスタント吹き出し、その後は Responses API で応答。
"""

import os
from flask import Blueprint, request, jsonify, session
from ..repository.session_repository import session_repository
from typing import List, Dict

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - runtime guard
    OpenAI = None

ai_chat_bp = Blueprint('ai_chat', __name__)


# ===== Streamlitと同等の振る舞いをするためのユーティリティ =====
RULES_BUBBLE_TEXT = (
    "【手続とルールのご案内】\n\n"
    "本システムでは以下のルールに基づいて評価を行います：\n\n"
    "**評価基準**\n"
    "- 5項目（学業成績、研究能力、コミュニケーション、リーダーシップ、将来性）の加重平均\n"
    "- 総合判定：あなたの重み配分 + 参加者評価者2名の多数決\n\n"
    "**重要な制約**\n"
    "- AIは結果を変更できません\n"
    "- 誤読・見落としがあれば異議申し立てで確認します\n"
    "- すべての評価根拠を透明に開示します\n\n"
    "**今後の流れ**\n"
    "1. あなたの重視点の確認\n"
    "2. 合格・不合格の観点整理と質問・異議機会\n"
    "3. 最終結果の要約\n"
)


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAIクライアント初期化に失敗しました（APIキー/パッケージ確認）")
    return OpenAI(api_key=api_key)


def _get_system_prompt() -> str:
    # prompts/system_prompt.txt > env SYSTEM_PROMPT > fallback
    default_path = os.path.join(os.getcwd(), "prompts", "system_prompt.txt")
    if os.path.exists(default_path):
        try:
            with open(default_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    env_prompt = os.getenv("SYSTEM_PROMPT")
    if env_prompt and env_prompt.strip():
        return env_prompt
    return (
        "あなたは合否判断の合意形成を支援するAIファシリテータです。"
        "ユーザーの重視点と2名の参加者の観点を踏まえ、簡潔に状況整理し、1つの質問のみを行ってください。"
    )


def _build_initial_messages(weights: Dict[str, int]) -> List[Dict[str, str]]:
    # ルール案内 + 重み確認（Streamlitの初期2バブルと同等）
    weights = weights or {}
    top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    top_criteria = [top[0][0], top[1][0]] if len(top) >= 2 else ([top[0][0], ""] if top else ["学業成績", "研究能力"])
    weights_text = (
        "【あなたの重視点について】\n\n"
        "UIで設定された重み配分を確認しました：\n"
        f"- 学業成績: {weights.get('学業成績', 30)}%\n"
        f"- 研究能力: {weights.get('研究能力', 25)}%  \n"
        f"- コミュニケーション: {weights.get('コミュニケーション', 20)}%\n"
        f"- リーダーシップ: {weights.get('リーダーシップ', 10)}%\n"
        f"- 将来性: {weights.get('将来性', 15)}%\n\n"
        f"あなたが特に{top_criteria[0]}を重視される理由について、詳しくお聞かせください。\n"
        "この学生の評価においてなぜこれらの項目を重要と考えられたのでしょうか？\n\n"
        "なお、参加者評価者2名もそれぞれ異なる基準を持って評価を行っています。"
    )
    return [
        {"role": "assistant", "content": RULES_BUBBLE_TEXT},
        {"role": "assistant", "content": weights_text},
    ]


def _build_responses_input(messages: List[Dict[str, str]], system_text: str, weights: Dict[str, int]):
    # Streamlit版と同じ構造に変換（typeは input_text/output_text）
    input_seq: List[Dict] = [
        {"role": "system", "content": [{"type": "input_text", "text": system_text}]}
    ]

    decision_data_json = (
        '{\n'
        '  "student_info": {\n'
        '    "name": "田中太郎",\n'
        '    "student_id": "S2024001",\n'
        '    "scores": {\n'
        f'      "学業成績": 85,\n'
        f'      "研究能力": 78,\n'
        f'      "コミュニケーション": 82,\n'
        f'      "リーダーシップ": 65,\n'
        f'      "将来性": 79\n'
        '    }\n'
        '  },\n'
        '  "user_weights": {\n'
        f'    "学業成績": {weights.get("学業成績", 30)},\n'
        f'    "研究能力": {weights.get("研究能力", 25)},\n'
        f'    "コミュニケーション": {weights.get("コミュニケーション", 20)},\n'
        f'    "リーダーシップ": {weights.get("リーダーシップ", 10)},\n'
        f'    "将来性": {weights.get("将来性", 15)}\n'
        '  },\n'
        '  "user_decision": "合格",\n'
        '  "participant_decisions": {\n'
        '    "participant1": {"decision": "不合格"},\n'
        '    "participant2": {"decision": "不合格"}\n'
        '  }\n'
        '}\n'
    )
    input_seq.append({
        "role": "system",
        "content": [{"type": "input_text", "text": "# 生徒/意思決定データ\n" + decision_data_json}],
    })

    for idx, m in enumerate(messages):
        # 初期の2つのassistantバブルはLLMへは渡さない
        if idx in (0, 1) and m.get("role") == "assistant":
            continue
        role = m.get("role", "user")
        ctype = "output_text" if role == "assistant" else "input_text"
        input_seq.append({"role": role, "content": [{"type": ctype, "text": m.get("content", "")}]} )

    return input_seq


def _call_llm(messages: List[Dict[str, str]], weights: Dict[str, int], model: str = "gpt-4.1") -> str:
    client = _get_openai_client()
    system_text = _get_system_prompt()
    input_payload = _build_responses_input(messages, system_text, weights)
    resp = client.responses.create(
        model=model,
        input=input_payload,
        temperature=0.4,
        max_output_tokens=1024,
        top_p=1
    )
    text = getattr(resp, "output_text", None)
    if text:
        return text
    try:
        return resp.output[0].content[0].text  # type: ignore[attr-defined]
    except Exception:
        return str(resp)


@ai_chat_bp.route('/setup_chat', methods=['POST'])
def setup_chat():
    """初期2バブルをサーバー側で生成し、セッションに保持"""
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400

    data = request.get_json(silent=True) or {}
    weights = data.get('weights') or {}
    decision = data.get('decision', '未定')

    # セッションへ保存（必要なら他画面でも参照可）
    session['user_weights'] = weights
    session['user_decision'] = decision

    initial_msgs = _build_initial_messages(weights)
    session['messages'] = list(initial_msgs)  # 新規開始
    session['conversation_count'] = 0

    return jsonify({'success': True, 'messages': initial_msgs})


@ai_chat_bp.route('/ai_chat', methods=['POST'])
def ai_chat():
    """LLMへの1ターン問い合わせ（Streamlit相当の入力方式）"""
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400

    payload = request.get_json(silent=True) or {}
    user_message = (payload.get('message') or '').strip()
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    # 会話メモリ
    messages: List[Dict[str, str]] = session.get('messages') or []
    if not messages:
        # 未初期化の場合は空の初期バブルを作る
        weights = session.get('user_weights', {})
        messages = _build_initial_messages(weights)

    messages.append({"role": "user", "content": user_message})

    try:
        weights = session.get('user_weights', {})
        model = os.getenv('OPENAI_RESPONSES_MODEL', 'gpt-4.1')
        assistant_text = _call_llm(messages, weights, model=model)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    messages.append({"role": "assistant", "content": assistant_text})
    session['messages'] = messages
    session['conversation_count'] = int(session.get('conversation_count', 0)) + 1

    # 既存のログテーブルへ保存（簡易）
    try:
        session_repository.save_ai_chat_turn(
            session_id=session['session_id'],
            turn=session['conversation_count'],
            user_message=user_message,
            ai_response=assistant_text,
            satisfaction_scores={},
            pj_state={}
        )
    except Exception:
        pass

    return jsonify({'success': True, 'message': assistant_text, 'turn': session['conversation_count']})


# 互換用途の軽量APIは不要になったため削除


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


# LangGraph関連エンドポイントは削除


@ai_chat_bp.route('/reset_chat', methods=['POST'])
def reset_chat():
    """セッションの会話をリセット（LangGraph要素は破棄）"""
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400

    session.pop('messages', None)
    session['conversation_count'] = 0
    return jsonify({'success': True, 'message': 'Chat reset successfully'})

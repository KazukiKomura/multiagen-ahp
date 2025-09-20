"""
AI Chat routes (LangGraph-free version).
Streamlitの `streamlit_simple_chat.py` と同等の吹き出し・やり取りに統一。
初期2メッセージは静的アシスタント吹き出し、その後は Responses API で応答。
"""

import os
from flask import Blueprint, request, jsonify, session
from ..repository.session_repository import session_repository
from ..utils import argumentation_engine  # 論理エンジンをインポート
from typing import List, Dict, Any
import json

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
    "- 5項目（学業成績、試験スコア、研究能力、推薦状、多様性）の加重平均\n"
    "- 総合判定：あなたの重み配分 + 参加者3名の多数決\n\n"
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
        "ユーザーの重視点（学業成績・試験スコア・研究能力・推薦状・多様性）と3名の参加者の観点を踏まえ、"
        "簡潔に状況整理し、1つの質問のみを行ってください。"
    )


def _build_initial_messages(weights: Dict[str, int]) -> List[Dict[str, str]]:
    # ルール案内 + 重み確認（Streamlitの初期2バブルと同等）
    weights = weights or {}
    criteria_order = ['学業成績', '試験スコア', '研究能力', '推薦状', '多様性']
    # 表示はUIの5項目に統一
    lines = [
        "【あなたの重視点について】\n",
        "UIで設定された重み配分を確認しました：\n",
    ]
    for c in criteria_order:
        v = weights.get(c)
        v = int(v) if isinstance(v, (int, float, str)) and str(v).isdigit() else (20 if c != '学業成績' else 20)
        lines.append(f"- {c}: {v}%\n")
    # 上位強調
    top = sorted([(k, int(weights.get(k, 0))) for k in criteria_order], key=lambda kv: kv[1], reverse=True)
    top_name = top[0][0] if top else '学業成績'
    lines += [
        "\n",
        f"あなたが特に{top_name}を重視される理由について、詳しくお聞かせください。\n",
        "この学生の評価においてなぜこれらの項目を重要と考えられたのでしょうか？\n\n",
        "なお、参加者3名もそれぞれ異なる基準を持って評価を行っています。",
    ]
    weights_text = ''.join(lines)
    return [
        {"role": "assistant", "content": RULES_BUBBLE_TEXT},
        {"role": "assistant", "content": weights_text},
    ]


def _build_responses_input(messages: List[Dict[str, str]], system_text: str, context: Dict[str, Any]):
    # Streamlit版と同じ構造に変換（typeは input_text/output_text）
    input_seq: List[Dict] = [
        {"role": "system", "content": [{"type": "input_text", "text": system_text}]}
    ]

    decision_data_json = json.dumps(context, ensure_ascii=False, indent=2)
    input_seq.append({
        "role": "system",
        "content": [{"type": "input_text", "text": "# セッションコンテキスト\n" + decision_data_json}],
    })

    for idx, m in enumerate(messages):
        # 初期の2つのassistantバブルはLLMへは渡さない
        if idx in (0, 1) and m.get("role") == "assistant":
            continue
        role = m.get("role", "user")
        ctype = "output_text" if role == "assistant" else "input_text"
        input_seq.append({"role": role, "content": [{"type": ctype, "text": m.get("content", "")}]} )

    return input_seq


def _call_llm(messages: List[Dict[str, str]], context: Dict[str, Any], model: str = "gpt-4.1") -> str:
    client = _get_openai_client()
    system_text = _get_system_prompt()
    input_payload = _build_responses_input(messages, system_text, context)

    # Optional debug of payload
    try:
        if str(os.getenv('DEBUG_LLM_CONTEXT', '')).lower() in ('1', 'true', 'yes', 'on'):
            print("===== DEBUG: LLM INPUT PAYLOAD (truncated preview) =====")
            import itertools as _it
            # Print safely up to certain length
            import json as _json
            payload_str = _json.dumps(input_payload, ensure_ascii=False)
            print(payload_str[:4000] + ("..." if len(payload_str) > 4000 else ""))
            print("===== END PAYLOAD =====")
    except Exception:
        pass
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

    # セッション/DBからLLM用のコンテキストを構築
    def build_llm_context() -> Dict[str, Any]:
        session_id = session['session_id']
        sdata = session_repository.get_session(session_id) or {}
        decision_data = sdata.get('decision_data', {})
        questionnaire = sdata.get('questionnaire_data', {})
        student = sdata.get('student_data') or session.get('student_info') or {}

        # 参加者の決定はセッション固定値があればそれを利用
        participants = session.get('participant_decisions') or decision_data.get('participant_decisions') or []
        participant_opinions = decision_data.get('participant_opinions') or []
        # コンテキスト本体
        ctx: Dict[str, Any] = {
            'session_id': session_id,
            'condition': session.get('condition'),
            'trial': session.get('trial'),
            'student_info': student,
            'user_initial_decision': decision_data.get('user_decision') or session.get('user_decision'),
            'user_initial_weights': decision_data.get('user_weights') or session.get('user_weights') or {},
            'user_final_decision': decision_data.get('final_decision'),
            'user_final_weights': decision_data.get('final_weights'),
            'participant_decisions': participants,
            'participant_opinions': participant_opinions,
            'group_outcome': decision_data.get('group_outcome'),
            'questionnaire': questionnaire,
        }
        return ctx

    try:
        model = os.getenv('OPENAI_RESPONSES_MODEL', 'gpt-4.1')
        ctx = build_llm_context()

        # === 論理エンジンによる議論分析 ===
        try:
            # 1. 論理エンジンを実行して、議論の構造を分析
            arguments = argumentation_engine.extract_atomic_arguments(ctx)
            attacks = argumentation_engine.determine_attacks(arguments)
            debate_summary = argumentation_engine.summarize_debate(arguments, attacks)

            # 2. 分析結果をLLMのコンテキストに追加
            ctx['argumentation_analysis'] = debate_summary
            
            # 3. 議論構造に基づく推奨質問も生成
            focused_question = argumentation_engine.generate_focused_question(debate_summary, messages)
            ctx['suggested_question'] = focused_question

            # デバッグ出力（論理エンジン分析結果）
            if str(os.getenv('DEBUG_LLM_CONTEXT', '')).lower() in ('1', 'true', 'yes', 'on'):
                print("\n===== 論理エンジン分析結果 =====")
                print(f"抽出された主張数: {len(arguments)}")
                print(f"攻撃関係数: {len(attacks)}")
                print(f"論点: {debate_summary.get('key_conflict_point', 'N/A')}")
                print(f"推奨質問: {focused_question}")
                print("===== 分析結果終了 =====\n")
        except Exception as analysis_error:
            print(f"論理エンジン分析エラー: {analysis_error}")
            # 分析失敗時はデフォルトの分析結果を設定
            ctx['argumentation_analysis'] = {
                "key_conflict_point": "分析データが不足しています。",
                "user_claim_summary": "ユーザーの主張を分析中です。"
            }

        # Debug print + persist last context (opt-in by env or always safe)
        try:
            if str(os.getenv('DEBUG_LLM_CONTEXT', '')).lower() in ('1', 'true', 'yes', 'on'):
                print("\n===== DEBUG: LLM CONTEXT (current session) =====")
                print(json.dumps(ctx, ensure_ascii=False, indent=2))
                print("===== END CONTEXT =====\n")
        except Exception:
            pass

        # Save last context to DB for inspection in admin or scripts
        try:
            sdata = session_repository.get_session(session['session_id']) or {}
            ai_chat_data = sdata.get('ai_chat_data', {}) or {}
            ai_chat_data['last_llm_context'] = ctx
            session_repository.update_session(session['session_id'], ai_chat_data=ai_chat_data)
        except Exception:
            pass

        assistant_text = _call_llm(messages, ctx, model=model)
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

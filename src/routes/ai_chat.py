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
    "- 5項目（学業成績、基礎能力テスト、実践経験、推薦・評価、学歴・所属）の加重平均\n"
    "- 総合判定：あなたの重み配分 + AI 3名の多数決（一次通過/見送り）\n\n"
    "**重要な制約**\n"
    "- AIは結果を変更できません\n"
    "- 誤読・見落としがあれば異議申し立てで確認します\n"
    "- すべての評価根拠を透明に開示します\n\n"
    "**今後の流れ**\n"
    "1. あなたの重視点の確認\n"
    "2. 一次通過・見送りの観点整理と質問・異議機会\n"
    "3. 最終結果の要約\n"
)


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAIクライアント初期化に失敗しました（APIキー/パッケージ確認）")
    return OpenAI(api_key=api_key)


def _create_openai_client_with_key(api_key: str):
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
        "あなたは一次選考判断の合意形成を支援するAIファシリテータです。"
        "あなたの重視点（学業成績・基礎能力テスト・実践経験・推薦・評価・学歴・所属）と"
        "3名のAIの観点を踏まえ、簡潔に状況整理し、1つの質問のみを行ってください。"
    )


def _sanitize_llm_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """LLM入力用に不要/秘匿キーを除去したディープコピーを返す。

    除外キー（ネスト内も対象）:
      - questionnaire, algorithm_type, reason, condition, session_id
    """
    EXCLUDE = {"questionnaire", "algorithm_type", "reason", "condition", "session_id"}

    def _clean(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in EXCLUDE:
                    continue
                out[k] = _clean(v)
            return out
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        return obj

    try:
        return _clean(dict(ctx))
    except Exception:
        return {k: v for k, v in (ctx or {}).items() if k not in EXCLUDE}


def _display_label(name: str) -> str:
    """UI表示用の基準名マッピング。"""
    try:
        return '学歴・所属' if name == '志望動機・フィット' else name
    except Exception:
        return name


def _build_initial_messages(weights: Dict[str, int], context: Dict[str, Any] = None) -> List[Dict[str, str]]:
    """
    初期メッセージを構築（argumentation_engine.pyの分析結果を直接使用）
    """
    weights = weights or {}

    # コンテキストがある場合は論理エンジンを実行して分析結果を取得
    if context:
        try:
            import src.utils.argumentation_engine as argumentation_engine

            print(f"[DEBUG] 初期バブル生成: コンテキスト確認")
            print(f"  - あなたの判断: {context.get('user_initial_decision')}")
            print(f"  - あなたの重み: {context.get('user_initial_weights')}")
            print(f"  - 参加者意見数: {len(context.get('participant_opinions', []))}")

            # 論理エンジンを実行（ai_chat.pyと同じ流れ）
            arguments = argumentation_engine.extract_atomic_arguments(context)
            attacks = argumentation_engine.determine_attacks(arguments)
            user_weights = context.get('user_initial_weights') or context.get('user_final_weights') or {}

            print(f"[DEBUG] 抽出された主張数: {len(arguments)}")
            print(f"[DEBUG] 攻撃関係数: {len(attacks)}")

            if user_weights:
                debate_summary = argumentation_engine.summarize_debate(arguments, attacks, user_weights)
            else:
                debate_summary = argumentation_engine.summarize_debate(arguments, attacks)

            print(f"[DEBUG] 使用アルゴリズム: {debate_summary.get('algorithm_type', 'legacy')}")

            # 新アルゴリズムの詳細分析結果を使用
            if debate_summary.get('algorithm_type') == 'two_track_ranking_salience':
                detailed_analysis = debate_summary.get('detailed_analysis', {})
                conflict_points = detailed_analysis.get('conflict_points', [])
                analysis_overview = detailed_analysis.get('analysis_overview', {})
                user_claim_summary = detailed_analysis.get('user_claim_summary', '')

                # 「ユーザー」を「あなた」に変換
                user_claim_summary = user_claim_summary.replace('ユーザーは', 'あなたは').replace('ユーザー', 'あなた')

                print(f"[DEBUG] 検出された対立点数: {len(conflict_points)}")

                lines = [
                    "## 📊 議論状況の分析\n\n",
                    f"**{user_claim_summary}**\n\n",
                    f"**AI**: {analysis_overview.get('total_participants', 3)}名の評価者（あなたとの価値観の近さで分類）\n",
                    f"**対立論点**: {analysis_overview.get('conflict_points_found', 0)}件の主要な違い\n\n"
                ]

                # 価値観の群分けを表示
                participant_opinions = context.get('participant_opinions', [])
                if participant_opinions:
                    lines.append("### 🎯 各AIの判断\n")
                    for opinion in participant_opinions:
                        bot_id = opinion.get('bot_id', 0)
                        participant_name = f"AI{bot_id + 1}"
                        decision = opinion.get('decision', '不明')
                        weights_info = opinion.get('weights', {})

                        if weights_info:
                            top_criterion = max(weights_info, key=weights_info.get, default='不明')
                            top_weight = weights_info.get(top_criterion, 0)

                            user_decision = context.get('user_initial_decision', '一次通過')
                            agreement = "✅ あなたと同じ判断" if decision == user_decision else "❌ あなたと異なる判断"

                            lines.append(f"- **{participant_name}**: {decision} {agreement}\n")
                            lines.append(f"  最重視: {_display_label(top_criterion)}（{top_weight}%）\n")

                    lines.append("\n")

                # 論点詳細を表示（技術的情報を除去）
                if conflict_points:
                    lines.append("### 🔍 注目すべき違い\n")
                    for i, point in enumerate(conflict_points, 1):
                        # データを安全に取得
                        criterion = _display_label(point.get('criterion', '不明'))
                        user_weight = point.get('user_weight', 0)
                        opponent_weight = point.get('opponent_weight', 0)

                        # opponent情報を取得
                        opponent_info = point.get('top_opponent', {})
                        opponent_source = opponent_info.get('source', 'participant1')
                        opponent_claim = opponent_info.get('claim', '不明')

                        # participant1 → AI1 に変換
                        opponent_name = opponent_source.replace('participant', 'AI')

                        # グループラベルを相対的分割に合わせて変換
                        group_label = point.get('group', '不明な群')
                        if group_label == '価値観が近い群':
                            group_explanation = f'あなたに最も近い価値観を持つAI'
                        elif group_label == '価値観が異なる群':
                            group_explanation = f'あなたと異なる価値観を持つAI'
                        else:
                            group_explanation = group_label.replace('価値観が近い群', '近い価値観のAI').replace('価値観が異なる群', '異なる価値観のAI')

                        lines.extend([
                            f"**違い{i}: {criterion}への評価**\n",
                            f"- あなた: {context.get('user_initial_decision', '一次通過')}（{user_weight}%重視）\n",
                            f"- {opponent_name}: {opponent_claim}（{opponent_weight}%重視）\n",
                            f"- 関係: {group_explanation}\n\n"
                        ])

                lines.extend([
                    "---\n\n",
                    "上記の状況を踏まえて、**あなたの判断理由**をお聞かせください。\n",
                    "特に同じような価値観の人と判断が分かれた点について、どのような考えで決められたのでしょうか？"
                ])

                initial_text = ''.join(lines)
                print(f"[DEBUG] 初期バブル生成完了（新アルゴリズム使用）")

                return [{"role": "assistant", "content": initial_text}]

            else:
                # 既存アルゴリズムの結果を使用
                print(f"[DEBUG] 既存アルゴリズムの結果を使用")
                user_claim = debate_summary.get('user_claim_summary', '')
                key_conflict = debate_summary.get('key_conflict_point', '')

                # 「ユーザー」を「あなた」に変換
                user_claim = user_claim.replace('ユーザーは', 'あなたは').replace('ユーザー', 'あなた')

                lines = [
                    "## 📊 議論分析結果\n\n",
                    f"**{user_claim}**\n\n",
                    f"**主要な違い**: {key_conflict}\n\n",
                    "---\n\n",
                    "上記の分析を踏まえて、あなたの判断理由について詳しくお聞かせください。"
                ]

                initial_text = ''.join(lines)
                return [{"role": "assistant", "content": initial_text}]

        except Exception as e:
            print(f"[ERROR] 初期メッセージでの論理エンジン実行エラー: {e}")
            import traceback
            traceback.print_exc()

    # フォールバック: 従来の重み確認メッセージ
    print(f"[DEBUG] フォールバック: 従来メッセージを使用")
    criteria_order = ['学業成績', '基礎能力テスト', '実践経験', '推薦・評価', '志望動機・フィット']
    lines = [
        "【あなたの重視点について】\n",
        "画面上で設定された重み配分を確認しました：\n",
    ]

    for c in criteria_order:
        v = weights.get(c, 0)
        v = int(v) if isinstance(v, (int, float, str)) and str(v).isdigit() else 20
        lines.append(f"- {_display_label(c)}: {v}%\n")

    def _val_for(k: str) -> int:
        try:
            val = weights.get(k, 0)
            return int(val) if str(val).isdigit() else 0
        except Exception:
            return 0

    top = sorted([(k, _val_for(k)) for k in criteria_order], key=lambda kv: kv[1], reverse=True)
    top_name = _display_label(top[0][0]) if top else '学業成績'
    lines += [
        "\n",
        f"あなたが特に{top_name}を重視される理由について、詳しくお聞かせください。\n",
        "この学生の評価においてなぜこれらの項目を重要と考えられたのでしょうか？\n\n",
        "なお、AI 3名もそれぞれ異なる基準を持って評価を行っています。",
    ]

    weights_text = ''.join(lines)
    return [{"role": "assistant", "content": weights_text}]


def _build_responses_input(messages: List[Dict[str, str]], system_text: str, context: Dict[str, Any]):
    # Streamlit版と同じ構造に変換（typeは input_text/output_text）
    input_seq: List[Dict] = [
        {"role": "system", "content": [{"type": "input_text", "text": system_text}]}
    ]

    safe_ctx = _sanitize_llm_context(context or {})
    decision_data_json = json.dumps(safe_ctx, ensure_ascii=False, indent=2)
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


def _call_llm(messages: List[Dict[str, str]], context: Dict[str, Any], model: str = "gpt-5-mini") -> str:
    client = _get_openai_client()
    system_text = _get_system_prompt()
    input_payload = _build_responses_input(messages, system_text, context)

    # AIへの入力全体の文字数を計測してログ出力
    try:
        print(f"[AI_IO] AI入力文字数: {len(json.dumps(input_payload, ensure_ascii=False))}")
    except Exception:
        pass

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
    def _do_call(_client):
        print(f"[model]{model}")
        resp = _client.responses.create(
            model=model,
            input=input_payload,
            temperature=0.0,
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

    assistant_text = None

    # 1) プライマリキーで実行
    try:
        assistant_text = _do_call(client)
    except Exception as primary_error:
        # 2) セカンダリキーが設定されていればフェイルオーバー
        api_key2 = os.getenv("OPENAI_API_KEY2")
        if api_key2:
            try:
                if str(os.getenv('DEBUG_LLM_CONTEXT', '')).lower() in ('1', 'true', 'yes', 'on'):
                    print("[LLM FAILOVER] プライマリ失敗。OPENAI_API_KEY2 で再試行します。")
                alt_client = _create_openai_client_with_key(api_key2)
                assistant_text = _do_call(alt_client)
            except Exception as secondary_error:
                # どちらも失敗した場合は、元のエラー内容と併せて上げる
                raise RuntimeError(f"Primary OpenAI failed: {primary_error}; Secondary failed: {secondary_error}")
        else:
            # セカンダリがない/利用不可なら元のエラーをそのまま
            raise

    if assistant_text is None:
        raise RuntimeError("AIレスポンスが取得できませんでした。")

    print(f"[AI_IO] AI出力文字数: {len(assistant_text)}")

    return assistant_text


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

    # コンテキストを構築（論理エンジン用）
    def build_context() -> Dict[str, Any]:
        session_id = session['session_id']
        sdata = session_repository.get_session(session_id) or {}
        decision_data = sdata.get('decision_data', {})
        student = sdata.get('student_data') or session.get('student_info') or {}

        participant_opinions = decision_data.get('participant_opinions') or []
        participant_decisions = decision_data.get('participant_decisions') or []

        ctx = {
            'session_id': session_id,
            'student_info': student,
            'user_initial_decision': decision,
            'user_initial_weights': weights,
            'participant_decisions': participant_decisions,
            'participant_opinions': participant_opinions,
        }
        return ctx

    try:
        context = build_context()
        initial_msgs = _build_initial_messages(weights, context)
    except Exception as e:
        print(f"初期メッセージ生成エラー: {e}")
        # エラー時はコンテキストなしで生成
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
            'participant_decisions': participants,
            'participant_opinions': participant_opinions,
            'group_outcome': decision_data.get('group_outcome'),
            'questionnaire': questionnaire,
        }
        return ctx

    try:
        model = os.getenv('OPENAI_RESPONSES_MODEL', 'gpt-5-mini')
        ctx = build_llm_context()

        # === 論理エンジンによる議論分析 ===
        try:
            # 1. 論理エンジンを実行して、議論の構造を分析
            arguments = argumentation_engine.extract_atomic_arguments(ctx)
            attacks = argumentation_engine.determine_attacks(arguments)

            # ユーザーの重み配分を取得（新アルゴリズム用）
            user_weights = ctx.get('user_initial_weights') or ctx.get('user_final_weights') or {}

            # 2. 新アルゴリズム（2本立てランキング型サリエンス）を適用
            if user_weights:
                debate_summary = argumentation_engine.summarize_debate(arguments, attacks, user_weights)
            else:
                # 重み情報がない場合は既存アルゴリズムを使用
                debate_summary = argumentation_engine.summarize_debate(arguments, attacks)

            # 3. 分析結果をLLMのコンテキストに追加
            ctx['argumentation_analysis'] = debate_summary

            # 4. 議論構造に基づく推奨質問も生成
            focused_question = argumentation_engine.generate_focused_question(debate_summary, messages)
            ctx['suggested_question'] = focused_question

            # デバッグ出力（論理エンジン分析結果）
            if str(os.getenv('DEBUG_LLM_CONTEXT', '')).lower() in ('1', 'true', 'yes', 'on'):
                print("\n===== 論理エンジン分析結果 =====")
                print(f"抽出された主張数: {len(arguments)}")
                print(f"攻撃関係数: {len(attacks)}")
                print(f"使用アルゴリズム: {debate_summary.get('algorithm_type', 'legacy')}")
                print(f"論点: {debate_summary.get('key_conflict_point', 'N/A')}")
                print(f"推奨質問: {focused_question}")
                if debate_summary.get('algorithm_type') == 'two_track_ranking_salience':
                    detailed = debate_summary.get('detailed_analysis', {})
                    print(f"検出された対立点数: {len(detailed.get('conflict_points', []))}")
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
                print("\n===== DEBUG: LLM CONTEXT (sanitized) =====")
                print(json.dumps(_sanitize_llm_context(ctx), ensure_ascii=False, indent=2))
                print("===== END CONTEXT =====\n")
        except Exception:
            pass

        # Save last context to DB for inspection in admin or scripts
        try:
            sdata = session_repository.get_session(session['session_id']) or {}
            ai_chat_data = sdata.get('ai_chat_data', {}) or {}
            ai_chat_data['last_llm_context'] = _sanitize_llm_context(ctx)
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

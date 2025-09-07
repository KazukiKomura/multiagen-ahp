import os
import streamlit as st
from typing import List, Dict, Tuple
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Allow the app to render a helpful error if package missing


PAGE_TITLE = "簡易AIチャットボット（手続き説明を最初に表示）"
DEFAULT_SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"


RULES_BUBBLE_TEXT = """【手続とルールのご案内】

本システムでは以下のルールに基づいて評価を行います：

**評価基準**
- 5項目（学業成績、研究能力、コミュニケーション、リーダーシップ、将来性）の加重平均
- 総合判定：あなたの重み配分 + 参加者評価者2名の多数決

**重要な制約**
- AIは結果を変更できません
- 誤読・見落としがあれば異議申し立てで確認します
- すべての評価根拠を透明に開示します

**今後の流れ**
1. あなたの重視点の確認
2. 合格・不合格の観点整理と質問・異議機会
3. 最終結果の要約
"""


def get_system_prompt() -> str:
    """システムプロンプトを.envから読み込む（ファイル指定にも対応）"""
    # .env の読み込み
    load_dotenv()

    # 優先度: SYSTEM_PROMPT_FILE > SYSTEM_PROMPT
    file_path = os.getenv("SYSTEM_PROMPT_FILE")
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass

    # 既定の同梱ファイル
    if os.path.exists(DEFAULT_SYSTEM_PROMPT_PATH):
        try:
            with open(DEFAULT_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass

    env_prompt = os.getenv("SYSTEM_PROMPT")
    if env_prompt and env_prompt.strip():
        return env_prompt

    # フォールバック（簡易版）
    return (
        "あなたは合否判断の合意形成を支援するAIファシリテータです。"
        "ユーザーの重視点と2名の参加者の観点を踏まえ、簡潔に状況整理し、1つの質問のみを行ってください。"
    )


def _init_state():
    if "system_text" not in st.session_state:
        st.session_state.system_text = get_system_prompt()

    if "weights" not in st.session_state:
        # 既定値（例）
        st.session_state.weights = {
            "学業成績": 30,
            "研究能力": 25,
            "コミュニケーション": 20,
            "リーダーシップ": 10,
            "将来性": 15,
        }

    if "messages" not in st.session_state:
        # 1) 最初の吹き出し：ルール案内
        msgs = [{"role": "assistant", "content": RULES_BUBBLE_TEXT}]

        # 2) 次の吹き出し：重み配分確認と質問（LLMには送らない）
        w = st.session_state.weights
        # 上位2項目
        top = sorted(w.items(), key=lambda kv: kv[1], reverse=True)[:2]
        top_criteria = [top[0][0], top[1][0]] if len(top) >= 2 else [top[0][0], ""]
        weights_text = (
            "【あなたの重視点について】\n\n"
            "UIで設定された重み配分を確認しました：\n"
            f"- 学業成績: {w['学業成績']}%\n"
            f"- 研究能力: {w['研究能力']}%  \n"
            f"- コミュニケーション: {w['コミュニケーション']}%\n"
            f"- リーダーシップ: {w['リーダーシップ']}%\n"
            f"- 将来性: {w['将来性']}%\n\n"
            f"あなたが特に{top_criteria[0]}を重視される理由について、詳しくお聞かせください。\n"
            "この学生の評価においてなぜこれらの項目を重要と考えられたのでしょうか？\n\n"
            "なお、参加者評価者2名もそれぞれ異なる基準を持って評価を行っています。"
        )
        msgs.append({"role": "assistant", "content": weights_text})

        st.session_state.messages = msgs


def _get_openai_client():
    # .env を読み込んでからキーを参照
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が環境変数に設定されていません。")
    if OpenAI is None:
        raise RuntimeError("openai パッケージが見つかりません。requirements に追加し、インストールしてください。")
    return OpenAI(api_key=api_key)


def _build_responses_input(messages: List[Dict], system_text: str):
    # OpenAI Responses API の input 形式に変換
    input_seq = [
        {
            "role": "system",
            # Responses API は 'text' ではなく 'input_text' を要求
            "content": [{"type": "input_text", "text": system_text}]
        }
    ]
    # 動的な意思決定データ（UIの重み等）を system で追加
    w = st.session_state.get("weights", {})
    decision_data_json = (
        '{\n'
        '  "student_info": {\n'
        '    "name": "田中太郎",\n'
        '    "student_id": "S2024001",\n'
        '    "scores": {\n'
        '      "学業成績": 85,\n'
        '      "研究能力": 78,\n'
        '      "コミュニケーション": 82,\n'
        '      "リーダーシップ": 65,\n'
        '      "将来性": 79\n'
        '    }\n'
        '  },\n'
        '  "user_weights": {\n'
        f'    "学業成績": {w.get("学業成績", 30)},\n'
        f'    "研究能力": {w.get("研究能力", 25)},\n'
        f'    "コミュニケーション": {w.get("コミュニケーション", 20)},\n'
        f'    "リーダーシップ": {w.get("リーダーシップ", 10)},\n'
        f'    "将来性": {w.get("将来性", 15)}\n'
        '  },\n'
        '  "user_decision": "合格",\n'
        '  "participant_decisions": {\n'
        '    "participant1": {\n'
        '      "decision": "不合格",\n'
        '      "weights": {\n'
        '        "リーダーシップ": 40,\n'
        '        "総合バランス": 30,\n'
        '        "学業成績": 15,\n'
        '        "研究能力": 10,\n'
        '        "将来性": 5\n'
        '      }\n'
        '    },\n'
        '    "participant2": {\n'
        '      "decision": "不合格",\n'
        '      "weights": {\n'
        '        "学業成績": 35,\n'
        '        "研究能力": 30,\n'
        '        "コミュニケーション": 20,\n'
        '        "リーダーシップ": 10,\n'
        '        "将来性": 5\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '}\n'
    )
    input_seq.append({
        "role": "system",
        "content": [{"type": "input_text", "text": "# 生徒/意思決定データ\n" + decision_data_json}]
    })
    # 最初の2つの assistant バブル（ルール案内 + 重み確認）は画面表示専用。LLMへは投入しない。
    for idx, m in enumerate(messages):
        if idx in (0, 1) and m.get("role") == "assistant":
            continue
        role = m["role"]
        # user/system は 'input_text'、assistant は過去出力として 'output_text'
        content_type = "output_text" if role == "assistant" else "input_text"
        input_seq.append({
            "role": role,
            "content": [{"type": content_type, "text": m["content"]}]
        })
    return input_seq


def _call_llm(client: "OpenAI", messages: List[Dict], system_text: str, model: str = "gpt-4.1") -> str:
    input_payload = _build_responses_input(messages, system_text)
    resp = client.responses.create(
        model=model,
        input=input_payload,
        temperature=0.4,
        max_output_tokens=1024,
        top_p=1
    )
    # SDK v1.35+ は output_text を提供
    text = getattr(resp, "output_text", None)
    if text:
        return text
    # 念の為のフォールバック
    try:
        return resp.output[0].content[0].text  # type: ignore[attr-defined]
    except Exception:
        return str(resp)


def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🤖", layout="centered")
    st.title(PAGE_TITLE)

    _init_state()

    with st.expander("環境設定", expanded=False):
        st.write(".env から OPENAI_API_KEY・SYSTEM_PROMPT を読み込みます。必要に応じて設定してください。")
        model = st.text_input("モデル", value="gpt-4.1", help="OpenAI Responses API 対応モデル名")

    with st.sidebar:
        st.subheader("重み設定（合計100%）")
        w = st.session_state.weights
        col1, col2 = st.columns(2)
        with col1:
            w_gaku = st.number_input("学業成績", min_value=0, max_value=100, value=int(w["学業成績"]))
            w_com = st.number_input("コミュニケーション", min_value=0, max_value=100, value=int(w["コミュニケーション"]))
            w_fut = st.number_input("将来性", min_value=0, max_value=100, value=int(w["将来性"]))
        with col2:
            w_ken = st.number_input("研究能力", min_value=0, max_value=100, value=int(w["研究能力"]))
            w_lead = st.number_input("リーダーシップ", min_value=0, max_value=100, value=int(w["リーダーシップ"]))

        total = w_gaku + w_ken + w_com + w_lead + w_fut
        st.caption(f"現在の合計: {total}%")
        apply_btn = st.button("重みを適用して会話をリセット")
        if apply_btn:
            st.session_state.weights = {
                "学業成績": int(w_gaku),
                "研究能力": int(w_ken),
                "コミュニケーション": int(w_com),
                "リーダーシップ": int(w_lead),
                "将来性": int(w_fut),
            }
            # 初期2バブルを最新の重みで再生成、履歴は破棄
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.experimental_rerun()

    # 既存メッセージの描画
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 入力欄
    user_input = st.chat_input("メッセージを入力...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            client = _get_openai_client()
            assistant_text = _call_llm(client, st.session_state.messages, st.session_state.system_text, model=model)
        except Exception as e:
            assistant_text = f"エラーが発生しました: {e}"

        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)


if __name__ == "__main__":
    main()

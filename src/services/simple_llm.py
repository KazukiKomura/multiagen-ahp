import os
from typing import Any, Dict


class SimpleLLMResponder:
    """
    極力シンプルなLLM呼び出し。
    - 使う情報はユーザーの事前入力（合否=decision, 重み=weights）のみ
    - それを深掘りする短い応答（1承認+1質問）を日本語で返す
    - 外部情報・他プロフィール要素（GPAなど）は参照しない
    - 出力はプレーンテキスト（JSONではない）
    """

    def __init__(self) -> None:
        # ご要望に合わせ既定を gpt-5-mini に。未対応環境向けにフォールバックも用意。
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-chat-latest")
        # カンマ区切りで複数指定可能（先頭から順に試行）
        self.fallback_models = [m.strip() for m in os.getenv(
            "OPENAI_FALLBACK_MODELS",
            "gpt-4o-mini,gpt-4o,gpt-3.5-turbo"
        ).split(",") if m.strip()]

    def _get_llm(self):
        try:
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            print(f"[SimpleLLM] OpenAI init error: {type(e).__name__}: {e}")
            raise RuntimeError("OpenAIクライアント初期化に失敗しました")

    def generate(self, user_message: str, decision_data: Dict[str, Any], fallback_decision: str = None) -> str:
        client = self._get_llm()
        if client is None:
            raise RuntimeError("OpenAIクライアント初期化に失敗しました")

        decision = (decision_data or {}).get('user_decision') or fallback_decision or '未定'
        weights = (decision_data or {}).get('user_weights', {})

        system_content = (
            "あなたは簡潔で敬意ある日本語アシスタントです。"
            "ユーザーが事前に示した『合否の判断』と『重み(スクロールバー)』にのみ基づいて対話を深掘りします。"
            "外部情報や他のプロフィール要素には言及しません。"
            "出力はプレーンテキスト1–2文のみ（JSONや箇条書きは禁止）。"
            "構成: 1文目で相手の意図の承認/要約、2文目で1つだけ具体的な質問。合計150字以内。"
        )

        user_content = f"判断: {decision}, 重み: {weights}, メッセージ: {user_message}"
        
        input_data = [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "input_text",
                        "text": system_content
                    }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "input_text",
                        "text": user_content
                    }
                ]
            }
        ]

        # モデル存在/権限エラー時のフォールバックを実装
        tried = []
        last_err = None
        for m in [self.model] + self.fallback_models:
            if not m or m in tried:
                continue
            tried.append(m)
            try:
                # GPT-5系モデルの場合は新しいresponses.create APIを使用
                if m.startswith("gpt-5"):
                    resp = client.responses.create(
                        model=m,
                        input=input_data,
                        text={
                            "format": {"type": "text"},
                            "verbosity": "medium"
                        },
                        tools=[],
                        temperature=0.7,
                        max_output_tokens=512,
                        top_p=1,
                        store=True,
                        include=["web_search_call.action.sources"]
                    )
                    # GPT-5 responses構造に合わせて修正
                    if hasattr(resp, 'output_text'):
                        content = resp.output_text
                    elif hasattr(resp, 'text') and hasattr(resp.text, 'content'):
                        content = resp.text.content
                    elif hasattr(resp, 'choices') and resp.choices:
                        content = resp.choices[0].message.content
                    else:
                        content = str(resp)
                else:
                    # 従来モデルは既存のchat.completions.create APIを使用
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ]
                    resp = client.chat.completions.create(
                        model=m,
                        messages=messages,
                        temperature=0.2,
                    )
                    content = (resp.choices[0].message.content or "").strip()
                    
                print(f"[SimpleLLM][RAW] model={m}: {content}")
                if content:
                    return content
            except Exception as e:
                last_err = e
                print(f"[SimpleLLM][Error] model={m}: {type(e).__name__}: {e}")

        # すべて失敗
        raise RuntimeError(f"LLM呼び出しに失敗しました（試行: {tried}, 最終エラー: {last_err}")

    def check_japanese_quality(self, user_message: str) -> Dict[str, Any]:
        """
        ユーザーの回答が適切な日本語になっているかをチェック（tinyLLM機能）
        フローの「きちんとした日本語になっているか、次のフェーズに進んでいいかのチェック」を実装
        """
        client = self._get_llm()
        if client is None:
            return {"is_valid": True, "reason": "LLM初期化失敗のためスキップ"}
        
        system_content = (
            "あなたは日本語の品質をチェックするアシスタントです。"
            "ユーザーの回答が以下の基準を満たしているかを判定してください："
            "1. 日本語として文法的に正しい 2. 意味が理解できる 3. 適切な長さである"
            "判定結果をJSON形式で返してください: {\"is_valid\": true/false, \"reason\": \"理由\"}"
        )
        
        user_content = f"以下の文章をチェックしてください: 「{user_message}」"
        
        try:
            # 最もシンプルなモデルを使用（効率化のため）
            model = self.fallback_models[0] if self.fallback_models else "gpt-3.5-turbo"
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=200
            )
            
            content = (resp.choices[0].message.content or "").strip()
            
            # JSONパースを試行
            import json
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # JSON形式でない場合は内容から判定
                is_valid = "true" in content.lower() or "有効" in content or "適切" in content
                return {
                    "is_valid": is_valid,
                    "reason": content if len(content) < 100 else "品質チェック完了"
                }
                
        except Exception as e:
            print(f"[SimpleLLM][QualityCheck] Error: {e}")
            # エラー時はデフォルトで通す
            return {"is_valid": True, "reason": "チェック機能エラーのためスキップ"}

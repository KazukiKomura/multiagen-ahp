"""
LLM utilities for procedural justice flow.

Contains LLM client initialization and judgment functions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _get_openai_client():
    try:
        import os
        from openai import OpenAI
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            return None
        return OpenAI(api_key=key)
    except Exception as e:
        raise ValueError(f"Failed to initialize OpenAI client: {e}")


def judge_meaningful(text: str) -> bool:
    """Tiny LLMでの具体性判定（失敗時はFalse）。"""
    m = (text or '').strip()
    if not m:
        return False
    client = _get_openai_client()
    if not client:
        return False
    import os, json as _json
    from .prompts import MEANINGFUL_SYSTEM_PROMPT
    model = os.getenv('OPENAI_MIN_JUDGE_MODEL', 'gpt-4o-mini')
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MEANINGFUL_SYSTEM_PROMPT},
                {"role": "user", "content": _json.dumps({"text": m, "criterion": "言語的に成立し、重視点や理由などの具体言及"}, ensure_ascii=False)}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        data = _json.loads(resp.choices[0].message.content)
        return bool(data.get('meaningful', False))
    except Exception:
        return False


def classify_intent(text: str, rule_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Tiny LLMベースのインテント分類（question/appeal/other）。"""
    m = (text or '').strip()
    if not m:
        return {'intent': 'other', 'confidence': 0.0}
    client = _get_openai_client()
    if not client:
        return {'intent': 'other', 'confidence': 0.0}
    import os, json as _json
    from .prompts import INTENT_SYSTEM_PROMPT
    model = os.getenv('OPENAI_MIN_JUDGE_MODEL', 'gpt-4o-mini')
    system = INTENT_SYSTEM_PROMPT
    user = {"text": m, "options": ["question","appeal","other"], "rule_summary": rule_summary or {}}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": _json.dumps(user, ensure_ascii=False)}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        data = _json.loads(resp.choices[0].message.content)
        intent = data.get('intent') if isinstance(data, dict) else 'other'
        conf = float(data.get('confidence', 0)) if isinstance(data, dict) else 0.0
        return {'intent': intent if intent in ('question','appeal','other') else 'other', 'confidence': conf}
    except Exception:
        return {'intent': 'other', 'confidence': 0.0}
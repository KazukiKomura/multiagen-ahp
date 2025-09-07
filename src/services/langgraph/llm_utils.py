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
    """Intent classification with safe heuristics first, LLM fallback second.

    Returns a dict: {"intent": "question|appeal|other", "confidence": float}
    Heuristics are conservative to avoid false "appeal" positives.
    """
    m = (text or '').strip()
    if not m:
        return {'intent': 'other', 'confidence': 0.0}

    low = m.lower()

    # 1) Deterministic question detection (Japanese + symbols)
    # Typical markers: question marks, polite question endings, interrogatives
    question_markers = ['?', '？', 'か？', 'ますか', 'ですか', 'でしょうか', 'できますか', '教えて']
    interrogatives = ['なぜ', 'なんで', 'どうして', 'どうやって', 'どのように', 'いつ', 'どこ', 'だれ', 'どれ', 'どちら', '何', 'なんですか']
    if any(tok in m for tok in question_markers) or any(tok in m for tok in interrogatives):
        return {'intent': 'question', 'confidence': 0.95}

    # 2) Deterministic appeal detection (explicit keywords only)
    # Only treat as appeal when the user clearly requests reconsideration/correction
    appeal_keywords = [
        '異議', '不服', '再考', '再評価', '再審', '見直', '取り消', '訂正', 'やり直',
        '誤り', '誤って', '見落とし', '覆して', '覆る', '再判定', '申し立て', '申立',
        '訴え', '変更して', '変えて', 'appeal', 'contest'
    ]
    # Negation patterns that explicitly state "no appeal"
    appeal_negations = [
        '異議なし', '異議はありません', '異議ありません', '異議はない', '異議ない',
        '異議はしません', '異議はございません', '異議ございません', '異議を申し立てません'
    ]
    if any(n in m for n in appeal_negations):
        return {'intent': 'other', 'confidence': 0.9}
    if any(k in m for k in appeal_keywords):
        return {'intent': 'appeal', 'confidence': 0.9}

    # 3) Obvious acknowledgements that should NOT be appeal
    acknowledgements = ['わかりました', '了解', '理解しました', 'はい', '承知', 'OK', 'ok', '大丈夫です']
    if any(a.lower() in low for a in acknowledgements):
        return {'intent': 'other', 'confidence': 0.8}

    # 4) Simple justification phrases often used for preferences (not appeals)
    # e.g., "〜が重要だからです", "〜を重視するためです"
    justification_markers = ['だからです', 'ためです', '重視', '重要', '理由は']
    if any(j in m for j in justification_markers):
        return {'intent': 'other', 'confidence': 0.7}

    # 5) Fallback to tiny LLM if available for nuance
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
        # Be conservative with appeals unless confidence is reasonably high
        if intent == 'appeal' and conf < 0.6:
            return {'intent': 'other', 'confidence': conf}
        return {'intent': intent if intent in ('question','appeal','other') else 'other', 'confidence': conf}
    except Exception:
        return {'intent': 'other', 'confidence': 0.0}

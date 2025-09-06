"""
Node handlers for the procedural justice flow.

Contains all the node processing logic for different stages of the conversation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from .llm_utils import FlowState, judge_meaningful, classify_intent

if TYPE_CHECKING:
    from .procedural_justice import LGFlow


class NodeHandlers:
    """Handles all node processing for the procedural justice flow."""
    
    def __init__(self, flow: 'LGFlow'):
        self.flow = flow
    
    def node_rules(self, state: FlowState) -> FlowState:
        if state.get('rules_shown'):
            # skip re-sending; move to next stored route or Prefs by default
            state['route'] = state.get('route') or 'Prefs'
            return state
        state['ai_response'] = (
            "【手続とルールのご案内】\n"
            "基準＝UIで定義された5項目の加重評価／閾値＝各項目2.5点以上。"
            "AIは結果を変更できません（誤読・見落としは異議で確認します）。\n"
            "流れ：①重視点の確認→②賛否の観点整理と質問・異議→③要請結果（ある場合）→④まとめ。"
        )
        state['rules_shown'] = True
        state['route'] = 'Prefs'
        self.flow._log(state, 'Stage0_Rules', '', state['ai_response'])
        return state

    def node_prefs(self, state: FlowState) -> FlowState:
        weights = state.get('user_weights') or {}
        last = state.get('last_user_input', '')
        payload = {"user_weights": weights, "last_user_text": last}
        if not last or not judge_meaningful(last):
            # LLMで短い再確認/深掘りを生成
            msg = self.flow._llm_generate('stage1_prefs_prompt', payload)
            state['ai_response'] = msg
            state['route'] = 'Prefs'  # wait for meaningful input
            self.flow._log(state, 'Stage1_Preferences_Prompt', last, msg)
            return state
        # meaningful -> LLMで了解文
        ack = self.flow._llm_generate('stage1_prefs_ack', payload)
        state['ai_response'] = ack
        state['route'] = 'Analysis'
        self.flow._log(state, 'Stage1_Preferences_Ack', last, ack)
        return state

    def node_analysis(self, state: FlowState) -> FlowState:
        last = state.get('last_user_input', '')
        # ここではLLMに全面委譲し、構造化はプロンプト側の指示に任せる

        # If user just arrived, present analysis via LLM
        if not last:
            # 集計: 他評価者の結論と2対1状況（簡易）
            bots = state.get('bot_evaluators', {}) or {}
            bot_summary = {k: v.get('decision') for k, v in bots.items() if isinstance(v, dict)}
            ratio = None
            try:
                user = '合格'
                others = list(bot_summary.values())
                ratio = f"{others.count('不合格')}対{1 if user=='合格' else 0}"
            except Exception:
                ratio = None
            payload = {
                "profile_facts": state.get('student_info', {}),
                "user_weights": state.get('user_weights', {}),
                "minority": True,
                "bot_evaluators": bot_summary,
                "ratio": ratio
            }
            resp = self.flow._llm_generate('stage2_present', payload)
            state['ai_response'] = resp
            state['route'] = 'Analysis'
            self.flow._log(state, 'Stage2_Analysis_Present', last, state['ai_response'])
            return state

        # Intent classification by tiny LLM (question / appeal / other)
        cls = classify_intent(last, state.get('rule_summary', {}))
        intent = cls.get('intent', 'other')

        # If question -> provide brief answer from rule/threshold
        if intent == 'question':
            rs = state.get('rule_summary', {})
            th = state.get('threshold')
            payload = {"question": last, "rule_summary": rs, "threshold": th}
            resp = self.flow._llm_generate('stage2_answer', payload)
            state['ai_response'] = resp
            state['route'] = 'Analysis'  # remain; next input routes forward
            self.flow._log(state, 'Stage2_Analysis_Answer', last, state['ai_response'])
            return state

        # If appeal/request
        if intent == 'appeal':
            state['appeal_made'] = True
            payload = {"appeal_text": last}
            resp = self.flow._llm_generate('stage2_appeal_ack', payload)
            state['ai_response'] = resp
            state['route'] = 'RequestFeedback'
            self.flow._log(state, 'Stage2_Analysis_AppealAccepted', last, state['ai_response'])
            return state

        # Otherwise: if meaningful enough, proceed
        if judge_meaningful(last):
            resp = self.flow._llm_generate('stage2_proceed', {"last": last})
            state['ai_response'] = resp
            state['route'] = 'WrapUp'
            self.flow._log(state, 'Stage2_Analysis_Proceed', last, state['ai_response'])
            return state

        # Not meaningful -> brief re-ask (LLMに短く依頼)
        resp = self.flow._llm_generate('stage2_weak_reprompt', {"last": last})
        state['ai_response'] = resp
        state['route'] = 'Analysis'
        self.flow._log(state, 'Stage2_Analysis_Reprompt', last, state['ai_response'])
        return state

    def node_request_feedback(self, state: FlowState) -> FlowState:
        resp = self.flow._llm_generate('stage2_5_feedback', {"appeal_made": state.get('appeal_made', False)})
        state['ai_response'] = resp
        state['route'] = 'WrapUp'
        self.flow._log(state, 'Stage2_5_RequestFeedback', state.get('last_user_input',''), state['ai_response'])
        return state

    def node_wrapup(self, state: FlowState) -> FlowState:
        resp = self.flow._llm_generate('stage3_wrapup', {"minority": True})
        state['ai_response'] = resp
        state['route'] = 'END'
        self.flow._log(state, 'Stage3_WrapUp', state.get('last_user_input',''), state['ai_response'])
        return state
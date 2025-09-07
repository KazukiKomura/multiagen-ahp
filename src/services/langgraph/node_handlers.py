"""
Node handlers for the procedural justice flow.

Contains all the node processing logic for different stages of the conversation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import FlowState
from .llm_utils import judge_meaningful, classify_intent

if TYPE_CHECKING:
    from .flow_engine import LGFlow


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
        def _is_proceed_signal(text: str) -> bool:
            t = (text or '').strip()
            if not t:
                return False
            low = t.lower()
            proceed_markers = [
                '次へ', '次に進', '進めて', '進んで', '続けて', 'まとめへ',
                'ok', 'OK', 'オーケー', 'おっけ', 'おっけい', 'おっけー'
            ]
            return any(m in t for m in proceed_markers) or 'ok' in low

        weights = state.get('user_weights') or {}
        last = state.get('last_user_input', '')
        payload = {"user_weights": weights, "last_user_text": last}

        # ユーザーの重視点の言語化をまず確実に済ませる
        confirmatory_responses = ['わかりました', '了解', '理解しました', 'はい', 'わかった']
        is_confirm_only = any(conf in last for conf in confirmatory_responses) if last else False

        # まだPrefsの確定OKが出ていない場合は進まない
        if not state.get('prefs_ok', False):
            if not last or not judge_meaningful(last) or is_confirm_only:
                # 深掘りリクエストを継続
                msg = self.flow._llm_generate('stage1_prefs_prompt', payload)
                state['ai_response'] = msg
                state['route'] = 'Prefs'
                self.flow._log(state, 'Stage1_Preferences_Prompt', last, msg)
                return state

            # 明示的に進行OKが出た場合のみ次へ
            if _is_proceed_signal(last):
                state['prefs_ok'] = True
            else:
                # 了解文＋「次へ進めて良いか確認」
                ack = self.flow._llm_generate('stage1_prefs_ack', payload)
                confirm_line = 'この内容で観点整理（次のステップ）に進めてよろしいですか？「次へ」などでお知らせください。'
                state['ai_response'] = f"{ack}\n\n{confirm_line}"
                state['route'] = 'Prefs'
                self.flow._log(state, 'Stage1_Preferences_Ack_WaitConfirm', last, state['ai_response'])
                return state

        # ここまでで prefs_ok がTrueならAnalysis用の提示を生成
        bots = state.get('bot_evaluators', {}) or {}
        bot_summary = {k: v.get('decision') for k, v in bots.items() if isinstance(v, dict)}

        ratio = None
        try:
            user = '合格'
            others = list(bot_summary.values())
            ratio = f"{others.count('不合格')}対{1 if user=='合格' else 0}"
        except Exception:
            ratio = None

        analysis_payload = {
            "profile_facts": state.get('student_info', {}),
            "user_weights": state.get('user_weights', {}),
            "minority": True,
            "bot_evaluators": bot_summary,
            "ratio": ratio
        }
        analysis = self.flow._llm_generate('stage2_present', analysis_payload)
        state['ai_response'] = analysis
        state['route'] = 'Analysis'
        self.flow._log(state, 'Stage1_to_Stage2_Present', last, state['ai_response'])
        return state

    def node_analysis(self, state: FlowState) -> FlowState:
        def _is_proceed_signal(text: str) -> bool:
            t = (text or '').strip()
            if not t:
                return False
            low = t.lower()
            proceed_markers = [
                '次へ', 'まとめへ', '進めて', '進んで', '次に進', '続けて',
                'ok', 'OK', 'オーケー', 'おっけ', 'おっけい', 'おっけー'
            ]
            return any(m in t for m in proceed_markers) or 'ok' in low

        last = state.get('last_user_input', '')
        
        # ここではLLMに全面委譲し、構造化はプロンプト側の指示に任せる
        # stage2_present は既に node_prefs で処理済み

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

        # 明示的な進行OKがある場合のみWrapUpへ
        if _is_proceed_signal(last):
            resp = self.flow._llm_generate('stage2_proceed', {"last": last})
            state['ai_response'] = resp
            state['route'] = 'WrapUp'
            self.flow._log(state, 'Stage2_Analysis_Proceed', last, state['ai_response'])
            return state

        # まだ質問・異議・進行の明示なし -> 短く促す
        if judge_meaningful(last):
            prompt = '追加の質問または異議があれば1点に絞ってお聞かせください。なければ「次へ」とお伝えください。'
            state['ai_response'] = prompt
            state['route'] = 'Analysis'
            self.flow._log(state, 'Stage2_Analysis_WaitConfirm', last, state['ai_response'])
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

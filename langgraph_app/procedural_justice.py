"""
LangGraph implementation of the staged conversational flow.

Stages (see docs/requirements/procedural_justice_langgraph.md):
 0: Rules (fixed message)
 1: Preferences confirmation & deep dive (minimal judge)
 2: Minority framing, symmetric pros/cons, questions & appeal
 2.5: Request feedback (if appeal)
 3: Wrap-up

If the official `langgraph` package is unavailable, the flow falls back to a
single-node runner that preserves the same step-by-step semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

# Logging only; we do not reuse the FSM now
import sys
sys.path.append('src')
from services.session_logger import SessionLogger  # noqa: E402

from .llm_utils import FlowState, judge_meaningful, classify_intent
from .node_handlers import NodeHandlers








@dataclass
class LGFlow:
    enable_logging: bool = True
    session_id: Optional[str] = None
    logger: Optional[SessionLogger] = field(init=False, default=None)
    _llm_meta: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.enable_logging:
            self.logger = SessionLogger(self.session_id)
        self.handlers = NodeHandlers(self)

    # ---- LLM helpers (src/services/procedural_justice.py 準拠の最小版) ----
    def _get_llm(self):
        try:
            import os
            from openai import OpenAI
            # Prefer GPT-5 by default for response generation
            model = os.getenv("OPENAI_MODEL", "gpt-5-chat-latest")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            fallbacks = [m.strip() for m in os.getenv(
                "OPENAI_FALLBACK_MODELS",
                "gpt-5-chat-latest,gpt-5-reasoner-latest,gpt-4o,gpt-4o-mini"
            ).split(",") if m.strip()]
            return client, model, fallbacks
        except Exception:
            raise Exception("Failed to initialize LLM")

    def _llm_generate(self, stage: str, payload: Dict[str, Any]) -> str:
        import json, time
        from .prompts import SYSTEM_PROMPT, ROLE_INSTRUCTIONS
        system = SYSTEM_PROMPT
        role_instr = ROLE_INSTRUCTIONS.get(stage, "簡潔に。")

        client, model, fallbacks = self._get_llm()
        self._llm_meta = {"stage": stage, "tried": [], "used": False, "model": None, "duration_ms": 0, "error": None}
        tried = []
        for m in [model] + fallbacks:
            if not client or not m:
                continue
            tried.append(m)
            try:
                t0 = time.time()
                if m.startswith("gpt-5"):
                    input_data = [
                        {"role": "system", "content": [{"type": "input_text", "text": system + "\n" + role_instr}]},
                        {"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]}
                    ]
                    resp = client.responses.create(
                        model=m,
                        input=input_data,
                        text={"format": {"type": "text"}},
                        tools=[], temperature=0.2, max_output_tokens=768, top_p=1,
                        store=False
                    )
                    raw = getattr(resp, 'output_text', None) or getattr(getattr(resp, 'text', None), 'content', None) or str(resp)
                else:
                    resp = client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": system + "\n" + role_instr},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                        ],
                        temperature=0.2,
                        response_format={"type": "json_object"}
                    )
                    raw = resp.choices[0].message.content
                dt = (time.time() - t0) * 1000
                data = json.loads(raw)
                items = data.get('candidates', []) if isinstance(data, dict) else []
                for it in items:
                    if isinstance(it, str) and it.strip():
                        self._llm_meta.update({"stage": stage, "tried": tried[:], "used": True, "model": m, "duration_ms": dt, "error": None})
                        return it.strip()[:600]
                # if empty, continue to fallback
            except Exception as e:
                self._llm_meta.update({"stage": stage, "tried": tried[:], "used": False, "model": m, "error": f"{type(e).__name__}: {e}"})
                continue
        # fallback minimal
        if not client:
            self._llm_meta.update({"stage": stage, "tried": tried[:], "used": False, "model": None, "error": "no_client"})
        return "（生成に失敗しました。もう一度入力をお願いします）"

    def _log(self, state: FlowState, action: str, user_input: str, response: str) -> None:
        if not self.logger:
            return
        try:
            user_ctx = {
                "stage": state.get('stage'),
                "route": state.get('route'),
                "appeal_made": state.get('appeal_made', False),
                "llm": self._llm_meta,
            }
            self.logger.log_turn(
                turn_number=state.get('turn', 0),
                action=action,
                user_input=user_input or '',
                user_context=user_ctx,
                ai_candidates=[],
                selected_response=response,
                judge_evaluation={"candidates_count": 0, "selected_index": 0, "action": action, "block": False},
                watchdog_evaluation={"overall": 0},
                validation_results=None,
                processing_time_ms=self._llm_meta.get('duration_ms', 0)
            )
        except Exception:
            pass

    # ---- Node handlers -----------------------------------------------------
    def node_rules(self, state: FlowState) -> FlowState:
        return self.handlers.node_rules(state)

    def node_prefs(self, state: FlowState) -> FlowState:
        return self.handlers.node_prefs(state)

    # -- hard-coded整形は排除し、LLMに委譲 --

    def node_analysis(self, state: FlowState) -> FlowState:
        return self.handlers.node_analysis(state)

    def node_request_feedback(self, state: FlowState) -> FlowState:
        return self.handlers.node_request_feedback(state)

    def node_wrapup(self, state: FlowState) -> FlowState:
        return self.handlers.node_wrapup(state)

    # ---- Graph runner ------------------------------------------------------
    def compile(self):
        try:
            from langgraph.graph import StateGraph, START, END
            g = StateGraph(FlowState)
            g.add_node('Rules', self.handlers.node_rules)
            g.add_node('Prefs', self.handlers.node_prefs)
            g.add_node('Analysis', self.handlers.node_analysis)
            g.add_node('RequestFeedback', self.handlers.node_request_feedback)
            g.add_node('WrapUp', self.handlers.node_wrapup)
            # Router from START to the node specified by state['route'] (default Rules)
            def router(s: FlowState) -> str:
                return s.get('route', 'Rules')
            g.add_conditional_edges(START, router, {
                'Rules': 'Rules',
                'Prefs': 'Prefs',
                'Analysis': 'Analysis',
                'RequestFeedback': 'RequestFeedback',
                'WrapUp': 'WrapUp'
            })
            # One node per invoke: each node goes directly to END
            g.add_edge('Rules', END)
            g.add_edge('Prefs', END)
            g.add_edge('Analysis', END)
            g.add_edge('RequestFeedback', END)
            g.add_edge('WrapUp', END)
            return g.compile()
        except Exception:
            # Fallback single-step executor preserving semantics
            class Fallback:
                def __init__(self, flow: 'LGFlow'):
                    self.flow = flow
                def invoke(self, state: FlowState) -> FlowState:
                    route = state.get('route') or 'Rules'
                    fn = {
                        'Rules': self.flow.handlers.node_rules,
                        'Prefs': self.flow.handlers.node_prefs,
                        'Analysis': self.flow.handlers.node_analysis,
                        'RequestFeedback': self.flow.handlers.node_request_feedback,
                        'WrapUp': self.flow.handlers.node_wrapup,
                    }.get(route, self.flow.handlers.node_rules)
                    return fn(state)
            return Fallback(self)

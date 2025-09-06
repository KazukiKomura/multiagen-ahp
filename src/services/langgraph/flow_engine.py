"""
LangGraph flow engine for procedural justice conversations.

Core LGFlow implementation with LLM integration and graph compilation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .models import FlowState
from .node_handlers import NodeHandlers
from .prompts import SYSTEM_PROMPT, ROLE_INSTRUCTIONS


@dataclass
class LGFlow:
    """LangGraph-based procedural justice flow engine."""
    
    enable_logging: bool = True
    session_id: Optional[str] = None
    logger: Optional[Any] = field(init=False, default=None)
    handlers: NodeHandlers = field(init=False)
    _llm_meta: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.enable_logging:
            # Import here to avoid circular dependency
            try:
                from ...repository.langgraph_repository import LangGraphLogger
                self.logger = LangGraphLogger(self.session_id)
            except ImportError:
                self.logger = None
        self.handlers = NodeHandlers(self)

    def _get_llm(self):
        """Get OpenAI client and model configuration."""
        try:
            import os
            from openai import OpenAI
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
        """Generate LLM response for given stage and payload."""
        system = SYSTEM_PROMPT
        role_instr = ROLE_INSTRUCTIONS.get(stage, "簡潔に。")

        client, model, fallbacks = self._get_llm()
        self._llm_meta = {"stage": stage, "tried": [], "used": False, "model": None, 
                         "duration_ms": 0, "error": None}
        
        for m in [model] + fallbacks:
            if not client or not m:
                continue
            self._llm_meta["tried"].append(m)
            
            try:
                t0 = time.time()
                if m.startswith("gpt-5"):
                    input_data = [
                        {"role": "system", "content": [{"type": "input_text", "text": system + "\n" + role_instr}]},
                        {"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]}
                    ]
                    resp = client.responses.create(
                        model=m, input=input_data, text={"format": {"type": "text"}},
                        tools=[], temperature=0.2, max_output_tokens=768, top_p=1, store=False
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
                        self._llm_meta.update({"used": True, "model": m, "duration_ms": dt, "error": None})
                        return it.strip()[:600]
                        
            except Exception as e:
                self._llm_meta.update({"used": False, "model": m, "error": f"{type(e).__name__}: {e}"})
                continue

        return "（生成に失敗しました。もう一度入力をお願いします）"

    def _log(self, state: FlowState, action: str, user_input: str, response: str) -> None:
        """Log conversation turn."""
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
                ai_response=response,
                processing_time_ms=self._llm_meta.get('duration_ms', 0)
            )
        except Exception:
            pass

    def compile(self):
        """Compile LangGraph or return fallback executor."""
        try:
            from langgraph.graph import StateGraph, START, END
            g = StateGraph(FlowState)
            g.add_node('Rules', self.handlers.node_rules)
            g.add_node('Prefs', self.handlers.node_prefs)
            g.add_node('Analysis', self.handlers.node_analysis)
            g.add_node('RequestFeedback', self.handlers.node_request_feedback)
            g.add_node('WrapUp', self.handlers.node_wrapup)
            
            def router(s: FlowState) -> str:
                return s.get('route', 'Rules')
                
            g.add_conditional_edges(START, router, {
                'Rules': 'Rules', 'Prefs': 'Prefs', 'Analysis': 'Analysis',
                'RequestFeedback': 'RequestFeedback', 'WrapUp': 'WrapUp'
            })
            
            for node in ['Rules', 'Prefs', 'Analysis', 'RequestFeedback', 'WrapUp']:
                g.add_edge(node, END)
                
            return g.compile()
            
        except Exception:
            # Fallback executor
            class Fallback:
                def __init__(self, flow: LGFlow):
                    self.flow = flow
                    
                def invoke(self, state: FlowState) -> FlowState:
                    route = state.get('route') or 'Rules'
                    fn_map = {
                        'Rules': self.flow.handlers.node_rules,
                        'Prefs': self.flow.handlers.node_prefs,
                        'Analysis': self.flow.handlers.node_analysis,
                        'RequestFeedback': self.flow.handlers.node_request_feedback,
                        'WrapUp': self.flow.handlers.node_wrapup,
                    }
                    fn = fn_map.get(route, self.flow.handlers.node_rules)
                    return fn(state)
                    
            return Fallback(self)
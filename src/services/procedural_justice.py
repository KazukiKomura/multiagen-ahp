"""
Procedural Justice System - 手続き的公正システム
Thibaut & Walker (1975) 完全準拠型LLMシステム

このシステムは有限状態機械（FSM）による優先度制御を使用し、
Voice, Neutrality, Transparency, Respect, Consistency の5つの要素を
完全に保証する手続き的公正システムを実装します。
"""

import json
import os
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    JST = ZoneInfo("Asia/Tokyo")
except Exception:
    JST = None
from typing import Any, Dict, List, Optional


class ProceduralJusticeSystem:
    """手続き的公正システム - 有限状態機械による実装"""
    
    def __init__(self, enable_logging: bool = True, session_id: Optional[str] = None):
        self.max_turns = 5
        self.transparency_deadline = 3
        self._model = os.getenv("OPENAI_MODEL", "gpt-5-chat-latest")
        # カンマ区切りで複数指定可能（先頭から順に試行）
        self.fallback_models = [m.strip() for m in os.getenv(
            "OPENAI_FALLBACK_MODELS",
            "gpt-4o-mini,gpt-4o,gpt-3.5-turbo"
        ).split(",") if m.strip()]
        
        # セッションログ機能
        self.enable_logging = enable_logging
        self.session_logger = None
        if enable_logging:
            from .session_logger import SessionLogger
            self.session_logger = SessionLogger(session_id)
        
        # 起動時診断ログ（キー本体は出力しない）
        api_key_present = bool(os.getenv("OPENAI_API_KEY"))
        print(f"[PJ] Init LLM config: model={self._model}, OPENAI_API_KEY set={api_key_present}")
        if enable_logging:
            print(f"[PJ] Session logging enabled: {self.session_logger.session_id if self.session_logger else 'None'}")
        else:
            print("[PJ] Session logging disabled")

    def _get_llm(self):
        """OpenAIクライアント取得（例外時はNone, 診断ログを出力）"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return client, self._model
        except Exception as e:
            print(f"[PJ][LLM] Client init failed: {type(e).__name__}: {e}")
            return None, self._model
        
    def execute_turn(self, message, decision, state, session_data):
        """1ターンの実行 - 優先度制御付きFSM"""
        import time
        
        start_time = time.time()
        turn_number = state.get('turn', 0)
        state_before = state.copy()
        
        # セッションメタデータを初期化時に設定
        if self.session_logger and turn_number == 1:
            self.session_logger.set_session_metadata({
                "user_decision": decision,
                "user_weights": session_data.get('decision_data', {}).get('user_weights', {}),
                "profile_facts": session_data.get('student_info', {}),
                "rule_summary": session_data.get('rule_summary', {}),
                "threshold": session_data.get('threshold'),
                "scenario_type": "procedural_justice_test",
                "test_mode": True
            })
        
        # 優先度順でアクション決定: Respect > Voice > Transparency(期限) > Neutrality > Appeal > 要約/終了
        action = self._determine_action(message, decision, state)
        
        # LLM-G: 候補文生成（1件のみ）
        candidates = self._generate_candidates(action, message, decision, state, session_data)
        # 単一候補のみ使用
        best_response = candidates[0] if candidates else ""
        
        # Judge結果を記録（ログ用）
        judge_evaluation = {"candidates_count": 1, "selected_index": 0, "action": action, "block": False}
        
        # 状態更新
        updated_state = self._update_state(action, best_response, state)
        
        # LLM-W: 満足度5軸監視
        satisfaction_scores = self._llm_watchdog(message, best_response, updated_state)
        
        # 補修システム（各軸1回まで）
        original_response = best_response
        if satisfaction_scores['overall'] < 1.2:
            repair_response = self._execute_repair(satisfaction_scores, updated_state, session_data)
            if repair_response:
                best_response += " " + repair_response
        
        # 処理時間を計算
        processing_time = (time.time() - start_time) * 1000
        
        # JSONログ記録
        if self.session_logger:
            user_context = {
                "state_before": state_before,
                "state_after": updated_state.copy(),
                "decision": decision,
                "minority_status": state.get('minority_status', False),
                "user_concern_level": 'high' if '心配' in message or '不安' in message else 'medium',
                "stage_context": {
                    'VoiceElicitation': '重視点聴取段階',
                    'NeutralityPresentation': '観点別分析段階', 
                    'AppealOffer': '異議申立て段階',
                    'SummaryAndClose': '手続完了段階'
                }.get(action, 'ルール説明段階'),
                "repair_applied": best_response != original_response
            }
            
            self.session_logger.log_turn(
                turn_number=turn_number,
                action=action,
                user_input=message,
                user_context=user_context,
                ai_candidates=[],
                selected_response=best_response,
                judge_evaluation=judge_evaluation,
                watchdog_evaluation=satisfaction_scores,
                processing_time_ms=processing_time
            )
        
        # 従来のログ記録（既存機能との互換性）
        self._log_turn(state['turn'], action, updated_state, satisfaction_scores)
        
        return {
            'response': best_response,
            'state': updated_state,
            'satisfaction_scores': satisfaction_scores,
            'action': action,
            'processing_time_ms': processing_time
        }
    
    def close_session(self, summary: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """セッションを終了してJSONファイルに保存"""
        if not self.session_logger:
            return None
        
        if summary is None:
            # デフォルトのサマリーを生成
            summary = self.session_logger.generate_analytics_summary()
        
        self.session_logger.set_session_summary(summary)
        filepath = self.session_logger.save_to_json()
        
        print(f"[PJ] Session closed and saved: {filepath}")
        return filepath
    
    def get_session_data(self) -> Dict[str, Any]:
        """現在のセッションデータを取得"""
        if self.session_logger:
            return self.session_logger.get_session_data()
        return {}
    
    def get_session_id(self) -> Optional[str]:
        """セッションIDを取得"""
        if self.session_logger:
            return self.session_logger.session_id
        return None
    
    def _determine_action(self, message, decision, state):
        """優先度に基づくアクション決定 - FSM状態遷移"""
        
        # Priority 1: Respect - 不適切発言の検出・対応
        if self._needs_respect_action(message):
            return 'RespectCorrection'

        # Priority 2: Transparency - セッション開始時は必ず先行実施
        if state.get('turn', 0) == 1 and not state['invariants']['Transparency']:
            return 'TransparencyExplanation'

        # Priority 3: Voice - ユーザー入力が具体的でない場合のみ簡易促し
        if not state['invariants']['Voice']:
            if self._is_input_specific(message, state):
                state['invariants']['Voice'] = True
                state['voice_summary_ack'] = True
            else:
                return 'VoiceElicitation'  # 具体化の軽い促しのみ実施

        # Priority 4: Transparency（未了なら実施）
        if not state['invariants']['Transparency']:
            return 'TransparencyExplanation'  # 「基準=..., 閾値=..., AIは結果変更不可」
        
        # Priority 5: Neutrality - 対称的4要素提示
        if not state['invariants']['Neutrality']:
            return 'NeutralityPresentation'  # 「合格有利/不利, 不合格有利/不利」
        
        # Priority 6: Appeal - 異議申立て機会
        if not state['appeal_offered'] and state['turn'] >= 2:
            return 'AppealOffer'  # 「1点だけ誤読・見落とし・追加情報があれば...」
        
        # Priority 7: 要約・終了条件
        if state['turn'] >= self.max_turns or self._all_invariants_satisfied(state):
            return 'SummaryAndClose'  # 「実施：重視点確認→対称提示→説明→Appeal」
        
        # デフォルト: 継続対話
        return 'ContinueDialogue'
    
    def _generate_candidates(self, action, message, decision, state, session_data):
        """LLM-G: アクション別候補文生成（k=2-3, 300字以内）"""
        try:
            client, model = self._get_llm()
            if client is None:
                raise RuntimeError("LLM client not available")

            # 役割マッピングと役割別指示
            role = {
                'VoiceElicitation': 'AskPref',
                'NeutralityPresentation': 'ProCon',
                'TransparencyExplanation': 'ExplainG',
                'AppealOffer': 'Appeal',
                'SummaryAndClose': 'SummarizeClose'
            }.get(action, 'Generic')

            system = (
                "あなたは手続的公正を担保するAIファシリテーターです。"
                "少数派の立場にあるユーザーでも尊重され、納得できる対話を提供してください。"
                "UIで与えられた事実のみを引用し、構造化された丁寧な応答を心がけてください。"
                "合否のどちらか一方だけを推しません。"
                "必要に応じて【見出し】や■箇条書きを使用し、視認性を高めてください。"
                '出力は必ずJSONのみで、{"candidates":[...]} を返し、配列要素は文字列のみ（オブジェクト禁止）。'
            )
            
            if role == 'AskPref':
                # 入力の具体性に応じて、促しを最小限に抑える
                is_specific = self._is_input_specific(message, state)
                if not is_specific:
                    role_instr = (
                        "目的: ユーザーの個別の重視点を理解し、その選択の背景にある価値観を丁寧に聴取する。"
                        "出力要件：短く一問だけ具体化を促す（例：最も重視する1点と、その理由を一文で）。"
                        "■ 構造: 【見出し】→既存重み配分への一言→一問だけ質問"
                        "■ 制約: 200字以内、質問は1つだけ、押し付けない語調"
                        "入力: {user_weights, last_user_text}"
                    )
                else:
                    role_instr = (
                        "目的: 既に具体的な入力があるため、理解の表明と次段階への橋渡しのみ行う。"
                        "出力要件：簡潔な要旨要約と了解の一言。追加質問はしない。"
                        "■ 制約: 150字以内、確認表現を入れる（例：この理解で問題ないか）"
                        "入力: {user_weights, last_user_text}"
                    )
            elif role == 'ProCon':
                role_instr = (
                    "目的: ユーザーが重視する観点を反映させながら、合格/不合格の双方に同フォーマットで長短1を提示。"
                    "出力要件: 『合格に有利：…／合格に不利：…』『不合格に有利：…／不合格に不利：…』。"
                    "ユーザーの重視点（user_weightsで高い値）に特に言及すること。"
                    "数値は入力に含まれる値のみ使用。300字以内。入力: {profile_facts, extracted_reasons, user_weights}"
                )
            elif role == 'ExplainG':
                role_instr = (
                    "目的: 外部基準・閾値・AI非介入の明示。"
                    "出力要件: 『基準＝…』『閾値＝…』『AIは結果を変更できません』。誤り/見落としは次ターンAppealで受付。300字以内。"
                    "入力: {rule_summary, threshold}"
                )
            elif role == 'Appeal':
                role_instr = (
                    "目的: 異議申立てを1回保証。"
                    "出力要件: 誤読/見落としの1点の申請依頼。反映可/不可の判断と理由を後続で告知予定。300字以内。"
                )
            elif role == 'SummarizeClose':
                role_instr = (
                    "目的: 手続的公正プロセス全体を振り返り、ユーザーの個別選択への理解を示しながら納得感と満足感を提供する。"
                    
                    "出力要件："
                    "■ 構造: プロセス完了宣言→個別選択への言及→段階別確認→手続的公正原則への言及→感謝表明"
                    "■ 必須要素："
                    "• 【手続的公正プロセス完了のお知らせ】などの見出し"
                    "• ユーザーが重視された具体的な観点への理解（「あなたが特に〇〇を重視されていることを踏まえ」等）"
                    "• 『実施手順：重視点確認→対称提示→説明→Appeal』の明記"
                    "• 1-5段階それぞれの完了確認（✓マーク使用推奨）"
                    "• Voice、Neutrality、Transparency等の原則への言及"
                    "• ユーザーの個別選択を尊重した要点を箇条書きで2-4行"
                    "• 丁寧な締めの感謝表明"
                    "■ 制約: 500字以内、完結した総括文章"
                    
                    "入力: {process_summary, decision_data, profile_facts, user_weights}"
                )
            else:  # Generic - ルール説明等
                role_instr = (
                    "目的: ルール説明時は構造化された詳細情報を提供し、手続的公正への信頼を築く。"
                    
                    "出力要件（ルール説明時）："
                    "■ 構造: 【見出し】→評価方法→手続きフロー→ユーザー権利の順序"
                    "■ 必須要素："
                    "• 【評価システムのルールをご説明します】の見出し"
                    "• ■ 評価方法: 加重平均、多数決の説明"
                    "• ■ 手続きの流れ: 1-5段階の明示（1.ルール説明 2.重視理由聴取 3.分析実施 4.追加要請可能 5.手続完了）"
                    "• ■ あなたの権利: 重み変更、再分析要請、透明性保証"
                    "■ 制約: 600字以内、構造化（■、•、数字使用）"
                    
                    "その他の場面では文脈に沿った丁寧で中立な短文。300字以内。"
                )
            
            # 追加入力の構築
            decision_data = (session_data or {}).get('decision_data', {})
            user_weights = decision_data.get('user_weights', {})
            profile_facts = (session_data or {}).get('student_info', {})
            rule_summary = (session_data or {}).get('rule_summary', {})
            threshold = (session_data or {}).get('threshold')
            
            # ユーザーの重視点の分析
            priority_analysis = self._analyze_user_priorities(user_weights, message, decision_data)
            
            # UI数値のホワイトリストを状態に格納
            def _collect_nums(obj, out):
                if isinstance(obj, dict):
                    for v in obj.values():
                        _collect_nums(v, out)
                elif isinstance(obj, list):
                    for v in obj:
                        _collect_nums(v, out)
                else:
                    if isinstance(obj, (int, float)):
                        out.add(str(obj))
            ui_numbers = set()
            _collect_nums(profile_facts, ui_numbers)
            _collect_nums(user_weights, ui_numbers)
            if threshold is not None:
                _collect_nums(threshold, ui_numbers)
            state['_ui_numbers'] = sorted(ui_numbers)

            # コンテキスト情報の拡充
            user_context = {
                'user_concern_level': 'high' if '心配' in message or '不安' in message else 'medium',
                'minority_status': True if '少数' in message or '2対1' in message else False,
                'stage_context': {
                    'VoiceElicitation': '重視点聴取段階',
                    'NeutralityPresentation': '観点別分析段階', 
                    'AppealOffer': '異議申立て段階',
                    'SummaryAndClose': '手続完了段階'
                }.get(action, 'ルール説明段階'),
                'priority_analysis': priority_analysis
            }

            user = {
                'role': role,
                'action': action,
                'message': message,
                'decision': decision,
                'last_user_text': (session_data or {}).get('last_user_text') or message,
                'user_weights': user_weights,
                'profile_facts': profile_facts,
                'extracted_reasons': state.get('extracted_reasons'),
                'rule_summary': rule_summary,
                'threshold': threshold,
                'context': user_context,
                'priority_analysis': priority_analysis,
                'constraints': {
                    'max_candidates': 3,
                    'max_chars': 600 if role in ['SummarizeClose', 'Generic'] else 400 if role == 'AskPref' else 300,
                    'string_array_only': True
                },
                'output_format': {'candidates': ['string']}
            }
            print(f"[PJ][LLM] _generate_candidates call: model={model}, action={action}, turn={state.get('turn')}")
            print(f"[PJ][LLM][INPUT] System: {system[:200]}...")
            print(f"[PJ][LLM][INPUT] Role instruction: {role_instr[:200]}...")
            print(f"[PJ][LLM][INPUT] User data: {json.dumps(user, ensure_ascii=False)[:300]}...")
            
            # フォールバック機能を含むLLM呼び出し
            tried = []
            last_err = None
            raw = None
            
            for m in [model] + self.fallback_models:
                if not m or m in tried:
                    continue
                tried.append(m)
                try:
                    # GPT-5系モデルの場合は新しいresponses.create APIを使用
                    if m.startswith("gpt-5"):
                        # responses.createのための入力データ変換
                        input_data = [
                            {
                                "role": "system", 
                                "content": [{"type": "input_text", "text": system + "\n" + role_instr}]
                            },
                            {
                                "role": "user", 
                                "content": [{"type": "input_text", "text": json.dumps(user, ensure_ascii=False)}]
                            }
                        ]
                        resp = client.responses.create(
                            model=m,
                            input=input_data,
                            text={
                                "format": {"type": "text"},
                                "verbosity": "medium"
                            },
                            tools=[],
                            temperature=0.4,
                            max_output_tokens=1024,
                            top_p=1,
                            store=True,
                            include=["web_search_call.action.sources"]
                        )
                        # GPT-5 responses構造に合わせてレスポンス取得
                        if hasattr(resp, 'output_text'):
                            raw = resp.output_text
                        elif hasattr(resp, 'text') and hasattr(resp.text, 'content'):
                            raw = resp.text.content
                        elif hasattr(resp, 'choices') and resp.choices:
                            raw = resp.choices[0].message.content
                        else:
                            raw = str(resp)
                    else:
                        # 従来モデルは既存のchat.completions.create APIを使用
                        resp = client.chat.completions.create(
                            model=m,
                            messages=[
                                {"role": "system", "content": system + "\n" + role_instr},
                                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
                            ],
                            temperature=0.4,
                            response_format={"type": "json_object"}
                        )
                        raw = resp.choices[0].message.content
                    
                    if raw:
                        break
                except Exception as e:
                    last_err = e
                    print(f"[PJ][LLM][Error] _generate_candidates model={m}: {type(e).__name__}: {e}")
            
            if not raw:
                raise RuntimeError(f"LLM呼び出しに失敗しました（試行: {tried}, 最終エラー: {last_err}")
            print(f"[PJ][LLM][OUTPUT] _generate_candidates response: {raw}")
            data = json.loads(raw)
            items = data.get('candidates', [])
            # 正規化: 文字列のみへ（単一候補に簡略化）
            candidates: List[str] = []
            for it in items:
                if isinstance(it, str):
                    candidates.append(it)
                elif isinstance(it, dict):
                    s = it.get('response') or it.get('text') or ''
                    if isinstance(s, str) and s.strip():
                        candidates.append(s)
            # サニティ：文字数トリム（制約に応じて）
            max_chars = user['constraints']['max_chars']
            trimmed = [c[:max_chars] for c in candidates if isinstance(c, str) and c.strip()]
            # 単一候補のみ保持
            if trimmed:
                trimmed = [trimmed[0]]
            print(f"[PJ][LLM] _generate_candidates ok: got={len(trimmed)} candidates")
            if not trimmed:
                # 1回だけ再試行（出力形式の強制を強める）
                print("[PJ][LLM] _generate_candidates retry: empty; enforce string array")
                role_instr_retry = role_instr + " 必ず1〜3件の非空文字列の配列 candidates を返すこと。オブジェクトは禁止。"
                
                raw2 = None
                for m in [model] + self.fallback_models:
                    if not m:
                        continue
                    try:
                        if m.startswith("gpt-5"):
                            input_data_retry = [
                                {
                                    "role": "system", 
                                    "content": [{"type": "input_text", "text": system + "\n" + role_instr_retry}]
                                },
                                {
                                    "role": "user", 
                                    "content": [{"type": "input_text", "text": json.dumps(user, ensure_ascii=False)}]
                                }
                            ]
                            resp2 = client.responses.create(
                                model=m,
                                input=input_data_retry,
                                text={
                                    "format": {"type": "text"},
                                    "verbosity": "medium"
                                },
                                tools=[],
                                temperature=0.2,
                                max_output_tokens=1024,
                                top_p=1,
                                store=True,
                                include=["web_search_call.action.sources"]
                            )
                            if hasattr(resp2, 'output_text'):
                                raw2 = resp2.output_text
                            elif hasattr(resp2, 'text') and hasattr(resp2.text, 'content'):
                                raw2 = resp2.text.content
                            elif hasattr(resp2, 'choices') and resp2.choices:
                                raw2 = resp2.choices[0].message.content
                            else:
                                raw2 = str(resp2)
                        else:
                            resp2 = client.chat.completions.create(
                                model=m,
                                messages=[
                                    {"role": "system", "content": system + "\n" + role_instr_retry},
                                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
                                ],
                                temperature=0.2,
                                response_format={"type": "json_object"}
                            )
                            raw2 = resp2.choices[0].message.content
                        
                        if raw2:
                            break
                    except Exception as e:
                        print(f"[PJ][LLM][Error] _generate_candidates retry model={m}: {type(e).__name__}: {e}")
                        continue
                print(f"[PJ][LLM][RAW] _generate_candidates response (retry): {raw2}")
                data2 = json.loads(raw2)
                items2 = data2.get('candidates', [])
                candidates2: List[str] = []
                for it in items2:
                    if isinstance(it, str):
                        candidates2.append(it)
                    elif isinstance(it, dict):
                        s = it.get('response') or it.get('text') or ''
                        if isinstance(s, str) and s.strip():
                            candidates2.append(s)
                trimmed = [c[:max_chars] for c in candidates2 if isinstance(c, str) and c.strip()]
                if trimmed:
                    trimmed = [trimmed[0]]
                print(f"[PJ][LLM] _generate_candidates retry result: got={len(trimmed)} candidates")
                if not trimmed:
                    raise RuntimeError("LLM候補生成が空です（再試行後）")
            return trimmed[:1]
        except Exception as e:
            print(f"[PJ][LLM][Error] _generate_candidates: {type(e).__name__}: {e}")
            raise
    
    def _analyze_user_priorities(self, user_weights, message, decision_data):
        """ユーザーの重視点と選択の背景を分析"""
        try:
            analysis = {
                'top_priorities': [],
                'concern_areas': [],
                'choice_reasoning': '',
                'personal_values': []
            }
            
            if not user_weights:
                return analysis
                
            # 重みの高い項目を特定
            sorted_weights = sorted(user_weights.items(), key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
            
            # 上位重視項目の特定
            if sorted_weights:
                top_weight = sorted_weights[0][1] if sorted_weights[0][1] else 0
                for item, weight in sorted_weights[:3]:
                    weight_val = float(weight) if isinstance(weight, (int, float)) else 0
                    if weight_val >= top_weight * 0.8:  # 最高値の80%以上を重視項目とする
                        analysis['top_priorities'].append({
                            'factor': item,
                            'weight': weight_val,
                            'importance': 'high' if weight_val == top_weight else 'medium-high'
                        })
            
            # メッセージからの懸念領域抽出
            concern_keywords = {
                '研究': 'research_ability',
                '学業': 'academic_performance', 
                '成績': 'academic_performance',
                '試験': 'test_scores',
                '推薦': 'recommendation',
                '多様性': 'diversity',
                '心配': 'general_concern',
                '不安': 'general_concern'
            }
            
            for keyword, area in concern_keywords.items():
                if keyword in message:
                    analysis['concern_areas'].append(area)
            
            # ユーザーの価値観推測
            if analysis['top_priorities']:
                top_factor = analysis['top_priorities'][0]['factor']
                value_mapping = {
                    'academic_performance': '学業的卓越性',
                    'research_ability': '研究への情熱',
                    'test_scores': '客観的評価',
                    'recommendation': '人間関係・信頼性',
                    'diversity': '多様性・包括性'
                }
                analysis['personal_values'] = [value_mapping.get(top_factor, '個人的基準')]
            
            # 選択の理由推測
            if len(analysis['top_priorities']) > 1:
                analysis['choice_reasoning'] = 'バランス重視型の判断'
            elif analysis['top_priorities']:
                analysis['choice_reasoning'] = f"{analysis['top_priorities'][0]['factor']}を最重要視する判断"
            else:
                analysis['choice_reasoning'] = '慎重な総合判断'
                
            return analysis
            
        except Exception as e:
            print(f"[PJ] _analyze_user_priorities error: {e}")
            return {
                'top_priorities': [],
                'concern_areas': [],
                'choice_reasoning': '',
                'personal_values': []
            }

    def _generate_neutrality_four_elements(self, decision, student_info):
        """中立性: 4要素対称提示の生成"""
        gpa = student_info.get('gpa', 3.5)
        gre_q = student_info.get('gre_quant', 160)
        
        return f"""対称的観点の提示：
・合格に有利：学業成績GPA{gpa}が基準以上、研究能力が平均的水準
・合格に不利：試験スコア定量{gre_q}に改善余地、多様性要素が限定的
・不合格に有利：基準の厳格適用の重要性、他候補との公平比較の必要性  
・不合格に不利：個別事情への配慮の価値、将来性への期待の重要性""".strip()
    
    def _generate_neutrality_alternative(self, decision, student_info):
        """中立性の代替バージョン"""
        return "多角的な観点から検討することが重要です。賛否両論を公平に検討していきましょう。"
    
    # ハード検証は今回廃止（LLM出力を直接採用）
    
    def _llm_judge(self, candidates, action, state):
        """LLM-J: ルーブリック評価による最良候補選択（LLM採点）"""
        if not candidates:
            raise RuntimeError("評価対象候補が空です")
        try:
            client, model = self._get_llm()
            if client is None:
                raise RuntimeError("LLM client not available")

            # アクション別の必須条件（満たさない候補は低得点またはblock対象）
            if action == 'NeutralityPresentation':
                action_req = "必須: 『合格に有利/合格に不利/不合格に有利/不合格に不利』の4ラベルが全て含まれること。"
            elif action == 'TransparencyExplanation':
                action_req = "必須: 『AIは結果を変更できません』の一文を含む。"
            elif action == 'VoiceElicitation':
                action_req = "必須: 相手の要旨の1文要約＋同意確認の1問を含む。"
            elif action == 'SummaryAndClose':
                action_req = "必須: 『実施手順：重視点確認→対称提示→説明→Appeal』を含み、箇条書きの要点を持つ。"
            else:
                action_req = ""

            system = (
                "あなたは審査員です。候補文を次のルーブリックで0–2点採点し、JSONのみ返してください。"
                "Rubric: R1 Voice, R2 Neutrality, R3 Accuracy, R4 Transparency, R5 Brevity (各0–2, 等重み)。"
                "Block: Appeal拒否/結果変更の示唆/攻撃・差別/外部情報の持込。" + action_req +
                '出力: {"R1":int,"R2":int,"R3":int,"R4":int,"R5":int,"block":bool,"notes":string,"best_index":int} のJSONのみ。'
            )
            user = {
                'action': action,
                'candidates': candidates,
                'state': state
            }
            print(f"[PJ][LLM] _llm_judge call: model={model}, action={action}, n={len(candidates)}")
            print(f"[PJ][LLM][INPUT] Judge system: {system[:200]}...")
            print(f"[PJ][LLM][INPUT] Judge candidates: {candidates}")
            
            # フォールバック機能を含むLLM呼び出し
            tried = []
            last_err = None
            raw = None
            
            for m in [model] + self.fallback_models:
                if not m or m in tried:
                    continue
                tried.append(m)
                try:
                    if m.startswith("gpt-5"):
                        input_data = [
                            {
                                "role": "system", 
                                "content": [{"type": "input_text", "text": system}]
                            },
                            {
                                "role": "user", 
                                "content": [{"type": "input_text", "text": json.dumps(user, ensure_ascii=False)}]
                            }
                        ]
                        resp = client.responses.create(
                            model=m,
                            input=input_data,
                            text={
                                "format": {"type": "text"},
                                "verbosity": "medium"
                            },
                            tools=[],
                            temperature=0,
                            max_output_tokens=1024,
                            top_p=1,
                            store=True,
                            include=["web_search_call.action.sources"]
                        )
                        if hasattr(resp, 'output_text'):
                            raw = resp.output_text
                        elif hasattr(resp, 'text') and hasattr(resp.text, 'content'):
                            raw = resp.text.content
                        elif hasattr(resp, 'choices') and resp.choices:
                            raw = resp.choices[0].message.content
                        else:
                            raw = str(resp)
                    else:
                        resp = client.chat.completions.create(
                            model=m,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
                            ],
                            temperature=0,
                            response_format={"type": "json_object"}
                        )
                        raw = resp.choices[0].message.content
                    
                    if raw:
                        break
                except Exception as e:
                    last_err = e
                    print(f"[PJ][LLM][Error] _llm_judge model={m}: {type(e).__name__}: {e}")
            
            if not raw:
                raise RuntimeError(f"LLM呼び出しに失敗しました（試行: {tried}, 最終エラー: {last_err}")
            print(f"[PJ][LLM][OUTPUT] _llm_judge response: {raw}")
            data = json.loads(raw)
            idx = int(data.get('best_index', 0))
            if 0 <= idx < len(candidates):
                print(f"[PJ][LLM] _llm_judge ok: best_index={idx}")
                return candidates[idx]
            return candidates[0]
        except Exception as e:
            print(f"[PJ][LLM][Error] _llm_judge: {type(e).__name__}: {e}")
            raise
    
    def _llm_watchdog(self, user_message, ai_response, state):
        """LLM-W: 満足度5軸推定＋未達インバリアント検出＋補修提案（JSON）"""
        try:
            client, model = self._get_llm()
            if client is None:
                raise RuntimeError("LLM client not available")

            system = (
                "見張りLLMとして、直近のユーザー発話・AI応答・状態を踏まえ、"
                "満足度5軸(V,N,T,C,R)の推定、未達インバリアント、次の推奨アクション、閉幕可否を返してください。"
                '出力はJSONのみで以下の形: {"satisfaction":{"V":0-2,"N":0-2,"T":0-2,"C":0-2,"R":0-2,"overall":number},"missing_invariants":[...],"recommend_next":string,"ready_to_close":bool}'
            )
            user = {
                'user_message': user_message,
                'ai_response': ai_response,
                'state': state
            }
            print(f"[PJ][LLM] _llm_watchdog call: model={model}, last_action={state.get('last_action')}")
            print(f"[PJ][LLM][INPUT] Watchdog system: {system[:200]}...")
            print(f"[PJ][LLM][INPUT] Watchdog user_message: {user_message[:100]}...")
            print(f"[PJ][LLM][INPUT] Watchdog ai_response: {ai_response[:100]}...")
            
            # フォールバック機能を含むLLM呼び出し
            tried = []
            last_err = None
            raw = None
            
            for m in [model] + self.fallback_models:
                if not m or m in tried:
                    continue
                tried.append(m)
                try:
                    if m.startswith("gpt-5"):
                        input_data = [
                            {
                                "role": "system", 
                                "content": [{"type": "input_text", "text": system}]
                            },
                            {
                                "role": "user", 
                                "content": [{"type": "input_text", "text": json.dumps(user, ensure_ascii=False)}]
                            }
                        ]
                        resp = client.responses.create(
                            model=m,
                            input=input_data,
                            text={
                                "format": {"type": "text"},
                                "verbosity": "medium"
                            },
                            tools=[],
                            temperature=0,
                            max_output_tokens=1024,
                            top_p=1,
                            store=True,
                            include=["web_search_call.action.sources"]
                        )
                        if hasattr(resp, 'output_text'):
                            raw = resp.output_text
                        elif hasattr(resp, 'text') and hasattr(resp.text, 'content'):
                            raw = resp.text.content
                        elif hasattr(resp, 'choices') and resp.choices:
                            raw = resp.choices[0].message.content
                        else:
                            raw = str(resp)
                    else:
                        resp = client.chat.completions.create(
                            model=m,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
                            ],
                            temperature=0,
                            response_format={"type": "json_object"}
                        )
                        raw = resp.choices[0].message.content
                    
                    if raw:
                        break
                except Exception as e:
                    last_err = e
                    print(f"[PJ][LLM][Error] _llm_watchdog model={m}: {type(e).__name__}: {e}")
            
            if not raw:
                raise RuntimeError(f"LLM呼び出しに失敗しました（試行: {tried}, 最終エラー: {last_err}")
            print(f"[PJ][LLM][RAW] _llm_watchdog response: {raw}")
            # フェンス付きJSON等を許容するクリーニング
            def _clean_json_response(txt: str) -> str:
                if not txt:
                    return "{}"
                s = txt.strip()
                # 汎用: 最初の{から最後の}を抽出（```json ... ``` を含むケースも拾う）
                start = s.find('{')
                end = s.rfind('}')
                if start != -1 and end != -1 and end > start:
                    return s[start:end+1]
                return s
            try:
                data = json.loads(raw)
            except Exception:
                data = json.loads(_clean_json_response(raw))
            sat = data.get('satisfaction', {}) or {}
            V = int(sat.get('V', 1)); N = int(sat.get('N', 1)); T = int(sat.get('T', 1)); C = int(sat.get('C', 1)); R = int(sat.get('R', 2))
            overall = float(sat.get('overall', (V + N + T + C + R) / 5))
            print(f"[PJ][LLM] _llm_watchdog ok: V={V},N={N},T={T},C={C},R={R},overall={overall:.2f}")
            # 追加情報（ログのみ）
            print(f"[PJ][LLM] missing_invariants={data.get('missing_invariants')}, recommend_next={data.get('recommend_next')}, ready_to_close={data.get('ready_to_close')}")
            return {'V': V, 'N': N, 'T': T, 'C': C, 'R': R, 'overall': overall}
        except Exception as e:
            print(f"[PJ][LLM][Error] _llm_watchdog: {type(e).__name__}: {e}")
            raise
    
    def _execute_repair(self, satisfaction_scores, state, session_data):
        """補修システム：低スコア軸への1回限り介入"""
        repairs = []
        
        if satisfaction_scores['V'] < 1 and not state.get('voice_repaired', False):
            repairs.append("あなたのご意見をもう少し詳しくお聞かせください。")
            state['voice_repaired'] = True
            
        if satisfaction_scores['N'] < 1 and not state.get('neutrality_repaired', False):
            repairs.append("様々な角度からの検討が重要ですね。")
            state['neutrality_repaired'] = True
            
        if satisfaction_scores['T'] < 1 and not state.get('transparency_repaired', False):
            repairs.append("プロセスについて不明な点があればお聞きします。")
            state['transparency_repaired'] = True
        
        return " ".join(repairs) if repairs else None

    # --- Minimal input-specificity judge -------------------------------------------------
    def _is_input_specific(self, message: str, state: Dict[str, Any]) -> bool:
        """ユーザー入力が具体的かを判定。最小限のLLM判定を試し、失敗時はヒューリスティック。

        具体性の目安:
        - 対象領域（学業成績/試験/研究能力/推薦/多様性 等）への明示的言及
        - 理由・根拠・背景（〜だから、〜ため、理由、根拠、具体例）
        - 数値や固有名、比較などの具体化
        """
        try:
            m = (message or '').strip()
            if len(m) < 5:
                return False
            # まず軽量ヒューリスティック（速い）
            domain_kw = ['学業成績', '試験', '研究能力', '推薦', '多様性', 'GPA', 'スコア', '計画書', '教授', '研究']
            reason_kw = ['理由', '根拠', 'ため', 'から', '具体', '例えば']
            hits = sum(1 for k in domain_kw if k in m) + sum(1 for k in reason_kw if k in m)
            if len(m) >= 30 and hits >= 2:
                return True

            # 最小LLM判定（JSONで true/false 返答）。失敗時はヒューリスティックにフォールバック
            client, model = self._get_llm()
            if client is None:
                # API未設定など
                return hits >= 1 and len(m) >= 40

            prompt = {
                'instruction': '以下のユーザー入力が具体的か判定してください。基準: 対象領域への言及や理由・根拠・数値等の具体化があるか。JSONのみで返答。',
                'schema': {'specific': True},
                'message': m
            }

            raw = None
            try:
                if model.startswith('gpt-5'):
                    input_data = [
                        {"role": "system", "content": [{"type": "input_text", "text": "JSON {\"specific\": bool} で返す。"}]},
                        {"role": "user", "content": [{"type": "input_text", "text": json.dumps(prompt, ensure_ascii=False)}]}
                    ]
                    resp = client.responses.create(model=model, input=input_data, text={"format": {"type": "json_schema", "json_schema": {"name": "spec", "schema": {"type": "object", "properties": {"specific": {"type": "boolean"}}, "required": ["specific"]}}}}, temperature=0)
                    raw = getattr(resp, 'output_text', None) or getattr(getattr(resp, 'text', None), 'content', None) or str(resp)
                else:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "Return only JSON like {\"specific\": true|false}."},
                            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
                        ],
                        temperature=0,
                        response_format={"type": "json_object"}
                    )
                    raw = resp.choices[0].message.content
                data = json.loads(raw)
                return bool(data.get('specific', False))
            except Exception:
                # LLMエラー時はヒューリスティック
                return hits >= 1 and len(m) >= 40
        except Exception:
            return False
    
    # ヘルパーメソッド群
    def _needs_respect_action(self, message):
        toxic_patterns = ['馬鹿', 'アホ', 'くそ', 'むかつく']
        return any(pattern in message.lower() for pattern in toxic_patterns)
    
    def _all_invariants_satisfied(self, state):
        required = ['Voice', 'Neutrality', 'Transparency', 'Respect', 'Consistency']
        return all(state['invariants'].get(inv, False) for inv in required) and state.get('appeal_offered', False)
    
    def _update_state(self, action, response, state):
        """状態遷移の実行"""
        state['last_action'] = action
        
        if action == 'VoiceElicitation':
            # 応答内の確認表現が含まれる場合はVoice充足
            ack_markers = ['合っていますか', 'よろしいでしょうか', 'よろしいですか', '合ってますか', 'ご確認', '確認']
            if ('合っていますか' in response) or any(k in response for k in ack_markers):
                state['voice_summary_ack'] = True
                state['invariants']['Voice'] = True
            
        elif action == 'NeutralityPresentation':
            state['invariants']['Neutrality'] = True
            
        elif action == 'TransparencyExplanation':
            state['invariants']['Transparency'] = True
            state['rule_explained_turn'] = state['turn']
            
        elif action == 'AppealOffer':
            state['appeal_offered'] = True
        
        return state
    
    def _log_turn(self, turn, action, state, scores):
        """監査用ログ記録"""
        log_entry = {
            "turn": turn,
            "action": action,
            "invariants": state['invariants'],
            "scores": scores,
            "appeal_offered": state.get('appeal_offered', False),
            "timestamp": (datetime.now(JST).isoformat() if JST else datetime.now().isoformat())
        }
        # 実装では外部ログシステムに記録
        print(f"PJ-LOG: {log_entry}")
    
    # スコアリングヘルパー
    def _score_voice_reflection(self, text, action):
        return 2 if action == 'VoiceElicitation' and ('合っていますか' in text or '重視' in text) else 1
    
    def _score_neutral_vocabulary(self, text):
        biased_words = ['正しい', '間違い', '同感', '反対']
        return 0 if any(word in text for word in biased_words) else 2
    
    def _score_accuracy(self, text):
        return 2  # 実装では事実チェックロジック
    
    def _score_transparency(self, text, action):
        if action == 'TransparencyExplanation':
            return 2 if '変更' in text and ('できません' in text or '権限' in text) else 1
        return 1
    
    def _score_conciseness(self, text):
        return 2 if len(text) <= 200 else (1 if len(text) <= 300 else 0)
    
    def _infer_main_criterion(self, decision, student_info):
        """決定から主要基準を推論"""
        if decision == '合格':
            return '学業成績の優秀さ'
        else:
            return '基準との適合性'
    
    def _get_user_priority(self, state):
        return state.get('user_priority', '確認済み基準')
    
    def _check_neutrality_completeness(self, state):
        return state.get('neutrality_elements', {}).get('four_elements_shown', False)
    
    def _contains_disrespectful_content(self, text):
        return False  # 実装では敬語・丁寧語チェック

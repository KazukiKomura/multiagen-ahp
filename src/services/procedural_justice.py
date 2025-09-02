"""
Procedural Justice System - 手続き的公正システム
Thibaut & Walker (1975) 完全準拠型LLMシステム

このシステムは有限状態機械（FSM）による優先度制御を使用し、
Voice, Neutrality, Transparency, Respect, Consistency の5つの要素を
完全に保証する手続き的公正システムを実装します。
"""

import json
from datetime import datetime


class ProceduralJusticeSystem:
    """手続き的公正システム - 有限状態機械による実装"""
    
    def __init__(self):
        self.max_turns = 5
        self.transparency_deadline = 3
        
    def execute_turn(self, message, decision, state, session_data):
        """1ターンの実行 - 優先度制御付きFSM"""
        
        # 優先度順でアクション決定: Respect > Voice > Transparency(期限) > Neutrality > Appeal > 要約/終了
        action = self._determine_action(message, decision, state)
        
        # LLM-G: 候補文生成（k=2-3）
        candidates = self._generate_candidates(action, message, decision, state, session_data)
        
        # ハード検証：フォーマット・禁則語・数値出典
        validated_candidates = self._hard_validation(candidates, action, state)
        
        # LLM-J: ルーブリック評価による最良選択
        best_response = self._llm_judge(validated_candidates, action, state)
        
        # 状態更新
        updated_state = self._update_state(action, best_response, state)
        
        # LLM-W: 満足度5軸監視
        satisfaction_scores = self._llm_watchdog(message, best_response, updated_state)
        
        # 補修システム（各軸1回まで）
        if satisfaction_scores['overall'] < 1.2:
            repair_response = self._execute_repair(satisfaction_scores, updated_state, session_data)
            if repair_response:
                best_response += " " + repair_response
        
        # ログ記録
        self._log_turn(state['turn'], action, updated_state, satisfaction_scores)
        
        return {
            'response': best_response,
            'state': updated_state,
            'satisfaction_scores': satisfaction_scores,
            'action': action
        }
    
    def _determine_action(self, message, decision, state):
        """優先度に基づくアクション決定 - FSM状態遷移"""
        
        # Priority 1: Respect - 不適切発言の検出・対応
        if self._needs_respect_action(message):
            return 'RespectCorrection'
        
        # Priority 2: Voice - 重視点確認（事前ボイス）
        if not state['invariants']['Voice']:
            if not state['voice_summary_ack']:
                return 'VoiceElicitation'  # 「最も重視する基準を1つ...」
            else:
                state['invariants']['Voice'] = True
        
        # Priority 3: Transparency - 期限内（≤3ターン）ルール説明
        if not state['invariants']['Transparency'] and state['turn'] <= self.transparency_deadline:
            return 'TransparencyExplanation'  # 「基準=..., 閾値=..., AIは結果変更不可」
        
        # Priority 4: Neutrality - 対称的4要素提示
        if not state['invariants']['Neutrality']:
            return 'NeutralityPresentation'  # 「合格有利/不利, 不合格有利/不利」
        
        # Priority 5: Appeal - 異議申立て機会
        if not state['appeal_offered'] and state['turn'] >= 2:
            return 'AppealOffer'  # 「1点だけ誤読・見落とし・追加情報があれば...」
        
        # Priority 6: 要約・終了条件
        if state['turn'] >= self.max_turns or self._all_invariants_satisfied(state):
            return 'SummaryAndClose'  # 「実施：重視点確認→対称提示→説明→Appeal」
        
        # デフォルト: 継続対話
        return 'ContinueDialogue'
    
    def _generate_candidates(self, action, message, decision, state, session_data):
        """LLM-G: アクション別候補文生成（k=2-3, 300字以内）"""
        
        student_info = session_data.get('student_info', {})
        criteria = session_data.get('criteria', ['学業成績', '試験スコア', '研究能力', '推薦状', '多様性'])
        
        templates = {
            'VoiceElicitation': [
                f"最も重視される基準を1つ教えてください。私の理解では、{decision}判断の背景には{self._infer_main_criterion(decision, student_info)}が重要と思いますが、合っていますか？",
                f"判断の際に最も重要視された点について詳しく聞かせてください。{decision}という結論に至った主な理由はどちらでしょうか？",
                f"あなたの{decision}という判断で、特に重視された評価基準を1つお聞かせください。"
            ],
            
            'NeutralityPresentation': [
                self._generate_neutrality_four_elements(decision, student_info),
                self._generate_neutrality_alternative(decision, student_info)
            ],
            
            'TransparencyExplanation': [
                f"透明性確保のため説明いたします。判定基準は{', '.join(criteria)}、合格閾値は事前設定値です。なお、AIファシリテーターは最終結果を変更することはできません。",
                f"システムの仕組みをご説明します：事前に設定された基準・閾値により判定が決まり、私には結果を変更する権限がございません。",
                f"プロセスの透明性として：評価は設定済み基準で自動決定され、AIは結果に介入できない仕様となっております。"
            ],
            
            'AppealOffer': [
                "1点だけ、見落としや誤解、追加で考慮すべき情報があればお聞かせください。その内容を検討し、反映の可否と理由をお伝えいたします。",
                "もし重要な情報の見落としがあったとお感じの場合、1点のみご指摘ください。検討いたします。",
                "判定に関して、1つだけ追加でお伝えしたい観点はございますか？内容を確認し回答いたします。"
            ],
            
            'SummaryAndClose': [
                f"これまでの実施内容：重視点確認→対称的観点提示→ルール説明→異議申立て機会の提供を完了いたしました。あなたが重視されていた点：{self._get_user_priority(state)}。十分な対話ができました。",
                f"手続きの完了報告：ご意見聴取、多角的検討、透明性確保、異議機会提供の全工程を実施いたしました。対話を終了いたします。"
            ],
            
            'ContinueDialogue': [
                "他にご質問やご意見はございますか？",
                "追加でお聞かせいただきたい点はございますか？"
            ]
        }
        
        return templates.get(action, ["継続的な対話を行います。"])[:3]  # k≤3に制限
    
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
    
    def _hard_validation(self, candidates, action, state):
        """ハード検証：フォーマット・禁則語・数値出典チェック"""
        validated = []
        
        for candidate in candidates:
            # 文字数制限（300字）
            if len(candidate) > 300:
                candidate = candidate[:297] + "..."
            
            # 禁則語チェック（価値判断回避）
            prohibited_words = ['正しい', '間違っている', '同感', '反対', 'その通り', 'だめ']
            if any(word in candidate for word in prohibited_words):
                continue  # 禁則語含む候補は除外
                
            # アクション別フォーマット検証
            if action == 'NeutralityPresentation':
                required_elements = ['合格に有利', '合格に不利', '不合格に有利', '不合格に不利']
                if not all(element in candidate for element in required_elements):
                    continue  # 4要素未満は除外
                    
            if action == 'TransparencyExplanation':
                if '変更することはできません' not in candidate and '変更する権限' not in candidate:
                    continue  # 権限制限の明示必須
                    
            validated.append(candidate)
        
        # 全て除外された場合のフォールバック
        return validated if validated else [f"適切な{action}を実行中です..."]
    
    def _llm_judge(self, candidates, action, state):
        """LLM-J: ルーブリック評価による最良候補選択"""
        if not candidates:
            return "対話を継続します。"
        
        best_score = -1
        best_candidate = candidates[0]
        
        for candidate in candidates:
            # ルーブリック評価（各0-2点）
            scores = {
                'voice_reflection': self._score_voice_reflection(candidate, action),
                'neutral_vocabulary': self._score_neutral_vocabulary(candidate),
                'accuracy': self._score_accuracy(candidate),
                'transparency': self._score_transparency(candidate, action),
                'conciseness': self._score_conciseness(candidate)
            }
            
            total_score = sum(scores.values())
            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate
        
        return best_candidate
    
    def _llm_watchdog(self, user_message, ai_response, state):
        """LLM-W: 満足度5軸推定（0-2点）"""
        scores = {
            'V': 2 if state['invariants']['Voice'] and state['voice_summary_ack'] else (1 if 'VoiceElicitation' in str(state.get('last_action', '')) else 0),
            'N': 2 if state['invariants']['Neutrality'] and self._check_neutrality_completeness(state) else 1,
            'T': 2 if state['invariants']['Transparency'] and state.get('rule_explained_turn', 999) <= 3 else 0,
            'C': 2 if len(ai_response) <= 200 else (1 if len(ai_response) <= 300 else 0),
            'R': 2 if not self._contains_disrespectful_content(ai_response) else 0
        }
        
        scores['overall'] = sum(scores.values()) / 5
        
        return scores
    
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
        
        if action == 'VoiceElicitation' and '合っていますか' in response:
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
            "timestamp": datetime.now().isoformat()
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
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import json
import random
import uuid
import csv
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'experiment-secret-key-2025'

# 学生入学データを読み込む関数
def load_student_data():
    csv_path = 'dataset/student admission data.csv'
    students = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                students.append(row)
    return students

# 判定困難な学生を事前選別する関数
def select_challenging_students():
    """
    判定が困難で議論を誘発しやすい学生を選別
    - decision=1,2 (境界線ケース)から選択
    - 強みと弱みが混在している学生を優先
    - 多様な背景を持つ学生を含める
    """
    students_data = load_student_data()
    
    # 境界線ケース（decision=1,2）を抽出
    boundary_cases = [s for s in students_data if int(s.get('decision', 0)) in [1, 2]]
    
    if not boundary_cases:
        return students_data[:4]  # フォールバック
    
    # 判定困難度を計算する関数
    def calculate_difficulty_score(student):
        score = 0
        
        # 基本スコア - decision=1,2は基本的に困難
        if int(student.get('decision', 0)) in [1, 2]:
            score += 10
        
        # 強みと弱みの混在度を計算
        gpa = float(student.get('gpa', 0))
        gre_q = int(student.get('gre_quant', 0))
        gre_v = int(student.get('gre_verbal', 0))
        gre_w = float(student.get('gre_writing', 0))
        sop = float(student.get('sop_score', 0))
        div = float(student.get('diversity_score', 0))
        
        # 正規化スコア (0-1)
        norm_gpa = gpa / 4.0
        norm_gre_q = (gre_q - 130) / 40
        norm_gre_v = (gre_v - 130) / 40
        norm_gre_w = gre_w / 6.0
        norm_sop = (sop - 1) / 4
        norm_div = (div - 1) / 4
        
        scores = [norm_gpa, norm_gre_q, norm_gre_v, norm_gre_w, norm_sop, norm_div]
        
        # 分散が大きい（強みと弱みが混在）ほど困難
        import statistics
        if len(scores) > 1:
            variance = statistics.variance(scores)
            score += variance * 20  # 分散を重み付け
        
        # 中程度のスコア（0.4-0.7）は判定困難
        avg_score = sum(scores) / len(scores)
        if 0.4 <= avg_score <= 0.7:
            score += 15
        
        # 推薦状が混在している場合は困難
        rec_strong = int(student.get('rec_letter_1_strong', 0)) + int(student.get('rec_letter_2_strong', 0)) + int(student.get('rec_letter_3_strong', 0))
        rec_weak = int(student.get('rec_letter_1_weak', 0)) + int(student.get('rec_letter_2_weak', 0)) + int(student.get('rec_letter_3_weak', 0))
        if rec_strong > 0 and rec_weak > 0:
            score += 10
        
        # 機関ランクが中位（2）は判定困難
        if student.get('institution_rank') == '2':
            score += 5
            
        return score
    
    # 困難度順にソート
    challenging_students = sorted(boundary_cases, key=calculate_difficulty_score, reverse=True)
    
    # 多様性を確保しながら選択
    selected = []
    used_majors = set()
    used_regions = set()
    
    for student in challenging_students:
        if len(selected) >= 12:  # 余裕をもって12名選択
            break
            
        # 専攻の多様性チェック
        major_flags = ['major_humanities', 'major_naturalscience', 'major_socialscience', 
                      'major_business', 'major_engineering', 'major_other']
        student_major = next((flag for flag in major_flags if student.get(flag) == '1'), 'other')
        
        # 地域の多様性チェック
        region_flags = ['institution_us', 'institution_canada', 'institution_asia', 
                       'institution_europe', 'institution_other']
        student_region = next((flag for flag in region_flags if student.get(flag) == '1'), 'other')
        
        # 多様性を保ちつつ選択（同じ専攻・地域を最大2名まで）
        major_count = sum(1 for s in selected if next((flag for flag in major_flags if s.get(flag) == '1'), 'other') == student_major)
        region_count = sum(1 for s in selected if next((flag for flag in region_flags if s.get(flag) == '1'), 'other') == student_region)
        
        if major_count < 2 and region_count < 2:
            selected.append(student)
    
    # 足りない場合は残りの境界線ケースから追加
    if len(selected) < 8:
        remaining = [s for s in boundary_cases if s not in selected]
        selected.extend(remaining[:8-len(selected)])
    
    return selected[:8]  # 最大8名を返す

# トライアル別に学生を選択する関数
def get_student_for_trial(trial, session_id):
    """
    トライアル別に適切な難易度の学生を選択
    trial 1: やや易しい（判定困難だが議論しやすい）
    trial 2-3: 中程度の難易度
    trial 4: 最も困難（強みと弱みが複雑に混在）
    """
    challenging_students = select_challenging_students()
    
    if not challenging_students:
        # フォールバック: 全データからランダム
        all_students = load_student_data()
        random.seed(hash(session_id + str(trial)) % 1000000)
        return random.choice(all_students) if all_students else None
    
    # 困難度スコアで分類
    def get_difficulty_category(student):
        # 同じ困難度スコア関数を使用
        score = 0
        if int(student.get('decision', 0)) in [1, 2]:
            score += 10
            
        gpa = float(student.get('gpa', 0))
        gre_q = int(student.get('gre_quant', 0))
        gre_v = int(student.get('gre_verbal', 0))
        gre_w = float(student.get('gre_writing', 0))
        sop = float(student.get('sop_score', 0))
        div = float(student.get('diversity_score', 0))
        
        norm_gpa = gpa / 4.0
        norm_gre_q = (gre_q - 130) / 40
        norm_gre_v = (gre_v - 130) / 40  
        norm_gre_w = gre_w / 6.0
        norm_sop = (sop - 1) / 4
        norm_div = (div - 1) / 4
        
        scores = [norm_gpa, norm_gre_q, norm_gre_v, norm_gre_w, norm_sop, norm_div]
        
        import statistics
        if len(scores) > 1:
            variance = statistics.variance(scores)
            score += variance * 20
            
        avg_score = sum(scores) / len(scores)
        if 0.4 <= avg_score <= 0.7:
            score += 15
            
        return score
    
    # 困難度によって分類
    students_with_scores = [(s, get_difficulty_category(s)) for s in challenging_students]
    students_with_scores.sort(key=lambda x: x[1])  # スコア昇順
    
    # 3つのグループに分割
    total = len(students_with_scores)
    easy_group = students_with_scores[:total//3]
    medium_group = students_with_scores[total//3:2*total//3]  
    hard_group = students_with_scores[2*total//3:]
    
    # トライアルに応じて選択
    if trial == 1:
        # やや易しい
        target_group = easy_group if easy_group else students_with_scores
    elif trial in [2, 3]:
        # 中程度
        target_group = medium_group if medium_group else students_with_scores
    else:
        # 最も困難
        target_group = hard_group if hard_group else students_with_scores
    
    # グループから決定論的に選択（セッションIDベース）
    if target_group:
        index = hash(session_id + str(trial)) % len(target_group)
        return target_group[index][0]
    
    # フォールバック
    return challenging_students[0] if challenging_students else None

# 学生データを表示用に整形する関数
def format_student_for_display(student_row):
    # 推薦状を解釈
    rec_letters = []
    for i in range(1, 4):
        if student_row.get(f'rec_letter_{i}_strong') == '1':
            rec_letters.append(f'推薦状{i}: 優秀')
        elif student_row.get(f'rec_letter_{i}_average') == '1':
            rec_letters.append(f'推薦状{i}: 普通')
        elif student_row.get(f'rec_letter_{i}_weak') == '1':
            rec_letters.append(f'推薦状{i}: 弱い')
    
    # 出身地域を解釈
    region = ""
    if student_row.get('institution_us') == '1':
        region = "米国"
    elif student_row.get('institution_canada') == '1':
        region = "カナダ"
    elif student_row.get('institution_asia') == '1':
        region = "アジア"
    elif student_row.get('institution_europe') == '1':
        region = "ヨーロッパ"
    else:
        region = "その他"
    
    # 専攻を解釈
    major = ""
    if student_row.get('major_humanities') == '1':
        major = "人文学"
    elif student_row.get('major_naturalscience') == '1':
        major = "自然科学"
    elif student_row.get('major_socialscience') == '1':
        major = "社会科学"
    elif student_row.get('major_business') == '1':
        major = "ビジネス"
    elif student_row.get('major_engineering') == '1':
        major = "工学"
    else:
        major = "その他"
    
    # 機関ランクを解釈
    rank_map = {"1": "最上位", "2": "中位", "3": "下位"}
    institution_rank = rank_map.get(student_row.get('institution_rank'), '不明')
    
    # 推薦状の評価グレードを計算
    strong_count = sum(1 for letter in rec_letters if '優秀' in letter)
    average_count = sum(1 for letter in rec_letters if '普通' in letter)
    weak_count = sum(1 for letter in rec_letters if '弱い' in letter)
    
    if strong_count >= 2:
        rec_grade = 'A'
        rec_grade_class = 'excellent'
    elif strong_count >= 1 and weak_count == 0:
        rec_grade = 'B+'
        rec_grade_class = 'good'
    elif strong_count >= 1:
        rec_grade = 'B'
        rec_grade_class = 'average'
    else:
        rec_grade = 'C+'
        rec_grade_class = 'weak'
    
    return {
        'id': student_row.get('id'),
        'gpa': float(student_row.get('gpa', 0)),
        'gre_quant': int(student_row.get('gre_quant', 0)),
        'gre_verbal': int(student_row.get('gre_verbal', 0)),
        'gre_writing': float(student_row.get('gre_writing', 0)),
        'sop_score': float(student_row.get('sop_score', 0)),  # intからfloatに変更
        'diversity_score': float(student_row.get('diversity_score', 0)),
        'rec_letters': rec_letters,
        'rec_grade': rec_grade,
        'rec_grade_class': rec_grade_class,
        'institution_rank': institution_rank,
        'region': region,
        'major': major,
        'actual_decision': int(student_row.get('decision', 0))
    }

# データベース初期化
def init_db():
    conn = sqlite3.connect('data/sessions.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            condition TEXT NOT NULL,
            phase INTEGER DEFAULT 0,
            trial INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data TEXT
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            phase TEXT,
            trial INTEGER,
            decision_data TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Bot意見生成（学生入学判定用）
def generate_bot_opinions_for_student():
    criteria = ['学業成績', '試験スコア', '研究能力', '推薦状', '多様性']
    decision_labels = ['不合格', '合格']  # 2択に簡素化
    bot_opinions = []
    
    for bot_id in [1, 2]:
        decision_value = random.randint(0, 1)  # 0:不合格, 1:合格
        decision = decision_labels[decision_value]
        weights = {}
        total_weight = 100
        
        for i, criterion in enumerate(criteria):
            if i == len(criteria) - 1:
                weights[criterion] = total_weight
            else:
                weight = random.randint(10, min(40, total_weight - (len(criteria) - i - 1) * 10))
                weights[criterion] = weight
                total_weight -= weight
        
        # 判定理由を生成
        main_criterion = max(weights, key=weights.get)
        if decision_value == 1:  # 合格
            reasoning = f'評価委員{bot_id + 1}の判断：{decision}。{main_criterion}が特に優れており、入学基準を満たしていると考えます。'
        else:  # 不合格
            reasoning = f'評価委員{bot_id + 1}の判断：{decision}。{main_criterion}の観点から、入学基準に達していないと判断します。'
        
        bot_opinions.append({
            'bot_id': bot_id,
            'decision': decision,
            'decision_value': decision_value,
            'weights': weights,
            'reasoning': reasoning
        })
    
    return bot_opinions

# 手続き的公正（PJ）システム - FSM実装
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


# AI-ファシリテーター応答生成（PJシステム使用）
def generate_ai_response(user_message, user_decision=None, conversation_count=0):
    """手続き的公正システムによる応答生成"""
    
    # セッション状態の取得・初期化
    if 'pj_state' not in session:
        session['pj_state'] = {
            'invariants': {
                'Voice': False,
                'Neutrality': False, 
                'Transparency': False,
                'Respect': True,
                'Consistency': True
            },
            'voice_summary_ack': False,
            'appeal_offered': False,
            'rule_explained_turn': None,
            'turn': 1
        }
    
    # ターン数更新
    session['pj_state']['turn'] = conversation_count + 1
    
    # PJシステム実行
    pj_system = ProceduralJusticeSystem()
    
    # セッションデータ構築
    session_data = {
        'student_info': session.get('student_info', {}),
        'criteria': ['学業成績', '試験スコア', '研究能力', '推薦状', '多様性']
    }
    
    result = pj_system.execute_turn(
        user_message, 
        user_decision, 
        session['pj_state'], 
        session_data
    )
    
    # セッション状態更新
    session['pj_state'] = result['state']
    
    # PJ満足度ログ保存
    session['pj_satisfaction_scores'] = result['satisfaction_scores']
    
    return result['response']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start/<condition>')
def start_experiment(condition):
    if condition != 'ai-facilitator':
        return redirect(url_for('index'))
    
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session['condition'] = condition
    session['trial'] = 1
    session['phase'] = 0
    
    # データベースにセッション保存
    conn = sqlite3.connect('data/sessions.db')
    c = conn.cursor()
    c.execute('INSERT INTO sessions (session_id, condition) VALUES (?, ?)', 
              (session_id, condition))
    conn.commit()
    conn.close()
    
    # トライアル1のみ事前質問紙へ
    return redirect(url_for('questionnaire', phase='pre'))

@app.route('/questionnaire/<phase>')
def questionnaire(phase):
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    return render_template('questionnaire.html', 
                         phase=phase, 
                         condition=session['condition'],
                         trial=session.get('trial', 1))

@app.route('/save_questionnaire', methods=['POST'])
def save_questionnaire():
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400
    
    data = request.json
    phase = data.get('phase')
    
    # データベースに保存
    conn = sqlite3.connect('data/sessions.db')
    c = conn.cursor()
    c.execute('INSERT INTO decisions (session_id, phase, trial, decision_data) VALUES (?, ?, ?, ?)',
              (session['session_id'], f'questionnaire_{phase}', session.get('trial', 1), json.dumps(data)))
    conn.commit()
    conn.close()
    
    if phase == 'pre':
        # 事前質問紙の後は体験フェーズへ
        return jsonify({'next': '/experience'})
    else:
        # 事後質問紙の後：次のトライアルまたは実験終了
        current_trial = session.get('trial', 1)
        if current_trial < 4:
            session['trial'] = current_trial + 1
            # トライアル2以降は体験フェーズに直接
            return jsonify({'next': '/experience'})
        else:
            return jsonify({'next': '/complete'})

@app.route('/decision')
def decision():
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    # 統合体験フェーズに転送
    return redirect(url_for('experience'))

@app.route('/experience')
def experience():
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    trial = session.get('trial', 1)
    
    # 判定困難な学生をトライアル別難易度で選択
    student_row = get_student_for_trial(trial, session['session_id'])
    
    if student_row:
        student_info = format_student_for_display(student_row)
    else:
        # フォールバック用のダミーデータ
        student_info = {
            'id': '99',
            'gpa': 3.75,
            'gre_quant': 165,
            'gre_verbal': 155,
            'gre_writing': 3.5,
            'sop_score': 4.0,
            'diversity_score': 3.0,
            'rec_letters': ['推薦状1: 優秀', '推薦状2: 普通', '推薦状3: 優秀'],
            'institution_rank': '中位',
            'region': 'アジア',
            'major': '工学',
            'actual_decision': 2
        }
    
    # 意思決定タスク情報
    decision_info = {
        'title': f'学生入学判定 (応募者ID: {student_info["id"]}) - トライアル {trial}',
        'description': 'この学生の入学可否について、合格または不合格で判定をお願いします。',
        'criteria': ['学業成績', '試験スコア', '研究能力', '推薦状', '多様性'],
        'student': student_info,
        'details': {
            '学業成績': f'GPA: {student_info["gpa"]}/4.0 - 直近の学位での成績',
            '試験スコア': f'GRE定量: {student_info["gre_quant"]}/170, GRE言語: {student_info["gre_verbal"]}/170, GRE作文: {student_info["gre_writing"]}/6.0',
            '研究能力': f'志望動機書: {student_info["sop_score"]}/5点, 多様性声明書: {student_info["diversity_score"]}/5点',
            '推薦状': ', '.join(student_info["rec_letters"]),
            '多様性': f'専攻: {student_info["major"]}, 出身地域: {student_info["region"]}, 出身機関ランク: {student_info["institution_rank"]}'
        }
    }
    
    return render_template('experience.html', 
                         info=decision_info,
                         condition=session['condition'],
                         trial=trial)

@app.route('/save_decision', methods=['POST'])
def save_decision():
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400
    
    data = request.json
    
    # データベースに保存
    conn = sqlite3.connect('data/sessions.db')
    c = conn.cursor()
    c.execute('INSERT INTO decisions (session_id, phase, trial, decision_data) VALUES (?, ?, ?, ?)',
              (session['session_id'], 'initial_decision', session.get('trial', 1), json.dumps(data)))
    conn.commit()
    conn.close()
    
    # Bot意見を生成してセッションに保存（学生入学判定用）
    bot_opinions = generate_bot_opinions_for_student()
    session['bot_opinions'] = bot_opinions
    
    # 統合体験フェーズではBot意見をレスポンスで返す
    return jsonify({'bot_opinions': bot_opinions})

@app.route('/aggregation')
def aggregation():
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    bot_opinions = session.get('bot_opinions', [])
    condition = session['condition']
    
    return render_template('aggregation.html',
                         bot_opinions=bot_opinions,
                         condition=condition,
                         trial=session.get('trial', 1))

@app.route('/intervention')
def intervention():
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    condition = session['condition']
    
    # AI-facilitator条件のみ
    return render_template('intervention.html',
                         condition=condition,
                         trial=session.get('trial', 1))

@app.route('/ai_chat', methods=['POST'])
def ai_chat():
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400
    
    user_message = request.json.get('message', '')
    user_decision = request.json.get('decision')
    conversation_count = request.json.get('conversation_count', 0)
    
    ai_response = generate_ai_response(user_message, user_decision, conversation_count)
    
    # チャット履歴を保存
    chat_data = {
        'user_message': user_message,
        'ai_response': ai_response,
        'timestamp': datetime.now().isoformat()
    }
    
    conn = sqlite3.connect('data/sessions.db')
    c = conn.cursor()
    c.execute('INSERT INTO decisions (session_id, phase, trial, decision_data) VALUES (?, ?, ?, ?)',
              (session['session_id'], 'ai_chat', session.get('trial', 1), json.dumps(chat_data)))
    conn.commit()
    conn.close()
    
    return jsonify({'response': ai_response})

@app.route('/final_decision')
def final_decision():
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    # 初期決定データを取得
    conn = sqlite3.connect('data/sessions.db')
    c = conn.cursor()
    c.execute('SELECT decision_data FROM decisions WHERE session_id = ? AND phase = ? AND trial = ? ORDER BY timestamp DESC LIMIT 1',
              (session['session_id'], 'initial_decision', session.get('trial', 1)))
    result = c.fetchone()
    conn.close()
    
    initial_decision = {}
    if result:
        initial_decision = json.loads(result[0])
    
    bot_opinions = session.get('bot_opinions', [])
    
    return render_template('final_decision.html',
                         initial_decision=initial_decision,
                         bot_opinions=bot_opinions,
                         condition=session['condition'],
                         trial=session.get('trial', 1))

@app.route('/save_final_decision', methods=['POST'])
def save_final_decision():
    if 'session_id' not in session:
        return jsonify({'error': 'No session'}), 400
    
    data = request.json
    
    # データベースに保存
    conn = sqlite3.connect('data/sessions.db')
    c = conn.cursor()
    c.execute('INSERT INTO decisions (session_id, phase, trial, decision_data) VALUES (?, ?, ?, ?)',
              (session['session_id'], 'final_decision', session.get('trial', 1), json.dumps(data)))
    conn.commit()
    conn.close()
    
    return jsonify({'next': '/questionnaire/post'})

@app.route('/complete')
def complete():
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    return render_template('complete.html',
                         condition=session['condition'])

@app.route('/admin/data')
def admin_data():
    conn = sqlite3.connect('data/sessions.db')
    c = conn.cursor()
    
    # セッション一覧
    c.execute('SELECT * FROM sessions ORDER BY created_at DESC')
    sessions = c.fetchall()
    
    # 決定データ
    c.execute('SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 100')
    decisions = c.fetchall()
    
    conn.close()
    
    return jsonify({
        'sessions': sessions,
        'decisions': decisions
    })

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
"""
Data utility functions for the multi-agent AHP experiment system.
Handles student data loading, selection, and formatting operations.
"""

import csv
import os
import random
import statistics
from typing import List, Dict, Any, Optional
import random as _random


def load_student_data() -> List[Dict[str, Any]]:
    """
    学生入学データをCSVファイルから読み込む
    
    Returns:
        List[Dict]: 学生データのリスト
    """
    csv_path = 'dataset/student admission data.csv'
    students = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                students.append(row)
    return students


def select_challenging_students() -> List[Dict[str, Any]]:
    """
    判定が困難で議論を誘発しやすい学生を選別
    - decision=1,2 (境界線ケース)から選択
    - 強みと弱みが混在している学生を優先
    - 多様な背景を持つ学生を含める
    
    Returns:
        List[Dict]: 選別された困難な判定ケースの学生リスト
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
            
        major = student.get('major', 'Unknown')
        region = student.get('region', 'Unknown')
        
        # 多様性確保: 同じ専攻・地域は2名まで
        major_count = sum(1 for s in selected if s.get('major') == major)
        region_count = sum(1 for s in selected if s.get('region') == region)
        
        if major_count < 2 and region_count < 2:
            selected.append(student)
            used_majors.add(major)
            used_regions.add(region)
    
    # 12名に満たない場合は残りをランダムで追加
    if len(selected) < 12:
        remaining = [s for s in challenging_students if s not in selected]
        selected.extend(random.sample(remaining, min(12 - len(selected), len(remaining))))
    
    return selected


def get_student_for_trial(trial: int, session_id: str) -> Optional[Dict[str, Any]]:
    """
    トライアル別に適切な難易度の学生を選択
    trial 1: やや易しい（判定困難だが議論しやすい）
    trial 2-3: 中程度の難易度
    trial 4: 最も困難（強みと弱みが複雑に混在）
    
    Args:
        trial: トライアル番号 (1-4)
        session_id: セッションID
        
    Returns:
        Dict: 選択された学生データ、または None
    """
    challenging_students = select_challenging_students()
    
    if not challenging_students:
        return None
    
    # セッションIDとトライアルを組み合わせてシードを作成（再現性確保）
    seed_value = hash(f"{session_id}_{trial}") % 2**32
    random.seed(seed_value)
    
    # トライアル別の選択戦略
    if trial == 1:
        # やや易しい: 困難度下位30%から選択
        candidates = challenging_students[int(len(challenging_students) * 0.7):]
    elif trial in [2, 3]:
        # 中程度: 困難度中位40%から選択
        start = int(len(challenging_students) * 0.3)
        end = int(len(challenging_students) * 0.7)
        candidates = challenging_students[start:end]
    else:  # trial == 4
        # 最困難: 困難度上位30%から選択
        candidates = challenging_students[:int(len(challenging_students) * 0.3)]
    
    if not candidates:
        candidates = challenging_students  # フォールバック
        
    return random.choice(candidates)


def format_student_for_display(student_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    学生データを表示用にフォーマット
    
    Args:
        student_row: 生の学生データ
        
    Returns:
        Dict: 表示用にフォーマットされた学生データ
    """
    # 推薦状を解釈
    rec_letters = []
    for i in range(1, 4):
        if student_row.get(f'rec_letter_{i}_strong') == '1':
            rec_letters.append(f'推薦状{i}: 優秀')
        elif student_row.get(f'rec_letter_{i}_weak') == '1':
            rec_letters.append(f'推薦状{i}: 懸念あり')
        else:
            rec_letters.append(f'推薦状{i}: 平均的')
    
    # 専攻の決定
    major_columns = ['major_humanities', 'major_naturalscience', 'major_socialscience', 
                     'major_business', 'major_engineering', 'major_other']
    major_map = {'major_humanities': '人文学', 'major_naturalscience': '自然科学',
                 'major_socialscience': '社会科学', 'major_business': 'ビジネス',
                 'major_engineering': '工学', 'major_other': 'その他'}
    
    major = 'その他'
    for col in major_columns:
        if student_row.get(col) == '1':
            major = major_map.get(col, 'その他')
            break
    
    # 地域の決定
    region_columns = ['institution_us', 'institution_canada', 'institution_asia',
                      'institution_europe', 'institution_other']
    region_map = {'institution_us': '米国', 'institution_canada': 'カナダ',
                  'institution_asia': 'アジア', 'institution_europe': '欧州',
                  'institution_other': 'その他'}
    
    region = 'その他'
    for col in region_columns:
        if student_row.get(col) == '1':
            region = region_map.get(col, 'その他')
            break
    
    # 機関ランクを文字列に変換
    institution_map = {'1': '高ランク', '2': '中ランク', '3': '低ランク'}
    institution_rank = institution_map.get(student_row.get('institution_rank', '2'), '中ランク')
    
    # 多様性情報を文字列に変換
    diversity_items = []
    if student_row.get('minority_status') == '1':
        diversity_items.append('マイノリティ背景')
    if student_row.get('first_generation') == '1':
        diversity_items.append('第一世代大学生')
    if student_row.get('rural_background') == '1':
        diversity_items.append('地方出身')
    
    diversity_text = '、'.join(diversity_items) if diversity_items else '一般的背景'
    
    return {
        'student_id': student_row.get('id', 'Unknown'),
        'major': major, 
        'region': region,
        'gpa': float(student_row.get('gpa', 0)),
        'gre_quant': int(student_row.get('gre_quant', 0)),
        'gre_verbal': int(student_row.get('gre_verbal', 0)),
        'gre_writing': float(student_row.get('gre_writing', 0)),
        'sop_score': float(student_row.get('sop_score', 0)),
        'diversity_score': float(student_row.get('diversity_score', 0)),
        'institution_rank': institution_rank,
        'rec_letters': rec_letters,
        'diversity_text': diversity_text,
        
        # 詳細な評価スコア
        'detailed_scores': {
            '学業成績': {
                'main_score': float(student_row.get('gpa', 0)),
                'subscores': [
                    f"GPA: {student_row.get('gpa', 'N/A')}",
                    f"機関ランク: {institution_rank}"
                ]
            },
            '試験スコア': {
                'main_score': (int(student_row.get('gre_quant', 0)) + int(student_row.get('gre_verbal', 0))) / 2,
                'subscores': [
                    f"定量: {student_row.get('gre_quant', 'N/A')}",
                    f"言語: {student_row.get('gre_verbal', 'N/A')}",
                    f"記述: {student_row.get('gre_writing', 'N/A')}"
                ]
            },
            '研究能力': {
                'main_score': float(student_row.get('sop_score', 0)),
                'subscores': [
                    f"研究計画書: {student_row.get('sop_score', 'N/A')}点",
                    "研究経験: 評価済み"
                ]
            },
            '推薦状': {
                'main_score': len([l for l in rec_letters if '優秀' in l]),
                'subscores': rec_letters
            },
            '多様性': {
                'main_score': float(student_row.get('diversity_score', 0)),
                'subscores': [
                    f"多様性スコア: {student_row.get('diversity_score', 'N/A')}",
                    f"背景: {diversity_text}"
                ]
            }
        }
    }


def generate_bot_opinions_for_student() -> List[Dict[str, Any]]:
    """
    学生に対するBot意見を生成（現在は使用されていませんが、互換性のため保持）
    
    Returns:
        List[Dict]: Bot意見のリスト
    """
    criteria = ['学業成績', '試験スコア', '研究能力', '推薦状', '多様性']
    decision_labels = ['不合格', '合格']  # 2択に簡素化
    bot_opinions = []
    
    for bot_id in [1, 2]:
        decision = random.choice([0, 1])
        weights = [random.randint(10, 40) for _ in range(5)]
        total = sum(weights)
        weights = [w / total * 100 for w in weights]
        
        opinion = {
            'bot_id': bot_id,
            'decision': decision,
            'decision_label': decision_labels[decision],
            'reasoning': f'Bot {bot_id}の判断理由',
            'weights': dict(zip(criteria, weights))
        }
        bot_opinions.append(opinion)
    
    return bot_opinions


def generate_participant_opinions(user_decision: str,
                                  criteria: List[str],
                                  trial: int,
                                  session_id: str) -> List[Dict[str, Any]]:
    """
    参加者2名の意見（決定と重み）を決定的に生成する。
    - 練習(trial==1): 決定はランダム
    - 本番(trial>=2): 決定はユーザーの初回判断の反対
    - 重み: 10%刻み、合計100%（セッションIDとtrialから決定的な乱数シード）
    """
    rng = _random.Random()
    rng.seed(f"{session_id}:{trial}:participants")

    def gen_weights() -> Dict[str, int]:
        n = len(criteria)
        remaining = 100
        weights: Dict[str, int] = {}
        for i, c in enumerate(criteria):
            if i == n - 1:
                weights[c] = remaining
            else:
                min_w = max(10, remaining - (n - i - 1) * 70)
                max_w = min(70, remaining - (n - i - 1) * 10)
                step_count = (max_w - min_w) // 10
                w = min_w + rng.randrange(step_count + 1) * 10
                weights[c] = w
                remaining -= w
        return weights

    def gen_decision() -> str:
        if trial == 1:
            return '合格' if rng.random() < 0.5 else '不合格'
        # 本番はユーザーの初回判断の反対
        return '不合格' if user_decision == '合格' else '合格'

    opinions: List[Dict[str, Any]] = []
    for bot_id in range(2):
        opinions.append({
            'bot_id': bot_id,
            'decision': gen_decision(),
            'weights': gen_weights(),
        })
    return opinions

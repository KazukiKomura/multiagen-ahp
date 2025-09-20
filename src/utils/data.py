"""
Data utility functions for the multi-agent AHP experiment system.
Handles student data loading, selection, and formatting operations.
"""

import csv
import os
import random
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




def get_student_for_trial(trial: int, session_id: str) -> Optional[Dict[str, Any]]:
    """
    トライアル別に学生を選択（4名の固定データ用）
    trial 1（練習）: 1人目固定
    trial 2-4（本番）: 2-4人目をセッションIDベースでランダム順序
    
    Args:
        trial: トライアル番号 (1-4)
        session_id: セッションID
        
    Returns:
        Dict: 選択された学生データ、または None
    """
    students_data = load_student_data()
    
    if len(students_data) < 4:
        return None
    
    if trial == 1:
        # 練習: 1人目固定
        return students_data[0]
    else:
        # 本番: 2-4人目をセッションIDベースでシャッフルして順次選択
        remaining_students = students_data[1:4]  # 2-4人目 (index 1,2,3)
        
        # セッションIDでシード固定（再現性確保）
        seed_value = hash(session_id) % 2**32
        random.seed(seed_value)
        
        # 2-4人目をシャッフル
        shuffled_students = remaining_students.copy()
        random.shuffle(shuffled_students)
        
        # trial 2,3,4 → shuffled index 0,1,2
        trial_index = trial - 2
        if trial_index < len(shuffled_students):
            return shuffled_students[trial_index]
    
    return None


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
    
    for bot_id in [1, 2, 3]:
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
    参加者3名の意見（決定と重み）を決定的に生成する。
    - 練習(trial==1): 決定はランダム（2-2にならないよう調整）
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

    def gen_decisions_for_trial() -> List[str]:
        if trial == 1:
            # 練習: ランダムだが2-2にならないよう調整（ユーザー1名+参加者3名=4名）
            # パターン: 3-1 または 1-3 になるよう調整
            user_choice = user_decision
            participant_choices = []
            
            # まず2名をランダムに決定
            for _ in range(2):
                participant_choices.append('合格' if rng.random() < 0.5 else '不合格')
            
            # 3名目は2-2を避けるよう調整
            user_count = 1 if user_choice == '合格' else 0
            participant_pass_count = sum(1 for d in participant_choices if d == '合格')
            total_pass_count = user_count + participant_pass_count
            
            # 現在のカウントで最終的に2-2になるかチェック
            if total_pass_count == 2:
                # 2-2になってしまうので、3名目で調整
                third_choice = '不合格' if user_choice == '合格' else '合格'
            else:
                # 2-2にならないので、3名目はランダム
                third_choice = '合格' if rng.random() < 0.5 else '不合格'
            
            participant_choices.append(third_choice)
            return participant_choices
        else:
            # 本番: 全員ユーザーの初回判断の反対
            opposite = '不合格' if user_decision == '合格' else '合格'
            return [opposite, opposite, opposite]

    decisions = gen_decisions_for_trial()
    opinions: List[Dict[str, Any]] = []
    for bot_id in range(3):
        opinions.append({
            'bot_id': bot_id,
            'decision': decisions[bot_id],
            'weights': gen_weights(),
        })
    return opinions

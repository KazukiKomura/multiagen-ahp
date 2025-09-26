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
    csv_path = 'data/dataset/student admission data.csv'
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
    学生データを表示用にフォーマット（日本の新卒採用向け、元データ値のまま）

    Args:
        student_row: 生の学生データ

    Returns:
        Dict: 表示用にフォーマットされた学生データ
    """
    # 推薦状を解釈（面接評価に変更）
    rec_letters = []
    for i in range(1, 4):
        if student_row.get(f'rec_letter_{i}_strong') == '1':
            rec_letters.append(f'面接評価{i}: 優秀')
        elif student_row.get(f'rec_letter_{i}_weak') == '1':
            rec_letters.append(f'面接評価{i}: 懸念あり')
        else:
            # 「平均的」ではなく「普通」を返す
            rec_letters.append(f'面接評価{i}: 普通')

    # 専攻の決定（日本の学部分類）
    major_columns = ['major_humanities', 'major_naturalscience', 'major_socialscience',
                     'major_business', 'major_engineering', 'major_other']
    major_map = {'major_humanities': '文系（人文・社会）', 'major_naturalscience': '理系（自然科学）',
                 'major_socialscience': '文系（社会科学）', 'major_business': '商学・経済',
                 'major_engineering': '理系（工学）', 'major_other': 'その他'}

    major = 'その他'
    for col in major_columns:
        if student_row.get(col) == '1':
            major = major_map.get(col, 'その他')
            break

    # 地域の決定（日本の大学群分類）
    region_columns = ['institution_us', 'institution_canada', 'institution_asia',
                      'institution_europe', 'institution_other']
    region_map = {'institution_us': '国立上位', 'institution_canada': '国立中位',
                  'institution_asia': '私立上位', 'institution_europe': '私立中位',
                  'institution_other': 'その他'}

    region = 'その他'
    for col in region_columns:
        if student_row.get(col) == '1':
            region = region_map.get(col, 'その他')
            break

    # 機関ランクを文字列に変換（日本の大学ランク）
    institution_map = {'1': 'Sランク', '2': 'Aランク', '3': 'Bランク'}
    institution_rank = institution_map.get(student_row.get('institution_rank', '2'), 'Aランク')

    # 多様性情報を文字列に変換（日本の採用文脈）
    diversity_items = []
    if student_row.get('minority_status') == '1':
        diversity_items.append('地方出身')
    if student_row.get('first_generation') == '1':
        diversity_items.append('第一世代大学生')
    if student_row.get('rural_background') == '1':
        diversity_items.append('留学経験あり')

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

        # 詳細な評価スコア（元データのまま）
        'detailed_scores': {
            '学業成績': {
                'main_score': float(student_row.get('gpa', 0)),
                'subscores': [
                    f"GPA: {student_row.get('gpa', 'N/A')}/4.0",
                    f"大学ランク: {institution_rank}"
                ]
            },
            '基礎能力テスト': {
                'main_score': (int(student_row.get('gre_quant', 0)) + int(student_row.get('gre_verbal', 0))) / 2,
                'subscores': [
                    f"数理: {student_row.get('gre_quant', 'N/A')}/170",
                    f"言語: {student_row.get('gre_verbal', 'N/A')}/170",
                    f"適性: {student_row.get('gre_writing', 'N/A')}/6.0"
                ]
            },
            '実践経験': {
                'main_score': float(student_row.get('sop_score', 0)),
                'subscores': [
                    f"インターン: {student_row.get('sop_score', 'N/A')}/5.0",
                    f"プロジェクト: {student_row.get('diversity_score', 'N/A')}/5.0"
                ]
            },
            '推薦・評価': {
                'main_score': len([l for l in rec_letters if '優秀' in l]),
                'subscores': rec_letters
            },
            '志望動機・フィット': {
                'main_score': float(student_row.get('diversity_score', 0)),
                'subscores': [
                    f"志望動機: {student_row.get('diversity_score', 'N/A')}/5.0",
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
    decision_labels = ['見送り', '一次通過']  # 2択に簡素化
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
                                  user_weights: Dict[str, int],
                                  criteria: List[str],
                                  trial: int,
                                  session_id: str) -> List[Dict[str, Any]]:
    """
    参加者3名の意見（決定と重み）を決定的に生成する。
    - 練習(trial==1): 決定はランダム（2-2にならないよう調整）
    - 本番(trial>=2): 決定はユーザーの初回判断の反対
    - 重み: ユーザーの重みとの距離に応じてclose/medium/farの3種類を生成
    """
    rng = _random.Random()
    rng.seed(f"{session_id}:{trial}:participants")

    def calculate_weight_distance(weights1: Dict[str, int], weights2: Dict[str, int]) -> float:
        """2つの重み配分間のユークリッド距離を計算"""
        total_squared_diff = 0
        for criterion in weights1.keys():
            diff = weights1[criterion] - weights2.get(criterion, 0)
            total_squared_diff += diff * diff
        return (total_squared_diff ** 0.5)

    def gen_weights_by_distance(user_weights: Dict[str, int],
                               criteria: List[str],
                               distance_level: str,
                               rng) -> Dict[str, int]:
        """ユーザーの重みからの距離に応じて重みを生成"""
        n = len(criteria)
        weights: Dict[str, int] = {}
        remaining = 100

        if distance_level == "close":
            # ユーザーの重みから±15%以内で調整
            for i, criterion in enumerate(criteria):
                if i == n - 1:
                    weights[criterion] = max(0, remaining)
                else:
                    user_weight = user_weights.get(criterion, 20)
                    min_w = max(10, min(70, user_weight - 15))
                    max_w = min(70, max(10, user_weight + 15))
                    min_w = max(min_w, remaining - (n - i - 1) * 70)
                    max_w = min(max_w, remaining - (n - i - 1) * 10)

                    if min_w <= max_w and remaining > 0:
                        step_count = (max_w - min_w) // 10
                        w = min_w + rng.randrange(step_count + 1) * 10
                        w = min(w, remaining - (n - i - 1) * 10)  # 残りを考慮
                    else:
                        w = max(10, min(remaining - (n - i - 1) * 10, min_w))
                    weights[criterion] = w
                    remaining -= w

        elif distance_level == "medium":
            # ユーザーの重みから±30%以内で調整
            for i, criterion in enumerate(criteria):
                if i == n - 1:
                    weights[criterion] = max(0, remaining)
                else:
                    user_weight = user_weights.get(criterion, 20)
                    min_w = max(10, min(70, user_weight - 30))
                    max_w = min(70, max(10, user_weight + 30))
                    min_w = max(min_w, remaining - (n - i - 1) * 70)
                    max_w = min(max_w, remaining - (n - i - 1) * 10)

                    if min_w <= max_w and remaining > 0:
                        step_count = (max_w - min_w) // 10
                        w = min_w + rng.randrange(step_count + 1) * 10
                        w = min(w, remaining - (n - i - 1) * 10)  # 残りを考慮
                    else:
                        w = max(10, min(remaining - (n - i - 1) * 10, min_w))
                    weights[criterion] = w
                    remaining -= w

        elif distance_level == "far":
            # ユーザーが重視する基準を低く、軽視する基準を高く設定（逆相関）
            for i, criterion in enumerate(criteria):
                if i == n - 1:
                    weights[criterion] = max(0, remaining)
                else:
                    user_weight = user_weights.get(criterion, 20)
                    # ユーザーの重みが高いほどAIの重みを低く設定
                    target_weight = max(10, min(70, 80 - user_weight))
                    min_w = max(10, target_weight - 10)
                    max_w = min(70, target_weight + 10)
                    min_w = max(min_w, remaining - (n - i - 1) * 70)
                    max_w = min(max_w, remaining - (n - i - 1) * 10)

                    if min_w <= max_w and remaining > 0:
                        step_count = (max_w - min_w) // 10
                        w = min_w + rng.randrange(step_count + 1) * 10
                        w = min(w, remaining - (n - i - 1) * 10)  # 残りを考慮
                    else:
                        w = max(10, min(remaining - (n - i - 1) * 10, min_w))
                    weights[criterion] = w
                    remaining -= w
        else:
            # デフォルト：完全ランダム（元のロジック）
            for i, criterion in enumerate(criteria):
                if i == n - 1:
                    weights[criterion] = max(0, remaining)
                else:
                    min_w = max(10, remaining - (n - i - 1) * 70)
                    max_w = min(70, remaining - (n - i - 1) * 10)
                    if min_w <= max_w and remaining > 0:
                        step_count = (max_w - min_w) // 10
                        w = min_w + rng.randrange(step_count + 1) * 10
                        w = min(w, remaining - (n - i - 1) * 10)  # 残りを考慮
                    else:
                        w = max(10, min(remaining - (n - i - 1) * 10, min_w))
                    weights[criterion] = w
                    remaining -= w

        # 最終検証：合計が100になるよう10%刻みで正規化
        total = sum(weights.values())
        if total != 100:
            # 10%刻みで分配（必ず合計100%になるよう調整）
            criteria_list = list(weights.keys())
            n = len(criteria_list)
            
            # 各項目の重要度に基づいて10%刻みで分配
            factor = 100.0 / total
            temp_weights = {}
            for criterion, weight in weights.items():
                temp_weights[criterion] = weight * factor
            
            # 重要度順にソートして10%刻みで分配
            sorted_items = sorted(temp_weights.items(), key=lambda x: x[1], reverse=True)
            adjusted_weights = {}
            remaining = 100
            
            # 最初のn-1項目は10%刻みで分配
            for i, (criterion, temp_weight) in enumerate(sorted_items):
                if i == n - 1:
                    # 最後の項目は残りを割り当て（10%以上を保証）
                    adjusted_weights[criterion] = max(10, remaining)
                else:
                    # 10%刻みで分配、最小10%、残り分配可能な範囲内
                    max_possible = remaining - (n - i - 1) * 10  # 他の項目に最低10%ずつ残す
                    adjusted_weight = min(max_possible, max(10, round(temp_weight / 10) * 10))
                    adjusted_weights[criterion] = adjusted_weight
                    remaining -= adjusted_weight
            
            weights = adjusted_weights

        return weights

    def gen_decisions_for_trial() -> List[str]:
        if trial == 1:
            # 練習: ランダムだが2-2にならないよう調整（ユーザー1名+参加者3名=4名）
            # パターン: 3-1 または 1-3 になるよう調整
            user_choice = user_decision
            participant_choices = []

            # まず2名をランダムに決定
            for _ in range(2):
                participant_choices.append('一次通過' if rng.random() < 0.5 else '見送り')

            # 3名目は2-2を避けるよう調整
            user_count = 1 if user_choice == '一次通過' else 0
            participant_pass_count = sum(1 for d in participant_choices if d == '一次通過')
            total_pass_count = user_count + participant_pass_count

            # 現在のカウントで最終的に2-2になるかチェック
            if total_pass_count == 2:
                # 2-2になってしまうので、3名目で調整
                third_choice = '見送り' if user_choice == '一次通過' else '一次通過'
            else:
                # 2-2にならないので、3名目はランダム
                third_choice = '一次通過' if rng.random() < 0.5 else '見送り'

            participant_choices.append(third_choice)
            return participant_choices
        else:
            # 本番: 全員ユーザーの初回判断の反対
            opposite = '見送り' if user_decision == '一次通過' else '一次通過'
            return [opposite, opposite, opposite]

    decisions = gen_decisions_for_trial()
    distance_levels = ["close", "medium", "far"]
    opinions: List[Dict[str, Any]] = []

    for bot_id in range(3):
        bot_weights = gen_weights_by_distance(
            user_weights,
            criteria,
            distance_levels[bot_id],
            rng
        )

        opinions.append({
            'bot_id': bot_id,
            'decision': decisions[bot_id],
            'weights': bot_weights,
        })

    return opinions

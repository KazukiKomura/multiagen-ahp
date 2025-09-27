# src/utils/argumentation_engine.py

from typing import List, Dict, Any, Tuple, Optional

def extract_atomic_arguments(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    セッションコンテキストから、各参加者の「主張」を構造化データとして抽出します。
    主張は {id, source, claim, reason_criterion, weights, ...} の辞書形式です。
    """
    args = []
    
    # ユーザーの主張を抽出
    user_decision = context.get('user_initial_decision')
    user_weights = context.get('user_initial_weights', {})
    
    if user_decision and user_weights:
        # 最も重視する基準を根拠とする
        top_criterion = max(user_weights, key=user_weights.get, default=None)
        if top_criterion:
            args.append({
                "id": "arg_user",
                "source": "user",
                "claim": user_decision,
                "reason_criterion": top_criterion,
                "reason_weight": user_weights[top_criterion],
                "weights": user_weights,  # 新アルゴリズム用
                "all_weights": user_weights  # 既存アルゴリズム用（互換性維持）
            })

    # 参加者ボットの主張を抽出
    participant_opinions = context.get('participant_opinions', [])
    participant_decisions = context.get('participant_decisions', [])
    
    # participant_opinionsがある場合（完全なデータ）
    if participant_opinions:
        for i, opinion in enumerate(participant_opinions):
            p_decision = opinion.get('decision')
            p_weights = opinion.get('weights', {})
            
            if p_decision and p_weights:
                top_criterion = max(p_weights, key=p_weights.get, default=None)
                if top_criterion:
                    args.append({
                        "id": f"arg_p{i+1}",
                        "source": f"participant{i+1}",
                        "claim": p_decision,
                        "reason_criterion": top_criterion,
                        "reason_weight": p_weights[top_criterion],
                        "weights": p_weights,  # 新アルゴリズム用
                        "all_weights": p_weights,  # 既存アルゴリズム用（互換性維持）
                        "bot_id": opinion.get('bot_id', i)
                    })
    
    # participant_decisionsのみがある場合（簡易データ）- デフォルト重み付けを仮定
    elif participant_decisions:
        # 実際の基準名に合わせてデフォルトパターンを修正
        default_weights_patterns = [
            {'学業成績': 20, '基礎能力テスト': 10, '実践経験': 30, '推薦・評価': 25, '志望動機・フィット': 15},  # 実践経験重視
            {'学業成績': 35, '基礎能力テスト': 25, '実践経験': 15, '推薦・評価': 15, '志望動機・フィット': 10},  # 学業重視
            {'学業成績': 15, '基礎能力テスト': 15, '実践経験': 20, '推薦・評価': 30, '志望動機・フィット': 20}   # 推薦・多様性重視
        ]
        
        for i, decision in enumerate(participant_decisions):
            if decision and i < len(default_weights_patterns):
                weights = default_weights_patterns[i]
                top_criterion = max(weights, key=weights.get)
                args.append({
                    "id": f"arg_p{i+1}",
                    "source": f"participant{i+1}",
                    "claim": decision,
                    "reason_criterion": top_criterion,
                    "reason_weight": weights[top_criterion],
                    "weights": weights,  # 新アルゴリズム用
                    "all_weights": weights,  # 既存アルゴリズム用（互換性維持）
                    "bot_id": i
                })
    
    return args


def determine_attacks(arguments: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    主張リストから、主張間の「攻撃関係」をタプルのリストとして特定します。
    例: ('arg_p1', 'arg_user') -> p1の主張がuserの主張を攻撃している
    """
    attacks = []
    for arg1 in arguments:
        for arg2 in arguments:
            if arg1['id'] == arg2['id']:
                continue
            
            # 結論が異なる場合、互いに攻撃関係にあると定義
            if arg1['claim'] != arg2['claim']:
                attacks.append((arg1['id'], arg2['id']))
    
    return list(set(attacks))  # 重複を削除して返す


def analyze_weight_conflicts(arguments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    重み付けの対立を分析し、最も大きな価値観の対立を特定する
    """
    if len(arguments) < 2:
        return {"conflict_type": "insufficient_data"}
    
    user_arg = next((arg for arg in arguments if arg['source'] == 'user'), None)
    if not user_arg:
        return {"conflict_type": "no_user_argument"}
    
    # ユーザーと異なる判断をしたAI（参加者）を特定
    opposing_args = [arg for arg in arguments if arg['claim'] != user_arg['claim']]
    
    if not opposing_args:
        return {
            "conflict_type": "consensus",
            "message": "全員が同じ判断に達しており、明確な対立はありません。"
        }
    
    # 最も重み配分が異なるAI（参加者）を特定
    max_weight_diff = 0
    main_opponent = None
    conflict_criterion = None
    
    for opp_arg in opposing_args:
        for criterion in user_arg['all_weights'].keys():
            user_weight = user_arg['all_weights'].get(criterion, 0)
            opp_weight = opp_arg['all_weights'].get(criterion, 0)
            weight_diff = abs(user_weight - opp_weight)
            
            if weight_diff > max_weight_diff:
                max_weight_diff = weight_diff
                main_opponent = opp_arg
                conflict_criterion = criterion
    
    return {
        "conflict_type": "weight_conflict",
        "user_criterion": user_arg['reason_criterion'],
        "user_weight": user_arg['reason_weight'],
        "opponent_criterion": main_opponent['reason_criterion'] if main_opponent else None,
        "opponent_weight": main_opponent['reason_weight'] if main_opponent else None,
        "max_conflict_criterion": conflict_criterion,
        "weight_difference": max_weight_diff,
        "opponent_source": main_opponent['source'] if main_opponent else None
    }

def calculate_value_rank_distance(user_weights: Dict[str, float], participant_weights: Dict[str, float]) -> float:
    """
    価値順位の距離を計算します（ケンドールのτ距離を使用）。
    
    Args:
        user_weights: ユーザーの重み配分
        participant_weights: 参加者の重み配分
    
    Returns:
        正規化された順位距離（0=完全一致, 1=完全逆順）
    """
    # 重みでソートして順位を取得
    user_ranking = sorted(user_weights.items(), key=lambda x: x[1], reverse=True)
    participant_ranking = sorted(participant_weights.items(), key=lambda x: x[1], reverse=True)
    
    # 基準名を順位にマッピング
    user_rank_map = {criterion: i for i, (criterion, _) in enumerate(user_ranking)}
    participant_rank_map = {criterion: i for i, (criterion, _) in enumerate(participant_ranking)}
    
    # ケンドールのτ距離を計算
    n = len(user_weights)
    discordant_pairs = 0
    total_pairs = n * (n - 1) // 2
    
    criteria = list(user_weights.keys())
    for i in range(n):
        for j in range(i + 1, n):
            criterion_i, criterion_j = criteria[i], criteria[j]
            
            # ユーザーでのi,jの順序
            user_i_before_j = user_rank_map[criterion_i] < user_rank_map[criterion_j]
            # 参加者でのi,jの順序
            participant_i_before_j = participant_rank_map[criterion_i] < participant_rank_map[criterion_j]
            
            # 順序が異なる場合はdiscordant
            if user_i_before_j != participant_i_before_j:
                discordant_pairs += 1
    
    # 正規化（0-1の範囲に）
    return discordant_pairs / total_pairs if total_pairs > 0 else 0.0


def group_participants_by_value_similarity(arguments: List[Dict[str, Any]], user_weights: Dict[str, float], threshold: float = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    価値順位の類似度に基づいて参加者を相対的に群分けします。
    
    Args:
        arguments: 全参加者の主張リスト
        user_weights: ユーザーの重み配分
        threshold: 非推奨 - 相対的分割のため使用されません
    
    Returns:
        'similar': 価値観が近い群, 'different': 価値観が異なる群
    """
    print(f"[DEBUG] 群分け開始 - 相対的分割アプローチ")
    user_arg = next((arg for arg in arguments if arg['source'] == 'user'), None)
    if not user_arg:
        print(f"[DEBUG] ユーザー主張が見つかりません")
        return {'similar': [], 'different': []}
    
    print(f"[DEBUG] ユーザー重み: {user_weights}")
    
    # 全参加者の距離を計算
    participants_with_distances = []
    for arg in arguments:
        if arg['source'] == 'user':
            continue
            
        # 価値順位距離を計算
        distance = calculate_value_rank_distance(user_weights, arg['weights'])
        participants_with_distances.append((arg, distance))
        print(f"[DEBUG] {arg['source']}: 距離={distance:.3f}, 重み={arg['weights']}")
    
    if not participants_with_distances:
        print(f"[DEBUG] 参加者がいません")
        return {'similar': [], 'different': []}
    
    # 距離でソート（近い順）
    participants_with_distances.sort(key=lambda x: x[1])
    
    # 相対的分割：最も近い参加者を「近い群」に、残りを「異なる群」に
    # 最低1人は「近い群」、最低1人は「異なる群」になるよう調整
    total_count = len(participants_with_distances)
    if total_count == 1:
        # 1人の場合は「近い群」に分類
        similar_group = [participants_with_distances[0][0]]
        different_group = []
    elif total_count == 2:
        # 2人の場合は1人ずつ
        similar_group = [participants_with_distances[0][0]]
        different_group = [participants_with_distances[1][0]]
    else:
        # 3人以上の場合は最も近い1人を「近い群」、残りを「異なる群」に
        similar_count = 1
        similar_group = [p[0] for p in participants_with_distances[:similar_count]]
        different_group = [p[0] for p in participants_with_distances[similar_count:]]
    
    print(f"[DEBUG] 相対分割結果:")
    for arg in similar_group:
        distance = next(d for p, d in participants_with_distances if p == arg)
        print(f"[DEBUG]   近い群: {arg['source']} (距離={distance:.3f})")
    
    for arg in different_group:
        distance = next(d for p, d in participants_with_distances if p == arg)
        print(f"[DEBUG]   異なる群: {arg['source']} (距離={distance:.3f})")
        
    print(f"[DEBUG] 群分け結果 - 近い群: {len(similar_group)}人, 異なる群: {len(different_group)}人")
    return {
        'similar': similar_group,
        'different': different_group
    }


def calculate_criterion_salience_scores(arguments: List[Dict[str, Any]], groups: Dict[str, List[Dict[str, Any]]], user_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    基準ごとのサリエンス（顕著性）スコアを計算します。
    
    Args:
        arguments: 全参加者の主張リスト
        groups: 群分け結果
        user_weights: ユーザーの重み配分
    
    Returns:
        群別・基準別のサリエンススコア
    """
    print(f"[DEBUG] サリエンススコア計算開始")
    user_arg = next((arg for arg in arguments if arg['source'] == 'user'), None)
    if not user_arg:
        return {'similar': {}, 'different': {}}
    
    results = {'similar': {}, 'different': {}}
    
    for group_name, group_members in groups.items():
        print(f"[DEBUG] {group_name}群の計算開始 - メンバー数: {len(group_members)}")
        criterion_scores = {}
        
        for criterion in user_weights.keys():
            total_salience = 0.0
            print(f"[DEBUG]   {criterion}の計算:")
            
            for participant in group_members:
                # 価値係数（重み配分の重要度）
                value_coefficient = user_weights[criterion] / 100.0
                
                # 重み差分（ユーザーとの重み差の絶対値）
                weight_difference = abs(user_weights[criterion] - participant['weights'][criterion])
                
                # サリエンススコア = 価値係数 × 重み差分
                salience = value_coefficient * weight_difference
                print(f"[DEBUG]     {participant['source']}: 係数={value_coefficient:.2f} × 重み差={weight_difference} = {salience:.2f}")
                total_salience += salience
            
            criterion_scores[criterion] = total_salience
            print(f"[DEBUG]   {criterion}: 合計サリエンス = {total_salience:.2f}")
        
        results[group_name] = criterion_scores
        print(f"[DEBUG] {group_name}群スコア: {criterion_scores}")
    
    return results


def extract_top_conflict_points(salience_scores: Dict[str, Dict[str, float]], groups: Dict[str, List[Dict[str, Any]]], arguments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    各群から最もサリエンスの高い論点を抽出します（2本立て）。
    
    Args:
        salience_scores: 群別・基準別サリエンススコア
        groups: 群分け結果
        arguments: 全参加者の主張リスト
    
    Returns:
        抽出された論点リスト（最大2件）
    """
    print(f"[DEBUG] 論点抽出開始")
    conflict_points = []
    user_arg = next((arg for arg in arguments if arg['source'] == 'user'), None)
    if not user_arg:
        print(f"[DEBUG] ユーザー主張なし - 論点抽出終了")
        return conflict_points
    
    for group_name, criterion_scores in salience_scores.items():
        print(f"[DEBUG] {group_name}群から論点抽出 - スコア: {criterion_scores}")
        if not criterion_scores:
            print(f"[DEBUG]   スコアなし - スキップ")
            continue
            
        # 最高サリエンスの基準を特定
        top_criterion = max(criterion_scores.items(), key=lambda x: x[1])
        criterion_name, salience_score = top_criterion
        print(f"[DEBUG]   最高サリエンス基準: {criterion_name} (スコア: {salience_score:.2f})")
        
        # その基準で最も対立しているAI（参加者）を特定
        group_members = groups[group_name]
        if not group_members:
            print(f"[DEBUG]   メンバーなし - スキップ")
            continue
            
        # ユーザーとの重み差が最大のAI（参加者）を選択
        max_weight_diff = 0
        top_opponent = None
        
        for participant in group_members:
            weight_diff = abs(user_arg['weights'][criterion_name] - participant['weights'][criterion_name])
            print(f"[DEBUG]     {participant['source']}: {criterion_name}重み差 = {weight_diff}")
            if weight_diff > max_weight_diff:
                max_weight_diff = weight_diff
                top_opponent = participant
        
        if top_opponent:
            print(f"[DEBUG]   最大対立相手: {top_opponent['source']} (重み差: {max_weight_diff})")
            # 寄与者リスト（同じ基準を重視するAI）
            contributors = []
            for participant in group_members:
                # その基準の重要度が平均以上のAIを寄与者とする
                avg_weight = sum(participant['weights'].values()) / len(participant['weights'])
                if participant['weights'][criterion_name] >= avg_weight:
                    contributors.append({
                        'participant_id': participant['source'],
                        'weight': participant['weights'][criterion_name],
                        'contribution': participant['weights'][criterion_name] / 100.0
                    })
            
            # 群名を読みやすい形式に変換（表示用）
            group_label = "価値観が近い群" if group_name == 'similar' else "価値観が異なる群"

            # 内部表現は 'similar' / 'different' を保持し、表示は後段で変換する
            conflict_point = {
                'group': group_name,  # 'similar' or 'different'
                'criterion': criterion_name,
                'salience_score': salience_score,
                'user_weight': user_arg['weights'][criterion_name],
                'opponent_weight': top_opponent['weights'][criterion_name],
                'weight_difference': max_weight_diff,
                'top_opponent': top_opponent,
                'contributors': contributors[:3]  # 上位3名まで
            }
            
            conflict_points.append(conflict_point)
            print(f"[DEBUG]   論点追加: {criterion_name} ({group_label})")
        else:
            print(f"[DEBUG]   対立相手なし - スキップ")
    
    # サリエンススコア順でソート
    conflict_points.sort(key=lambda x: x['salience_score'], reverse=True)
    print(f"[DEBUG] 論点抽出完了 - 抽出数: {len(conflict_points)}")
    
    return conflict_points[:2]  # 最大2件  # 最大2件


def generate_two_track_debate_summary(conflict_points: List[Dict[str, Any]], arguments: List[Dict[str, Any]], user_weights: Dict[str, float]) -> Dict[str, Any]:
    """
    2本立て論点に基づく討論サマリーを生成します。
    
    Args:
        conflict_points: 抽出された論点リスト
        arguments: 全参加者の主張リスト
        user_weights: ユーザーの重み配分
    
    Returns:
        新アルゴリズムによる分析結果
    """
    user_arg = next((arg for arg in arguments if arg['source'] == 'user'), None)
    if not user_arg or not conflict_points:
        return {
            "algorithm_type": "two_track_ranking_salience",
            "conflict_points": [],
            "user_claim_summary": "ユーザーの主張が見つかりませんでした。"
        }
    
    # 各論点の詳細情報を構築
    detailed_points = []
    for i, point in enumerate(conflict_points):
        group_label = "価値観が近い群" if point['group'] == 'similar' else "価値観が異なる群"
        
        # 対立の性質を分析
        if point['user_weight'] > point['opponent_weight']:
            conflict_nature = f"ユーザーは{point['criterion']}をより重視（{point['user_weight']}% vs {point['opponent_weight']}%）"
        else:
            conflict_nature = f"ユーザーは{point['criterion']}を軽視（{point['user_weight']}% vs {point['opponent_weight']}%）"
        
        # 互換性のため、重みや相手情報を明示的に含める
        detailed_point = {
            "rank": i + 1,
            "criterion": point['criterion'],
            "group": group_label,
            "salience_score": round(point['salience_score'], 2),
            "conflict_nature": conflict_nature,
            "weight_difference": point['weight_difference'],
            # 既存レンダラ（ai_chat.py）が参照するフィールド
            "user_weight": point.get('user_weight'),
            "opponent_weight": point.get('opponent_weight'),
            # 相手情報を2系統で提供（互換性確保）
            "top_opponent": {
                "source": point['top_opponent'].get('source'),
                "claim": point['top_opponent'].get('claim')
            },
            "top_opponent_id": point['top_opponent']['source'],
            "top_opponent_claim": point['top_opponent']['claim'],
            "contributors": point['contributors']
        }
        
        detailed_points.append(detailed_point)
    
    # 全体サマリー
    total_participants = len([arg for arg in arguments if arg['source'] != 'user'])
    similar_count = len([p for p in conflict_points if p['group'] == 'similar'])
    different_count = len([p for p in conflict_points if p['group'] == 'different'])
    
    return {
        "algorithm_type": "two_track_ranking_salience",
        "user_claim_summary": f"ユーザーは「{user_arg['reason_criterion']}」({user_arg['reason_weight']}%)を最重視して「{user_arg['claim']}」と判断しています。",
        "analysis_overview": {
            "total_participants": total_participants,
            "conflict_points_found": len(conflict_points),
            "similar_group_conflicts": similar_count,
            "different_group_conflicts": different_count
        },
        "conflict_points": detailed_points,
        "suggested_question_direction": _generate_two_track_question_direction(detailed_points)
    }


def _display_label(name: str) -> str:
    """UI表示用の基準名マッピング"""
    try:
        return '学歴・所属' if name == '志望動機・フィット' else name
    except Exception:
        return name


def _generate_two_track_question_direction(detailed_points: List[Dict[str, Any]]) -> str:
    """
    2本立て論点に基づく質問方向性を生成します。
    """
    if not detailed_points:
        return "対立点が見つからないため、重み配分の根拠について"
    
    if len(detailed_points) == 1:
        point = detailed_points[0]
        criterion_display = _display_label(point['criterion'])
        return f"最大の論点である「{criterion_display}」の評価について、{point['group']}との対立理由について"
    
    # 2つの論点がある場合
    point1, point2 = detailed_points[0], detailed_points[1]
    criterion1_display = _display_label(point1['criterion'])
    criterion2_display = _display_label(point2['criterion'])
    return f"「{criterion1_display}」({point1['group']})と「{criterion2_display}」({point2['group']})の2つの論点について、それぞれの重視理由について"


def summarize_debate(arguments: List[Dict[str, Any]], attacks: List[Tuple[str, str]], user_weights: Dict[str, float] = None) -> Dict[str, str]:
    """
    分析結果を、LLMが理解しやすい自然言語のサマリーに変換します。
    新しいアルゴリズム（2本立てランキング型サリエンス）を適用します。
    """
    user_arg = next((arg for arg in arguments if arg['source'] == 'user'), None)
    if not user_arg:
        return {"key_conflict_point": "ユーザーの主張が見つかりませんでした。"}

    # user_weightsが提供されない場合は、既存のロジックを使用
    if user_weights is None:
        return _legacy_summarize_debate(arguments, attacks)
    
    # 新アルゴリズム: 2本立てランキング型サリエンス
    print(f"[DEBUG] 新アルゴリズム開始 - 2本立てランキング型サリエンス")
    try:
        # Step 1: 価値順位による群分け
        print(f"[DEBUG] Step 1: 価値順位による群分け")
        groups = group_participants_by_value_similarity(arguments, user_weights)
        
        # Step 2: 基準ごとのサリエンススコア計算
        print(f"[DEBUG] Step 2: 基準ごとのサリエンススコア計算")
        salience_scores = calculate_criterion_salience_scores(arguments, groups, user_weights)
        
        # Step 3: 各群からTop-1論点を抽出
        print(f"[DEBUG] Step 3: 各群からTop-1論点を抽出")
        conflict_points = extract_top_conflict_points(salience_scores, groups, arguments)
        
        # Step 4: 2本立て討論サマリー生成
        print(f"[DEBUG] Step 4: 2本立て討論サマリー生成")
        analysis_result = generate_two_track_debate_summary(conflict_points, arguments, user_weights)
        
        print(f"[DEBUG] 新アルゴリズム完了 - 抽出された論点数: {len(conflict_points)}")
        # 既存形式との互換性を保つため、キー名を調整
        return {
            "key_conflict_point": _format_conflict_points_for_legacy(analysis_result),
            "user_claim_summary": analysis_result["user_claim_summary"],
            "suggested_question_direction": analysis_result["suggested_question_direction"],
            "algorithm_type": "two_track_ranking_salience",
            "detailed_analysis": analysis_result
        }
        
    except Exception as e:
        # エラー時は既存ロジックにフォールバック
        print(f"新アルゴリズムでエラーが発生しました: {e}")
        return _legacy_summarize_debate(arguments, attacks)


def _legacy_summarize_debate(arguments: List[Dict[str, Any]], attacks: List[Tuple[str, str]]) -> Dict[str, str]:
    """
    既存の分析ロジック（互換性維持のため）
    """
    user_arg = next((arg for arg in arguments if arg['source'] == 'user'), None)
    
    # 重み対立分析
    weight_analysis = analyze_weight_conflicts(arguments)
    
    if weight_analysis["conflict_type"] == "consensus":
        return {
            "key_conflict_point": "全AIが同じ判断に達しており、対立点はありません。重み配分の根拠を深掘りすると良いでしょう。",
            "user_claim_summary": f"ユーザーは「{user_arg['reason_criterion']}」({user_arg['reason_weight']}%)を最重視して「{user_arg['claim']}」と判断しています。",
            "suggested_question_direction": "この重み配分を選んだ具体的な理由や、他の基準との比較検討について"
        }
    
    # ユーザーの主張に反論している主張を特定
    counter_args = [
        next(arg for arg in arguments if arg['id'] == attacker_id)
        for attacker_id, target_id in attacks if target_id == user_arg['id']
    ]

    if not counter_args:
        return {
            "key_conflict_point": "明確な対立点はなく、判断は一致しています。",
            "user_claim_summary": f"ユーザーは「{user_arg['reason_criterion']}」({user_arg['reason_weight']}%)を根拠に「{user_arg['claim']}」と判断しています。"
        }

    # 最も重み配分が対立しているAIを特定
    main_opponent = counter_args[0]
    if weight_analysis["conflict_type"] == "weight_conflict":
        opponent_info = next((arg for arg in counter_args 
                            if arg['source'] == weight_analysis["opponent_source"]), counter_args[0])
        main_opponent = opponent_info
    
    if user_arg['reason_criterion'] == main_opponent['reason_criterion']:
        # CASE 1: 最重視基準が同じで、結論が異なる場合
        criterion = user_arg['reason_criterion']
        key_conflict_point = (
            f"最大の論点は、同じ『{criterion}』を重視しながらも、結論が異なる点（評価の解釈の違い）です。"
        )
        suggested_question_direction = (
            f"同じ『{criterion}』という基準を見ても判断が分かれた理由、つまり評価の解釈（通過ライン）がどう違うのかについて"
        )
    else:
        # CASE 2: 最重視基準が異なる場合（既存のロジック）
        user_criterion = user_arg['reason_criterion']
        opponent_criterion = main_opponent['reason_criterion']
        key_conflict_point = (
            f"最大の論点は『{user_criterion}』重視 vs 『{opponent_criterion}』重視の価値観対立です。"
        )
        suggested_question_direction = (
            f"なぜ{user_criterion}を{opponent_criterion}より重視するのか、具体的な根拠について"
        )

    summary = {
        "user_claim_summary": f"ユーザーは「{user_arg['reason_criterion']}」({user_arg['reason_weight']}%)を最重視して「{user_arg['claim']}」と判断しています。",
        "opponent_claim_summary": f"AI{main_opponent['source'][-1]}は「{main_opponent['reason_criterion']}」({main_opponent['reason_weight']}%)を最重視して「{main_opponent['claim']}」と判断しています。",
        "key_conflict_point": key_conflict_point,
        "weight_difference": weight_analysis.get("weight_difference", 0),
        "suggested_question_direction": suggested_question_direction
    }
    
    return summary


def _format_conflict_points_for_legacy(analysis_result: Dict[str, Any]) -> str:
    """
    新アルゴリズムの結果を既存形式のkey_conflict_pointに変換
    """
    if not analysis_result.get("conflict_points"):
        return "明確な対立点は検出されませんでした。"
    
    points = analysis_result["conflict_points"]
    
    if len(points) == 1:
        point = points[0]
        return f"最大の論点は「{point['criterion']}」({point['group']})での対立です。{point['conflict_nature']}"
    
    elif len(points) == 2:
        point1, point2 = points[0], points[1]
        return f"2つの主要論点: 1)「{point1['criterion']}」({point1['group']}) 2)「{point2['criterion']}」({point2['group']})での対立があります。"
    
    return "複数の論点で対立が検出されました。"


def generate_focused_question(debate_summary: Dict[str, str], conversation_history: List[Dict[str, str]]) -> str:
    """
    論証分析結果と会話履歴を基に、核心を突く質問を1つ生成する
    """
    # 過去の質問を確認（重複回避）
    past_questions = [msg['content'] for msg in conversation_history if msg.get('role') == 'assistant']
    
    conflict_point = debate_summary.get("key_conflict_point", "")
    question_direction = debate_summary.get("suggested_question_direction", "")
    
    # 対立構造に基づく質問生成
    if "価値観対立" in conflict_point:
        if any("なぜ" in q for q in past_questions):
            return f"{question_direction}について、他の選択肢と比較してどう考えますか？"
        else:
            return f"{question_direction}について、詳しく聞かせてください。"
    
    elif "重み配分" in conflict_point:
        return "この重み配分で最も重視した判断基準について、具体的な根拠を教えてください。"
    
    else:
        return "あなたの判断で最も決定的だった要素は何でしょうか？"

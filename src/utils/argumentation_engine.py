# src/utils/argumentation_engine.py

from typing import List, Dict, Any, Tuple, Optional

def extract_atomic_arguments(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    セッションコンテキストから、各参加者の「主張」を構造化データとして抽出します。
    主張は {id, source, claim, reason_criterion, ...} の辞書形式です。
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
                "all_weights": user_weights
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
                        "all_weights": p_weights,
                        "bot_id": opinion.get('bot_id', i)
                    })
    
    # participant_decisionsのみがある場合（簡易データ）- デフォルト重み付けを仮定
    elif participant_decisions:
        default_weights_patterns = [
            {'学業成績': 20, '試験スコア': 10, '研究能力': 30, '推薦状': 25, '多様性': 15},  # 研究重視
            {'学業成績': 35, '試験スコア': 25, '研究能力': 15, '推薦状': 15, '多様性': 10}   # 学業重視
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
                    "all_weights": weights,
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
    
    # ユーザーと異なる判断をした参加者を特定
    opposing_args = [arg for arg in arguments if arg['claim'] != user_arg['claim']]
    
    if not opposing_args:
        return {
            "conflict_type": "consensus",
            "message": "全員が同じ判断に達しており、明確な対立はありません。"
        }
    
    # 最も重み配分が異なる参加者を特定
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


def summarize_debate(arguments: List[Dict[str, Any]], attacks: List[Tuple[str, str]]) -> Dict[str, str]:
    """
    分析結果を、LLMが理解しやすい自然言語のサマリーに変換します。
    """
    user_arg = next((arg for arg in arguments if arg['source'] == 'user'), None)
    if not user_arg:
        return {"key_conflict_point": "ユーザーの主張が見つかりませんでした。"}

    # 重み対立分析
    weight_analysis = analyze_weight_conflicts(arguments)
    
    if weight_analysis["conflict_type"] == "consensus":
        return {
            "key_conflict_point": "全参加者が同じ判断に達しており、対立点はありません。重み配分の根拠を深掘りすると良いでしょう。",
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

    # 最も重み配分が対立している参加者を特定
    main_opponent = counter_args[0]
    if weight_analysis["conflict_type"] == "weight_conflict":
        opponent_info = next((arg for arg in counter_args 
                            if arg['source'] == weight_analysis["opponent_source"]), counter_args[0])
        main_opponent = opponent_info
    
    # <<< [ADD] START: ここから条件分岐を追加 >>>
    if user_arg['reason_criterion'] == main_opponent['reason_criterion']:
        # CASE 1: 最重視基準が同じで、結論が異なる場合
        criterion = user_arg['reason_criterion']
        key_conflict_point = (
            f"最大の論点は、同じ『{criterion}』を重視しながらも、結論が異なる点（評価の解釈の違い）です。"
        )
        # student_infoから具体的なスコアを取得できるとさらに良い
        # score_info = context.get('student_info', {}).get('detailed_scores', {}).get(criterion, {}).get('main_score', 'N/A')
        suggested_question_direction = (
            f"同じ『{criterion}』という基準を見ても判断が分かれた理由、つまり評価の解釈（合格ライン）がどう違うのかについて"
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
    # <<< [ADD] END: 条件分岐はここまで >>>

    summary = {
        "user_claim_summary": f"ユーザーは「{user_arg['reason_criterion']}」({user_arg['reason_weight']}%)を最重視して「{user_arg['claim']}」と判断しています。",
        "opponent_claim_summary": f"参加者{main_opponent['source'][-1]}は「{main_opponent['reason_criterion']}」({main_opponent['reason_weight']}%)を最重視して「{main_opponent['claim']}」と判断しています。",
        "key_conflict_point": key_conflict_point,  # 修正後の値を使用
        "weight_difference": weight_analysis.get("weight_difference", 0),
        "suggested_question_direction": suggested_question_direction  # 修正後の値を使用
    }
    
    return summary


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

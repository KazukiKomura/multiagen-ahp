"""
論点分析ユーティリティ
ai_chat.pyから抽出した分析ロジックを独立関数として提供
"""
from typing import Dict, Any, List


def _display_label(name: str) -> str:
    """UI表示用の基準名マッピング（ai_chat.pyと統一）"""
    try:
        return '学歴・所属' if name == '志望動機・フィット' else name
    except Exception:
        return name


def analyze_debate_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    セッションコンテキストから論点分析を実行
    
    Args:
        context: セッションコンテキスト（user_initial_decision, participant_opinions等）
    
    Returns:
        分析結果辞書（conflict_points, user_claim_summary, analysis_overview等）
    """
    if not context:
        return {"error": "コンテキストが提供されていません"}
    
    try:
        import src.utils.argumentation_engine as argumentation_engine
        
        print(f"[DEBUG] 論点分析開始")
        print(f"  - あなたの判断: {context.get('user_initial_decision')}")
        print(f"  - あなたの重み: {context.get('user_initial_weights')}")
        print(f"  - 参加者意見数: {len(context.get('participant_opinions', []))}")
        
        # 論理エンジンを実行
        arguments = argumentation_engine.extract_atomic_arguments(context)
        attacks = argumentation_engine.determine_attacks(arguments)
        user_weights = context.get('user_initial_weights') or context.get('user_final_weights') or {}
        
        print(f"[DEBUG] 抽出された主張数: {len(arguments)}")
        print(f"[DEBUG] 攻撃関係数: {len(attacks)}")
        
        if user_weights:
            debate_summary = argumentation_engine.summarize_debate(arguments, attacks, user_weights)
        else:
            debate_summary = argumentation_engine.summarize_debate(arguments, attacks)
            
        print(f"[DEBUG] 使用アルゴリズム: {debate_summary.get('algorithm_type', 'legacy')}")
        
        # 新アルゴリズムの詳細分析結果を使用
        if debate_summary.get('algorithm_type') == 'two_track_ranking_salience':
            detailed_analysis = debate_summary.get('detailed_analysis', {})
            conflict_points = detailed_analysis.get('conflict_points', [])
            analysis_overview = detailed_analysis.get('analysis_overview', {})
            user_claim_summary = detailed_analysis.get('user_claim_summary', '')
            
            # 「ユーザー」を「あなた」に変換
            user_claim_summary = user_claim_summary.replace('ユーザーは', 'あなたは').replace('ユーザー', 'あなた')
            
            print(f"[DEBUG] 検出された対立点数: {len(conflict_points)}")
            
            return {
                "success": True,
                "algorithm_type": "two_track_ranking_salience",
                "user_claim_summary": user_claim_summary,
                "analysis_overview": analysis_overview,
                "conflict_points": conflict_points,
                "detailed_analysis": detailed_analysis,
                "key_conflict_point": debate_summary.get('key_conflict_point', ''),
                "suggested_question_direction": detailed_analysis.get('suggested_question_direction', '')
            }
        else:
            # 従来アルゴリズムの場合
            return {
                "success": True,
                "algorithm_type": "legacy",
                "key_conflict_point": debate_summary.get('key_conflict_point', ''),
                "user_claim_summary": debate_summary.get('user_claim_summary', ''),
                "suggested_question": debate_summary.get('suggested_question', '')
            }
            
    except Exception as e:
        print(f"[ERROR] 論点分析エラー: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def format_analysis_for_display(analysis_result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    分析結果をUI表示用にフォーマット
    
    Args:
        analysis_result: analyze_debate_context()の戻り値
        context: セッションコンテキスト（participant_opinions等を含む）
    
    Returns:
        UI表示用フォーマット済み辞書
    """
    if not analysis_result.get("success"):
        return {
            "markdown_content": "❌ 論点分析中にエラーが発生しました",
            "conflict_count": 0
        }
    
    if analysis_result.get("algorithm_type") == "two_track_ranking_salience":
        # 新アルゴリズム用フォーマット（AIチャット版と統一）
        conflict_points = analysis_result.get("conflict_points", [])
        analysis_overview = analysis_result.get("analysis_overview", {})
        user_claim_summary = analysis_result.get("user_claim_summary", "")
        
        lines = [
            "## 📊 議論状況の分析\n\n",
            f"**{user_claim_summary}**\n\n",
            f"**AI**: {analysis_overview.get('total_participants', 3)}名の評価者（あなたとの価値観の近さで分類）\n",
            f"**対立論点**: {analysis_overview.get('conflict_points_found', 0)}件の主要な違い\n\n"
        ]
        
        # AI参加者の判断を表示（AIチャット版と同じ形式）
        if context and context.get('participant_opinions'):
            participant_opinions = context.get('participant_opinions', [])
            user_decision = context.get('user_initial_decision', '一次通過')
            
            lines.append("### 🎯 各AIの判断\n")
            for i, opinion in enumerate(participant_opinions[:3]):
                ai_name = f"AI{i + 1}"
                decision = opinion.get('decision', '不明')
                weights_info = opinion.get('weights', {})
                
                if weights_info:
                    top_criterion = max(weights_info, key=weights_info.get, default='不明')
                    top_weight = weights_info.get(top_criterion, 0)
                    
                    agreement = "✅ あなたと同じ判断" if decision == user_decision else "❌ あなたと異なる判断"
                    
                    lines.append(f"- **{ai_name}**: {decision} {agreement}\n")
                    lines.append(f"  最重視: {_display_label(top_criterion)}（{top_weight}%）\n")
            
            lines.append("\n")
        
        # 論点詳細を表示（AIチャット版と同じ形式）
        if conflict_points:
            lines.append("### 🔍 注目すべき違い\n")
            for i, point in enumerate(conflict_points[:2], 1):  # 上位2つまで
                criterion = _display_label(point.get('criterion', '不明'))
                user_weight = point.get('user_weight', 0)
                opponent_weight = point.get('opponent_weight', 0)
                
                # opponent情報を取得
                opponent_info = point.get('top_opponent', {})
                opponent_source = opponent_info.get('source', 'participant1')
                opponent_claim = opponent_info.get('claim', '不明')
                
                # participant1 → AI1 に変換
                opponent_name = opponent_source.replace('participant', 'AI')
                
                # グループラベルを相対的分割に合わせて変換
                group_label = point.get('group', '不明な群')
                if group_label == '価値観が近い群':
                    group_explanation = f'あなたに最も近い価値観を持つAI'
                elif group_label == '価値観が異なる群':
                    group_explanation = f'あなたと異なる価値観を持つAI'
                else:
                    group_explanation = group_label.replace('価値観が近い群', '近い価値観のAI').replace('価値観が異なる群', '異なる価値観のAI')
                
                lines.extend([
                    f"**違い{i}: {criterion}への評価**\n",
                    f"- あなた: {context.get('user_initial_decision', '一次通過')}（{user_weight}%重視）\n",
                    f"- {opponent_name}: {opponent_claim}（{opponent_weight}%重視）\n",
                    f"- 関係: {group_explanation}\n\n"
                ])
        
        suggested_direction = analysis_result.get("suggested_question_direction", "")
        if suggested_direction:
            lines.append("### 💭 推奨される議論の方向性\n")
            lines.append(f"{suggested_direction}\n\n")
        
        return {
            "markdown_content": "".join(lines),
            "conflict_count": len(conflict_points),
            "analysis_overview": analysis_overview
        }
    
    else:
        # 従来アルゴリズム用フォーマット（AIチャット版と統一）
        user_claim = analysis_result.get('user_claim_summary', '')
        key_conflict = analysis_result.get('key_conflict_point', '')
        suggested_question = analysis_result.get('suggested_question', '')
        
        lines = [
            "## 📊 議論分析結果\n\n",
            f"**{user_claim}**\n\n",
            f"**主要な違い**: {key_conflict}\n\n"
        ]
        
        if suggested_question:
            lines.append("### 💭 推奨質問\n")
            lines.append(f"{suggested_question}\n\n")
        
        return {
            "markdown_content": "".join(lines),
            "conflict_count": 1
        }
"""
Main routes for the multi-agent AHP experiment system.
Handles general pages, questionnaires, and decision forms.
"""

from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
import uuid
from ..repository.session_repository import session_repository
from ..utils.data import get_student_for_trial, format_student_for_display

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Landing page with condition selection"""
    return render_template('index.html')


@main_bp.route('/start/<condition>')
def start_experiment(condition):
    """Start the experiment with selected condition"""
    if condition != 'ai-facilitator':
        return redirect(url_for('main.index'))
    
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session['condition'] = condition
    session['trial'] = 1
    
    # Create session in database
    session_repository.create_session(session_id, condition)
    
    # 困難な学生ケースを事前に選択してセッションに保存
    challenging_student = get_student_for_trial(1, session_id)
    if challenging_student:
        formatted_student = format_student_for_display(challenging_student)
        session['student_info'] = formatted_student
        session_repository.update_session(session_id, student_data=formatted_student)
    
    return redirect(url_for('main.questionnaire', phase='pre'))


@main_bp.route('/questionnaire/<phase>')
def questionnaire(phase):
    """Display questionnaire forms"""
    if 'session_id' not in session:
        return redirect(url_for('main.index'))
    
    return render_template('questionnaire.html', 
                         phase=phase, 
                         condition=session['condition'],
                         trial=session.get('trial', 1))


@main_bp.route('/save_questionnaire', methods=['POST'])
def save_questionnaire():
    """Save questionnaire responses"""
    print(f"質問紙保存リクエスト受信")
    
    if 'session_id' not in session:
        print("エラー: セッションIDが見つかりません")
        return jsonify({'error': 'No session'}), 400
    
    try:
        data = request.json
        print(f"受信データ: {data}")
        
        phase = data.get('phase')
        responses = data.get('responses', {})
        
        session_id = session['session_id']
        print(f"セッションID: {session_id}, フェーズ: {phase}")
        
        # Retrieve existing questionnaire data
        session_data = session_repository.get_session(session_id)
        questionnaire_data = session_data.get('questionnaire_data', {}) if session_data else {}
        
        # Update with new responses
        questionnaire_data[phase] = responses
        
        print(f"保存する質問紙データ: {questionnaire_data}")
        
        # Save to database
        success = session_repository.update_session(
            session_id,
            questionnaire_data=questionnaire_data,
            phase=phase
        )
        
        if success:
            next_url = None
            if phase == 'pre':
                # 事前質問紙の次は練習セッション（trial=1）の体験フェーズ
                next_url = url_for('main.experience')
            else:
                # 事後質問紙（練習後や各本番後）
                current_trial = session.get('trial', 1)
                if current_trial < 4:
                    # 次のトライアルへ（本番1->2, 2->3）
                    session['trial'] = current_trial + 1
                    # 次の学生情報をロード
                    from ..utils.data import get_student_for_trial, format_student_for_display
                    session.pop('student_info', None)
                    next_student = get_student_for_trial(session['trial'], session_id)
                    if next_student:
                        formatted = format_student_for_display(next_student)
                        session['student_info'] = formatted
                        session_repository.update_session(session_id, student_data=formatted)
                    next_url = url_for('main.experience')
                else:
                    # 最終（本番3=trial4）終了後は完了ページへ
                    next_url = url_for('main.complete')

            print(f"成功: 次のURL = {next_url}")
            return jsonify({'success': True, 'next_url': next_url})
        else:
            print("エラー: データベース保存に失敗")
            return jsonify({'error': 'Failed to save questionnaire'}), 500
            
    except Exception as e:
        print(f"例外エラー: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@main_bp.route('/decision')
def decision():
    """Legacy route - redirect to experience"""
    if 'session_id' not in session:
        return redirect(url_for('main.index'))
    
    # 統合体験フェーズに転送
    return redirect(url_for('main.experience'))


@main_bp.route('/experience')
def experience():
    """Integrated experience phase"""
    if 'session_id' not in session:
        return redirect(url_for('main.index'))
    
    trial = session.get('trial', 1)
    
    # 学生情報が未設定の場合は取得
    if 'student_info' not in session:
        challenging_student = get_student_for_trial(trial, session['session_id'])
        if challenging_student:
            formatted_student = format_student_for_display(challenging_student)
            session['student_info'] = formatted_student
            session_repository.update_session(session['session_id'], student_data=formatted_student)
        else:
            return "学生データの取得に失敗しました", 500
    
    # Create info object expected by template
    student_info = session['student_info']
    info = {
        'title': '入学審査意思決定',
        'description': '以下の応募者の情報を基に、入学許可の可否を判断してください。',
        'criteria': ['学業成績', '試験スコア', '研究能力', '推薦状', '多様性'],
        'student': {
            'id': student_info.get('student_id', 'Unknown'),
            'major': student_info.get('major', 'Unknown'),
            'region': student_info.get('region', 'Unknown'),
            'institution_rank': student_info.get('institution_rank', '中ランク'),
            'gpa': student_info.get('gpa', 3.0),
            'gre_quant': student_info.get('gre_quant', 160),
            'gre_verbal': student_info.get('gre_verbal', 160),
            'gre_writing': student_info.get('gre_writing', 4.0),
            'sop_score': student_info.get('sop_score', 3.0),
            'diversity_score': student_info.get('diversity_score', 3.0),
            'rec_letters': student_info.get('rec_letters', []),
            'detailed_scores': student_info.get('detailed_scores', {})
        }
    }
    
    return render_template('experience.html',
                         trial=trial,
                         student_info=student_info,
                         info=info,
                         condition=session['condition'])


@main_bp.route('/save_decision', methods=['POST'])
def save_decision():
    """Save decision and weights"""
    print("意思決定保存リクエスト受信")
    
    if 'session_id' not in session:
        print("エラー: セッションIDが見つかりません")
        return jsonify({'error': 'No session'}), 400
    
    try:
        data = request.json
        print(f"受信した意思決定データ: {data}")
        
        session_id = session['session_id']
        
        # Get existing decision data
        session_data = session_repository.get_session(session_id)
        decision_data = session_data.get('decision_data', {}) if session_data else {}
        
        # Update decision data（初回判断）
        decision_data.update({
            'user_decision': data.get('decision'),
            'user_weights': data.get('weights', {}),
            'reasoning': data.get('reasoning', ''),
            'trial': session.get('trial', 1),
            'timestamp': data.get('timestamp')
        })

        # 参加者の事前判断・重みを決定して永続化
        criteria = ['学業成績', '試験スコア', '研究能力', '推薦状', '多様性']
        from ..utils.data import generate_participant_opinions
        trial = session.get('trial', 1)
        user_initial = decision_data.get('user_decision')
        if user_initial:
            # chat_test からのオーバーライド（任意）
            incoming = data.get('participant_opinions')
            def _is_valid(op):
                try:
                    if op.get('decision') not in ['合格', '不合格']:
                        return False
                    w = op.get('weights', {})
                    keys = set(w.keys())
                    return keys == set(criteria) and all(isinstance(w[k], (int, float)) for k in criteria)
                except Exception:
                    return False
            if isinstance(incoming, list) and len(incoming) >= 2 and _is_valid(incoming[0]) and _is_valid(incoming[1]):
                opinions = incoming[:2]
            else:
                opinions = generate_participant_opinions(user_initial, criteria, trial, session_id)

            decision_data['participant_opinions'] = opinions
            decision_data['participant_decisions'] = [op['decision'] for op in opinions]
            session['participant_decisions'] = decision_data['participant_decisions']
        
        print(f"保存する決定データ: {decision_data}")
        
        # Save to database
        success = session_repository.update_session(
            session_id,
            decision_data=decision_data
        )
        
        if success:
            print("成功: 初回判断・参加者意見を保存")
            return jsonify({'success': True, 'participant_opinions': decision_data.get('participant_opinions', [])})
        else:
            print("エラー: データベース保存に失敗")
            return jsonify({'error': 'Failed to save decision'}), 500
            
    except Exception as e:
        print(f"意思決定保存中の例外エラー: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@main_bp.route('/aggregation')
def aggregation():
    """Bot opinion aggregation phase (legacy)"""
    if 'session_id' not in session:
        return redirect(url_for('main.index'))
    
    bot_opinions = session.get('bot_opinions', [])
    condition = session['condition']
    
    return render_template('aggregation.html', 
                         bot_opinions=bot_opinions, 
                         condition=condition)


@main_bp.route('/intervention')
def intervention():
    """AI intervention phase (legacy)"""
    if 'session_id' not in session:
        return redirect(url_for('main.index'))
    
    condition = session['condition']
    
    return render_template('intervention.html', 
                         condition=condition,
                         student_info=session.get('student_info', {}))


@main_bp.route('/final_decision')
def final_decision():
    """Final decision review page"""
    if 'session_id' not in session:
        return redirect(url_for('main.index'))
    
    # 初期決定データを取得
    session_data = session_repository.get_session(session['session_id'])
    decision_data = session_data.get('decision_data', {}) if session_data else {}
    
    return render_template('final_decision.html',
                         condition=session['condition'],
                         student_info=session.get('student_info', {}),
                         initial_decision=decision_data.get('user_decision'),
                         initial_weights=decision_data.get('user_weights', {}),
                         initial_reasoning=decision_data.get('reasoning', ''))


@main_bp.route('/save_final_decision', methods=['POST'])
def save_final_decision():
    """Save final decision after AI interaction"""
    print(f"最終決定保存リクエスト受信")
    
    if 'session_id' not in session:
        print("エラー: セッションIDが見つかりません")
        return jsonify({'error': 'No session'}), 400
    
    try:
        data = request.json
        print(f"受信した最終決定データ: {data}")
        session_id = session['session_id']
        
        # Get existing decision data
        session_data = session_repository.get_session(session_id)
        decision_data = session_data.get('decision_data', {}) if session_data else {}
        
        # Update with final decision
        decision_data.update({
            'final_decision': data.get('final_decision'),
            'final_weights': data.get('final_weights', {}),
            'change_reasoning': data.get('change_reasoning', ''),
            'confidence': data.get('confidence'),
            'final_timestamp': data.get('timestamp')
        })
        
        print(f"保存する最終決定データ: {decision_data}")
        
        # Save to database
        success = session_repository.update_session(
            session_id,
            decision_data=decision_data,
            phase='post_questionnaire'
        )
        
        if success:
            # 進行制御:
            # - トライアル1（練習）終了後は本番セッション1へ（事後質問紙は無し）
            # - トライアル2,3,4は最終決定後に事後質問紙へ
            current_trial = session.get('trial', 1)
            if current_trial == 1:
                # 次のトライアルへ進める
                session['trial'] = 2
                # 次の学生情報を準備（存在すれば保存）
                from ..utils.data import get_student_for_trial, format_student_for_display
                session.pop('student_info', None)
                next_student = get_student_for_trial(2, session_id)
                if next_student:
                    formatted = format_student_for_display(next_student)
                    session['student_info'] = formatted
                    session_repository.update_session(session_id, student_data=formatted)
                next_url = url_for('main.experience')
            else:
                # 本番セッションでは、事後質問紙の前に最終結果アナウンスページを表示
                next_url = url_for('main.final_outcome')

            print(f"成功: 次のURL = {next_url}")
            return jsonify({'success': True, 'next': next_url})
        else:
            print("エラー: データベース保存に失敗")
            return jsonify({'error': 'Failed to save final decision'}), 500
            
    except Exception as e:
        print(f"最終決定保存中の例外エラー: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@main_bp.route('/complete')
def complete():
    """Completion page"""
    if 'session_id' not in session:
        return redirect(url_for('main.index'))
    
    return render_template('complete.html',
                         condition=session['condition'])


@main_bp.route('/final_outcome')
def final_outcome():
    """Announce the final decision (majority vote) before post-questionnaire.

    Majority is calculated among: user's final decision and two participant opinions.
    For main sessions (trial>=2), participants' decisions are set to be the opposite
    of the user's initial decision (as used in the comparison view).
    """
    if 'session_id' not in session:
        return redirect(url_for('main.index'))

    trial = session.get('trial', 1)
    if trial == 1:
        # 練習ではこのページはスキップ
        return redirect(url_for('main.experience'))

    # 直近の決定データを取得
    session_data = session_repository.get_session(session['session_id']) or {}
    decision_data = session_data.get('decision_data', {})

    user_initial = decision_data.get('user_decision')  # フェーズ1
    user_final = decision_data.get('final_decision')   # フェーズ4

    # 参加者（ボット）の決定は固定ルール（変更しない）
    votes = []
    if user_final:
        votes.append({'who': 'あなた', 'decision': user_final})

    # 本番セッションの参加者判断
    # 1) セッションに固定済み（save_decision時）のものがあればそれを使用
    # 2) なければ「初回判断の反対」を採用
    part_decisions = session.get('participant_decisions')
    if isinstance(part_decisions, list) and len(part_decisions) >= 2:
        votes.append({'who': '参加者 1', 'decision': part_decisions[0]})
        votes.append({'who': '参加者 2', 'decision': part_decisions[1]})
    elif user_initial:
        opposite = '合格' if user_initial == '不合格' else '不合格'
        votes.append({'who': '参加者 1', 'decision': opposite})
        votes.append({'who': '参加者 2', 'decision': opposite})

    # 多数決の計算
    counts = {'合格': 0, '不合格': 0}
    for v in votes:
        if v['decision'] in counts:
            counts[v['decision']] += 1
    outcome = '合格' if counts['合格'] >= counts['不合格'] else '不合格'

    # 保存（任意）
    decision_data['group_outcome'] = outcome
    session_repository.update_session(session['session_id'], decision_data=decision_data)

    return render_template('final_outcome.html',
                           trial=trial,
                           votes=votes,
                           outcome=outcome,
                           condition=session.get('condition'))


@main_bp.route('/admin/data')
def admin_data():
    """Admin data view"""
    sessions = session_repository.get_all_sessions()
    return render_template('admin_data.html', sessions=sessions)


@main_bp.route('/chat_test')
def chat_test():
    """AIチャット機能のテストページ（論理エンジン統合版）"""
    return render_template('chat_test.html')

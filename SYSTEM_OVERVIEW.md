# Multi-Agent AHP 実験システム - 完全再現ガイド

## 🎯 システム概要

本システムは、**学生入学審査におけるAI支援意思決定**の実験的研究プラットフォームです。主に以下の3つの実験条件を通じて、**議論支援の効果**を定量的に測定することを目的としています。

### 核心コンセプト
- **AHP（階層化意思決定プロセス）**: 5つの評価基準による重み付け判断
- **多エージェント議論**: ユーザーと3つのAI参加者による意見交換
- **段階的議論支援の削減**: フル支援→静的分析→支援なしの3条件比較

## 🧪 実験条件と研究設計

### 実験条件の詳細

| 条件 | ブランチ | 議論支援の種類 | フェーズ数 | 主要特徴 |
|------|----------|----------------|------------|----------|
| **Condition 1** | `feat/condition1` | **対話型ファシリテーション** | 4 | ✅ AIチャット機能<br>✅ リアルタイム対話<br>✅ 最低4回の会話要求 |
| **Condition 2** | `feat/condition2_ahponly` | **静的論点分析** | 4 | ❌ チャット無効化<br>✅ 自動論点整理表示<br>✅ 衝突点の可視化 |
| **Condition 3** | `feat/condition3_ahp_augless` | **支援なし** | 3 | ❌ 議論支援フェーズ削除<br>❌ 論点分析なし<br>⚡ 直接最終決定 |

### 実験仮説
1. **H1**: 対話型支援（Condition1）が最も意思決定変化を促進する
2. **H2**: 静的分析（Condition2）でも一定の効果が得られる
3. **H3**: 支援なし（Condition3）では意思決定変化が最小となる

## 🏗️ システムアーキテクチャ

### 全体構成
```
multiagentahp/
├── src/                    # アプリケーションコア
│   ├── app.py             # Flask アプリケーション ファクトリー
│   ├── routes/            # HTTP エンドポイント層
│   │   ├── main.py        # メインフロー（4フェーズ管理）
│   │   └── ai_chat.py     # AI対話システム（条件1専用）
│   ├── repository/        # データアクセス層
│   │   └── session_repository.py  # SQLite セッション管理
│   ├── utils/             # ビジネスロジック層
│   │   ├── data.py        # 学生選択・AI意見生成
│   │   ├── argumentation_engine.py    # 論点抽出エンジン
│   │   └── argumentation_analysis.py  # 条件2専用分析
│   ├── templates/         # Jinja2 HTMLテンプレート
│   │   ├── base.html      # 共通レイアウト
│   │   ├── experience.html # 統合実験画面（核心）
│   │   └── questionnaire.html # 事前・事後質問紙
│   └── static/            # CSS・JavaScript
├── data/                  # SQLite データベース
│   └── sessions.db        # 実験データ永続化
├── dataset/               # 学生データセット
│   └── student admission data.csv
├── run.py                 # アプリケーション起動スクリプト
└── requirements.txt       # Python依存関係
```

### 技術スタック詳細

| 層 | 技術 | 役割 |
|----|----- |------|
| **フロントエンド** | HTML5, CSS3, Vanilla JavaScript | 単一ページアプリケーション |
| **バックエンド** | Python 3.8+, Flask 2.0+ | RESTful API, セッション管理 |
| **データ** | SQLite3 | 軽量データベース、実験ログ |
| **AI処理** | OpenAI API (GPT-4) | 論点分析、対話生成 |
| **デプロイ** | Docker, Docker Compose | コンテナ化された環境 |

## 📊 データモデルとフロー

### データベーススキーマ

#### sessions テーブル
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,        -- UUID形式のセッション識別子
    condition TEXT NOT NULL,            -- "condition1" | "condition2" | "condition3"
    trial INTEGER NOT NULL,             -- 練習(1) or 本番(2+)
    phase TEXT DEFAULT 'questionnaire', -- 現在のフェーズ
    student_data TEXT,                  -- JSON: 選択された学生情報
    questionnaire_data TEXT,            -- JSON: 事前質問紙回答
    decision_data TEXT,                 -- JSON: 意思決定の全履歴
    ai_chat_data TEXT,                  -- JSON: チャット履歴（条件1のみ）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### decision_data の詳細構造
```json
{
    "user_decision": "一次通過",              // 初期判断
    "user_weights": {                         // 初期重み付け
        "学業成績": 25,
        "志望動機・フィット": 20,
        "課外活動": 15,
        "推薦状": 20,
        "面接評価": 20
    },
    "participant_opinions": [                 // AI参加者の意見
        {
            "bot_id": 0,
            "decision": "見送り",
            "weights": { "学業成績": 40, ... },
            "reasoning": "成績面での懸念..."
        }
    ],
    "final_decision": "見送り",               // 最終判断
    "final_weights": { ... },                // 最終重み付け
    "confidence": 75,                        // 信頼度（1-100）
    "decision_change": true,                 // 判断変更の有無
    "weight_change": 15.2                    // 重み変更度（距離）
}
```

### 実験フロー（全条件共通）

#### フェーズ1: 初期意思決定
```javascript
// 1. 学生情報の提示
studentData = getStudentForTrial(trial);

// 2. 5基準による重み付け（スライダーUI）
weights = {
    "学業成績": userInput,      // 合計100%になるよう制約
    "志望動機・フィット": userInput,
    "課外活動": userInput,
    "推薦状": userInput,
    "面接評価": userInput
};

// 3. 一次通過/見送りの判断
decision = userChoice; // "一次通過" | "見送り"
```

#### フェーズ2: 他者意見確認
```javascript
// AI参加者3名の意見生成
botOpinions = generateParticipantOpinions(studentData);

// 各AI参加者の表示
for (let ai of botOpinions) {
    display({
        decision: ai.decision,           // 判断
        weights: ai.weights,            // 重み付け
        reasoning: ai.reasoning         // 判断理由
    });
}
```

#### フェーズ3: 議論支援（条件依存）

##### Condition 1: 対話型ファシリテーション
```javascript
// AIファシリテーターとのチャット
async function sendMessage(userMessage) {
    const response = await fetch('/ai_chat', {
        method: 'POST',
        body: JSON.stringify({
            message: userMessage,
            decision: initialDecision,
            conversation_count: currentTurn
        })
    });
    
    const aiResponse = await response.json();
    addMessage(aiResponse.message, false);
    
    // 最低4回の会話が必要
    if (conversationCount >= 4) {
        showNextPhaseButton();
    }
}
```

##### Condition 2: 静的論点分析
```javascript
// 論点分析の自動実行と表示
async function loadArgumentationAnalysis() {
    const analysis = await fetch('/api/argumentation_analysis');
    const result = await analysis.json();
    
    if (result.success) {
        // Markdown形式の分析結果を表示
        displayAnalysis(result.formatted.markdown_content);
    }
}
```

##### Condition 3: フェーズスキップ
```javascript
// フェーズ2から直接フェーズ4へ
function nextPhase() {
    if (currentPhase === 1) {
        showPhase(2); // 直接最終決定へ
    }
}
```

#### フェーズ4: 最終決定
```javascript
// 最終判断と重み付けの再設定
finalDecision = {
    decision: userFinalChoice,
    weights: userFinalWeights,
    confidence: userConfidence,
    reasoning: userReasoning
};

// 変化量の計算
decisionChange = (initialDecision !== finalDecision.decision);
weightChange = calculateWeightDistance(initialWeights, finalWeights);
```

## 🔧 キーコンポーネント解説

### 1. 学生選択アルゴリズム

```python
# src/utils/data.py
def get_student_for_trial(trial):
    """戦略的困難ケース選択"""
    students = load_student_data()
    
    # 境界線ケースのフィルタリング
    boundary_cases = [s for s in students if s['decision'] in [1, 2]]
    
    # 困難度スコアの計算
    for student in boundary_cases:
        scores = [student[criterion] for criterion in CRITERIA]
        student['difficulty'] = calculate_difficulty_score(scores)
    
    # 困難度順にソートして選択
    sorted_students = sorted(boundary_cases, key=lambda x: x['difficulty'], reverse=True)
    return sorted_students[trial - 1]  # トライアル番号に応じて選択

def calculate_difficulty_score(scores):
    """判断困難度の算出"""
    variance = np.var(scores)           # 評価のばらつき
    mean_distance = abs(np.mean(scores) - 2.5)  # 中央値からの距離
    return variance * (1 + mean_distance)       # 複合困難度スコア
```

### 2. AI参加者意見生成

```python
def generate_participant_opinions(student_data):
    """多様な価値観を持つAI参加者の生成"""
    
    # 3つの異なる重み付けパターン
    weight_patterns = [
        {"学業成績": 45, "志望動機・フィット": 25, "課外活動": 10, "推薦状": 10, "面接評価": 10},  # 成績重視
        {"学業成績": 15, "志望動機・フィット": 40, "課外活動": 25, "推薦状": 10, "面接評価": 10},  # 適性重視
        {"学業成績": 20, "志望動機・フィット": 15, "課外活動": 15, "推薦状": 25, "面接評価": 25}   # バランス型
    ]
    
    opinions = []
    for i, weights in enumerate(weight_patterns):
        # 重み付きスコア計算
        weighted_score = sum(student_data[criterion] * (weights[criterion] / 100) 
                            for criterion in weights)
        
        # 閾値による判断（個体差を付与）
        threshold = 2.3 + (i * 0.2)  # AI毎に異なる基準
        decision = "一次通過" if weighted_score > threshold else "見送り"
        
        opinions.append({
            "bot_id": i,
            "decision": decision,
            "weights": weights,
            "reasoning": generate_reasoning(student_data, weights, decision)
        })
    
    return opinions
```

### 3. 論点分析エンジン（条件2の核心）

```python
# src/utils/argumentation_engine.py
def extract_atomic_arguments(context):
    """議論から原子的主張を抽出"""
    user_decision = context['user_initial_decision']
    user_weights = context['user_initial_weights']
    participants = context['participant_opinions']
    
    arguments = []
    
    # ユーザーの主張
    arguments.append({
        'source': 'user',
        'claim': user_decision,
        'weights': user_weights,
        'type': 'decision'
    })
    
    # AI参加者の主張
    for i, participant in enumerate(participants):
        arguments.append({
            'source': f'participant{i+1}',
            'claim': participant['decision'],
            'weights': participant['weights'],
            'type': 'decision'
        })
    
    return arguments

def determine_attacks(arguments):
    """主張間の攻撃関係を特定"""
    attacks = []
    
    for i, arg1 in enumerate(arguments):
        for j, arg2 in enumerate(arguments):
            if i != j and arg1['claim'] != arg2['claim']:
                # 意見対立による攻撃関係
                attacks.append({
                    'attacker': arg2['source'],
                    'target': arg1['source'],
                    'type': 'decision_conflict',
                    'strength': calculate_conflict_strength(arg1, arg2)
                })
    
    return attacks

def calculate_conflict_strength(arg1, arg2):
    """衝突の強度計算"""
    # 重み付けの距離を計算
    weight_distance = sum(abs(arg1['weights'][criterion] - arg2['weights'][criterion]) 
                         for criterion in arg1['weights']) / 100
    
    # 判断の対立度
    decision_conflict = 1.0 if arg1['claim'] != arg2['claim'] else 0.0
    
    return weight_distance * decision_conflict
```

### 4. 条件別実装の詳細

#### Condition 1: ai_chat.py
```python
@ai_chat_bp.route('/ai_chat', methods=['POST'])
def ai_chat():
    """対話型ファシリテーション"""
    data = request.get_json()
    user_message = data.get('message', '')
    decision = data.get('decision', '')
    conversation_count = data.get('conversation_count', 0)
    
    # 手続き的公正システムによる応答生成
    # （詳細は元のREADMEを参照）
    
    return jsonify({
        'message': ai_response,
        'conversation_count': conversation_count + 1
    })
```

#### Condition 2: argumentation_analysis.py
```python
@main_bp.route('/api/argumentation_analysis', methods=['GET'])
def get_argumentation_analysis():
    """静的論点分析API"""
    session_data = session_repository.get_session(session['session_id'])
    
    # コンテキスト構築
    context = {
        'user_initial_decision': session_data['decision_data']['user_decision'],
        'user_initial_weights': session_data['decision_data']['user_weights'],
        'participant_opinions': session_data['decision_data']['participant_opinions']
    }
    
    # 論点分析実行
    analysis_result = analyze_debate_context(context)
    formatted_result = format_analysis_for_display(analysis_result, context)
    
    return jsonify({
        'success': True,
        'analysis': analysis_result,
        'formatted': formatted_result,
        'markdown_content': formatted_result['markdown_content']
    })
```

#### Condition 3: HTMLテンプレート分岐
```html
<!-- experience.html -->
{% if condition != "condition3" %}
<div class="phase-content" data-phase="2">
    <!-- 議論支援フェーズ -->
</div>
{% endif %}

<div class="phase-content" data-phase="{% if condition == 'condition3' %}2{% else %}3{% endif %}">
    <!-- 最終決定フェーズ -->
</div>
```

## 🚀 システム構築・実行手順

### 1. 環境構築

```bash
# リポジトリクローン
git clone https://github.com/your-repo/multiagentahp.git
cd multiagentahp

# Python環境構築
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt

# 環境変数設定
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "FLASK_ENV=development" >> .env
```

### 2. データベース初期化

```bash
# データディレクトリ作成
mkdir -p data

# 学生データの配置
# dataset/student admission data.csv が必要

# データベース確認
python check_db.py
```

### 3. 実験条件の切り替え

```bash
# Condition 1 (対話型)
git checkout feat/condition1

# Condition 2 (静的分析)
git checkout feat/condition2_ahponly

# Condition 3 (支援なし)
git checkout feat/condition3_ahp_augless
```

### 4. アプリケーション起動

```bash
# 開発サーバー起動
python run.py

# または直接Flask実行
python -m src.app

# ブラウザで http://localhost:5000 にアクセス
```

### 5. Docker環境（オプション）

```bash
# Docker環境での起動
docker-compose up -d

# ログ確認
docker-compose logs -f

# 停止
docker-compose down
```

## 📈 実験データ分析

### データエクスポート

```bash
# 全セッションをCSV出力
python export_sessions_csv.py

# 特定セッションの詳細表示
python dump_session.py <session_id>

# チャットログの確認
python check_db.py chat <session_id>
```

### 主要評価指標

#### 1. 意思決定変化率
```python
decision_change_rate = (changed_decisions / total_decisions) * 100
```

#### 2. 重み付け変化量
```python
def calculate_weight_distance(initial_weights, final_weights):
    return sum(abs(initial_weights[criterion] - final_weights[criterion]) 
               for criterion in initial_weights) / 100
```

#### 3. 信頼度変化
```python
confidence_improvement = final_confidence - initial_confidence
```

#### 4. 条件別比較
```sql
-- 条件別の意思決定変化率
SELECT 
    condition,
    COUNT(*) as total_sessions,
    SUM(CASE WHEN JSON_EXTRACT(decision_data, '$.decision_change') = 'true' 
             THEN 1 ELSE 0 END) as changed_decisions,
    ROUND(
        (SUM(CASE WHEN JSON_EXTRACT(decision_data, '$.decision_change') = 'true' 
                  THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2
    ) as change_rate_percent
FROM sessions 
WHERE trial > 1  -- 練習除外
GROUP BY condition;
```

## 🔧 カスタマイズとメンテナンス

### 新しい実験条件の追加

1. **新ブランチ作成**
```bash
git checkout -b feat/condition4_new_approach
```

2. **条件別ロジック実装**
```python
# src/routes/main.py
@main_bp.route('/experiment')
def experiment():
    condition = session.get('condition', 'condition1')
    
    if condition == 'condition4':
        # 新しい条件のロジック
        pass
    
    return render_template('experience.html', condition=condition)
```

3. **UIテンプレート分岐**
```html
<!-- src/templates/experience.html -->
{% if condition == 'condition4' %}
    <!-- 新条件専用UI -->
{% endif %}
```

### 学生データセットの更新

```python
# dataset/student admission data.csv の形式
# student_id,学業成績,志望動機・フィット,課外活動,推薦状,面接評価,decision
# 001,3.5,4.0,2.8,3.2,4.1,1

# 新しいデータセット追加時
def validate_student_data(csv_path):
    """データセットの整合性確認"""
    required_columns = ['student_id', '学業成績', '志望動機・フィット', 
                       '課外活動', '推薦状', '面接評価', 'decision']
    
    df = pd.read_csv(csv_path)
    assert all(col in df.columns for col in required_columns)
    assert df['decision'].isin([1, 2, 3]).all()  # 1:見送り, 2:保留, 3:通過
```

### パフォーマンス最適化

```python
# セッションデータのキャッシュ化
from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_student_data():
    return load_student_data()

# データベースインデックス追加
def optimize_database():
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_condition ON sessions(condition);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_trial ON sessions(trial);")
```

## ⚠️ 重要な注意事項

### セキュリティ
- **本システムは研究用途専用**です。本番環境での使用は想定していません
- OpenAI API キーは `.env` ファイルで管理し、リポジトリには含めないでください
- セッションデータには個人識別可能な情報を含めないでください

### データ整合性
- 実験中はブランチを切り替えないでください
- データベースのバックアップを定期的に取得してください
- セッション途中での条件変更は結果に影響します

### 実験倫理
- 被験者への適切なインフォームドコンセントが必要です
- データの匿名化と適切な管理を行ってください
- 実験結果の公表時は適切な倫理審査を受けてください

## 📚 理論的背景と関連研究

### 手続き的公正理論（Thibaut & Walker, 1975）
本システムの理論的基盤となる手続き的公正の5要素：

1. **Voice（発言機会）**: 意見表明の機会保証
2. **Neutrality（中立性）**: 偏見のない情報提示
3. **Transparency（透明性）**: 明確なルールと基準
4. **Respect（尊重）**: 参加者の尊厳維持
5. **Consistency（一貫性）**: 公平で一貫した手続き

### AHP（階層化意思決定プロセス）
Saaty (1980) のAHP理論を簡略化し、実験環境に適用：

- 5つの評価基準による多基準意思決定
- 重み付けによる選好の明示化
- 重み変化による学習効果の測定

### 議論マイニングとComputational Argumentation
- 自動論点抽出による議論構造の可視化
- 攻撃関係の特定による対立点の明確化
- 議論支援システムの効果測定

## 📞 サポートとトラブルシューティング

### よくある問題

1. **OpenAI API エラー**
```bash
# API キーの確認
echo $OPENAI_API_KEY

# レート制限エラーの場合は時間をおいて再実行
```

2. **データベースロックエラー**
```bash
# プロセス確認
ps aux | grep python

# データベースファイルの権限確認
ls -la data/sessions.db
```

3. **条件切り替えが反映されない**
```bash
# ブラウザキャッシュクリア
# または
# シークレットモードで実行
```

### ログとデバッグ

```python
# デバッグ用ログ出力
import logging
logging.basicConfig(level=logging.DEBUG)

# セッション状態の確認
@main_bp.route('/debug/session')
def debug_session():
    return jsonify(dict(session))
```

## 📄 ライセンスとクレジット

MIT License

本システムは学術研究目的で開発されました。商用利用や再配布の際は適切なクレジットを付与してください。

### 引用方法
```
@software{multiagent_ahp_2025,
  title={Multi-Agent AHP Experiment System},
  author={Research Team},
  year={2025},
  url={https://github.com/your-repo/multiagentahp}
}
```

---
*🤖 このドキュメントは実験システムの完全な再現を目的として作成されました*
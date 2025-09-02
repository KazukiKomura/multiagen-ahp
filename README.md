# Multi-Agent AHP 実験システム

## 概要

**手続き的公正理論に基づくAI支援型意思決定システム**

このシステムは、学生入学審査を題材とした意思決定実験を行うためのWebアプリケーションです。Thibaut & Walker (1975) の手続き的公正理論に基づいた**ProceduralJusticeSystem**を実装し、特に少数派の納得感向上を目的としています。

## 🎯 主要特徴

- **手続き的公正システム**: Voice、Neutrality、Transparency、Respect、Consistencyの5要素を完全実装
- **3層LLMアーキテクチャ**: LLM-G (生成)、LLM-J (評価)、LLM-W (監視) による品質保証
- **有限状態機械 (FSM)**: 優先度制御による対話フロー管理
- **リアルタイム満足度監視**: 5軸満足度スコアによる自動補修システム
- **統合UI**: 4段階の意思決定プロセスを単一ページで実現

## 🏗️ システム構成

### アーキテクチャ

```
├── src/
│   ├── services/           # ビジネスロジック層
│   │   └── procedural_justice.py   # 手続き的公正システム
│   ├── repository/         # データアクセス層
│   │   └── session_repository.py   # セッションデータ管理
│   ├── routes/            # Web層
│   │   ├── main.py        # メインルート
│   │   └── ai_chat.py     # AIチャット API
│   ├── utils/             # ユーティリティ層
│   │   └── data.py        # データ処理・学生選択
│   └── templates/         # HTMLテンプレート
│       ├── base.html      # ベーステンプレート
│       ├── experience.html # 統合実験画面
│       └── questionnaire.html # 質問紙
```

### 技術スタック

- **フロントエンド**: HTML5, CSS3, JavaScript (Vanilla)
- **バックエンド**: Python Flask (Repository Pattern)
- **データベース**: SQLite3 (sessions, ai_chat_logs)
- **AI実装**: 3層LLMアーキテクチャ (テンプレートベース)

## 🔬 実験フロー

### 現在の条件
- **AI-Facilitator条件**: 手続き的公正システムによる対話支援

### プロセス（単一トライアル）
1. **事前質問紙** → 気分・信頼度の測定
2. **統合体験フェーズ**:
   - Phase 1: 個人意思決定（判定 + 重み設定）
   - Phase 2: 他参加者意見確認
   - Phase 3: **AIファシリテーター対話** 🤖
   - Phase 4: 最終決定
3. **事後質問紙** → 満足度・有用性の測定

## 🚀 セットアップと実行

### 1. 依存関係のインストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd multiagentahp

# Python仮想環境の作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. アプリケーション実行

```bash
# 開発サーバーの起動
python run.py

# またはFlaskアプリの直接実行
python -m src.app

# ブラウザで http://localhost:5000 にアクセス
```

### 3. データベース確認（オプション）

```bash
# データベース内容の確認
python check_db.py                    # 基本情報
python check_db.py session <id>       # 特定セッション詳細
python check_db.py chat <session_id>  # AIチャットログ
python check_db.py help               # ヘルプ表示
```

## 🔧 手続き的公正システムの詳細

### FSM制御による優先度
1. **Respect** (最優先) → 不適切発言への対応
2. **Voice** → 重視点の確認・聴取
3. **Transparency** → ルール・制約の明示 (3ターン以内)
4. **Neutrality** → 4要素対称的観点の提示
5. **Appeal** → 異議申立て機会の提供
6. **Summary** → 手続き完了報告

### 3層LLMアーキテクチャ
```python
# LLM-G: 複数候補生成 (k=2-3)
candidates = generate_candidates(action, context)

# ハード検証: 禁則語・フォーマット・文字数制限
validated = hard_validation(candidates)

# LLM-J: ルーブリック評価による最良選択
best_response = llm_judge(validated, rubric)

# LLM-W: 満足度5軸監視 (V,N,T,C,R)
scores = llm_watchdog(response, state)

# 自動補修システム (スコア < 1.2)
if scores['overall'] < 1.2:
    repair = execute_repair(scores)
```

## 📊 データベース構造

### sessions テーブル
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    condition TEXT,                 -- 実験条件
    trial INTEGER,                  -- トライアル番号
    phase TEXT,                     -- 現在フェーズ
    student_data TEXT,              -- 学生情報 (JSON)
    questionnaire_data TEXT,        -- 質問紙回答 (JSON)
    decision_data TEXT,             -- 意思決定データ (JSON)
    ai_chat_data TEXT,             -- AIチャット履歴 (JSON)
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### ai_chat_logs テーブル
```sql
CREATE TABLE ai_chat_logs (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    turn INTEGER,                   -- ターン番号
    user_message TEXT,              -- ユーザーメッセージ
    ai_response TEXT,               -- AI応答
    satisfaction_scores TEXT,       -- 満足度スコア (JSON)
    pj_state TEXT,                  -- 手続き的公正状態 (JSON)
    timestamp TIMESTAMP
);
```

## 🎓 学生データについて

### データソース
- `dataset/student admission data.csv` - 実際の学生入学データ
- 戦略的選択アルゴリズムによる困難ケースの抽出

### 選択基準
- **境界線ケース**: decision=1,2 (合否判定が困難)
- **強みと弱みの混在**: 分散が大きい評価プロフィール
- **多様性確保**: 専攻・地域の偏りを防止
- **判定困難度スコア**: 総合的な困難度による順位付け

## 🔍 実験データの分析

### アクセス可能なデータ
1. **意思決定プロセス**: 初期判断 → 最終判断の変化
2. **重み付け変化**: 5基準の重要度変遷
3. **対話ログ**: AIとの完全な対話履歴
4. **満足度メトリクス**: ターン別5軸スコア
5. **手続き的公正達成度**: 各要素の完了状況

### エンドポイント
- `/admin/data` - 全セッションデータの閲覧
- `/pj_state` - 現在の手続き的公正状態
- `/chat_history` - セッション別チャット履歴

## ⚙️ カスタマイズ

### 1. 手続き的公正パラメータの調整
```python
# src/services/procedural_justice.py
class ProceduralJusticeSystem:
    def __init__(self):
        self.max_turns = 5              # 最大ターン数
        self.transparency_deadline = 3   # 透明性説明の期限
```

### 2. 学生選択アルゴリズムの変更
```python
# src/utils/data.py
def select_challenging_students():
    # 困難度計算ロジックの調整
    pass
```

### 3. UIデザインの変更
```css
/* src/static/css/style.css */
.category-card {
    /* カード表示のカスタマイズ */
}
```

## 📈 実験結果の評価指標

### 主要メトリクス
1. **手続き的公正達成度** (5要素 × 完了率)
2. **満足度スコア** (V,N,T,C,R軸 × ターン推移)
3. **意思決定変化** (初期 vs 最終判断)
4. **対話品質** (ターン数、内容の深度)
5. **少数派納得感** (主観的満足度、信頼度)

## 📚 理論的背景

### Thibaut & Walker (1975) 手続き的公正理論
- **Voice Effect**: 意見表明機会の保証
- **Neutrality**: 中立的・公平な情報提示
- **Trustworthiness**: 信頼できる透明なプロセス
- **Status Recognition**: 尊厳と敬意の維持

### CHI'25論文との差異
本システムは精度向上(+14.1%)ではなく、**納得感向上**を主目的としており、手続き的公正理論に特化した独自実装です。

## ⚠️ 注意事項

### セキュリティ
- 実験用途専用（本番環境不適合）
- セッションキーは `app.secret_key` で変更
- 個人情報の適切な管理

### データ移行
- 古い`decisions`テーブルが残存（31件）
- 新しい`sessions`テーブルに統合推奨

## 📄 ライセンス

MIT License

## 📞 サポート

実験に関するお問い合わせは実験担当者までご連絡ください。

---
*🤖 このREADMEは手続き的公正理論に基づくシステム実装の詳細を反映しています*
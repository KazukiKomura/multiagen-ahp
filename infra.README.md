# インフラ構築手順（Lightsail/EC2 + Docker）

このドキュメントは、現在のリポジトリ構成（`python run.py` で起動、Docker/Compose 利用）に合わせた手順です。

> ⚠️ **重要**: AWS Lightsail/EC2のNitroインスタンスでは、追加ディスクは`/dev/xvdf`ではなく**`/dev/nvme1n1`**として認識されます。古いドキュメントで`/dev/xvdf`が記載されている場合は`/dev/nvme1n1`に読み替えてください。

## 📋 目次
- [A) AWS CLI でのフルセットアップ（新規作成）](#a-aws-cli-でのフルセットアップ新規作成)
- [B) 既存インスタンス向け手順](#b-既存インスタンス向け手順)

---

# A) AWS CLI でのフルセットアップ（新規作成）

AWS CLIを使ってLightsailインスタンスを最初から作成する手順です。

## A-0) AWS CLI セットアップ

### ✅ 推奨: IAM Identity Center（SSO）で設定

```bash
# 1) プロファイルを作成
aws configure sso
# 聞かれる内容に回答：
# SSO start URL（管理者から共有されたURL）
# SSOのリージョン（例: ap-northeast-1 か us-west-2 等）
# AWSアカウントID
# ロール（AdministratorAccess など）
# デフォルトリージョン: ap-northeast-1 / 出力: json

# 2) ブラウザでログイン
aws sso login --profile <プロファイル名>

# 3) 確認
aws sts get-caller-identity --profile <プロファイル名>
```

### ◇ 代替: アクセスキーで設定（急ぎ用）

```bash
aws configure
# AWS Access Key ID / Secret Access Key を入力
# Default region: ap-northeast-1
# Default output: json
```

## A-1) SSH キー準備

```bash
# デフォルト鍵をダウンロード（地域別）
aws lightsail download-default-key-pair --region ap-northeast-1 > dkp.json
# ※ privateKeyBase64はBase64ではなく、PEM形式で\nエスケープされた形式
jq -r '.privateKeyBase64' dkp.json | sed 's/\\n/\n/g' > ~/.ssh/lightsail-apne1.pem
chmod 400 ~/.ssh/lightsail-apne1.pem
```

## A-2) Lightsail インスタンス作成

```bash
# 利用可能なブループリント確認
aws lightsail get-blueprints --region ap-northeast-1 \
  --query "blueprints[?contains(blueprintId, 'ubuntu')].[blueprintId,version]" --output table

# 利用可能なバンドル（プラン）確認  
aws lightsail get-bundles --region ap-northeast-1 --output table

# インスタンス作成（$5プラン推奨）
aws lightsail create-instances \
    --instance-names "multiagent-ahp" \
    --availability-zone "ap-northeast-1a" \
    --blueprint-id "ubuntu_22_04" \
    --bundle-id "medium_2_0" \
    --region ap-northeast-1 \
    --user-data "#!/bin/bash
apt-get update
apt-get install -y docker.io docker-compose-v2 git curl jq
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu"
```

## A-3) 静的IP割り当て

```bash
# 静的IP作成・割り当て
aws lightsail allocate-static-ip \
    --static-ip-name "multiagent-ahp-ip" \
    --region ap-northeast-1

aws lightsail attach-static-ip \
    --static-ip-name "multiagent-ahp-ip" \
    --instance-name "multiagent-ahp" \
    --region ap-northeast-1

# IPアドレス確認
STATIC_IP=$(aws lightsail get-static-ip \
    --static-ip-name "multiagent-ahp-ip" \
    --region ap-northeast-1 \
    --query 'staticIp.ipAddress' --output text)
echo "Static IP: $STATIC_IP"
```

## A-4) 追加ディスク作成・アタッチ

```bash
# 20GB ディスク作成
aws lightsail create-disk \
    --disk-name "multiagent-ahp-data" \
    --availability-zone "ap-northeast-1a" \
    --size-in-gb 20 \
    --region ap-northeast-1

# インスタンス起動完了まで待機
aws lightsail get-instance \
    --instance-name "multiagent-ahp" \
    --region ap-northeast-1 \
    --query 'instance.state.name'

# ディスクをインスタンスにアタッチ
aws lightsail attach-disk \
    --disk-name "multiagent-ahp-data" \
    --instance-name "multiagent-ahp" \
    --disk-path "/dev/xvdf" \
    --region ap-northeast-1
```

## A-5) ファイアウォール設定

```bash
# HTTPとHTTPSポートを開放
aws lightsail put-instance-public-ports \
    --instance-name "multiagent-ahp" \
    --region ap-northeast-1 \
    --port-infos '[
        {
            "fromPort": 22,
            "toPort": 22,
            "protocol": "tcp",
            "accessFrom": {
                "protocol": "tcp",
                "accessType": "public"
            }
        },
        {
            "fromPort": 80,
            "toPort": 80,
            "protocol": "tcp",
            "accessFrom": {
                "protocol": "tcp", 
                "accessType": "public"
            }
        },
        {
            "fromPort": 443,
            "toPort": 443,
            "protocol": "tcp",
            "accessFrom": {
                "protocol": "tcp",
                "accessType": "public"
            }
        },
        {
            "fromPort": 5002,
            "toPort": 5002,
            "protocol": "tcp",
            "accessFrom": {
                "protocol": "tcp",
                "accessType": "public"
            }
        }
    ]'
```

## A-6) SSH接続・初期セットアップ

```bash
# SSH接続
ssh -i ~/.ssh/lightsail-apne1.pem ubuntu@$STATIC_IP

# === 以下、サーバー内で実行 ===

# 追加ディスクの確認・セットアップ
lsblk
sudo mkfs.ext4 -F /dev/nvme1n1
sudo mkdir -p /mnt/data
sudo mount /dev/nvme1n1 /mnt/data
sudo chown ubuntu:ubuntu /mnt/data

# 自動マウント設定
UUID=$(sudo blkid -s UUID -o value /dev/nvme1n1)
echo "UUID=$UUID /mnt/data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab

# プロジェクトクローン（GitHub等から）
git clone https://github.com/[YOUR-REPO]/multiagentahp.git
cd multiagentahp

# 環境変数設定
cat > .env << EOF
OPENAI_API_KEY=your_api_key_here
OPENAI_API_KEY2=your_api_key2_here  
OPENAI_RESPONSES_MODEL=gpt-4.1
DEBUG_LLM_CONTEXT=0
FLASK_ENV=production
FLASK_DEBUG=0
EOF

# 本番デプロイ
docker compose -f docker-compose.prod.yml up -d --build
```

---

# B) 既存インスタンス向け手順

既にLightsail/EC2インスタンスが存在する場合の手順です。

## B-0) 前提
- OS: Ubuntu 22.04/24.04（Lightsail/EC2）
- ファイアウォール/セキュリティグループで必要ポートを開放済み
- このリポジトリがサーバに配置済み（`git clone` など）

## B-1) 追加ディスクの準備（任意）
1. ディスクを確認:
   - `lsblk -f` または `lsblk`
   - 通常、`nvme0n1`がルートディスク、`nvme1n1`が追加ディスク
   - 新しいボリュームが `nvme1n1` 等で表示され、FSTYPE が空なら未フォーマット
   - **注意**: `/dev/xvdf`は存在しません。必ず`/dev/nvme1n1`を使用してください
2. フォーマット（既存データが無いことを確認の上）:
   - `sudo mkfs.ext4 -F /dev/nvme1n1`
3. マウント:
   - `sudo mkdir -p /mnt/data`
   - `sudo mount /dev/nvme1n1 /mnt/data`
   - `sudo chown ubuntu:ubuntu /mnt/data`
4. 自動マウント（再起動後の維持）:
   - `UUID=$(sudo blkid -s UUID -o value /dev/nvme1n1)`
   - `echo "UUID=$UUID /mnt/data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab`
   - `sudo mount -a`

> 注意: ルートディスク（`/dev/nvme0n1`）には触れないでください。

## B-2) Docker/Compose のセットアップ
```
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
docker --version
docker compose version
```

## B-3) 環境変数の設定
このリポジトリのルートに `.env` を作成して、以下を設定します（Compose が読み込みます）。

```
OPENAI_API_KEY=sk-...
OPENAI_API_KEY2=sk-...             # 任意（フェイルオーバー）
OPENAI_RESPONSES_MODEL=gpt-4.1     # 任意（例: gpt-4o-mini）
DEBUG_LLM_CONTEXT=0                # 負荷テスト時は 0 推奨
```

## B-4) コンテナのビルドと起動
このリポジトリ直下で:

```bash
# 本番用構成でデプロイ（推奨）
docker compose -f docker-compose.prod.yml down --remove-orphans
docker compose -f docker-compose.prod.yml build --no-cache
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml logs -f

# または開発用（デバッグモード有効）
# docker compose down --remove-orphans
# docker compose build --no-cache
# docker compose up -d
```

本番用では Gunicorn を使用し、ポート80（HTTP）でアクセス可能です。開発用は Flask開発サーバーでポート5002です。

## B-5) 動作確認
**本番用構成の場合:**
- サーバ自身から: `curl http://localhost/` または `curl http://localhost:80/`
- 外部 PC から: ブラウザで `http://<LightsailのPublicDNSまたはIP>/`
- ファイアウォールでポート80を開放する必要があります

**開発用構成の場合:**
- サーバ自身から: `curl http://localhost:5002/`
- 外部 PC から: ブラウザで `http://<LightsailのPublicDNSまたはIP>:5002/`
- ファイアウォールでポート5002を開放する必要があります

## B-6) 簡易テスト（ホストPCからリモートに対して）
```
# LLMなし（疎通）
python3 tests/scenario_test_ai_chat.py --skip-llm --host <EC2-IP> --port 5002

# LLM込み（レート制限に注意）
python3 tests/load_test_ai_chat.py --host <EC2-IP> --port 5002 \
  --concurrency 50 --iterations 50 --turns 1 --jitter-ms 300
```

## B-7) よくあるハマり
- ブラウザで見えない: アプリが `127.0.0.1` で待受していないか（本リポジトリは `run.py` を `0.0.0.0` に修正済み）。SG/Firewall で 5002 開放を確認。
- healthcheck 失敗: コンテナ内で curl 不在だと失敗します（Dockerfile に curl を追加済み）。
- 429（レート制限）: `--jitter-ms` を付与、`--concurrency` を調整、`OPENAI_RESPONSES_MODEL` を変更、DEBUG を 0 に。

---

## 📝 費用見積もり（AWS Lightsail）

**A案（フルセットアップ）推奨構成:**
- Lightsail $5/月プラン (1GB RAM, 1 vCPU, 40GB SSD)
- 追加ディスク 20GB: $2/月
- 静的IP: 無料（アタッチ済みの場合）
- **合計: 月額 $7**

## 🔧 メンテナンス・運用

```bash
# ログ確認
docker compose -f docker-compose.prod.yml logs -f

# 再起動
docker compose -f docker-compose.prod.yml restart

# システム更新
sudo apt update && sudo apt upgrade -y

# データバックアップ
sudo tar czf backup-$(date +%Y%m%d).tar.gz /mnt/data
```

---

**手順の使い分け:**
- **A案**: 最初からLightsailで構築する場合（推奨・最安）
- **B案**: 既存のLightsail/EC2インスタンスを活用する場合

この手順は、現在の `Dockerfile.prod`（Gunicorn）と `docker-compose.prod.yml`に最適化されています。


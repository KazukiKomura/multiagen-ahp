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

# インスタンス作成（2GB RAM以上推奨）
aws lightsail create-instances \
    --instance-names "multiagent-ahp" \
    --availability-zone "ap-northeast-1a" \
    --blueprint-id "ubuntu_22_04" \
    --bundle-id "large_2_0" \
    --region ap-northeast-1 \
    --user-data '#!/bin/bash
apt-get update
apt-get install -y docker.io docker-compose-v2 git curl jq
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu'
```

## A-3) 静的IP割り当て

```bash
# 静的IP作成
aws lightsail allocate-static-ip \
    --static-ip-name "multiagent-ahp-ip" \
    --region ap-northeast-1

# インスタンス起動完了まで待機（重要）
echo "インスタンスの起動完了を待機中..."
while [ "$(aws lightsail get-instance --instance-name "multiagent-ahp" --region ap-northeast-1 --query 'instance.state.name' --output text)" != "running" ]; do
    echo "待機中..."
    sleep 10
done
echo "インスタンスが起動しました"

# 静的IPをインスタンスにアタッチ
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

# 静的IP環境変数を設定（後の手順で使用）
export STATIC_IP
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
            "cidrs": ["0.0.0.0/0"]
        },
        {
            "fromPort": 80,
            "toPort": 80,
            "protocol": "tcp",
            "cidrs": ["0.0.0.0/0"]
        },
        {
            "fromPort": 443,
            "toPort": 443,
            "protocol": "tcp",
            "cidrs": ["0.0.0.0/0"]
        },
        {
            "fromPort": 5002,
            "toPort": 5002,
            "protocol": "tcp",
            "cidrs": ["0.0.0.0/0"]
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

# 追加ディスクのデバイス名を確認（xvdf または nvme1n1）
# 出力結果に応じて適切なデバイス名を使用
if [ -b /dev/xvdf ]; then
    DISK_DEVICE="/dev/xvdf"
elif [ -b /dev/nvme1n1 ]; then
    DISK_DEVICE="/dev/nvme1n1"
else
    echo "追加ディスクが見つかりません"
    exit 1
fi

echo "使用するディスクデバイス: $DISK_DEVICE"

sudo mkfs.ext4 -F $DISK_DEVICE
sudo mkdir -p /mnt/data
sudo mount $DISK_DEVICE /mnt/data
sudo chown ubuntu:ubuntu /mnt/data

# 自動マウント設定
UUID=$(sudo blkid -s UUID -o value $DISK_DEVICE)
echo "UUID=$UUID /mnt/data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab

# プロジェクトファイル転送
# 方法1: ローカルからrsyncで転送（推奨・確実）
# ローカルマシン側で実行（SSHセッションを一度exit）:
exit

# ローカルマシンで実行:
cd /path/to/your/multiagentahp
rsync -avz --progress -e "ssh -i ~/.ssh/lightsail-apne1.pem" \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='data/sessions.db' \
  . ubuntu@$STATIC_IP:~/multiagentahp/

# 再度サーバーに接続
ssh -i ~/.ssh/lightsail-apne1.pem ubuntu@$STATIC_IP

# 方法2: HTTPSクローン（代替手段）
# git clone https://github.com/KazukiKomura/multiagentahp.git

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
- SSH接続が可能
- このリポジトリをサーバに配置する必要あり（下記手順参照）

## B-1) 追加ディスクの準備（任意）
1. ディスクを確認:
   - `lsblk -f` または `lsblk`
   - 追加ディスクは環境により`xvdf`または`nvme1n1`として認識される
   - 新しいボリュームがFSTYPE空で表示されれば未フォーマット
2. デバイス名の確認とフォーマット（既存データが無いことを確認の上）:
   ```bash
   # デバイス名を確認
   if [ -b /dev/xvdf ]; then
       DISK_DEVICE="/dev/xvdf"
   elif [ -b /dev/nvme1n1 ]; then
       DISK_DEVICE="/dev/nvme1n1"
   else
       echo "追加ディスクが見つかりません"
       exit 1
   fi
   echo "使用するディスクデバイス: $DISK_DEVICE"

   # フォーマット
   sudo mkfs.ext4 -F $DISK_DEVICE
   ```
3. マウント:
   - `sudo mkdir -p /mnt/data`
   - `sudo mount $DISK_DEVICE /mnt/data`
   - `sudo chown ubuntu:ubuntu /mnt/data`
4. 自動マウント（再起動後の維持）:
   - `UUID=$(sudo blkid -s UUID -o value $DISK_DEVICE)`
   - `echo "UUID=$UUID /mnt/data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab`
   - `sudo mount -a`

> 注意: ルートディスク（通常`xvda`または`nvme0n1`）には触れないでください。

## B-2) プロジェクトファイル転送

### 方法1: ローカルからrsync転送（推奨）

**ローカルマシン側で実行:**

```bash
# パブリックIPアドレスを確認・設定
export PUBLIC_IP="[あなたのLightsailパブリックIP]"

# プロジェクトディレクトリに移動
cd /path/to/your/multiagentahp

# rsyncで効率的に転送
rsync -avz --progress -e "ssh -i ~/.ssh/lightsail-apne1.pem" \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='data/sessions.db' \
  . ubuntu@35.77.244.101:~/multiagentahp/
```

### 方法2: SCP転送

```bash
# プロジェクト全体を転送
scp -r -i ~/.ssh/lightsail-apne1.pem . ubuntu@$PUBLIC_IP:~/multiagentahp/
```

### 方法3: GitHubからクローン

**サーバー側で実行（SSH接続後）:**

```bash
# HTTPSクローン（公開リポジトリ）
git clone https://github.com/KazukiKomura/multiagentahp.git

# またはSSHクローン（SSH鍵設定済みの場合）
# git clone git@github.com:KazukiKomura/multiagentahp.git

cd multiagentahp
```

**⚠️ 注意:** プライベートIPアドレス（172.x.x.x）ではなく、必ずパブリックIPアドレスを使用してください。

## B-3) Docker/Compose のセットアップ
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

## B-4) 環境変数の設定
このリポジトリのルートに `.env` を作成して、以下を設定します（Compose が読み込みます）。

```
OPENAI_API_KEY=sk-...
OPENAI_API_KEY2=sk-...             # 任意（フェイルオーバー）
OPENAI_RESPONSES_MODEL=gpt-4.1     # 任意（例: gpt-4o-mini）
DEBUG_LLM_CONTEXT=0                # 負荷テスト時は 0 推奨
```

## B-5) コンテナのビルドと起動
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

## B-6) 動作確認
**本番用構成の場合:**
- サーバ自身から: `curl http://localhost/` または `curl http://localhost:80/`
- 外部 PC から: ブラウザで `http://<LightsailのPublicDNSまたはIP>/`
- ファイアウォールでポート80を開放する必要があります

**開発用構成の場合:**
- サーバ自身から: `curl http://localhost:5002/`
- 外部 PC から: ブラウザで `http://<LightsailのPublicDNSまたはIP>:5002/`
- ファイアウォールでポート5002を開放する必要があります

## B-7) 簡易テスト（ホストPCからリモートに対して）
```
# LLMなし（疎通）
python3 tests/scenario_test_ai_chat.py --skip-llm --host <EC2-IP> --port 5002

# LLM込み（レート制限に注意）
python3 tests/load_test_ai_chat.py --host <EC2-IP> --port 5002 \
  --concurrency 50 --iterations 50 --turns 1 --jitter-ms 300
```

## B-8) よくあるハマり

### ファイル転送関連
- **SCP/rsync接続失敗**: プライベートIP（172.x.x.x）ではなく、パブリックIPを使用してください
- **SSH接続タイムアウト**: 静的IPが割り当てられていない場合があります。AWS CLIで静的IP確認・割り当てを行ってください
- **Permission denied**: SSH鍵のパスとファイル権限（400）を確認してください

### アプリケーション関連
- **ブラウザで見えない**: アプリが `127.0.0.1` で待受していないか（本リポジトリは `run.py` を `0.0.0.0` に修正済み）。ファイアウォールで必要ポート開放を確認
- **healthcheck 失敗**: コンテナ内で curl 不在だと失敗します（Dockerfile に curl を追加済み）
- **429（レート制限）**: `--jitter-ms` を付与、`--concurrency` を調整、`OPENAI_RESPONSES_MODEL` を変更、DEBUG を 0 に

---

## 📝 費用見積もり（AWS Lightsail）

**A案（フルセットアップ）推奨構成:**
- Lightsail $10/月プラン (2GB RAM, 1 vCPU, 60GB SSD)
- 追加ディスク 20GB: $2/月
- 静的IP: 無料（アタッチ済みの場合）
- **合計: 月額 $12**

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

# ローカルからファイル更新（ローカルマシンで実行）
rsync -avz --progress -e "ssh -i ~/.ssh/lightsail-apne1.pem" \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='data/' \
  . ubuntu@[パブリックIP]:~/multiagentahp/

# アプリケーション再デプロイ（サーバー側）
cd ~/multiagentahp
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d --build
```

---

**手順の使い分け:**
- **A案**: 最初からLightsailで構築する場合（推奨・最安）
- **B案**: 既存のLightsail/EC2インスタンスを活用する場合

この手順は、現在の `Dockerfile.prod`（Gunicorn）と `docker-compose.prod.yml`に最適化されています。

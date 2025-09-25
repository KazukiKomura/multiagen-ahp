FROM python:3.11-slim

# 作業ディレクトリの設定
WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をコピー
COPY requirements.txt .

# Pythonパッケージのインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# データディレクトリの作成
RUN mkdir -p data

# データベースファイルの権限設定
RUN chmod 777 data

# run.py は 5002 で起動
EXPOSE 5002

# アプリケーション起動（Gunicorn, 並列対応・最小変更）
# 4 workers, 8 threads each; tune per CPU/IO profile
CMD ["gunicorn", "-w", "4", "-k", "gthread", "--threads", "8", "--timeout", "120", "-b", "0.0.0.0:5002", "src.app:app"]

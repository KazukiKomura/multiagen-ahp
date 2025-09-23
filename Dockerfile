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

# アプリケーション起動
CMD ["python", "run.py"]

FROM python:3.11-slim

# 作業ディレクトリの設定
WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
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

# ポート5000を開放
EXPOSE 5000

# アプリケーション起動
CMD ["python", "app.py"]
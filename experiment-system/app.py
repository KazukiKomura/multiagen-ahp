from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    """Main page - redirect to simple AHP setup"""
    return redirect(url_for('simple_ahp'))

@app.route('/simple_ahp')
def simple_ahp():
    """Simple AHP comparison page"""
    return render_template('simple_ahp.html')

@app.route('/simple_game')
def simple_game():
    """Simple game interface"""
    return render_template('simple_game.html')

@app.route('/turn_game')
def turn_game():
    """Turn-based game interface"""
    return render_template('turn_based_game.html')

@app.route('/p2p_game')
def p2p_game():
    """P2P game interface with observation states"""
    return render_template('p2p_game.html')

@app.route('/results')
def results():
    """Results page (placeholder)"""
    return """
    <html>
    <head><title>結果</title></head>
    <body style="font-family: Arial; text-align: center; padding: 50px;">
        <h1>ゲーム終了</h1>
        <p>お疲れさまでした！</p>
        <p>結果の詳細はここに表示されます。</p>
        <button onclick="window.location.href='/'" style="padding: 10px 20px; font-size: 16px;">
            最初からやり直す
        </button>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("Simple University Selection Experiment")
    print("Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
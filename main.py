from flask import Flask, render_template
from mainweb import find_trades

app = Flask(__name__)
ft = find_trades()


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/tradesf/')
def trades():
    return render_template('main.html', ft=ft[0])


@app.route('/tradesff/')
def tradesff():
    return render_template('main.html', ft=ft[1])


@app.route('/tradesfb/')
def tradesfb():
    return render_template('main.html', ft=ft[2])


if __name__ == '__main__':
    app.run(debug=True, port=5501)

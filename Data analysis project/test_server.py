from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/result')
def result():
    output = request.form.to_dict()
    name = output["name"]
    return render_template("index.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)



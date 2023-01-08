from flask import Flask,jsonify,request
app = Flask(__name__)

@app.route("/api/perlib/<path>")
def perlib(path):
    return "Hello World"

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask
ECHO is on.
app = Flask(__name__)
ECHO is on.
@app.route("/")
def home():
    return "Flask test OK!"
ECHO is on.
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

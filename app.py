"""RPS Stratego - A tactical board game server combining Stratego hidden information with RPS combat."""

from flask import Flask

from routes import api_bp

app = Flask(__name__, static_folder="static")
app.register_blueprint(api_bp)

if __name__ == "__main__":
    print("Starting RPS Stratego server on http://localhost:5000")
    app.run(debug=True, port=5000)

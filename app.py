from flask import Flask, render_template
import livevideo_emodetection

app = Flask(__name__)


@app.route('/home')
def home():
    return "Welcome to detection space"


@app.route('detect')
def detect():
    return "Detection begins"


#app.add_url_rule('/',  livevideo_emodetection)

# def space1():
#   return "Detected the emotion of the person"


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, jsonify, request
from tag_predictor import getTags

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        question = request.json['question']
        tags = getTags([question])[0]
        return jsonify(tags)
    except KeyError:
        error_message = "Question not found in the request."
        return jsonify(error=error_message), 400

if __name__ == "__main__":
    app.run(debug=True)




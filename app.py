from flask import Flask, render_template, request, session
from ml_backend import suggestion, search_stack_overflow, update_visit_count

app = Flask(__name__)
app.secret_key = 'iXhqJcm23*ojb5bwBLT9QA(('

@app.route('/')
def index():
    update_visit_count(session)
    return render_template('index.html', suggested_tags=None, question_links=None,visit_count=session.get('visit_count', 0))

@app.route('/get_tags', methods=['POST'])
def get_tags():
    question = request.form['question']
    suggested_tags = suggestion(question)
    update_visit_count(session)
    return render_template('index.html', suggested_tags=suggested_tags, question_links=None, visit_count=session.get('visit_count', 0))

@app.route('/suggest_questions', methods=['POST'])
def suggest_questions():
    question = request.form['question']
    suggested_tags = suggestion(question)
    question_links = search_stack_overflow(suggested_tags)
    update_visit_count(session)
    return render_template('index.html', suggested_tags=suggested_tags, question_links=question_links,visit_count=session.get('visit_count', 0))

if __name__ == '__main__':
    app.run(debug=True)

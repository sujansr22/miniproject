from flask import Blueprint, request, jsonify, session, render_template, redirect, url_for, current_app
from extensions import mysql

questions_bp = Blueprint('questions', __name__)

def login_required_farmer(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.is_json:
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def login_required_consultant(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'consultant_id' not in session:
            if request.is_json:
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect(url_for('consultant.login'))
        return f(*args, **kwargs)
    return decorated_function

@questions_bp.route('/post', methods=['GET', 'POST'])
def post_question():
    if request.method == 'GET':
        return render_template('expert_mode.html') # Redirect or render unified view
    
    try:
        data = request.get_json()
        if not data.get('title') or not data.get('description'):
            return jsonify({'error': 'Title and description are required'}), 400
        
        # Use session user_id if logged in, otherwise None (NULL in DB)
        farmer_id = session.get('user_id')
        
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO questions (farmer_id, title, description) VALUES (%s, %s, %s)",
            (farmer_id, data['title'], data['description'])
        )
        mysql.connection.commit()
        cur.close()
        
        return jsonify({'message': 'Question posted successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@questions_bp.route('/<int:id>/answer', methods=['POST'])
def answer_question(id):
    try:
        data = request.get_json()
        if not data.get('answer_text'):
            return jsonify({'error': 'Answer text is required'}), 400
        
        # Use session consultant_id if logged in, otherwise None (NULL in DB)
        consultant_id = session.get('consultant_id')
        
        cur = mysql.connection.cursor()
        
        # Insert answer (Allowing NULL consultant_id for community answers)
        cur.execute(
            "INSERT INTO answers (question_id, consultant_id, answer_text) VALUES (%s, %s, %s)",
            (id, consultant_id, data['answer_text'])
        )
        
        # Update question status
        cur.execute("UPDATE questions SET status = 'answered' WHERE id = %s", (id,))
        
        mysql.connection.commit()
        cur.close()
        
        return jsonify({'message': 'Answer submitted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@questions_bp.route('/my')
@login_required_farmer
def my_questions():
    try:
        cur = mysql.connection.cursor()
        
        # Get questions and their answers if any
        cur.execute("""
            SELECT q.id, q.title, q.description, q.status, q.created_at, a.answer_text, a.created_at as answered_at, c.name as consultant_name
            FROM questions q
            LEFT JOIN answers a ON q.id = a.question_id
            LEFT JOIN consultants c ON a.consultant_id = c.id
            WHERE q.farmer_id = %s
            ORDER BY q.created_at DESC
        """, (session['user_id'],))
        
        questions = cur.fetchall()
        cur.close()
        
        return render_template('my_questions.html', questions=questions)
    except Exception as e:
        return str(e), 500
@questions_bp.route('/mode')
def expert_mode():
    try:
        cur = mysql.connection.cursor()
        
        # Get all questions and their answers for the community feed
        cur.execute("""
            SELECT q.id, q.title, q.description, q.status, q.created_at, q.farmer_id,
                   IFNULL(u.username, 'Guest') as farmer_name,
                   a.answer_text, a.created_at as answered_at, 
                   IFNULL(c.name, 'Community Member') as consultant_name, 
                   IFNULL(c.expertise, 'General') as consultant_expertise
            FROM questions q
            LEFT JOIN users u ON q.farmer_id = u.id
            LEFT JOIN answers a ON q.id = a.question_id
            LEFT JOIN consultants c ON a.consultant_id = c.id
            ORDER BY q.created_at DESC
        """)
        all_questions = cur.fetchall()
        cur.close()
        
        return render_template('expert_mode.html', questions=all_questions)
    except Exception as e:
        return str(e), 500

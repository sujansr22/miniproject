from flask import Blueprint, request, jsonify, session, render_template, redirect, url_for
from extensions import mysql
import bcrypt
from flask_mysqldb import MySQL

# Note: We need to access mysql from the main app, so we'll pass it in or use current_app
from flask import current_app

consultant_bp = Blueprint('consultant', __name__)

@consultant_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('consultant_register.html')
    
    try:
        data = request.get_json()
        if not all(k in data for k in ('name', 'email', 'password', 'expertise')):
            return jsonify({'error': 'All fields are required'}), 400
        
        name = data['name'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        expertise = data['expertise'].strip()
        
        cur = mysql.connection.cursor()
        
        # Check if email exists
        cur.execute("SELECT id FROM consultants WHERE email = %s", (email,))
        if cur.fetchone():
            cur.close()
            return jsonify({'error': 'Email already registered'}), 400
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Insert consultant
        cur.execute(
            "INSERT INTO consultants (name, email, password_hash, expertise) VALUES (%s, %s, %s, %s)",
            (name, email, password_hash, expertise)
        )
        mysql.connection.commit()
        cur.close()
        
        return jsonify({'message': 'Consultant registered successfully'}), 201
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@consultant_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('consultant_login.html')
    
    try:
        data = request.get_json()
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400
        
        email = data['email'].strip().lower()
        password = data['password']
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, name, email, password_hash FROM consultants WHERE email = %s", (email,))
        consultant = cur.fetchone()
        cur.close()
        
        if not consultant:
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Verify password
        if bcrypt.checkpw(password.encode('utf-8'), consultant['password_hash'].encode('utf-8')):
            session['consultant_id'] = consultant['id']
            session['consultant_name'] = consultant['name']
            session['consultant_email'] = consultant['email']
            return jsonify({'message': 'Login successful', 'redirect': url_for('questions.expert_mode')}), 200
        else:
            return jsonify({'error': 'Invalid email or password'}), 401
            
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@consultant_bp.route('/logout')
def logout():
    session.pop('consultant_id', None)
    session.pop('consultant_name', None)
    session.pop('consultant_email', None)
    return redirect(url_for('consultant.login'))

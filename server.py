#!/usr/bin/env python3
"""
RituCare Backend Server
A Flask-based backend for the RituCare women's health application.
Provides API endpoints for data persistence, user management, and health calculations.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime, timedelta
import uuid
import sqlite3
from contextlib import contextmanager

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for all routes

# Database setup
DATABASE = 'ritucare.db'

@contextmanager
def get_db():
    """Database connection context manager"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize database tables"""
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT,
                age INTEGER,
                height_cm REAL,
                weight_kg REAL,
                dietary_preferences TEXT, -- JSON string
                notification_prefs TEXT, -- JSON string
                cycle_settings TEXT, -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE,
                flow TEXT CHECK(flow IN ('light', 'medium', 'heavy')),
                symptoms TEXT, -- JSON string
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS pcos_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                assessment_data TEXT NOT NULL, -- JSON string
                score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommendations TEXT, -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS nutrition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date DATE NOT NULL,
                meal_type TEXT NOT NULL,
                food_items TEXT NOT NULL, -- JSON string
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        conn.commit()

# Initialize database on startup
init_db()

# Routes for serving static files
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/pages/<path:filename>')
def serve_pages(filename):
    return send_from_directory('pages', filename)

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('js', filename)

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

# API Routes

# User Authentication
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')  # In production, hash this!

        if not all([username, email, password]):
            return jsonify({'error': 'Missing required fields'}), 400

        with get_db() as conn:
            # Check if user exists
            existing = conn.execute(
                'SELECT id FROM users WHERE username = ? OR email = ?',
                (username, email)
            ).fetchone()

            if existing:
                return jsonify({'error': 'User already exists'}), 409

            # Create user
            cursor = conn.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password)
            )
            user_id = cursor.lastrowid
            conn.commit()

            return jsonify({
                'message': 'User registered successfully',
                'user_id': user_id
            }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        if not all([username, password]):
            return jsonify({'error': 'Missing credentials'}), 400

        with get_db() as conn:
            user = conn.execute(
                'SELECT id, username, email FROM users WHERE username = ? AND password_hash = ?',
                (username, password)
            ).fetchone()

            if not user:
                return jsonify({'error': 'Invalid credentials'}), 401

            # Update last login
            conn.execute(
                'UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?',
                (user['id'],)
            )
            conn.commit()

            return jsonify({
                'message': 'Login successful',
                'user': dict(user)
            }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Profile Management
@app.route('/api/profile/<int:user_id>', methods=['GET'])
def get_profile(user_id):
    try:
        with get_db() as conn:
            profile = conn.execute(
                'SELECT * FROM profiles WHERE user_id = ?',
                (user_id,)
            ).fetchone()

            if not profile:
                return jsonify({'error': 'Profile not found'}), 404

            return jsonify(dict(profile)), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile/<int:user_id>', methods=['POST'])
def update_profile(user_id):
    try:
        data = request.get_json()

        with get_db() as conn:
            # Check if profile exists
            existing = conn.execute(
                'SELECT id FROM profiles WHERE user_id = ?',
                (user_id,)
            ).fetchone()

            profile_data = {
                'name': data.get('name'),
                'age': data.get('age'),
                'height_cm': data.get('height_cm'),
                'weight_kg': data.get('weight_kg'),
                'dietary_preferences': json.dumps(data.get('dietary_preferences', {})),
                'notification_prefs': json.dumps(data.get('notification_prefs', {})),
                'cycle_settings': json.dumps(data.get('cycle_settings', {}))
            }

            if existing:
                # Update existing profile
                conn.execute('''
                    UPDATE profiles SET
                        name = ?, age = ?, height_cm = ?, weight_kg = ?,
                        dietary_preferences = ?, notification_prefs = ?, cycle_settings = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (
                    profile_data['name'], profile_data['age'], profile_data['height_cm'],
                    profile_data['weight_kg'], profile_data['dietary_preferences'],
                    profile_data['notification_prefs'], profile_data['cycle_settings'],
                    user_id
                ))
            else:
                # Create new profile
                conn.execute('''
                    INSERT INTO profiles (user_id, name, age, height_cm, weight_kg,
                                        dietary_preferences, notification_prefs, cycle_settings)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, profile_data['name'], profile_data['age'], profile_data['height_cm'],
                    profile_data['weight_kg'], profile_data['dietary_preferences'],
                    profile_data['notification_prefs'], profile_data['cycle_settings']
                ))

            conn.commit()
            return jsonify({'message': 'Profile updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Cycle Management
@app.route('/api/cycles/<int:user_id>', methods=['GET'])
def get_cycles(user_id):
    try:
        with get_db() as conn:
            cycles = conn.execute(
                'SELECT * FROM cycles WHERE user_id = ? ORDER BY start_date DESC',
                (user_id,)
            ).fetchall()

            return jsonify([dict(cycle) for cycle in cycles]), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cycles/<int:user_id>', methods=['POST'])
def add_cycle(user_id):
    try:
        data = request.get_json()

        cycle_data = {
            'user_id': user_id,
            'start_date': data.get('start_date'),
            'end_date': data.get('end_date'),
            'flow': data.get('flow'),
            'symptoms': json.dumps(data.get('symptoms', [])),
            'notes': data.get('notes', '')
        }

        if not cycle_data['start_date']:
            return jsonify({'error': 'Start date is required'}), 400

        with get_db() as conn:
            cursor = conn.execute('''
                INSERT INTO cycles (user_id, start_date, end_date, flow, symptoms, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                cycle_data['user_id'], cycle_data['start_date'], cycle_data['end_date'],
                cycle_data['flow'], cycle_data['symptoms'], cycle_data['notes']
            ))

            cycle_id = cursor.lastrowid
            conn.commit()

            return jsonify({
                'message': 'Cycle added successfully',
                'cycle_id': cycle_id
            }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cycles/<int:user_id>/<int:cycle_id>', methods=['PUT'])
def update_cycle(user_id, cycle_id):
    try:
        data = request.get_json()

        with get_db() as conn:
            # Check if cycle belongs to user
            cycle = conn.execute(
                'SELECT id FROM cycles WHERE id = ? AND user_id = ?',
                (cycle_id, user_id)
            ).fetchone()

            if not cycle:
                return jsonify({'error': 'Cycle not found'}), 404

            # Update cycle
            conn.execute('''
                UPDATE cycles SET
                    start_date = ?, end_date = ?, flow = ?, symptoms = ?, notes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
            ''', (
                data.get('start_date'), data.get('end_date'), data.get('flow'),
                json.dumps(data.get('symptoms', [])), data.get('notes', ''),
                cycle_id, user_id
            ))

            conn.commit()
            return jsonify({'message': 'Cycle updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cycles/<int:user_id>/<int:cycle_id>', methods=['DELETE'])
def delete_cycle(user_id, cycle_id):
    try:
        with get_db() as conn:
            # Check if cycle belongs to user
            cycle = conn.execute(
                'SELECT id FROM cycles WHERE id = ? AND user_id = ?',
                (cycle_id, user_id)
            ).fetchone()

            if not cycle:
                return jsonify({'error': 'Cycle not found'}), 404

            # Delete cycle
            conn.execute(
                'DELETE FROM cycles WHERE id = ? AND user_id = ?',
                (cycle_id, user_id)
            )

            conn.commit()
            return jsonify({'message': 'Cycle deleted successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# PCOS Assessment
@app.route('/api/pcos/<int:user_id>', methods=['GET'])
def get_pcos_assessments(user_id):
    try:
        with get_db() as conn:
            assessments = conn.execute(
                'SELECT * FROM pcos_assessments WHERE user_id = ? ORDER BY created_at DESC',
                (user_id,)
            ).fetchall()

            return jsonify([dict(assessment) for assessment in assessments]), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pcos/<int:user_id>', methods=['POST'])
def add_pcos_assessment(user_id):
    try:
        data = request.get_json()

        assessment_data = {
            'user_id': user_id,
            'assessment_data': json.dumps(data.get('assessment_data', {})),
            'score': data.get('score', 0),
            'risk_level': data.get('risk_level', 'Unknown'),
            'recommendations': json.dumps(data.get('recommendations', []))
        }

        with get_db() as conn:
            cursor = conn.execute('''
                INSERT INTO pcos_assessments (user_id, assessment_data, score, risk_level, recommendations)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                assessment_data['user_id'], assessment_data['assessment_data'],
                assessment_data['score'], assessment_data['risk_level'],
                assessment_data['recommendations']
            ))

            assessment_id = cursor.lastrowid
            conn.commit()

            return jsonify({
                'message': 'Assessment added successfully',
                'assessment_id': assessment_id
            }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Nutrition Logging
@app.route('/api/nutrition/<int:user_id>', methods=['GET'])
def get_nutrition_logs(user_id):
    try:
        date = request.args.get('date')
        query = 'SELECT * FROM nutrition_logs WHERE user_id = ?'
        params = [user_id]

        if date:
            query += ' AND date = ?'
            params.append(date)

        query += ' ORDER BY date DESC, created_at DESC'

        with get_db() as conn:
            logs = conn.execute(query, params).fetchall()
            return jsonify([dict(log) for log in logs]), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nutrition/<int:user_id>', methods=['POST'])
def add_nutrition_log(user_id):
    try:
        data = request.get_json()

        log_data = {
            'user_id': user_id,
            'date': data.get('date'),
            'meal_type': data.get('meal_type'),
            'food_items': json.dumps(data.get('food_items', [])),
            'notes': data.get('notes', '')
        }

        if not all([log_data['date'], log_data['meal_type']]):
            return jsonify({'error': 'Date and meal type are required'}), 400

        with get_db() as conn:
            cursor = conn.execute('''
                INSERT INTO nutrition_logs (user_id, date, meal_type, food_items, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                log_data['user_id'], log_data['date'], log_data['meal_type'],
                log_data['food_items'], log_data['notes']
            ))

            log_id = cursor.lastrowid
            conn.commit()

            return jsonify({
                'message': 'Nutrition log added successfully',
                'log_id': log_id
            }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health Calculations API
@app.route('/api/calculate/bmi', methods=['POST'])
def calculate_bmi():
    try:
        data = request.get_json()
        height_cm = data.get('height_cm')
        weight_kg = data.get('weight_kg')

        if not all([height_cm, weight_kg]):
            return jsonify({'error': 'Height and weight are required'}), 400

        height_m = height_cm / 100
        bmi = round(weight_kg / (height_m ** 2), 1)

        # BMI categories
        if bmi < 18.5:
            category = 'Underweight'
        elif bmi < 25:
            category = 'Normal'
        elif bmi < 30:
            category = 'Overweight'
        else:
            category = 'Obese'

        return jsonify({
            'bmi': bmi,
            'category': category
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate/cycle-predictions', methods=['POST'])
def calculate_cycle_predictions():
    try:
        data = request.get_json()
        cycles = data.get('cycles', [])

        if len(cycles) < 2:
            return jsonify({'error': 'At least 2 cycles needed for predictions'}), 400

        # Calculate average cycle length
        lengths = []
        for i in range(len(cycles) - 1):
            start1 = datetime.strptime(cycles[i]['start_date'], '%Y-%m-%d')
            start2 = datetime.strptime(cycles[i + 1]['start_date'], '%Y-%m-%d')
            length = (start2 - start1).days
            lengths.append(length)

        avg_length = sum(lengths) / len(lengths)

        # Predict next period
        last_start = datetime.strptime(cycles[-1]['start_date'], '%Y-%m-%d')
        next_period = last_start + timedelta(days=round(avg_length))

        # Predict ovulation (14 days before next period)
        ovulation = next_period - timedelta(days=14)

        return jsonify({
            'average_cycle_length': round(avg_length),
            'next_period_date': next_period.strftime('%Y-%m-%d'),
            'ovulation_date': ovulation.strftime('%Y-%m-%d'),
            'fertile_window_start': (ovulation - timedelta(days=2)).strftime('%Y-%m-%d'),
            'fertile_window_end': (ovulation + timedelta(days=2)).strftime('%Y-%m-%d')
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Data Export/Import
@app.route('/api/export/<int:user_id>', methods=['GET'])
def export_user_data(user_id):
    try:
        with get_db() as conn:
            # Get all user data
            profile = conn.execute(
                'SELECT * FROM profiles WHERE user_id = ?',
                (user_id,)
            ).fetchone()

            cycles = conn.execute(
                'SELECT * FROM cycles WHERE user_id = ?',
                (user_id,)
            ).fetchall()

            assessments = conn.execute(
                'SELECT * FROM pcos_assessments WHERE user_id = ?',
                (user_id,)
            ).fetchall()

            nutrition_logs = conn.execute(
                'SELECT * FROM nutrition_logs WHERE user_id = ?',
                (user_id,)
            ).fetchall()

            export_data = {
                'profile': dict(profile) if profile else None,
                'cycles': [dict(cycle) for cycle in cycles],
                'assessments': [dict(assessment) for assessment in assessments],
                'nutrition_logs': [dict(log) for log in nutrition_logs],
                'export_date': datetime.now().isoformat()
            }

            return jsonify(export_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/import/<int:user_id>', methods=['POST'])
def import_user_data(user_id):
    try:
        data = request.get_json()

        with get_db() as conn:
            # Import profile
            if data.get('profile'):
                profile = data['profile']
                conn.execute('''
                    INSERT OR REPLACE INTO profiles
                    (user_id, name, age, height_cm, weight_kg, dietary_preferences,
                     notification_prefs, cycle_settings, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    user_id, profile.get('name'), profile.get('age'),
                    profile.get('height_cm'), profile.get('weight_kg'),
                    json.dumps(profile.get('dietary_preferences', {})),
                    json.dumps(profile.get('notification_prefs', {})),
                    json.dumps(profile.get('cycle_settings', {}))
                ))

            # Import cycles
            for cycle in data.get('cycles', []):
                conn.execute('''
                    INSERT INTO cycles
                    (user_id, start_date, end_date, flow, symptoms, notes, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    user_id, cycle['start_date'], cycle.get('end_date'),
                    cycle.get('flow'), json.dumps(cycle.get('symptoms', [])),
                    cycle.get('notes', '')
                ))

            # Import assessments
            for assessment in data.get('assessments', []):
                conn.execute('''
                    INSERT INTO pcos_assessments
                    (user_id, assessment_data, score, risk_level, recommendations, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, assessment['assessment_data'], assessment['score'],
                    assessment['risk_level'], assessment['recommendations'],
                    assessment.get('created_at', datetime.now().isoformat())
                ))

            # Import nutrition logs
            for log in data.get('nutrition_logs', []):
                conn.execute('''
                    INSERT INTO nutrition_logs
                    (user_id, date, meal_type, food_items, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, log['date'], log['meal_type'],
                    json.dumps(log.get('food_items', [])),
                    log.get('notes', ''), log.get('created_at', datetime.now().isoformat())
                ))

            conn.commit()
            return jsonify({'message': 'Data imported successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting RituCare Backend Server...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

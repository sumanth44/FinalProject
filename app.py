from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory, make_response
import sqlite3
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime, timedelta
import pandas as pd
import base64
from werkzeug.utils import secure_filename
import json
from face_detection import face_detector
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.utils

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'faces'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            image_path TEXT NOT NULL,
            face_embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            status TEXT DEFAULT 'Present',
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/faces/<filename>')
def uploaded_file(filename):
    """Serve uploaded face images"""
    return send_from_directory(UPLOAD_FOLDER, filename)

def get_face_embedding(image_path):
    """Extract face embedding from image"""
    try:
        print(f"Extracting face embedding from {image_path}")
        # Load image
        image = face_recognition.load_image_file(image_path)
        print(f"Image loaded, shape: {image.shape}")
        
        # Get face locations first
        face_locations = face_recognition.face_locations(image, model="hog")
        print(f"Found {len(face_locations)} face locations")
        
        if len(face_locations) == 0:
            print("No faces found in image")
            return None
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        print(f"Generated {len(face_encodings)} face encodings")
        
        if len(face_encodings) > 0:
            embedding = face_encodings[0].tolist()
            print(f"Face embedding length: {len(embedding)}")
            return embedding
        return None
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/admin')
def admin_panel():
    """Admin panel for user management"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users ORDER BY created_at DESC')
    users = cursor.fetchall()
    conn.close()
    return render_template('admin.html', users=users)

@app.route('/add_user', methods=['POST'])
def add_user():
    """Add new user with face image"""
    name = request.form['name']
    phone = request.form['phone']
    
    if 'image' not in request.files:
        flash('No image file selected')
        return redirect(url_for('admin_panel'))
    
    file = request.files['image']
    if file.filename == '':
        flash('No image selected')
        return redirect(url_for('admin_panel'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract face embedding
        face_embedding = get_face_embedding(filepath)
        if face_embedding is None:
            flash('No face detected in the image. Please try again.')
            os.remove(filepath)  # Remove the file if no face detected
            return redirect(url_for('admin_panel'))
        
        # Save to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (name, phone, image_path, face_embedding)
            VALUES (?, ?, ?, ?)
        ''', (name, phone, filepath, json.dumps(face_embedding)))
        conn.commit()
        
        # Get the new user ID
        user_id = cursor.lastrowid
        conn.close()
        
        # Retrain the face recognition system with the new user
        user_data = {
            'user_id': user_id,
            'name': name,
            'phone': phone,
            'embedding': face_embedding
        }
        face_detector.add_new_face(user_data)
        
        flash('User added successfully!')
    else:
        flash('Invalid file type. Please upload a valid image.')
    
    return redirect(url_for('admin_panel'))

@app.route('/delete_user/<int:user_id>')
def delete_user(user_id):
    """Delete user and their image"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get image path before deleting
    cursor.execute('SELECT image_path FROM users WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    
    if result:
        image_path = result[0]
        # Delete from database
        cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
        conn.commit()
        
        # Delete image file
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Retrain the face recognition system after deletion
        cursor.execute('SELECT user_id, name, phone, face_embedding FROM users')
        users = cursor.fetchall()
        conn.close()
        
        user_data = []
        for user in users:
            user_id, name, phone, embedding_json = user
            if embedding_json:
                embedding = json.loads(embedding_json)
                user_data.append({
                    'user_id': user_id,
                    'name': name,
                    'phone': phone,
                    'embedding': embedding
                })
        
        face_detector.retrain_system(user_data)
        
        flash('User deleted successfully!')
    else:
        flash('User not found!')
    
    return redirect(url_for('admin_panel'))

@app.route('/attendance')
def attendance():
    """Attendance marking page"""
    return render_template('attendance.html')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    """Mark attendance for detected face"""
    data = request.get_json()
    user_id = data.get('user_id')
    
    if user_id:
        # Mark attendance in database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        today = datetime.now().date()
        today_str = str(today)
        current_time_str = datetime.now().strftime('%H:%M:%S')
        
        # Check if already marked today
        cursor.execute('''
            SELECT time FROM attendance 
            WHERE user_id = ? AND date = ?
        ''', (user_id, today_str))
        
        existing_record = cursor.fetchone()
        if existing_record:
            conn.close()
            return jsonify({
                'success': False, 
                'message': 'Attendance already marked today',
                'marked_time': existing_record[0]
            })
        
        # Insert attendance record
        cursor.execute('''
            INSERT INTO attendance (user_id, date, time, status)
            VALUES (?, ?, ?, ?)
        ''', (user_id, today_str, current_time_str, 'Present'))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Attendance marked successfully!'})
    
    return jsonify({'success': False, 'message': 'Invalid user ID'})

@app.route('/mark_multiple_attendance', methods=['POST'])
def mark_multiple_attendance():
    """Mark attendance for multiple users at once"""
    data = request.get_json()
    user_ids = data.get('user_ids', [])
    
    if not user_ids:
        return jsonify({'success': False, 'message': 'No user IDs provided'})
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    today = datetime.now().date()
    today_str = str(today)
    current_time_str = datetime.now().strftime('%H:%M:%S')
    
    results = []
    success_count = 0
    already_marked = []
    
    for user_id in user_ids:
        # Check if already marked today
        cursor.execute('''
            SELECT time FROM attendance 
            WHERE user_id = ? AND date = ?
        ''', (user_id, today_str))
        
        existing_record = cursor.fetchone()
        if existing_record:
            already_marked.append({
                'user_id': user_id,
                'time': existing_record[0]
            })
        else:
            # Insert attendance record
            cursor.execute('''
                INSERT INTO attendance (user_id, date, time, status)
                VALUES (?, ?, ?, ?)
            ''', (user_id, today_str, current_time_str, 'Present'))
            success_count += 1
    
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True,
        'message': f'Attendance marked for {success_count} users',
        'success_count': success_count,
        'already_marked': already_marked
    })

@app.route('/attendance_logs')
def attendance_logs():
    """View attendance logs"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.attendance_id, u.name, u.phone, a.date, a.time, a.status
        FROM attendance a
        JOIN users u ON a.user_id = u.user_id
        ORDER BY a.date DESC, a.time DESC
    ''')
    logs = cursor.fetchall()
    conn.close()
    return render_template('logs.html', logs=logs)

@app.route('/export_attendance')
def export_attendance():
    """Export attendance to CSV"""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT u.name, u.phone, a.date, a.time, a.status
            FROM attendance a
            JOIN users u ON a.user_id = u.user_id
            ORDER BY a.date DESC, a.time DESC
        ''')
        logs = cursor.fetchall()
        conn.close()
        
        # Create DataFrame
        df = pd.DataFrame(logs, columns=['Name', 'Phone', 'Date', 'Time', 'Status'])
        
        # Generate CSV content
        csv_content = df.to_csv(index=False)
        
        # Create response with CSV content
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=attendance_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
        
    except Exception as e:
        print(f"Error exporting CSV: {e}")
        return jsonify({'success': False, 'message': 'Error exporting attendance data'})

@app.route('/get_users')
def get_users():
    """Get all users for face recognition, auto-generating embeddings if missing"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, name, phone, image_path, face_embedding FROM users')
    users = cursor.fetchall()
    
    user_data = []
    for user in users:
        user_id, name, phone, image_path, embedding_json = user
        embedding = None
        if embedding_json:
            try:
                embedding = json.loads(embedding_json)
            except Exception:
                embedding = None
        
        # If embedding is missing, try to create it from stored image
        if embedding is None and image_path and os.path.exists(image_path):
            new_embedding = get_face_embedding(image_path)
            if new_embedding:
                embedding = new_embedding
                # Persist back to DB for future requests
                try:
                    cursor.execute('UPDATE users SET face_embedding = ? WHERE user_id = ?', (json.dumps(new_embedding), user_id))
                    conn.commit()
                except Exception as e:
                    print(f"Failed to save embedding for user {user_id}: {e}")
        
        if embedding is not None:
            user_data.append({
                'user_id': user_id,
                'name': name,
                'phone': phone,
                'embedding': embedding
            })
    
    conn.close()
    
    # Load users into face detector
    face_detector.load_known_faces(user_data)
    
    return jsonify(user_data)

@app.route('/detect_face', methods=['POST'])
def detect_face():
    """Detect and recognize faces in uploaded image"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data provided'})
        
        # Load users first
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT user_id, name, phone, face_embedding FROM users')
        users = cursor.fetchall()
        conn.close()
        
        user_data = []
        for user in users:
            user_id, name, phone, embedding_json = user
            if embedding_json:
                embedding = json.loads(embedding_json)
                user_data.append({
                    'user_id': user_id,
                    'name': name,
                    'phone': phone,
                    'embedding': embedding
                })
        
        print(f"Loaded {len(user_data)} users for face recognition")
        
        # Load users into face detector
        face_detector.load_known_faces(user_data)
        
        # Detect faces
        results = face_detector.detect_faces_in_image(image_data)
        
        print(f"Face detection results: {len(results)} faces detected")
        
        if results:
            # Return all recognized faces
            recognized_faces = []
            unknown_faces = 0
            
            for result in results:
                if result['user_id']:
                    recognized_faces.append({
                        'user_id': result['user_id'],
                        'name': result['name'],
                        'phone': result['phone'],
                        'confidence': result['confidence']
                    })
                    print(f"Recognized user: {result['name']} (confidence: {result['confidence']})")
                else:
                    unknown_faces += 1
            
            if recognized_faces:
                return jsonify({
                    'success': True,
                    'faces': recognized_faces,
                    'count': len(recognized_faces),
                    'unknown_count': unknown_faces
                })
            else:
                # No recognized faces
                print("Face detected but not recognized")
                return jsonify({
                    'success': False,
                    'message': 'Face detected but not recognized',
                    'unknown_count': unknown_faces
                })
        else:
            print("No face detected in image")
            return jsonify({
                'success': False,
                'message': 'No face detected'
            })
            
    except Exception as e:
        print(f"Error in face detection: {e}")
        return jsonify({
            'success': False,
            'message': 'Error processing image'
        })

@app.route('/user/<int:user_id>')
def user_details(user_id):
    """User details page with analytics"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get user info
    cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
    user = cursor.fetchone()
    
    if not user:
        flash('User not found!')
        return redirect(url_for('admin_panel'))
    
    # Get attendance records for this user
    cursor.execute('''
        SELECT date, time, status FROM attendance 
        WHERE user_id = ? 
        ORDER BY date DESC, time DESC
    ''', (user_id,))
    attendance_records = cursor.fetchall()
    
    # Get attendance statistics
    cursor.execute('''
        SELECT 
            COUNT(*) as total_days,
            SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) as present_days,
            SUM(CASE WHEN status = 'Absent' THEN 1 ELSE 0 END) as absent_days
        FROM attendance 
        WHERE user_id = ?
    ''', (user_id,))
    stats = cursor.fetchone()
    
    # Get recent attendance (last 30 days)
    thirty_days_ago = (datetime.now() - timedelta(days=30)).date()
    cursor.execute('''
        SELECT date, status FROM attendance 
        WHERE user_id = ? AND date >= ?
        ORDER BY date ASC
    ''', (user_id, thirty_days_ago))
    recent_attendance = cursor.fetchall()
    
    conn.close()
    
    # Calculate attendance percentage
    attendance_percentage = (stats[1] / stats[0] * 100) if stats[0] > 0 else 0
    
    return render_template('user_details.html', 
                         user=user, 
                         attendance_records=attendance_records,
                         stats=stats,
                         attendance_percentage=attendance_percentage,
                         recent_attendance=recent_attendance)

@app.route('/user/<int:user_id>/analytics')
def user_analytics(user_id):
    """Get analytics data for a specific user"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get user info
    cursor.execute('SELECT name FROM users WHERE user_id = ?', (user_id,))
    user_name = cursor.fetchone()[0]
    
    # Get attendance data for the last 30 days
    thirty_days_ago = (datetime.now() - timedelta(days=30)).date()
    cursor.execute('''
        SELECT date, status FROM attendance 
        WHERE user_id = ? AND date >= ?
        ORDER BY date ASC
    ''', (user_id, thirty_days_ago))
    attendance_data = cursor.fetchall()
    
    # Get weekly attendance data
    cursor.execute('''
        SELECT 
            strftime('%w', date) as weekday,
            COUNT(*) as total,
            SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) as present
        FROM attendance 
        WHERE user_id = ? AND date >= ?
        GROUP BY strftime('%w', date)
        ORDER BY weekday
    ''', (user_id, thirty_days_ago))
    weekly_data = cursor.fetchall()
    
    conn.close()
    
    # Create attendance trend chart
    dates = [record[0] for record in attendance_data]
    statuses = [record[1] for record in attendance_data]
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(attendance_data, columns=['date', 'status'])
    df['date'] = pd.to_datetime(df['date'])
    df['present'] = (df['status'] == 'Present').astype(int)
    
    # Create daily attendance trend
    daily_attendance = df.groupby('date')['present'].sum().reset_index()
    
    # Create charts
    charts = {}
    
    # 1. Daily attendance trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_attendance['date'],
        y=daily_attendance['present'],
        mode='lines+markers',
        name='Daily Attendance',
        line=dict(color='#3B82F6', width=3)
    ))
    fig.update_layout(
        title='Daily Attendance Trend (Last 30 Days)',
        xaxis_title='Date',
        yaxis_title='Present (1) / Absent (0)',
        template='plotly_white'
    )
    charts['daily_trend'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 2. Weekly attendance pattern
    weekday_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    weekly_df = pd.DataFrame(weekly_data, columns=['weekday', 'total', 'present'])
    weekly_df['weekday_name'] = [weekday_names[int(w)] for w in weekly_df['weekday']]
    weekly_df['attendance_rate'] = (weekly_df['present'] / weekly_df['total'] * 100).round(1)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=weekly_df['weekday_name'],
        y=weekly_df['attendance_rate'],
        name='Attendance Rate %',
        marker_color='#10B981'
    ))
    fig2.update_layout(
        title='Weekly Attendance Pattern',
        xaxis_title='Day of Week',
        yaxis_title='Attendance Rate (%)',
        template='plotly_white'
    )
    charts['weekly_pattern'] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 3. Overall statistics
    total_days = len(attendance_data)
    present_days = sum(1 for status in statuses if status == 'Present')
    attendance_rate = (present_days / total_days * 100) if total_days > 0 else 0
    
    stats = {
        'total_days': total_days,
        'present_days': present_days,
        'absent_days': total_days - present_days,
        'attendance_rate': round(attendance_rate, 1)
    }
    
    return jsonify({
        'charts': charts,
        'stats': stats,
        'user_name': user_name
    })

@app.route('/analytics')
def analytics():
    """Overall analytics dashboard"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get overall statistics
    cursor.execute('''
        SELECT 
            COUNT(DISTINCT user_id) as total_users,
            COUNT(*) as total_attendance_records,
            SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) as total_present,
            SUM(CASE WHEN status = 'Absent' THEN 1 ELSE 0 END) as total_absent
        FROM attendance
    ''')
    overall_stats = cursor.fetchone()
    
    # Get daily attendance for the last 30 days
    thirty_days_ago = (datetime.now() - timedelta(days=30)).date()
    cursor.execute('''
        SELECT 
            date,
            COUNT(DISTINCT user_id) as unique_users,
            SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) as present_count
        FROM attendance 
        WHERE date >= ?
        GROUP BY date
        ORDER BY date ASC
    ''', (thirty_days_ago,))
    daily_stats = cursor.fetchall()
    
    # Get user-wise attendance
    cursor.execute('''
        SELECT 
            u.name,
            COUNT(a.attendance_id) as total_records,
            SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) as present_count
        FROM users u
        LEFT JOIN attendance a ON u.user_id = a.user_id
        GROUP BY u.user_id, u.name
        ORDER BY present_count DESC
    ''')
    user_stats = cursor.fetchall()
    
    conn.close()
    
    return render_template('analytics.html', 
                         overall_stats=overall_stats,
                         daily_stats=daily_stats,
                         user_stats=user_stats)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=3000)

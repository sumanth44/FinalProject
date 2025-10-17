#!/usr/bin/env python3
"""
Script to add sample face data for testing
"""

import os
import sqlite3
import json
import numpy as np
from PIL import Image, ImageDraw
import face_recognition
from datetime import datetime, timedelta
import random

def create_sample_face_image(name, size=(200, 200)):
    """Create a sample face image for testing"""
    # Create a simple face-like image
    img = Image.new('RGB', size, color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple face
    # Face outline
    draw.ellipse([20, 20, size[0]-20, size[1]-20], fill='peachpuff', outline='black', width=2)
    
    # Eyes
    draw.ellipse([60, 60, 80, 80], fill='white', outline='black')
    draw.ellipse([120, 60, 140, 80], fill='white', outline='black')
    draw.ellipse([65, 65, 75, 75], fill='black')  # Left pupil
    draw.ellipse([125, 65, 135, 75], fill='black')  # Right pupil
    
    # Nose
    draw.polygon([(100, 90), (90, 120), (110, 120)], fill='peachpuff', outline='black')
    
    # Mouth
    draw.arc([70, 130, 130, 150], 0, 180, fill='red', width=3)
    
    return img

def get_face_embedding(image_path):
    """Extract face embedding from image"""
    try:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            return face_encodings[0].tolist()
        return None
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None

def add_sample_users():
    """Add sample users to the database"""
    # Ensure faces directory exists
    os.makedirs('faces', exist_ok=True)
    
    # Sample users
    users = [
        {'name': 'Alice Johnson', 'phone': '555-0101'},
        {'name': 'Bob Smith', 'phone': '555-0102'},
        {'name': 'Carol Davis', 'phone': '555-0103'},
        {'name': 'David Wilson', 'phone': '555-0104'},
        {'name': 'Eva Brown', 'phone': '555-0105'}
    ]
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    for i, user in enumerate(users, 1):
        # Create sample face image
        img = create_sample_face_image(user['name'])
        image_path = f'faces/sample_face_{i}.jpg'
        img.save(image_path)
        
        # Get face embedding
        embedding = get_face_embedding(image_path)
        if embedding is None:
            # Create a dummy embedding if face recognition fails
            embedding = np.random.rand(128).tolist()
        
        # Insert user into database
        cursor.execute('''
            INSERT INTO users (name, phone, image_path, face_embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (user['name'], user['phone'], image_path, json.dumps(embedding), datetime.now()))
    
    conn.commit()
    conn.close()
    print(f"Added {len(users)} sample users")

def add_sample_attendance():
    """Add sample attendance records"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get all users
    cursor.execute('SELECT user_id FROM users')
    user_ids = [row[0] for row in cursor.fetchall()]
    
    if not user_ids:
        print("No users found. Please add users first.")
        return
    
    # Add attendance records for the last 30 days
    today = datetime.now().date()
    attendance_records = []
    
    for user_id in user_ids:
        for days_ago in range(30):
            date = today - timedelta(days=days_ago)
            
            # Skip weekends for some users (simulate real attendance patterns)
            if date.weekday() >= 5 and random.random() < 0.7:
                continue
            
            # Randomly mark attendance (80% chance of present)
            status = 'Present' if random.random() < 0.8 else 'Absent'
            
            # Random time between 8 AM and 6 PM
            hour = random.randint(8, 18)
            minute = random.randint(0, 59)
            time = datetime.now().replace(hour=hour, minute=minute).time()
            
            attendance_records.append((user_id, str(date), str(time), status))
    
    # Insert all attendance records
    cursor.executemany('''
        INSERT INTO attendance (user_id, date, time, status)
        VALUES (?, ?, ?, ?)
    ''', attendance_records)
    
    conn.commit()
    conn.close()
    print(f"Added {len(attendance_records)} sample attendance records")

def main():
    """Main function to add sample data"""
    print("Adding sample data to the attendance system...")
    
    # Initialize database
    from app import init_db
    init_db()
    
    # Add sample users
    add_sample_users()
    
    # Add sample attendance
    add_sample_attendance()
    
    print("Sample data added successfully!")
    print("\nYou can now:")
    print("1. Run the Flask app: python app.py")
    print("2. Visit http://localhost:5000 to see the system")
    print("3. Check the admin panel to see the sample users")
    print("4. View analytics to see the sample data visualizations")

if __name__ == '__main__':
    main()

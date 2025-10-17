import pytest
import os
import tempfile
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import sqlite3
from datetime import datetime, timedelta

# Set up test environment
os.environ['TESTING'] = 'True'

from app import app, init_db, get_face_embedding
from face_detection import FaceDetectionSystem

@pytest.fixture
def client():
    """Create test client with temporary database"""
    # Create temporary database
    db_fd, app.config['DATABASE'] = tempfile.mkstemp()
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
    
    with app.test_client() as client:
        with app.app_context():
            init_db()
        yield client
    
    # Clean up
    os.close(db_fd)
    os.unlink(app.config['DATABASE'])

@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    return {
        'name': 'John Doe',
        'phone': '1234567890',
        'image_path': 'test_image.jpg'
    }

@pytest.fixture
def sample_attendance_data():
    """Sample attendance data for testing"""
    return {
        'user_id': 1,
        'date': datetime.now().date(),
        'time': datetime.now().time(),
        'status': 'Present'
    }

@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

class TestDatabase:
    """Test database operations"""
    
    def test_init_db(self, client):
        """Test database initialization"""
        with app.app_context():
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'users' in tables
            assert 'attendance' in tables
            
            conn.close()
    
    def test_add_user(self, client, sample_user_data, sample_image):
        """Test adding a user"""
        # Create test image file
        test_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test.jpg')
        sample_image.seek(0)
        with open(test_image_path, 'wb') as f:
            f.write(sample_image.read())
        
        # Mock face embedding
        mock_embedding = [0.1] * 128
        
        with app.app_context():
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (name, phone, image_path, face_embedding)
                VALUES (?, ?, ?, ?)
            ''', (sample_user_data['name'], sample_user_data['phone'], 
                  test_image_path, json.dumps(mock_embedding)))
            conn.commit()
            
            # Verify user was added
            cursor.execute('SELECT * FROM users WHERE name = ?', (sample_user_data['name'],))
            user = cursor.fetchone()
            
            assert user is not None
            assert user[1] == sample_user_data['name']
            assert user[2] == sample_user_data['phone']
            
            conn.close()
    
    def test_mark_attendance(self, client, sample_attendance_data):
        """Test marking attendance"""
        with app.app_context():
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            
            # Add test user first
            cursor.execute('''
                INSERT INTO users (name, phone, image_path, face_embedding)
                VALUES (?, ?, ?, ?)
            ''', ('Test User', '1234567890', 'test.jpg', json.dumps([0.1] * 128)))
            conn.commit()
            
            # Mark attendance
            cursor.execute('''
                INSERT INTO attendance (user_id, date, time, status)
                VALUES (?, ?, ?, ?)
            ''', (sample_attendance_data['user_id'], str(sample_attendance_data['date']),
                  str(sample_attendance_data['time']), sample_attendance_data['status']))
            conn.commit()
            
            # Verify attendance was marked
            cursor.execute('SELECT * FROM attendance WHERE user_id = ?', (1,))
            attendance = cursor.fetchone()
            
            assert attendance is not None
            assert attendance[1] == sample_attendance_data['user_id']
            assert attendance[4] == sample_attendance_data['status']
            
            conn.close()

class TestFlaskRoutes:
    """Test Flask routes"""
    
    def test_index_route(self, client):
        """Test home page route"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'AI-Powered Attendance System' in response.data
    
    def test_admin_panel_route(self, client):
        """Test admin panel route"""
        response = client.get('/admin')
        assert response.status_code == 200
        assert b'Admin Panel' in response.data
    
    def test_attendance_route(self, client):
        """Test attendance marking route"""
        response = client.get('/attendance')
        assert response.status_code == 200
        assert b'Mark Attendance' in response.data
    
    def test_attendance_logs_route(self, client):
        """Test attendance logs route"""
        response = client.get('/attendance_logs')
        assert response.status_code == 200
        assert b'Attendance Logs' in response.data
    
    def test_analytics_route(self, client):
        """Test analytics route"""
        response = client.get('/analytics')
        assert response.status_code == 200
        assert b'Analytics Dashboard' in response.data
    
    def test_get_users_route(self, client):
        """Test get users API route"""
        response = client.get('/get_users')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
    
    def test_add_user_route(self, client, sample_image):
        """Test add user route"""
        # Create test image file
        test_image_path = os.path.join(client.application.config['UPLOAD_FOLDER'], 'test.jpg')
        sample_image.seek(0)
        with open(test_image_path, 'wb') as f:
            f.write(sample_image.read())
        
        # Mock face embedding function
        def mock_get_face_embedding(path):
            return [0.1] * 128
        
        # Replace the function temporarily
        import app
        original_func = app.get_face_embedding
        app.get_face_embedding = mock_get_face_embedding
        
        try:
            data = {
                'name': 'Test User',
                'phone': '1234567890'
            }
            
            with open(test_image_path, 'rb') as f:
                response = client.post('/add_user', 
                                     data={**data, 'image': f},
                                     content_type='multipart/form-data')
            
            assert response.status_code == 302  # Redirect after successful add
        finally:
            # Restore original function
            app.get_face_embedding = original_func
    
    def test_mark_attendance_route(self, client):
        """Test mark attendance API route"""
        # First add a user
        with app.app_context():
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (name, phone, image_path, face_embedding)
                VALUES (?, ?, ?, ?)
            ''', ('Test User', '1234567890', 'test.jpg', json.dumps([0.1] * 128)))
            conn.commit()
            conn.close()
        
        # Mark attendance
        response = client.post('/mark_attendance', 
                             json={'user_id': 1},
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'success' in data
    
    def test_user_details_route(self, client):
        """Test user details route"""
        # First add a user
        with app.app_context():
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (name, phone, image_path, face_embedding)
                VALUES (?, ?, ?, ?)
            ''', ('Test User', '1234567890', 'test.jpg', json.dumps([0.1] * 128)))
            conn.commit()
            conn.close()
        
        response = client.get('/user/1')
        assert response.status_code == 200
        assert b'Test User' in response.data
    
    def test_user_analytics_route(self, client):
        """Test user analytics API route"""
        # First add a user and some attendance
        with app.app_context():
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (name, phone, image_path, face_embedding)
                VALUES (?, ?, ?, ?)
            ''', ('Test User', '1234567890', 'test.jpg', json.dumps([0.1] * 128)))
            
            # Add some attendance records
            today = datetime.now().date()
            for i in range(5):
                date = today - timedelta(days=i)
                cursor.execute('''
                    INSERT INTO attendance (user_id, date, time, status)
                    VALUES (?, ?, ?, ?)
                ''', (1, str(date), str(datetime.now().time()), 'Present'))
            
            conn.commit()
            conn.close()
        
        response = client.get('/user/1/analytics')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'charts' in data
        assert 'stats' in data

class TestFaceDetection:
    """Test face detection functionality"""
    
    def test_face_detection_system_init(self):
        """Test face detection system initialization"""
        detector = FaceDetectionSystem()
        assert detector.known_face_encodings == []
        assert detector.known_face_names == []
        assert detector.known_face_ids == []
        assert detector.known_face_phones == []
    
    def test_load_known_faces(self):
        """Test loading known faces"""
        detector = FaceDetectionSystem()
        users_data = [
            {
                'user_id': 1,
                'name': 'John Doe',
                'phone': '1234567890',
                'embedding': [0.1] * 128
            },
            {
                'user_id': 2,
                'name': 'Jane Smith',
                'phone': '0987654321',
                'embedding': [0.2] * 128
            }
        ]
        
        detector.load_known_faces(users_data)
        
        assert len(detector.known_face_encodings) == 2
        assert len(detector.known_face_names) == 2
        assert len(detector.known_face_ids) == 2
        assert len(detector.known_face_phones) == 2
        assert detector.known_face_names[0] == 'John Doe'
        assert detector.known_face_names[1] == 'Jane Smith'
    
    def test_detect_faces_in_image(self, sample_image):
        """Test face detection in image"""
        detector = FaceDetectionSystem()
        
        # Load some test faces
        users_data = [
            {
                'user_id': 1,
                'name': 'Test User',
                'phone': '1234567890',
                'embedding': [0.1] * 128
            }
        ]
        detector.load_known_faces(users_data)
        
        # Convert image to base64
        sample_image.seek(0)
        img_bytes = sample_image.read()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        image_data = f"data:image/jpeg;base64,{img_base64}"
        
        # Test detection (this will likely return no faces since it's a simple test image)
        results = detector.detect_faces_in_image(image_data)
        assert isinstance(results, list)

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_allowed_file(self, client):
        """Test file extension validation"""
        from app import allowed_file
        
        assert allowed_file('test.jpg') == True
        assert allowed_file('test.png') == True
        assert allowed_file('test.jpeg') == True
        assert allowed_file('test.gif') == True
        assert allowed_file('test.txt') == False
        assert allowed_file('test.pdf') == False
        assert allowed_file('test') == False

class TestIntegration:
    """Integration tests"""
    
    def test_full_workflow(self, client, sample_image):
        """Test complete workflow from user registration to attendance marking"""
        # 1. Add a user
        test_image_path = os.path.join(client.application.config['UPLOAD_FOLDER'], 'test.jpg')
        sample_image.seek(0)
        with open(test_image_path, 'wb') as f:
            f.write(sample_image.read())
        
        # Mock face embedding
        def mock_get_face_embedding(path):
            return [0.1] * 128
        
        import app
        original_func = app.get_face_embedding
        app.get_face_embedding = mock_get_face_embedding
        
        try:
            data = {
                'name': 'Integration Test User',
                'phone': '1234567890'
            }
            
            with open(test_image_path, 'rb') as f:
                response = client.post('/add_user', 
                                     data={**data, 'image': f},
                                     content_type='multipart/form-data')
            
            assert response.status_code == 302
            
            # 2. Mark attendance
            response = client.post('/mark_attendance', 
                                 json={'user_id': 1},
                                 content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] == True
            
            # 3. Check attendance logs
            response = client.get('/attendance_logs')
            assert response.status_code == 200
            assert b'Integration Test User' in response.data
            
            # 4. Check user details
            response = client.get('/user/1')
            assert response.status_code == 200
            assert b'Integration Test User' in response.data
            
        finally:
            app.get_face_embedding = original_func

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

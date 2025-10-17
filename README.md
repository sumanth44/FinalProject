# AI-Powered Attendance System

An automated attendance marking system using real-time face detection and recognition. This system allows administrators to add users with face images and automatically mark attendance using webcam-based face detection.

## Features

- **Real-time Face Detection**: Uses OpenCV and face_recognition library for accurate face detection
- **Face Recognition**: Compares detected faces with stored user embeddings
- **Admin Panel**: Add, update, and delete users with face images
- **Attendance Marking**: Automatic attendance marking with identity confirmation
- **Attendance Logs**: View and export attendance records
- **Modern UI**: Responsive design with Tailwind CSS
- **Local Storage**: All data stored locally using SQLite

## Technology Stack

- **Backend**: Python 3.x, Flask
- **Face Detection**: OpenCV, face_recognition library
- **Database**: SQLite
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Image Processing**: Pillow, NumPy

## Installation

1. **Clone or download the project**
   ```bash
   cd "Ai powered attendance system"
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   Open your browser and go to `http://localhost:5000`

## Project Structure

```
/project_root
│
├── /faces                 # Store user face images
│   ├── 1.jpg
│   ├── 2.jpg
│
├── /templates             # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── admin.html
│   ├── attendance.html
│   └── logs.html
│
├── app.py                 # Main Flask application
├── face_detection.py      # Face detection and recognition system
├── database.db            # SQLite database (created automatically)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Usage

### 1. Admin Panel

1. Go to the Admin Panel
2. Click "Add New User"
3. Enter user details (name, phone number)
4. Upload a clear face image
5. The system will automatically extract face embeddings for recognition

### 2. Mark Attendance

1. Go to the Mark Attendance page
2. Click "Start Detection"
3. Allow camera access when prompted
4. Position your face in front of the camera
5. The system will detect and recognize your face
6. Confirm your identity when prompted
7. Attendance will be automatically marked

### 3. View Attendance Logs

1. Go to the Attendance Logs page
2. View all attendance records
3. Use filters to search by date, name, or status
4. Export data to CSV if needed

## Database Schema

### Users Table
- `user_id`: Primary key
- `name`: User's full name
- `phone`: Phone number
- `image_path`: Path to stored face image
- `face_embedding`: Face encoding for recognition
- `created_at`: Timestamp

### Attendance Table
- `attendance_id`: Primary key
- `user_id`: Foreign key to users table
- `date`: Attendance date
- `time`: Attendance time
- `status`: Present/Absent status

## API Endpoints

- `GET /` - Home page
- `GET /admin` - Admin panel
- `POST /add_user` - Add new user
- `GET /delete_user/<user_id>` - Delete user
- `GET /attendance` - Attendance marking page
- `POST /mark_attendance` - Mark attendance
- `GET /attendance_logs` - View attendance logs
- `GET /export_attendance` - Export attendance to CSV
- `GET /get_users` - Get all users for recognition
- `POST /detect_face` - Detect faces in image

## Configuration

### Camera Settings
The system uses the default webcam. To change camera settings, modify the `getUserMedia` call in `attendance.html`:

```javascript
const stream = await navigator.mediaDevices.getUserMedia({ 
    video: { 
        width: 640, 
        height: 480,
        facingMode: 'user'  // Use front camera
    } 
});
```

### Face Recognition Tolerance
Adjust the face recognition tolerance in `face_detection.py`:

```python
matches = face_recognition.compare_faces(
    self.known_face_encodings, 
    face_encoding, 
    tolerance=0.6  # Lower = more strict
)
```

## Troubleshooting

### Common Issues

1. **Camera not working**
   - Ensure camera permissions are granted
   - Check if another application is using the camera
   - Try refreshing the page

2. **Face not detected**
   - Ensure good lighting
   - Position face directly in front of camera
   - Make sure face is clearly visible

3. **Face not recognized**
   - Check if user is registered in admin panel
   - Ensure face image quality is good
   - Try adjusting recognition tolerance

4. **Installation issues**
   - Make sure Python 3.x is installed
   - Install dependencies: `pip install -r requirements.txt`
   - Check if all required libraries are properly installed

### Dependencies Issues

If you encounter issues with face_recognition library:

```bash
# On macOS
brew install cmake
pip install face_recognition

# On Ubuntu/Debian
sudo apt-get install cmake
pip install face_recognition

# On Windows
# Install Visual Studio Build Tools first
pip install face_recognition
```

## Security Notes

- All data is stored locally
- No external API calls for face recognition
- Face images are stored in local filesystem
- Database is SQLite file-based

## Future Enhancements

- Multi-face detection support
- Real-time video streaming
- Advanced reporting features
- User authentication
- Cloud storage integration
- Mobile app support

## License

This project is open source and available under the MIT License.

## Support

For issues and questions, please check the troubleshooting section or create an issue in the project repository.
# FinalProject

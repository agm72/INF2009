import os
import cv2
import time
import shutil
import face_recognition
import pickle
import numpy as np
from imutils import paths
from flask import Flask, request, render_template, jsonify, session, send_from_directory, Response, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from modules.ultrasonic import get_distance
from modules.humidity_temp import get_temperature_humidity
from datetime import datetime, timedelta
import logging
logging.basicConfig(level=logging.DEBUG)

# Flask Setup
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Database Setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Upload Paths
UPLOAD_FOLDER = '/home/fuzzi/WebInterface/Users'
TEMP_FOLDER = '/home/fuzzi/WebInterface/temp'
ENCODINGS_FILE = "encodings.pickle"

# Ensure necessary folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ------------------ Database Models ------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    image1 = db.Column(db.String(200))
    image2 = db.Column(db.String(200))
    image3 = db.Column(db.String(200))
    image4 = db.Column(db.String(200))
    image5 = db.Column(db.String(200))

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    start_time = db.Column(db.DateTime, nullable=False)
    user = db.relationship('User', backref=db.backref('tasks', lazy=True))

with app.app_context():
    db.create_all()

# ------------------ Utility / Training ------------------
def train_model():
    """Train face recognition model using images in the Users folder."""
    print("[INFO] Training face recognition model...")
    imagePaths = list(paths.list_images(UPLOAD_FOLDER))
    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):
        print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}")
        user_id = os.path.basename(os.path.dirname(imagePath))
        user = db.session.get(User, int(user_id))
        if user:
            name = user.name
        else:
            print(f"[WARNING] No user found for ID {user_id}, skipping...")
            continue

        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    print("[INFO] Saving trained model...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open(ENCODINGS_FILE, "wb") as f:
        f.write(pickle.dumps(data))

    print("[INFO] Training complete!")

def recognize_face():
    """Use the camera to detect and recognize a face WITHOUT GUI calls."""
    print("[INFO] Loading face recognition model...")
    if not os.path.exists(ENCODINGS_FILE):
        print("[ERROR] No trained model found. Please register first!")
        return None

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.loads(f.read())

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera!")
        return None

    print("[INFO] Looking for a face...")
    max_attempts = 100
    attempts = 0

    while attempts < max_attempts:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = data["names"][best_match_index]
                print(f"[INFO] Recognized: {name}")
                cap.release()
                return name
        attempts += 1

    cap.release()
    return None

def gen_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Could not open camera for live feed!")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    camera.release()

# ------------------ Routes that serve HTML templates ------------------
@app.route('/')
def home():
    user = db.session.get(User, session['user_id']) if 'user_id' in session else None
    user_tasks = []
    now = datetime.utcnow()  
    if user:
        user_tasks = Task.query.filter_by(user_id=user.id).order_by(Task.start_time.asc()).all()

    return render_template('index.html', user=user, tasks=user_tasks, now=now, timedelta=timedelta)

@app.route('/face-login-page')
def face_login_page():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('face_login.html')

@app.route('/register', methods=['GET'])
def register_get():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('register.html')

# ------------------ JSON-based routes (POST/logic) ------------------
@app.route('/capture', methods=['POST'])
def capture():
    username = request.form.get('username')
    if not username:
        return jsonify({"success": False, "error": "Username is required"}), 400

    temp_folder = os.path.join(TEMP_FOLDER, username)
    os.makedirs(temp_folder, exist_ok=True)

    camera = cv2.VideoCapture(0)
    time.sleep(1)
    ret, frame = camera.read()
    if not ret:
        return jsonify({"success": False, "error": "Failed to capture image"}), 500

    existing_images = len(os.listdir(temp_folder))
    if existing_images >= 5:
        return jsonify({"success": False, "error": "You already have 5 images."}), 400

    filename = f"temp_img{existing_images+1}.png"
    filepath = os.path.join(temp_folder, filename)
    cv2.imwrite(filepath, frame)
    camera.release()

    return jsonify({
        "success": True,
        "message": "Image captured successfully",
        "image": filename,
        "path": f"/temp/{username}/{filename}"
    })

@app.route('/clear-images', methods=['POST'])
def clear_images():
    username = request.form.get('username')
    temp_folder = os.path.join(TEMP_FOLDER, username)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)
    return jsonify({"success": True, "message": "All images cleared successfully!"})

@app.route('/register', methods=['POST'])
def register_post():
    if 'user_id' in session:
        return jsonify({"success": False, "error": "Already logged in."})

    username = request.form.get('username', '')
    temp_folder = os.path.join(TEMP_FOLDER, username)

    if User.query.filter_by(name=username).first():
        return jsonify({"success": False, "error": "Username is already taken!"})

    image_files = os.listdir(temp_folder) if os.path.exists(temp_folder) else []
    if len(image_files) < 5:
        return jsonify({"success": False, "error": "You must capture at least 5 images!"})

    new_user = User(name=username)
    db.session.add(new_user)
    db.session.commit()

    user_folder = os.path.join(UPLOAD_FOLDER, str(new_user.id))
    os.makedirs(user_folder, exist_ok=True)

    image_paths = []
    for i, image_file in enumerate(sorted(image_files)):
        new_filename = f"user_{new_user.id}_img{i+1}.png"
        old_path = os.path.join(temp_folder, image_file)
        new_path = os.path.join(user_folder, new_filename)
        shutil.move(old_path, new_path)
        image_paths.append(new_filename)

    new_user.image1, new_user.image2, new_user.image3, new_user.image4, new_user.image5 = image_paths
    db.session.commit()

    shutil.rmtree(temp_folder, ignore_errors=True)
    train_model()

    return jsonify({
        "success": True,
        "message": "Registration successful! Please log in using face recognition."
    })

@app.route('/face-login', methods=['GET'])
def face_login():
    user_name = recognize_face()
    if not user_name:
        return jsonify({"success": False, "error": "Face not recognized. Try again."})

    user = User.query.filter_by(name=user_name).first()
    if not user:
        return jsonify({"success": False, "error": "No such user in database."})

    session['user_id'] = user.id
    return jsonify({"success": True, "message": f"Welcome back, {user_name}!"})

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('user_id', None)
    return jsonify({"success": True, "message": "You have been logged out."})

# ------------------ Task CRUD Routes ------------------
@app.route('/add_task', methods=['POST'])
def add_task():
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401

    user = db.session.get(User, session['user_id'])
    title = request.form.get('title', '').strip()
    description = request.form.get('description', '').strip()
    start_time_str = request.form.get('start_time', '').strip()

    if not title or not start_time_str:
        return jsonify({"success": False, "message": "Title and Start Date/Time are required."}), 400

    try:
        start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M')
    except ValueError:
        return jsonify({"success": False, "message": "Invalid date/time format."}), 400

    new_task = Task(
        user_id=user.id,
        title=title,
        description=description,
        start_time=start_time
    )
    db.session.add(new_task)
    db.session.commit()
    return jsonify({"success": True, "message": "Task added successfully."})

@app.route('/edit_task/<int:task_id>', methods=['POST'])
def edit_task(task_id):
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401

    task = Task.query.get_or_404(task_id)
    if task.user_id != session['user_id']:
        return jsonify({"success": False, "message": "Unauthorized"}), 403

    title = request.form.get('title', '').strip()
    description = request.form.get('description', '').strip()
    start_time_str = request.form.get('start_time', '').strip()

    if not title or not start_time_str:
        return jsonify({"success": False, "message": "Title and Start Date/Time are required."}), 400

    try:
        start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M')
    except ValueError:
        return jsonify({"success": False, "message": "Invalid date/time format."}), 400

    task.title = title
    task.description = description
    task.start_time = start_time
    db.session.commit()

    return jsonify({"success": True, "message": "Task updated successfully."})

@app.route('/delete_task/<int:task_id>', methods=['POST'])
def delete_task(task_id):
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401

    task = Task.query.get_or_404(task_id)
    if task.user_id != session['user_id']:
        return jsonify({"success": False, "message": "Unauthorized"}), 403

    db.session.delete(task)
    db.session.commit()
    return jsonify({"success": True, "message": "Task deleted successfully."})

@app.route('/api/tasks')
def api_tasks():
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in."}), 401

    user = db.session.get(User, session['user_id'])
    tasks = Task.query.filter_by(user_id=user.id).all()
    events = []
    for task in tasks:
        events.append({
            "id": task.id,
            "title": task.title,
            "start": task.start_time.isoformat(),
            "description": task.description
        })
    return jsonify(events)
# ------------------ File & Sensor Endpoints ------------------
@app.route('/users/<int:user_id>/<path:filename>')
def get_image(user_id, filename):
    user_folder = os.path.join(UPLOAD_FOLDER, str(user_id))
    file_path = os.path.join(user_folder, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_from_directory(user_folder, filename)

@app.route('/temp/<username>/<path:filename>')
def get_temp_image(username, filename):
    temp_folder = os.path.join(TEMP_FOLDER, username)
    file_path = os.path.join(temp_folder, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_from_directory(temp_folder, filename)

@app.route('/sensor-data')
def sensor_data():
    """Returns the distance measured by the ultrasonic sensor."""
    try:
        distance = get_distance()
        if distance == -1:
            return jsonify({"success": False, "error": "Measurement timeout! Check wiring."}), 500
        return jsonify({"success": True, "distance": distance})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/temp-humidity')
def temp_humidity():
    """API endpoint to return temperature and humidity readings."""
    temperature, humidity = get_temperature_humidity()

    if temperature is None or humidity is None:
        return jsonify({"success": False, "error": "Failed to read sensor data"}), 500

    return jsonify({"success": True, "temperature": temperature, "humidity": humidity})
# ------------------ Run Flask ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

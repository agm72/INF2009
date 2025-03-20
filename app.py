import os
import cv2
import time
import shutil
import face_recognition
import pickle
import numpy as np
import json  # For loading heart rate JSON data and user data JSON
from imutils import paths
from flask import Flask, request, render_template, jsonify, session, send_from_directory, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from modules.ultrasonic import get_distance
from modules.humidity_temp import get_temperature_humidity
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
HEARTRATE_FOLDER = 'Heartrate'  # Folder where heart rate JSON files are stored
USERDATA_FOLDER = 'UserData'    # Folder where user JSON data files are stored
ENCODINGS_FILE = "encodings.pickle"

# Ensure necessary folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(HEARTRATE_FOLDER, exist_ok=True)
os.makedirs(USERDATA_FOLDER, exist_ok=True)

# ------------------ Database Models ------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    schedule_link = db.Column(db.String(300), nullable=False)
    # Storing up to 5 images for face recognition
    image1 = db.Column(db.String(200))
    image2 = db.Column(db.String(200))
    image3 = db.Column(db.String(200))
    image4 = db.Column(db.String(200))
    image5 = db.Column(db.String(200))

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

# ------------------ Heart Rate Analytics Utilities ------------------
def load_heartrate_data(user_id):
    """Load heart rate data from a JSON file for the given user."""
    file_path = os.path.join(HEARTRATE_FOLDER, f"{user_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    return {}

def compute_heart_rate_stats(data):
    """Compute daily analytics (min, max, average) for the heart rate data."""
    stats = {}
    for date, readings in data.items():
        if readings:
            stats[date] = {
                "min": min(readings),
                "max": max(readings),
                "avg": round(sum(readings) / len(readings), 2)
            }
    return stats

# ------------------ User Data Analysis Utilities ------------------
def load_user_data(username):
    """Load user data JSON from the UserData folder based on the username."""
    file_path = os.path.join(USERDATA_FOLDER, f"{username}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    return None

def analyze_user_data(data):
    """
    Compute summary statistics for user data.
    Expects data in the format:
    {
      "username": "afiq",
      "age": "27",
      "records": [
          {"heartrate": "93 bpm", "temperature": "37", "humidity": "90%", "bodytemp": "35", "time_recorded": "2025-03-20 17:01:43"},
          ...
      ]
    }
    """
    if not data or "records" not in data:
        return None
    records = data["records"]
    heartrates = []
    temperatures = []
    humidities = []
    bodytemps = []
    times = []
    
    for rec in records:
        # Parse heart rate (removing "bpm")
        hr_str = rec.get("heartrate", "0 bpm")
        try:
            hr = int(hr_str.split()[0])
        except:
            hr = 0
        heartrates.append(hr)
        
        # Parse temperature
        temp_str = rec.get("temperature", "0")
        try:
            temp = float(temp_str)
        except:
            temp = 0
        temperatures.append(temp)
        
        # Parse humidity (remove "%" if present)
        hum_str = rec.get("humidity", "0%")
        try:
            hum = int(hum_str.replace("%", ""))
        except:
            hum = 0
        humidities.append(hum)
        
        # Parse body temperature
        bt_str = rec.get("bodytemp", "0")
        try:
            bt = float(bt_str)
        except:
            bt = 0
        bodytemps.append(bt)
        
        # Parse time
        time_str = rec.get("time_recorded", "")
        try:
            t = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except:
            t = None
        times.append(t)
    
    stats = {
        "heartrate": {
            "min": min(heartrates) if heartrates else None,
            "max": max(heartrates) if heartrates else None,
            "avg": round(sum(heartrates) / len(heartrates), 2) if heartrates else None
        },
        "temperature": {
            "min": min(temperatures) if temperatures else None,
            "max": max(temperatures) if temperatures else None,
            "avg": round(sum(temperatures) / len(temperatures), 2) if temperatures else None
        },
        "humidity": {
            "min": min(humidities) if humidities else None,
            "max": max(humidities) if humidities else None,
            "avg": round(sum(humidities) / len(humidities), 2) if humidities else None
        },
        "bodytemp": {
            "min": min(bodytemps) if bodytemps else None,
            "max": max(bodytemps) if bodytemps else None,
            "avg": round(sum(bodytemps) / len(bodytemps), 2) if bodytemps else None
        },
        # Prepare records for charting for each measurement as a list of [timestamp, value]
        "heartrate_records": [
            [t.strftime("%Y-%m-%d %H:%M:%S") if t else "", hr] 
            for t, hr in zip(times, heartrates)
        ],
        "temperature_records": [
            [t.strftime("%Y-%m-%d %H:%M:%S") if t else "", temp] 
            for t, temp in zip(times, temperatures)
        ],
        "humidity_records": [
            [t.strftime("%Y-%m-%d %H:%M:%S") if t else "", hum] 
            for t, hum in zip(times, humidities)
        ],
        "bodytemp_records": [
            [t.strftime("%Y-%m-%d %H:%M:%S") if t else "", bt] 
            for t, bt in zip(times, bodytemps)
        ]
    }
    return stats

# ------------------ Routes that serve HTML templates ------------------
@app.route('/')
def home():
    user = db.session.get(User, session['user_id']) if 'user_id' in session else None
    now = datetime.utcnow()
    heart_rate_stats = None
    user_data_analysis = None

    if user:
        # Load and analyze heart rate data
        hr_data = load_heartrate_data(user.id)
        if hr_data:
            heart_rate_stats = compute_heart_rate_stats(hr_data)
        
        # Load and analyze user JSON data from UserData folder
        user_json_data = load_user_data(user.name)
        if user_json_data:
            user_data_analysis = analyze_user_data(user_json_data)

    return render_template('index.html', 
                           user=user, 
                           now=now, 
                           timedelta=timedelta,
                           heart_rate_stats=heart_rate_stats,
                           user_data_analysis=user_data_analysis)

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

    username = request.form.get('username', '').strip()
    date_of_birth_str = request.form.get('date_of_birth', '').strip()
    schedule_link = request.form.get('schedule_link', '').strip()  # Get the schedule link
    temp_folder = os.path.join(TEMP_FOLDER, username)

    # Check if username is taken
    if User.query.filter_by(name=username).first():
        return jsonify({"success": False, "error": "Username is already taken!"})

    # Validate date_of_birth
    if not date_of_birth_str:
        return jsonify({"success": False, "error": "Date of Birth is required."})
    try:
        date_of_birth = datetime.strptime(date_of_birth_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({"success": False, "error": "Invalid Date of Birth format. Use YYYY-MM-DD."})

    # Validate schedule link
    if not schedule_link:
        return jsonify({"success": False, "error": "Schedule link is required."})

    # Validate images
    image_files = os.listdir(temp_folder) if os.path.exists(temp_folder) else []
    if len(image_files) < 5:
        return jsonify({"success": False, "error": "You must capture at least 5 images!"})

    # Create new user with the schedule link
    new_user = User(name=username, date_of_birth=date_of_birth, schedule_link=schedule_link)
    db.session.add(new_user)
    db.session.commit()

    user_folder = os.path.join(UPLOAD_FOLDER, str(new_user.id))
    os.makedirs(user_folder, exist_ok=True)

    # Move temp images to permanent user folder
    image_paths = []
    for i, image_file in enumerate(sorted(image_files)):
        new_filename = f"user_{new_user.id}_img{i+1}.png"
        old_path = os.path.join(temp_folder, image_file)
        new_path = os.path.join(user_folder, new_filename)
        shutil.move(old_path, new_path)
        image_paths.append(new_filename)

    new_user.image1, new_user.image2, new_user.image3, new_user.image4, new_user.image5 = image_paths
    db.session.commit()

    # Clean up temp folder
    shutil.rmtree(temp_folder, ignore_errors=True)

    # Retrain model with new user
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

# ------------------ Run Flask ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

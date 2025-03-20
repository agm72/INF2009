import os
import cv2
import time
import shutil
import face_recognition
import pickle
import numpy as np
import json
import pandas as pd
from imutils import paths
from flask import Flask, request, render_template, jsonify, session, send_from_directory, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = "supersecretkey"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

UPLOAD_FOLDER = '/home/fuzzi/WebInterface/Users'
TEMP_FOLDER = '/home/fuzzi/WebInterface/temp'
USERDATA_FOLDER = 'UserData'
ENCODINGS_FILE = "encodings.pickle"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(USERDATA_FOLDER, exist_ok=True)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    schedule_link = db.Column(db.String(300), nullable=False)
    image1 = db.Column(db.String(200))
    image2 = db.Column(db.String(200))
    image3 = db.Column(db.String(200))
    image4 = db.Column(db.String(200))
    image5 = db.Column(db.String(200))

with app.app_context():
    db.create_all()

def train_model():
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

def load_user_data(username):
    file_path = os.path.join(USERDATA_FOLDER, f"{username}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    return None

def analyze_with_pandas(data):
    if not data or "records" not in data:
        return None
    records = data["records"]
    df = pd.DataFrame(records)

    for col in ["heartrate", "temperature", "humidity", "bodytemp", "time_recorded"]:
        if col not in df.columns:
            df[col] = None

    df["heartrate"] = df["heartrate"].astype(str).str.replace(" bpm", "", regex=False)
    df["humidity"] = df["humidity"].astype(str).str.replace("%", "", regex=False)
    df["heartrate"] = pd.to_numeric(df["heartrate"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
    df["bodytemp"] = pd.to_numeric(df["bodytemp"], errors="coerce")

    df["time_recorded"] = pd.to_datetime(df["time_recorded"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    basic_stats = df[["heartrate", "temperature", "humidity", "bodytemp"]].describe().to_dict()
    df_records = df.to_dict(orient="records")

    return {
        "basic_stats": basic_stats,
        "df_records": df_records
    }

@app.route('/')
def home():
    user = db.session.get(User, session['user_id']) if 'user_id' in session else None
    now = datetime.utcnow()
    pandas_analysis = None

    if user:
        user_json_data = load_user_data(user.name)
        if user_json_data:
            pandas_analysis = analyze_with_pandas(user_json_data)

    return render_template(
        'index.html',
        user=user,
        now=now,
        timedelta=timedelta,
        pandas_analysis=pandas_analysis
    )

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
    schedule_link = request.form.get('schedule_link', '').strip()
    temp_folder = os.path.join(TEMP_FOLDER, username)

    if User.query.filter_by(name=username).first():
        return jsonify({"success": False, "error": "Username is already taken!"})

    if not date_of_birth_str:
        return jsonify({"success": False, "error": "Date of Birth is required."})
    try:
        date_of_birth = datetime.strptime(date_of_birth_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({"success": False, "error": "Invalid Date of Birth format. Use YYYY-MM-DD."})

    if not schedule_link:
        return jsonify({"success": False, "error": "Schedule link is required."})

    image_files = os.listdir(temp_folder) if os.path.exists(temp_folder) else []
    if len(image_files) < 5:
        return jsonify({"success": False, "error": "You must capture at least 5 images!"})

    new_user = User(name=username, date_of_birth=date_of_birth, schedule_link=schedule_link)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

import os
import cv2
import time
import shutil
import face_recognition
import pickle
import numpy as np
from imutils import paths
from flask import Flask, request, render_template, jsonify, session, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from modules.ultrasonic import get_distance
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

# Database Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
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

        # Extract user_id from folder name
        user_id = os.path.basename(os.path.dirname(imagePath))

        # Fetch username from database using user_id
        user = db.session.get(User, int(user_id))
        if user:
            name = user.name
        else:
            print(f"[WARNING] No user found for ID {user_id}, skipping...")
            continue

        # Load image and convert to RGB
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces and encode them
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

    # Load trained encodings
    if not os.path.exists(ENCODINGS_FILE):
        print("[ERROR] No trained model found. Please register first!")
        return None

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.loads(f.read())

    # Initialize camera
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
    # Optional: get user from session so you can pass to template
    user = db.session.get(User, session['user_id']) if 'user_id' in session else None
    return render_template('index.html', user=user)

@app.route('/face-login-page')
def face_login_page():
    return render_template('face_login.html')

@app.route('/register', methods=['GET'])
def register_get():
    return render_template('register.html')


# ------------------ JSON-based routes (POST/logic) ------------------

@app.route('/capture', methods=['POST'])
def capture():
    """Captures an image from the Pi camera and stores it in a temporary folder."""
    username = request.form.get('username')
    if not username:
        return jsonify({"success": False, "error": "Username is required"}), 400

    temp_folder = os.path.join(TEMP_FOLDER, username)
    os.makedirs(temp_folder, exist_ok=True)

    camera = cv2.VideoCapture(0)
    time.sleep(1)  # Allow camera to adjust

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
    """Clears all captured images for a user in the temp folder."""
    username = request.form.get('username')
    temp_folder = os.path.join(TEMP_FOLDER, username)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)

    return jsonify({"success": True, "message": "All images cleared successfully!"})

@app.route('/register', methods=['POST'])
def register_post():
    """Handle the actual registration (JSON response)."""
    if 'user_id' in session:
        return jsonify({"success": False, "error": "Already logged in."})

    username = request.form.get('username', '')
    temp_folder = os.path.join(TEMP_FOLDER, username)

    # Check if username already exists
    if User.query.filter_by(name=username).first():
        return jsonify({"success": False, "error": "Username is already taken!"})

    # Check images
    image_files = os.listdir(temp_folder) if os.path.exists(temp_folder) else []
    if len(image_files) < 5:
        return jsonify({"success": False, "error": "You must capture at least 5 images!"})

    # Create user in database
    new_user = User(name=username)
    db.session.add(new_user)
    db.session.commit()

    # Move images from temp/username -> Users/<user_id>
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

    # Clean up temp folder
    shutil.rmtree(temp_folder, ignore_errors=True)

    # Train model
    train_model()

    return jsonify({
        "success": True,
        "message": "Registration successful! Please log in using face recognition."
    })

@app.route('/face-login', methods=['GET'])
def face_login():
    """Perform face recognition login, return JSON response instead of redirect."""
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
    """Log out by clearing session, return JSON instead of redirect."""
    session.pop('user_id', None)
    return jsonify({"success": True, "message": "You have been logged out."})


# ------------------ File & Sensor Endpoints ------------------

@app.route('/users/<int:user_id>/<path:filename>')
def get_image(user_id, filename):
    """Serve user images stored in /Users/{user_id}/"""
    user_folder = os.path.join(UPLOAD_FOLDER, str(user_id))
    file_path = os.path.join(user_folder, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_from_directory(user_folder, filename)

@app.route('/temp/<username>/<path:filename>')
def get_temp_image(username, filename):
    """Serve temporary images before registration."""
    temp_folder = os.path.join(TEMP_FOLDER, username)
    file_path = os.path.join(temp_folder, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_from_directory(temp_folder, filename)

@app.route('/sensor-data')
def sensor_data():
    """API endpoint to return the distance measured by the ultrasonic sensor."""
    try:
        distance = get_distance()
        if distance == -1:
            return jsonify({"success": False, "error": "Measurement timeout! Check wiring."}), 500
        return jsonify({"success": True, "distance": distance})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return "You must be logged in to view this feed.", 403
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# ------------------ Run Flask ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

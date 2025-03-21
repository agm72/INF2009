import os
import cv2
import time
import pickle
import numpy as np
import json
import face_recognition
import paho.mqtt.client as mqtt
from datetime import date
from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from modules.ultrasonic import get_distance  # your ultrasonic sensor module

# ------------------ Database Setup ------------------
INSTANCE_FOLDER = os.path.join(os.getcwd(), "instance")
os.makedirs(INSTANCE_FOLDER, exist_ok=True)
DATABASE_PATH = os.path.join(INSTANCE_FOLDER, "database.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    date_of_birth = Column(Date, nullable=False)
    schedule_link = Column(String(300), nullable=False)

# ------------------ Face Recognition Setup ------------------
UPLOAD_FOLDER = '/home/fuzzi/WebInterface/Users'
ENCODINGS_FILE = "encodings.pickle"

def recognize_face():
    """
    Use the camera to detect and recognize a face.
    Returns the recognized username if found, otherwise returns None.
    """
    print("[INFO] Loading face recognition model...")
    if not os.path.exists(ENCODINGS_FILE):
        print("[ERROR] No trained model found. Please train the model first!")
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
    recognized_name = None

    while attempts < max_attempts and recognized_name is None:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                recognized_name = data["names"][best_match_index]
                print(f"[INFO] Recognized: {recognized_name}")
                break
        attempts += 1

    cap.release()
    if recognized_name is None:
        print("[INFO] Face not recognized within maximum attempts.")
    return recognized_name

def calculate_age(born):
    """
    Calculate age from the date of birth.
    """
    today = date.today()
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    return age

# ------------------ MQTT and Sensor Integrated Loop ------------------
def main():
    # MQTT client setup
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "Publisher")
    client.connect("localhost", 1883)

    # Parameters for sensor and recognition
    SENSOR_THRESHOLD = 50.0      # centimeters
    SENSOR_CHECK_INTERVAL = 5    # seconds between sensor checks
    FACE_CHECK_INTERVAL = 30     # seconds to wait before re-checking face
    last_face_check = 0

    current_user = None

    print("[INFO] Starting main loop. Continuously publishing MQTT messages.")

    while True:
        # Get distance measurement from ultrasonic sensor
        distance = get_distance()
        if distance == -1:
            print("[ERROR] Sensor error: Check wiring or sensor. Retrying...")
            time.sleep(SENSOR_CHECK_INTERVAL)
            continue

        print(f"[INFO] Measured distance: {distance} cm")

        # If sensor detects a person (distance below threshold)
        if distance <= SENSOR_THRESHOLD:
            current_time = time.time()
            # Run facial recognition if no user has been set yet or after a cooldown period
            if current_user is None or (current_time - last_face_check) > FACE_CHECK_INTERVAL:
                print("[INFO] Running facial recognition to update user data...")
                recognized_name = recognize_face()
                if recognized_name:
                    user = session.query(User).filter_by(name=recognized_name).first()
                    if user:
                        # Update only if the detected user is different from current
                        if current_user is None or (current_user and current_user.name != user.name):
                            print(f"[INFO] Updating current user to: {user.name}")
                            current_user = user
                        else:
                            print("[INFO] Same user detected. No update required.")
                    else:
                        print("[WARN] Recognized user not found in the database. Keeping previous user data.")
                else:
                    print("[INFO] No face recognized during check. Keeping current user data.")
                last_face_check = current_time
        else:
            # Sensor does not detect a person: continue publishing the last known user data
            print("[INFO] No user detected by sensor. Continuing with last known user data.")

        # Publish MQTT message if we have a current user
        if current_user:
            age = calculate_age(current_user.date_of_birth)
            message_data = {
                "username": current_user.name,
                "age": age,
                "schedule_link": current_user.schedule_link
            }
            message = json.dumps(message_data)
            client.publish("face/recognized", message)
            print(f"[INFO] Published MQTT message: {message}")
        else:
            print("[INFO] No current user data available to publish.")

        time.sleep(SENSOR_CHECK_INTERVAL)

if __name__ == "__main__":
    main()

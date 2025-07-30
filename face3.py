import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image
import urllib.request
import csv
import pandas as pd

# === Configuration ===
CAMERA_URL = "http://192.0.0.4:8080/shot.jpg"
KNOWN_FACE_DIR = "known_faces"
DATA_FILE = "face_data.json"
ATTENDANCE_FILE = "attendance_log.csv"

# === Setup folders and files ===
os.makedirs(KNOWN_FACE_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w') as f:
        json.dump({}, f)

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Mobile', 'Time'])

# === Load known faces ===
known_face_encodings = []
known_face_names = []
marked_attendance = set()
recent_unknown_encodings = []

with open(DATA_FILE, 'r') as f:
    face_data = json.load(f)

for name, info in face_data.items():
    path = os.path.join(KNOWN_FACE_DIR, info['image'])
    if os.path.exists(path):
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)

# === Streamlit UI ===
st.set_page_config(layout="centered")
st.title("📷 Mobile Face Recognition Attendance")
status = st.empty()
image_display = st.image([])

# === Functions ===
def log_attendance(name):
    if name in marked_attendance:
        return
    mobile = face_data[name]["mobile"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, mobile, now])
    marked_attendance.add(name)
    status.success(f"{name} marked present at {now}")

def save_new_face(frame, location, encoding):
    for recent_encoding in recent_unknown_encodings:
        if np.linalg.norm(encoding - recent_encoding) < 0.6:
            return
    with st.form("add_new_face", clear_on_submit=True):
        st.warning("Unknown face detected! Please enter details.")
        name = st.text_input("Name")
        mobile = st.text_input("Mobile")
        submitted = st.form_submit_button("Save")
        if submitted and name and mobile:
            filename = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            top, right, bottom, left = location
            face_img = frame[top:bottom, left:right]
            path = os.path.join(KNOWN_FACE_DIR, filename)
            cv2.imwrite(path, face_img)
            face_data[name] = {"mobile": mobile, "image": filename}
            with open(DATA_FILE, "w") as f:
                json.dump(face_data, f, indent=4)
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            recent_unknown_encodings.append(encoding)
            if len(recent_unknown_encodings) > 10:
                recent_unknown_encodings.pop(0)
            st.success(f"{name} added successfully.")

def get_frame():
    try:
        resp = urllib.request.urlopen(CAMERA_URL)
        img_np = np.array(bytearray(resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        status.error("Error accessing camera stream.")
        return None

# === Main Frame Update ===
frame = get_frame()
if frame is not None:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        status.warning("No face detected. Show your face clearly.")
    else:
        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, encoding)
            name = "Unknown"

            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            top, right, bottom, left = location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, top - 20), (right, top), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            if name == "Unknown":
                save_new_face(frame, location, encoding)
            else:
                log_attendance(name)

    img_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_display.image(img_display, channels="RGB")

# === Attendance Table ===
if os.path.exists(ATTENDANCE_FILE):
    st.markdown("## 📝 Attendance Log")
    df = pd.read_csv(ATTENDANCE_FILE)
    df = df.sort_values(by="Time", ascending=False)
    st.dataframe(df)
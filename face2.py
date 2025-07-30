import cv2
import face_recognition
import numpy as np
import os
import json
from tkinter import Tk, Label, simpledialog, messagebox, StringVar #streamlit
from PIL import Image, ImageTk
from datetime import datetime
import urllib.request
import csv

# Constants
CAMERA_URL = "http://192.0.0.4:8080/shot.jpg"  # Replace with your IP Webcam stream URL
KNOWN_FACE_DIR = "known_faces"
DATA_FILE = "face_data.json"
ATTENDANCE_FILE = "attendance_log.csv"

# Ensure required folders/files
os.makedirs(KNOWN_FACE_DIR, exist_ok=True)
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w') as f:
        json.dump({}, f)
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Mobile', 'Time'])

# Load known faces
known_face_encodings = []
known_face_names = []
face_data = {}

with open(DATA_FILE, "r") as f:
    face_data = json.load(f)

for name, info in face_data.items():
    img_path = os.path.join(KNOWN_FACE_DIR, info['image'])
    if os.path.exists(img_path):
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(name)

# GUI Setup
root = Tk()
root.title("Mobile Face Recognition")
label = Label(root)
label.pack()

status_var = StringVar()
status_label = Label(root, textvariable=status_var, font=("Arial", 12))
status_label.pack()

marked_attendance = set()
recent_unknown_encodings = []

def log_attendance(name):
    if name in marked_attendance:
        return
    mobile = face_data[name]['mobile']
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, mobile, now])
    marked_attendance.add(name)
    status_var.set(f"{name} marked present at {now}")

def save_new_face(frame, face_location, face_encoding):
    for recent_encoding in recent_unknown_encodings:
        if np.linalg.norm(face_encoding - recent_encoding) < 0.6:
            return
    name = simpledialog.askstring("New Face", "Enter name:")
    mobile = simpledialog.askstring("New Face", "Enter mobile:")
    if name and mobile:
        filename = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        top, right, bottom, left = face_location
        face_img = frame[top:bottom, left:right]
        path = os.path.join(KNOWN_FACE_DIR, filename)
        cv2.imwrite(path, face_img)
        face_data[name] = {"mobile": mobile, "image": filename}
        with open(DATA_FILE, "w") as f:
            json.dump(face_data, f, indent=4)
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
        messagebox.showinfo("Saved", f"Face data for {name} saved.")
        recent_unknown_encodings.append(face_encoding)
        if len(recent_unknown_encodings) > 10:
            recent_unknown_encodings.pop(0)

def get_frame():
    try:
        resp = urllib.request.urlopen(CAMERA_URL)
        img_np = np.array(bytearray(resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print("Error fetching frame:", e)
        return None

def update_frame():
    frame = get_frame()
    if frame is None:
        status_var.set("Camera not found or network error.")
        root.after(1000, update_frame)
        return

    # No downscaling — use full resolution
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        status_var.set("No face detected. Show face clearly to camera.")
    else:
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            top, right, bottom, left = face_location

            # Draw green box and label above
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, top - 20), (right, top), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            if name == "Unknown":
                save_new_face(frame, (top, right, bottom, left), face_encoding)
            else:
                log_attendance(name)

    # Convert image to display in Tkinter
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    root.after(100, update_frame)

def on_close():
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
update_frame()
root.mainloop()
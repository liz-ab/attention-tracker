import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Webcam
cap = cv2.VideoCapture(0)

# Counters
focused_time = 0
total_time = 0

start_time = time.time()

def get_eye_status(landmarks, w, h):
    # Left eye landmarks
    left = [33, 160, 158, 133, 153, 144]
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in left]

    # Vertical distances
    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))

    # Horizontal distance
    h_dist = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))

    ear = (v1 + v2) / (2.0 * h_dist)

    if ear < 0.2:
        return False  # eyes closed
    return True       # eyes open

def get_head_direction(landmarks, w, h):
    nose = landmarks[1]
    x = nose.x

    if x < 0.4:
        return "left"
    elif x > 0.6:
        return "right"
    else:
        return "center"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "No Face"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            eyes_open = get_eye_status(landmarks, w, h)
            direction = get_head_direction(landmarks, w, h)

            if eyes_open and direction == "center":
                status = "Focused"
                focused_time += 1
            else:
                status = "Distracted"

            total_time += 1

    # Calculate score
    score = 0
    if total_time > 0:
        score = (focused_time / total_time) * 100

    # Display
    cv2.putText(frame, f"Status: {status}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Score: {int(score)}%", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Attention Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
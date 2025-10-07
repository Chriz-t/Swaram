import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Indices for lips in MediaPipe face mesh
LIPS_IDX = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 310, 415
]))

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        
        # Get lip points
        lip_points = []
        for idx in LIPS_IDX:
            pt = landmarks.landmark[idx]
            x, y = int(pt.x * w), int(pt.y * h)
            lip_points.append((x, y))

        # Get bounding box around lips
        x_coords = [p[0] for p in lip_points]
        y_coords = [p[1] for p in lip_points]
        x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
        y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)

        # Crop mouth region
        mouth_roi = frame[y_min:y_max, x_min:x_max]

        # Preprocess: resize, grayscale, normalize
        mouth_processed = cv2.resize(mouth_roi, (112, 112))
        mouth_processed_gray = cv2.cvtColor(mouth_processed, cv2.COLOR_BGR2GRAY)
        mouth_processed_gray = mouth_processed_gray.astype(np.float32) / 255.0  # normalize to [0,1]

        # Show original frame with bounding box
        for pt in lip_points:
            cv2.circle(frame, pt, 1, (0, 255, 0), -1)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Show preprocessed mouth
        cv2.imshow('Mouth Region (Preprocessed)', mouth_processed_gray)
    
    # Show camera feed
    cv2.imshow('Webcam Feed', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#we want to implement an app for reading lip movements from a person who has lost the ability to speak and then in phase 1 we want to get live text on what they are saying in phase two we want live speach give me deatild steps to implement this project also point out the main difficulties and how to overcome them

#give me code for in python opencv to implement preprossing in phase 1 to take input from webcam
import cv2
import numpy as np
import mediapipe as mp
import os

# === User Inputs ===
WORD = input("Enter the word to record: ").strip()
NUM_SAMPLES = int(input("Enter the number of samples: ").strip())

# === Setup output folder ===
OUTPUT_DIR = "face_dataset"
word_dir = os.path.join(OUTPUT_DIR, WORD)
os.makedirs(word_dir, exist_ok=True)

print(f"\n[INFO] Recording word: '{WORD}'")
print(f"[INFO] Total samples to record: {NUM_SAMPLES}")
print("[INFO] Press 'r' to record a sample, 'q' to quit.\n")

# === Mediapipe Face Mesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Lip landmarks for mouth movement
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14

# All lip points (for bounding box visualization)
LIPS_IDX = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
    37, 0, 267, 269, 270, 409, 310, 415
]))

# === Webcam ===
cap = cv2.VideoCapture(0)
sample_count = 0
recording = False
active_recording = False
frames = []
silence_counter = 0
MOUTH_THRESHOLD = 8  # lip distance threshold, adjust if needed

def mouth_distance(landmarks, w, h):
    """Compute vertical distance between upper and lower lips"""
    upper = landmarks.landmark[UPPER_LIP_IDX]
    lower = landmarks.landmark[LOWER_LIP_IDX]
    y1, y2 = int(upper.y * h), int(lower.y * h)
    return abs(y2 - y1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        lip_points = [(int(pt.x * w), int(pt.y * h))
                      for idx, pt in enumerate(landmarks.landmark) if idx in LIPS_IDX]

        if lip_points:
            # bounding box
            x_coords = [p[0] for p in lip_points]
            y_coords = [p[1] for p in lip_points]
            x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
            y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)

            mouth_roi = frame[y_min:y_max, x_min:x_max]

            # preprocess
            mouth_processed = cv2.resize(mouth_roi, (112, 112))
            mouth_processed_gray = cv2.cvtColor(mouth_processed, cv2.COLOR_BGR2GRAY)
            mouth_processed_gray = mouth_processed_gray.astype(np.float32) / 255.0

            cv2.imshow('Mouth Region (Preprocessed)', mouth_processed_gray)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # --- Mouth movement detection ---
            dist = mouth_distance(landmarks, w, h)

            if recording:
                if dist > MOUTH_THRESHOLD:
                    # Mouth moved -> record frames
                    active_recording = True
                    frames.append(mouth_processed_gray)
                    silence_counter = 0
                elif active_recording:
                    # Mouth closed -> wait few frames then stop
                    silence_counter += 1
                    if silence_counter > 5:  # stop after ~5 closed frames
                        sample_count += 1
                        filename = os.path.join(word_dir, f"sample_{sample_count:03d}.npy")
                        np.save(filename, np.array(frames))
                        print(f"[SAVED] {filename} ({sample_count}/{NUM_SAMPLES})")

                        # reset
                        frames = []
                        recording = False
                        active_recording = False
                        silence_counter = 0

                        if sample_count >= NUM_SAMPLES:
                            print("\n[INFO] Finished recording all samples.")
                            break

    cv2.imshow('Webcam Feed', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and not recording and sample_count < NUM_SAMPLES:
        print(f"[READY] Waiting for mouth movement for sample {sample_count+1}/{NUM_SAMPLES}...")
        recording = True
        active_recording = False
        frames = []
        silence_counter = 0

cap.release()
cv2.destroyAllWindows()

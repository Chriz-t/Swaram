import cv2
import numpy as np
import mediapipe as mp
import os

# === User Inputs ===
WORD = input("Enter the word to record: ").strip()
NUM_SAMPLES = int(input("Enter the number of samples: ").strip())
SEQUENCE_LENGTH = 30   # frames per sample

# === Setup output folder ===
OUTPUT_DIR = "face_dataset"
word_dir = os.path.join(OUTPUT_DIR, WORD)
os.makedirs(word_dir, exist_ok=True)

print(f"\n[INFO] Recording word: '{WORD}'")
print(f"[INFO] Each sample = {SEQUENCE_LENGTH} frames")
print(f"[INFO] Total samples to record: {NUM_SAMPLES}")
print("[INFO] Press 'r' to record a sample, 'q' to quit.\n")

# === Mediapipe Face Mesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

LIPS_IDX = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
    37, 0, 267, 269, 270, 409, 310, 415
]))

# === Webcam ===
cap = cv2.VideoCapture(0)
sample_count = 0
recording = False
frames = []

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
            x_coords = [p[0] for p in lip_points]
            y_coords = [p[1] for p in lip_points]
            x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
            y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)

            mouth_roi = frame[y_min:y_max, x_min:x_max]

            # Preprocess
            mouth_processed = cv2.resize(mouth_roi, (112, 112))
            mouth_processed_gray = cv2.cvtColor(mouth_processed, cv2.COLOR_BGR2GRAY)
            mouth_processed_gray = mouth_processed_gray.astype(np.float32) / 255.0

            # Show preprocessed mouth
            cv2.imshow('Mouth Region (Preprocessed)', mouth_processed_gray)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # === If recording ===
            if recording:
                frames.append(mouth_processed_gray)
                if len(frames) == SEQUENCE_LENGTH:
                    sample_count += 1
                    filename = os.path.join(word_dir, f"sample_{sample_count:03d}.npy")
                    np.save(filename, np.array(frames))
                    print(f"[SAVED] {filename} ({sample_count}/{NUM_SAMPLES})")

                    frames = []
                    recording = False

                    if sample_count >= NUM_SAMPLES:
                        print("\n[INFO] Finished recording all samples.")
                        break

    cv2.imshow('Webcam Feed', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and not recording and sample_count < NUM_SAMPLES:
        print(f"[RECORDING] Sample {sample_count+1}/{NUM_SAMPLES}...")
        recording = True
        frames = []

cap.release()
cv2.destroyAllWindows()

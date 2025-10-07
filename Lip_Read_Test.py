import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import os

# =====================
# 1. Model Definition (must match training)
# =====================
class LipReadingModel(nn.Module):
    def __init__(self, num_classes):
        super(LipReadingModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten_dim = 128 * 14 * 14
        self.lstm = nn.LSTM(input_size=self.flatten_dim, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        final_out = lstm_out[:, -1, :]
        out = self.fc(final_out)
        return out

# =====================
# 2. Load Model
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update this path if needed
DATASET_DIR = r"C:\Users\HP\Desktop\MLProject\face_dataset"
MODEL_PATH = r"C:\Users\HP\lip_reading_model.pth"
CLASSES = sorted(os.listdir(DATASET_DIR))  # same as training class names

num_classes = len(CLASSES)
model = LipReadingModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded. Ready for live testing!")

# =====================
# 3. Mediapipe Setup
# =====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

LIPS_IDX = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
    37, 0, 267, 269, 270, 409, 310, 415
]))

# =====================
# 4. Webcam Loop
# =====================
SEQUENCE_LENGTH = 30
frames = []

cap = cv2.VideoCapture(0)

print("\n🎤 Speak a word (each prediction uses 30 frames)...")
print("Press 'q' to quit.\n")

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
            mouth_processed = cv2.resize(mouth_roi, (112, 112))
            mouth_processed_gray = cv2.cvtColor(mouth_processed, cv2.COLOR_BGR2GRAY)
            mouth_processed_gray = mouth_processed_gray.astype(np.float32) / 255.0

            frames.append(mouth_processed_gray)

            # draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # When we have enough frames -> predict
            if len(frames) == SEQUENCE_LENGTH:
                input_seq = np.expand_dims(frames, axis=1)  # (30, 1, 112, 112)
                input_seq = np.expand_dims(input_seq, axis=0)  # (1, 30, 1, 112, 112)
                input_seq = torch.tensor(input_seq, dtype=torch.float32).to(device)

                with torch.no_grad():
                    outputs = model(input_seq)
                    pred = torch.argmax(outputs, dim=1).item()
                    predicted_word = CLASSES[pred]

                print(f"📝 Predicted: {predicted_word}")
                cv2.putText(frame, f"Predicted: {predicted_word}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                frames = []  # reset for next word

    cv2.imshow("Live Lip Reading", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

# =========================
# Haar Cascade
# =========================
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# =========================
# PyTorch CNN Model
# =========================
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),

            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),

            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),

            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =========================
# Device & Model Load
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNN().to(device)
model.load_state_dict(torch.load("emotiondetector.pth", map_location=device))
model.eval()

# =========================
# Labels
# =========================
labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# =========================
# Image Preprocessing
# =========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

def extract_features(face_img):
    img = Image.fromarray(face_img)
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    return img

# =========================
# Webcam Emotion Detection
# =========================
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # FIX mirror image

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_tensor = extract_features(face)

        with torch.no_grad():
            output = model(face_tensor)
            pred = torch.argmax(output, dim=1).item()

        emotion = labels[pred]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            2,
            (0, 0, 255),
            2
        )

    cv2.imshow("Emotion Detection (PyTorch)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()

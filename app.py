import torch
from torchvision import transforms, models, datasets
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_dir = "dataset"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model.load_state_dict(torch.load("face_recognition_model.pth"))
model.eval()
model = model.to(device)

def predict_face(face_image):
    input_tensor = transform(face_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_idx = predicted.item()
    return dataset.classes[class_idx], confidence.item()

def detect_faces_and_classify(image_path, output_path="result.jpg"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).convert("RGB")

        label, confidence = predict_face(face_pil)

        label_with_confidence = f"{label} ({confidence:.2f})"

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(image, label_with_confidence, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imwrite(output_path, image)
    print(f"Result saved to {output_path}")

test_image_path = "test5.jpg"
detect_faces_and_classify(test_image_path)

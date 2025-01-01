import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
    
    return dataset.classes[class_idx], image

def draw_label(image, label, coordinates=(50, 50)):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(cv_image, label, coordinates, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return cv_image

def detect_faces_and_draw(image_path, label):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    image = draw_label(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), label, coordinates=(10, 30))

    cv2.imwrite("result.jpg", image)

test_image_path = "test5.jpg"

predicted_label, image = predict(test_image_path)

detect_faces_and_draw(test_image_path, predicted_label)

print(f"Classe prevista: {predicted_label}")

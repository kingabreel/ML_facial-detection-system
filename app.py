import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
    
    return dataset.classes[class_idx]

test_image_path = "test4.jpg"
result = predict(test_image_path)
print(f"Classe prevista: {result}")

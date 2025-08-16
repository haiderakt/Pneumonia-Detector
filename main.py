import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)
model.load_state_dict(torch.load("/model/best_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ["NORMAL", "PNEUMONIA"]

def predict(image):
    image = Image.fromarray(image).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    
    result = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Chest X-ray"),
    outputs=gr.Label(num_top_classes=2, label="Predictions"),
    title="Pneumonia Detector",
    description="Upload a chest X-ray image to test the trained model."
)

if __name__ == "__main__":
    demo.launch()

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import cv2

class EmotionDetector(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.backbone = mobilenet_v2(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class EmotionInference:
    def __init__(self, model_path, num_classes=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = EmotionDetector(num_classes=num_classes).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.prev_probs = None
        self.alpha = 0.6
        self.risk_weights = {
                "angry": 0.9,
                "fear": 0.7,
                "sad": 0.4,
                "neutral": 0.1,
                "happy": 0.05
            }

        self.classes = ["angry", "fear", "happy", "neutral", "sad"]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        img_tensor = self.transform(face_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            if self.prev_probs is None:
                self.prev_probs = probs
            else:
                self.prev_probs = self.alpha * probs + (1-self.alpha)*self.prev_probs
            pred = torch.argmax(self.prev_probs, dim=1).item()
            emotion_label = self.classes[pred]
            confidence = self.prev_probs[0][pred].item()

            risk_score = 0.0
            for i, cls in enumerate(self.classes):
                prob = self.prev_probs[0][i].item()
                risk_score += prob * self.risk_weights[cls]

            risk_score = max(0.0, min(1.0, risk_score))

            return emotion_label, confidence, risk_score
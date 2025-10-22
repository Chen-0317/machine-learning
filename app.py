import gradio as gr
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms, models
from pyngrok import ngrok
from config import *

# ============================================================
# 初始化 Ngrok
# ============================================================
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# ============================================================
# 模型載入
# ============================================================
if MODEL_NAME == "resnet34":
    model = models.resnet34(weights=None)
else:
    raise ValueError(f"不支援的模型：{MODEL_NAME}")

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ============================================================
# 前處理
# ============================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ============================================================
# 預測函式
# ============================================================
def predict(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    if confidence < THRESHOLD:
        return {"無法辨識，要不換一張?": 1.0}
    else:
        return {CLASS_NAMES[0]: float(probs[0]), CLASS_NAMES[1]: float(probs[1])}

# ============================================================
# Gradio 介面
# ============================================================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="numpy", 
        label="上傳圖片（貓或狗）",
        sources=["upload", "camera"],
        tool="editor"
    ),
    outputs=gr.Label(num_top_classes=2, label="分類結果"),
    title="🐱🐶 貓狗影像分類模型",
    description="上傳一張圖片，模型會預測是『貓』還是『狗』，並顯示預測結果。",
)

# ============================================================
# 啟動伺服器
# ============================================================
public_url = ngrok.connect(NGROK_PORT)
print("🌍 外部可訪問網址：", public_url)
demo.launch(debug=True, server_name="0.0.0.0", server_port=NGROK_PORT, share=False)

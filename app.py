import gradio as gr
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms, models
import os
from pyngrok import ngrok

ngrok.set_auth_token("34MmYmTrffRXpFTIlhtG8e692ta_6NccwSiUw6QcoMpxkg3u6")

MODEL_PATH = "best_model.pth"   # 模型路徑
THRESHOLD = 0.9                 # 信心指數
CLASSES = ["Cat", "Dog"]        # 分類標籤

# ============================================================
# 模型載入 (使用你訓練好的權重)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet34(weights=None)  # 不再使用預訓練
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # 這裡要放你儲存的最佳模型
model = model.to(device)
model.eval()

# ============================================================
# 前處理 Transform (要和訓練時一致)
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
    # 轉成灰階再轉回 3 channels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Apply transform
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    # ✅ 新增條件判斷
    if confidence < THRESHOLD:
        return {"無法辨識，要不換一張?": 1.0}  # Gradio 需要 dict，隨便給 1.0
    else:
        return {CLASSES[0]: float(probs[0]), CLASSES[1]: float(probs[1])}

# ============================================================
# 建立 Gradio 介面
# ============================================================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="numpy", 
        label="上傳圖片（貓或狗）",
        sources=["upload", "camera"],  # 同時支援上傳與拍照
        tool="editor"
    ),
    outputs=gr.Label(num_top_classes=2, label="分類結果"),
    title="🐱🐶 貓狗影像分類模型",
    description="上傳一張圖片，模型會預測是『貓』還是『狗』，並顯示預測結果。",
)

# 建立 ngrok 通道
public_url = ngrok.connect(7860)
print("🌍 外部可訪問網址：", public_url)

# demo.launch(debug=True, share=True)
demo.launch(debug=True, server_name="0.0.0.0", server_port=7860, share=False)

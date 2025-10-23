import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms, models
from config import *  # 匯入集中設定
import argparse

# ============================================================
# 命令列參數設定
# ============================================================
parser = argparse.ArgumentParser(description="Predict Cat vs Dog Images")
parser.add_argument("--image", type=str, help="Path to a single image for prediction")
parser.add_argument("--folder", type=str, help="Path to a folder containing multiple images")
args = parser.parse_args()

# ============================================================
# 檢查模型是否存在
# ============================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到模型檔案：{MODEL_PATH}\n請先執行 train.py 訓練模型。")

# ============================================================
# 模型載入
# ============================================================
print(f"🧠 載入模型：{MODEL_PATH}")
device = DEVICE

model = models.resnet34(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ============================================================
# 前處理 Transform（需與訓練一致）
# ============================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ============================================================
# 單張圖片預測函式
# ============================================================
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"⚠️ 找不到圖片：{img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 無法讀取圖片：{img_path}")
        return

    # 轉灰階→3通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    if confidence < THRESHOLD:
        print(f"🟡 圖片：{os.path.basename(img_path)} → 無法辨識（信心 {confidence:.2f}）")
    else:
        print(f"✅ 圖片：{os.path.basename(img_path)} → {CLASS_NAMES[pred_idx]}（信心 {confidence:.2f}）")

# ============================================================
# 批次或單張預測
# ============================================================
if args.image:
    predict_image(args.image)
elif args.folder:
    image_files = [f for f in os.listdir(args.folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        print("⚠️ 該資料夾內沒有圖片。")
    else:
        for img_name in image_files:
            predict_image(os.path.join(args.folder, img_name))
else:
    print("⚠️ 請使用 --image 或 --folder 參數指定要預測的圖片。")

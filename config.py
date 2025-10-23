import torch

# ============================================================
# 環境設定

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 資料與模型路徑

DATA_DIR = "data/train_grayscale"  # 訓練資料資料夾
MODEL_PATH = "best_model.pth"      # 訓練後的模型檔案
MODEL_NAME = "resnet34"

# ============================================================
# 訓練設定

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2
CLASS_NAMES = ["Cat", "Dog"]

# ============================================================
# 推論設定

THRESHOLD = 0.9  # 信心指數（預測用）

# ============================================================
# Ngrok / Gradio 設定

NGROK_AUTH_TOKEN = "34MmYmTrffRXpFTIlhtG8e692ta_6NccwSiUw6QcoMpxkg3u6"
NGROK_PORT = 7860
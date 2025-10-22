# train.py
import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from config import *  # ✅ 匯入集中設定

# ============================================================
# 檢查裝置
# ============================================================
print(f"使用裝置：{DEVICE}")

# ============================================================
# 自訂 Dataset
# ============================================================
class CatsDogsDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.cat_files = glob.glob(os.path.join(folder_path, 'gray_cat.*.jpg'))
        self.dog_files = glob.glob(os.path.join(folder_path, 'gray_dog.*.jpg'))
        self.all_files = self.cat_files + self.dog_files
        self.labels = [0] * len(self.cat_files) + [1] * len(self.dog_files)
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label

# ============================================================
# Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ============================================================
# Dataset & DataLoader
# ============================================================
full_dataset = CatsDogsDataset(TRAIN_DATA_PATH, transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ============================================================
# 模型設定 (ResNet34 Fine-tuning)
# ============================================================
resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
for param in resnet34.parameters():
    param.requires_grad = False
for name, param in resnet34.named_parameters():
    if "layer3" in name:
        param.requires_grad = True

num_ftrs = resnet34.fc.in_features
resnet34.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = resnet34.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ============================================================
# 訓練 + Early Stopping
# ============================================================
best_val_loss = float("inf")
trigger_times = 0
best_model_state = None
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # 驗證階段
    model.eval()
    val_loss, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    acc = accuracy_score(y_true, y_pred)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")

    # LR Scheduler
    scheduler.step(val_loss)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= PATIENCE:
            print("⏹️ Early stopping triggered")
            break

# ============================================================
# 儲存最佳模型
# ============================================================
if best_model_state:
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, MODEL_PATH)
    print(f"✅ 模型已儲存至：{MODEL_PATH}")

# ============================================================
# 報告與曲線
# ============================================================
print("\n✅ 分類報告：")
print(classification_report(y_true, y_pred))
print("✅ 最終準確率：", accuracy_score(y_true, y_pred))

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (ResNet34 + Early Stopping)")
plt.legend()
plt.show()

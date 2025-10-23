# 🐾 貓狗影像分類模型 (Cat vs Dog Image Classifier)

使用 PyTorch + Gradio 建立的貓狗影像分類器。  
使用者可以上傳圖片（或使用相機拍照），即時辨識圖中是 貓 還是 狗

---

## 🚀 專案特色

- 使用 ResNet34 作為主幹模型
- 提供 Gradio 互動介面，可本地運行或 Ngrok 外網訪問
- 支援上傳與相機拍照圖片
- 模型權重與參數集中管理（`config.py`）
- 推論速度快，部署簡易

---

## 📂 專案結構
```
📦 cat-dog-classifier/
│
├─ app.py             # Gradio Web 介面
├─ train.py           # 模型訓練（可帶參數）
├─ config.py          # 各種設定集中管理
├─ predict.py         # 終端機輸入圖片 → 輸出預測結果
├─ requirements.txt   # 套件需求
├─ best_model.pth     # 訓練好的模型 (請見下方下載說明)
├─ README.md          # 說明文件
└─ data               # 放訓練圖片

```

---

## 📦 模型下載說明

> ⚠️ 由於 GitHub 對檔案大小有限制（100MB），請手動下載模型檔。

- 📥 下載模型權重：https://drive.google.com/file/d/1J50pIQMn2UwX8xD-hbDsaSjOTG4Q4VoS/view?usp=drive_link
- 下載後請放置於專案根目錄，命名為：

```bash
best_model.pth
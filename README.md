# 🐾 貓狗影像分類模型 (Cat vs Dog Image Classifier)

使用 PyTorch + Gradio 建立的貓狗影像分類器。  
使用者可以上傳圖片（或使用相機拍照），即時辨識圖中是 貓 還是 狗

---

## 🚀 專案特色

- 使用 ResNet34 作為主要模型
- 提供 Gradio 互動介面，可本地運行或 Ngrok 外網訪問
- 支援上傳與相機拍照圖片
- 模型權重與參數集中管理（`config.py`）
- 推論速度快，部署簡易
- 支援自訂訓練資料與模型參數
- 
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

## ⚙️ 安裝環境

請先確認已安裝 **Python 3.8 以上版本**。  
然後依序執行以下命令：

```bash
git clone https://github.com/你的帳號/cat-dog-classifier.git
cd cat-dog-classifier
pip install -r requirements.txt
```

---

## 🧠 模型訓練

訓練時會自動偵測 GPU 或 CPU。  
輸入以下命令開始訓練：

```bash
python train.py
```

若要自訂參數（例如訓練週期、批次大小）：

```bash
python train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

---

## 📦 模型下載說明

> ⚠️ 由於 GitHub 對檔案大小有限制（100 MB），請手動下載模型檔案。

- 📥 **下載模型權重：**  
  👉 [best_model.pth（Google Drive）](https://drive.google.com/file/d/1J50pIQMn2UwX8xD-hbDsaSjOTG4Q4VoS/view?usp=drive_link)

- 下載後請將模型放置於專案根目錄，命名為：

```bash
best_model.pth
```

---

## 🐱🐶 訓練圖片下載說明

### 📥 方式一：從 Kaggle 下載  
1. 前往 Kaggle 官方資料集：  
   👉 [Dogs vs Cats Dataset (Kaggle)](https://www.kaggle.com/c/dogs-vs-cats/data)
2. 登入後點選 **Download** 下載 `train.zip`
3. 解壓縮後，將圖片放入下列資料夾：

```bash
cat-dog-classifier/data/
```

---

### ☁️ 方式二：從雲端下載（Google Drive）

若無法從 Kaggle 下載，可使用雲端共享資料：

- 📦 [Google Drive 資料夾下載連結](https://drive.google.com/drive/folders/1gH0nfG70_9LedWBs3lo5dxLG4soAc4bx?usp=sharing)

解壓縮後的結構如下：

```
data/
├─ cat.0.jpg
├─ cat.1.jpg
├─ dog.0.jpg
└─ dog.1.jpg
```

---

✅ **完成後目錄結構應如下：**

```
📦 cat-dog-classifier/
├─ best_model.pth
├─ train.py
├─ predict.py
├─ config.py
├─ app.py
├─ requirements.txt
├─ README.md
└─ data/
   ├─ cat.0.jpg
   ├─ dog.0.jpg
   └─ ...
```

---

## 🔍 預測圖片

使用命令列進行預測：

```bash
python predict.py --image_path data/dog.1.jpg
```

或啟動 Gradio 介面：

```bash
python app.py
```

執行後會在瀏覽器中開啟互動式頁面，可直接上傳圖片預測。

---
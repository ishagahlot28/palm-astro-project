# 🖐️ Palm Segmentation using Deep Learning

## 📘 Overview
This project is based on palm image segmentation using Python and deep learning.
It trains a U-Net model (with EfficientNet encoder) to detect and highlight the palm region from input images.

The project fulfills all assignment requirements for **Python Test 1 — Palm History Project**.

---

## 🚀 Features
- Organizes data (images and masks)
- Automatically generates dummy masks if missing
- Trains a segmentation model using PyTorch
- Saves checkpoints and best model
- Performs inference on palm images
- Displays and saves visual predictions

---

## 🧩 Folder Structure
palm-astro-project/
┣ 📂 data/
│ ┣ 📂 images/train/
│ ┗ 📂 masks/train/
┣ 📂 models/
│ ┗ 📜 best_model.pth
┣ 📂 output/
│ ┗ 📜 predicted_palm.png
┣ 📜 train.py
┣ 📜 inference.py
┣ 📜 make_masks.py
┣ 📜 check_data.py
┣ 📜 requirements.txt
┗ 📜 README.md


---

## ⚙️ Setup Instructions

### 1️⃣ Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt


<img width="1920" height="1080" alt="Screenshot (68)" src="https://github.com/user-attachments/assets/902f9f8e-d24e-46c4-9756-95114dcd6888" /># 🖐️ Palm Segmentation using Deep Learning

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
2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Prepare Dataset

Add your palm images to:

data/images/train/


If masks are missing, run:

python make_masks.py
4️⃣ Train the Model
python train.py --data_dir data --split train --epochs 2 --batch_size 1 --save_dir models

5️⃣ Run Inference / Prediction
python inference.py


Libraries Used

torch

torchvision

torchaudio

segmentation-models-pytorch

pillow

tqdm

matplotlib

numpy

🏁 Results
Input Palm Image	Predicted Palm Region




✅ Successfully trained and tested the palm segmentation model.
The project demonstrates model training, prediction, and visualization — fulfilling all assignment requirements.













pip install -r requirements.txt




# 🖐️ Palm Astro Project — Palm Image Segmentation using U-Net  

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 📘 Overview
This project performs **palm image segmentation** using a **U-Net deep learning model** built in **PyTorch**.  
The model takes a hand image and predicts the palm region mask — useful for:

- Palmistry analysis  
- Biometrics  
- Gesture recognition  
- Hand shape detection  

---

## 📂 Project Structure


palm-astro-project/
├── data/
│ ├── images/ # input palm images (train/val)
│ ├── masks/ # segmentation masks
├── models/ # trained model weights (.pth)
├── output/ # prediction outputs
├── utils/ # helper scripts
├── train.py # training script
├── inference.py # inference and visualization
├── make_masks.py # generate masks if missing
├── make_small_dataset.py # create 200-image dataset
├── requirements.txt # python dependencies
└── README.md

## 🧠 Model Details
- **Architecture:** U-Net  
- **Encoder:** EfficientNet-B0 (ImageNet pretrained)  
- **Loss Function:** Binary Cross Entropy  
- **Optimizer:** Adam  
- **Epochs Trained:** 10  
- **Batch Size:** 2  
- **Framework:** PyTorch + segmentation-models-pytorch  

---

## 📦 Dataset
Dataset used: **Human Palm Images** (from Kaggle)  
Download manually OR via CLI:



kaggle datasets download -d feyiamujo/human-palm-images


Unzip:



unzip human-palm-images.zip -d data/images


To create a smaller 200-image dataset:



python make_small_dataset.py


---

## ⚙️ Installation



git clone https://github.com/ishagahlot28/palm-astro-project.git

cd palm-astro-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


---

## 🚀 Training



python train.py --data_dir data --split small_train --epochs 10 --batch_size 2 --save_dir models


Model will be saved in:



models/best_model.pth


---

## 🧪 Inference



python inference.py


This will generate visualization similar to:

### Input vs Predicted Output
| Input Palm Image | Predicted Palm Region |
|------------------|----------------------|
| *(your input image)* | *(U-Net segmented palm mask)* |

---

## 📊 Sample Training Log



Epoch 1 - Avg Loss: 0.6389
Epoch 5 - Avg Loss: 0.1296
Epoch 10 - Avg Loss: 0.0390
Training complete — model saved to models/


---

## 📁 Outputs
- ✔ `best_model.pth` — final model  
- ✔ `checkpoint_epoch_*.pth` — intermediate  
- ✔ `output/` — visual segmentation results  

---

## 🧰 Tech Stack

- Python 3.10+  
- PyTorch  
- segmentation-models-pytorch  
- NumPy  
- Pillow  
- OpenCV  
- Matplotlib  
- TQDM  



---

## ✨ Author
**Isha Gahlot**  
🔗 GitHub: https://github.com/ishagahlot28  
📅 November 2025  
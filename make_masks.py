import os
from PIL import Image

# Paths
image_dir = "data/images"
train_img_dir = "data/images/train"
mask_dir = "data/masks/train"

# Create folders if missing
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# Move images into train folder
for f in os.listdir(image_dir):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        src = os.path.join(image_dir, f)
        dst = os.path.join(train_img_dir, f)
        if not os.path.exists(dst):
            os.rename(src, dst)
            print(f"📦 Moved {f} → train folder")

# Create dummy (black) masks for each image
for img_name in os.listdir(train_img_dir):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        mask_path = os.path.join(mask_dir, os.path.splitext(img_name)[0] + ".png")
        if not os.path.exists(mask_path):
            img_path = os.path.join(train_img_dir, img_name)
            img = Image.open(img_path)
            blank = Image.new("RGB", img.size, (0, 0, 0))
            blank.save(mask_path)
            print(f"✅ Created dummy mask for: {img_name}")

print("\n🎉 All images moved to data/images/train/")
print("🎨 Dummy masks created in data/masks/train/")

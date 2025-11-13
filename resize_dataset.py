# resize_dataset.py
from PIL import Image
import os

img_dir = "data/images/small_train"
mask_dir = "data/masks/small_train"

target_size = (256, 256)

def resize_all(folder):
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, f)
            try:
                img = Image.open(path)
                img = img.resize(target_size)
                img.save(path)
            except Exception as e:
                print("⚠️ Skipped:", f, "-", e)

print("🖼️ Resizing images...")
resize_all(img_dir)
print("🎭 Resizing masks...")
resize_all(mask_dir)
print("✅ Done resizing all images and masks to 256x256!")

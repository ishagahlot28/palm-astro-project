# make_small_dataset.py
import os
import shutil

src_img = "data/images/train"
src_mask = "data/masks/train"
dst_img = "data/small_train"
dst_mask = "data/small_masks"

os.makedirs(dst_img, exist_ok=True)
os.makedirs(dst_mask, exist_ok=True)

# collect images (sorted) and take first 200
imgs = sorted([f for f in os.listdir(src_img) if f.lower().endswith(('.jpg','.jpeg','.png'))])[:200]

copied = 0
for f in imgs:
    src_img_path = os.path.join(src_img, f)
    dst_img_path = os.path.join(dst_img, f)
    shutil.copy2(src_img_path, dst_img_path)

    mask_name = os.path.splitext(f)[0] + ".png"
    src_mask_path = os.path.join(src_mask, mask_name)
    dst_mask_path = os.path.join(dst_mask, mask_name)
    if os.path.exists(src_mask_path):
        shutil.copy2(src_mask_path, dst_mask_path)
    else:
        # create black mask if missing
        from PIL import Image
        img = Image.open(src_img_path)
        blank = Image.new("RGB", img.size, (0,0,0))
        blank.save(dst_mask_path)
    copied += 1

print(f"✅ Copied {copied} images and masks to:")
print(f"   images -> {dst_img}")
print(f"   masks  -> {dst_mask}")

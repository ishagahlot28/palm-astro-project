import os
from PIL import Image

# Paths
train_dir = "data/images/train"
mask_dir = "data/masks/train"

# Make sure directories exist
os.makedirs(mask_dir, exist_ok=True)

# Step 1 — Remove corrupted images
bad_files = []
for f in os.listdir(train_dir):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(train_dir, f)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            bad_files.append(f)
            os.remove(path)
            print(f"❌ Removed corrupted file: {f}")

print(f"\n✅ Step 1 done — removed {len(bad_files)} bad files.")

# Step 2 — Create dummy black masks
from PIL import Image

for f in os.listdir(train_dir):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(train_dir, f)
        mask_path = os.path.join(mask_dir, os.path.splitext(f)[0] + ".png")
        if not os.path.exists(mask_path):
            img = Image.open(img_path)
            blank = Image.new("RGB", img.size, (0, 0, 0))
            blank.save(mask_path)
            print(f"✅ Created mask for: {f}")

# Step 3 — Verify counts
img_files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"\n🖼️ Total images: {len(img_files)}")
print(f"🎭 Total masks : {len(mask_files)}")

if len(img_files) == len(mask_files):
    print("✅ All good — image and mask counts match!\n")
else:
    print("⚠️ Mismatch detected — rerun make_masks.py or check folders.\n")

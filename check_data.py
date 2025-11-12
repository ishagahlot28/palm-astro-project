import os

img_dir = "data/images/train"
mask_dir = "data/masks/train"

img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"🖼️ Images found: {len(img_files)}")
print(f"🎭 Masks found: {len(mask_files)}\n")

img_set = set(os.path.splitext(f)[0] for f in img_files)
mask_set = set(os.path.splitext(f)[0] for f in mask_files)

missing_masks = img_set - mask_set
missing_imgs = mask_set - img_set

if missing_masks:
    print("⚠️ Missing masks for:", missing_masks)
if missing_imgs:
    print("⚠️ Missing images for:", missing_imgs)

if not missing_masks and not missing_imgs:
    print("✅ All images and masks are perfectly matched!")

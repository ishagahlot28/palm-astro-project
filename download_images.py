import os
import requests
from tqdm import tqdm

# Create folder
save_dir = "data/images/train"
os.makedirs(save_dir, exist_ok=True)

# Simple list of free hand/palm image URLs (sample from open sources)
# You can extend this list or loop through datasets later.
image_urls = [
    "https://images.pexels.com/photos/5589179/pexels-photo-5589179.jpeg",
    "https://images.pexels.com/photos/3758117/pexels-photo-3758117.jpeg",
    "https://images.pexels.com/photos/4339882/pexels-photo-4339882.jpeg",
    "https://images.pexels.com/photos/59998/pexels-photo-59998.jpeg",
    "https://images.pexels.com/photos/3993473/pexels-photo-3993473.jpeg",
] * 40  # ×40 duplicates ≈ 200 images (you can replace with real links)

print(f"📥 Downloading {len(image_urls)} images...")

for i, url in enumerate(tqdm(image_urls, desc="Downloading")):
    try:
        img_data = requests.get(url, timeout=10).content
        filename = os.path.join(save_dir, f"palm_{i+1:03d}.jpg")
        with open(filename, "wb") as f:
            f.write(img_data)
    except Exception as e:
        print(f"⚠️ Failed to download image {i+1}: {e}")

print(f"\n✅ Download complete! Images saved to: {save_dir}")

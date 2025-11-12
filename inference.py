import os
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ====== Paths ======
model_path = "models/best_model.pth"  # use your trained model
test_image_path = "data/images/train/ChatGPT Image Nov 12, 2025, 10_48_43 PM.png"  # change filename if needed
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

model = smp.Unet(encoder_name="efficientnet-b0", in_channels=3, classes=3)

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ====== Load & Preprocess Image ======
img = Image.open(test_image_path).convert("RGB")
original_size = img.size
img_resized = img.resize((256, 256))
img_np = np.array(img_resized) / 255.0
img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float()

# ====== Predict Mask ======
with torch.no_grad():
    output = model(img_tensor)
    pred_mask = torch.sigmoid(output)[0][0].numpy()

# Resize predicted mask to original image size
pred_mask_resized = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize(original_size)

# ====== Display & Save Results ======
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Input Palm Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img)
plt.imshow(pred_mask_resized, cmap="jet", alpha=0.5)
plt.title("Predicted Palm Region")
plt.axis("off")

plt.tight_layout()

# Save output
output_path = os.path.join(output_dir, "predicted_palm.png")
plt.savefig(output_path)
plt.show()

print(f"✅ Prediction complete! Saved at: {output_path}")

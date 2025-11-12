# train.py
import os
import argparse
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# ---- Dataset ----
class PalmLinesDataset(Dataset):
    """
    Expects:
      images/  - rgb images (.jpg/.png)
      masks/   - mask images where each channel (R,G,B) corresponds to Life, Head, Heart (0/255)
    """
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('RGB')  # 3-channel mask
        img = np.array(img) / 255.0
        mask = np.array(mask) / 255.0
        img = torch.from_numpy(img).permute(2,0,1).float()
        mask = torch.from_numpy(mask).permute(2,0,1).float()
        return img, mask


# ---- Metrics & Loss ----
def dice_coef(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(2,3))
    denom = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2*inter + eps) / (denom + eps)
    return dice.mean()


# ---- Training Loop ----
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    img_files = sorted(glob(os.path.join(args.data_dir, 'images', args.split, '*.*')))
    mask_files = sorted(glob(os.path.join(args.data_dir, 'masks', args.split, '*.*')))
    assert len(img_files) == len(mask_files), "Images and masks count mismatch"

    dataset = PalmLinesDataset(img_files, mask_files)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
    )
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
    print("✅ Training complete — model saved to", args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Root data folder')
    parser.add_argument('--split', type=str, default='train', help='Subfolder name (train)')
    parser.add_argument('--save_dir', type=str, default='models', help='Where to save checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    train(args)

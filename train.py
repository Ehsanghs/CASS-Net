import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2

# Import custom modules
from models.cass_net import CASSNet
from utils.losses import CASSNetLoss
from utils.dataset import AISDataset

def load_ids(file_path):
    """Loads patient IDs from a text file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def calculate_dice_score(preds, targets, smooth=1e-6):
    """Calculates the Dice score for validation (Threshold = 0.5 as per Algorithm 1)."""
    preds_prob = torch.sigmoid(preds)
    preds_bin = (preds_prob > 0.5).float()
    
    intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth)
    return dice.mean().item()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load patient IDs
    train_ids = load_ids(args.train_ids)
    val_ids = load_ids(args.val_ids)
    
    # Transforms matching the manuscript description
    train_transform = A.Compose([
        A.PadIfNeeded(min_height=384, min_width=384, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomCrop(height=384, width=384),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.7),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.PadIfNeeded(min_height=384, min_width=384, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(height=384, width=384),
        ToTensorV2()
    ])
    
    # Datasets & Loaders
    train_ds = AISDataset(args.data_dir, patient_ids=train_ids, transform=train_transform)
    val_ds = AISDataset(args.data_dir, patient_ids=val_ids, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = CASSNet(num_input_channels=4).to(device)
    
    # Loss, Optimizer & Scheduler
    criterion = CASSNetLoss(total_epochs=args.epochs).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.5e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_dice = 0.0
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        
        # 10-Epoch Linear Warm-up
        if epoch < 10:
            lr_scale = 0.01 + 0.99 * (epoch / 10.0)
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * lr_scale
        
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Train")
        
        for batch in loop:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda'):
                preds = model(imgs) # Returns (out, aux1, aux2, aux3) during training
                loss = criterion(preds, masks, epoch)
            
            # Backward pass with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_dice = 0.0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Val  ")
            for batch in val_loop:
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                with torch.amp.autocast('cuda'):
                    preds = model(imgs) # Returns only 'out' during eval
                
                val_dice += calculate_dice_score(preds, masks)
                
        avg_val_dice = val_dice / len(val_loader)
        print(f"Epoch {epoch+1} Summary -> Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")
        
        # Scheduler Step (after warm-up)
        if epoch >= 10:
            scheduler.step(avg_val_dice)
            
        # Save Best Model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            save_path = os.path.join(args.save_dir, "cassnet_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
            }, save_path)
            print(f"--> Saved new best model with Val Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASS-Net Training Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to AISD dataset")
    parser.add_argument("--train_ids", type=str, default="splits/train_ids.txt", help="Path to training IDs list")
    parser.add_argument("--val_ids", type=str, default="splits/val_ids.txt", help="Path to validation IDs list")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model weights")
    parser.add_argument("--epochs", type=int, default=200, help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size (default: 6 for 16GB VRAM)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    args = parser.parse_args()
    
    train(args)

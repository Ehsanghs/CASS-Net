import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cass_net import CASSNet
from utils.losses import CASSNetLoss
from utils.dataset import AISDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transforms
    train_transform = A.Compose([
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.7),
        # ... add other augs from your script
        ToTensorV2()
    ])
    
    # Dataset & Loader
    # Note: You need to implement the ID splitting logic here or pass lists
    train_ds = AISDataset(args.data_dir, patient_ids=TRAIN_IDS, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = CASSNet(num_input_channels=4).to(device)
    
    # Loss & Optimizer
    criterion = CASSNetLoss(total_epochs=args.epochs).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.5e-2)
    
    # Loop
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in loop:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(imgs) # Returns (out, aux1, aux2, aux3)
            
            # Loss
            loss = criterion(preds, masks, epoch)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())
            
        # Validation logic here...
        
        # Save checkpoint
        torch.save(model.state_dict(), f"{args.save_dir}/cassnet_epoch_{epoch}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to AISD dataset")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    
    train(args)
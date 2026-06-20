import argparse
import csv
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cass_net import CASSNet
from utils.dataset import AISDataset
from utils.losses import CASSNetLoss


def load_ids(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")

    patient_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError(f"Duplicate patient IDs found in {path}")

    return patient_ids


def validate_splits(train_ids, val_ids):
    overlap = set(train_ids).intersection(val_ids)

    if overlap:
        raise ValueError(
            f"Training and validation splits overlap: {sorted(overlap)}"
        )

    if len(train_ids) != 305:
        raise ValueError(
            f"Expected 305 training patients, found {len(train_ids)}."
        )

    if len(val_ids) != 40:
        raise ValueError(
            f"Expected 40 validation patients, found {len(val_ids)}."
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_transforms():
    train_transform = A.Compose(
        [
            A.PadIfNeeded(
                min_height=384,
                min_width=384,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=1.0,
            ),
            A.RandomCrop(height=384, width=384, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.7,
            ),
            A.ElasticTransform(
                alpha=1.5,
                sigma=50,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.4,
            ),
            A.Affine(
                translate_percent=(-0.1, 0.1),
                scale=(0.9, 1.1),
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                cval_mask=0,
                p=0.5,
            ),
            A.GaussNoise(
                var_limit=(0.0005, 0.003),
                p=0.3,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.4,
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.2,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.PadIfNeeded(
                min_height=384,
                min_width=384,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=1.0,
            ),
            A.CenterCrop(height=384, width=384, p=1.0),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform


def dice_score(logits, targets, smooth=1e-6):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).float()

    predictions = predictions.reshape(predictions.size(0), -1)
    targets = (targets > 0.5).float().reshape(targets.size(0), -1)

    intersection = (predictions * targets).sum(dim=1)

    score = (
        2.0 * intersection + smooth
    ) / (
        predictions.sum(dim=1) + targets.sum(dim=1) + smooth
    )

    return score.mean().item()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    amp_enabled,
):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    processed_batches = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1} - train",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            outputs = model(images)
            loss = criterion(outputs, masks, epoch)
            main_output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch + 1}."
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_dice += dice_score(main_output.detach(), masks)
        processed_batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{total_dice / processed_batches:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    if processed_batches == 0:
        raise RuntimeError("No training batches were processed.")

    return (
        total_loss / processed_batches,
        total_dice / processed_batches,
    )


def validate(
    model,
    loader,
    criterion,
    device,
    epoch,
    amp_enabled,
):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    processed_batches = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1} - validation",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                outputs = model(images)
                loss = criterion(outputs, masks, epoch)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite validation loss at epoch {epoch + 1}."
                )

            total_loss += loss.item()
            total_dice += dice_score(outputs, masks)
            processed_batches += 1

            progress.set_postfix(
                loss=f"{total_loss / processed_batches:.4f}",
                dice=f"{total_dice / processed_batches:.4f}",
            )

    if processed_batches == 0:
        raise RuntimeError("No validation batches were processed.")

    return (
        total_loss / processed_batches,
        total_dice / processed_batches,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    warmup_scheduler,
    scaler,
    epoch,
    best_val_dice,
    patience_counter,
    val_loss,
    val_dice,
    args,
):
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_dice": best_val_dice,
        "patience_counter": patience_counter,
        "val_loss": val_loss,
        "val_dice": val_dice,
        "seed": args.seed,
        "arguments": vars(args),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    warmup_scheduler,
    scaler,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    warmup_scheduler.load_state_dict(
        checkpoint["warmup_scheduler_state_dict"]
    )

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return (
        int(checkpoint.get("epoch", 0)),
        float(checkpoint.get("best_val_dice", 0.0)),
        int(checkpoint.get("patience_counter", 0)),
    )


def append_history(path, row):
    path = Path(path)
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def resolve_device(requested_device):
    if requested_device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return torch.device(requested_device)


def train(args):
    set_seed(args.seed)

    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda"

    train_ids = load_ids(args.train_ids)
    val_ids = load_ids(args.val_ids)
    validate_splits(train_ids, val_ids)

    run_dir = Path(args.output_dir) / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(vars(args), file, indent=2)

    train_transform, val_transform = build_transforms()

    train_dataset = AISDataset(
        args.data_dir,
        patient_ids=train_ids,
        transform=train_transform,
        mode="train",
    )

    val_dataset = AISDataset(
        args.data_dir,
        patient_ids=val_ids,
        transform=val_transform,
        mode="validation",
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": amp_enabled,
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_options,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    model = CASSNet(
        num_input_channels=4,
        dropout_rate=args.dropout,
    ).to(device)

    criterion = CASSNetLoss(
        total_epochs=args.epochs,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    start_epoch = 0
    best_val_dice = 0.0
    patience_counter = 0

    if args.resume is not None:
        start_epoch, best_val_dice, patience_counter = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            warmup_scheduler,
            scaler,
            device,
        )

    best_checkpoint = run_dir / "best_checkpoint.pt"
    last_checkpoint = run_dir / "last_checkpoint.pt"
    history_path = run_dir / "training_history.csv"

    print(f"Device: {device}")
    print(f"Training patients: {len(train_ids)}")
    print(f"Validation patients: {len(val_ids)}")
    print(f"Seed: {args.seed}")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            amp_enabled,
        )

        val_loss, val_dice = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            amp_enabled,
        )

        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(val_dice)

        improved = val_dice > best_val_dice

        if improved:
            best_val_dice = val_dice
            patience_counter = 0
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]["lr"]

        append_history(
            history_path,
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "learning_rate": current_lr,
            },
        )

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scheduler,
            warmup_scheduler,
            scaler,
            epoch,
            best_val_dice,
            patience_counter,
            val_loss,
            val_dice,
            args,
        )

        if improved:
            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scheduler,
                warmup_scheduler,
                scaler,
                epoch,
                best_val_dice,
                patience_counter,
                val_loss,
                val_dice,
                args,
            )

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train Dice {train_dice:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val Dice {val_dice:.4f} | "
            f"best {best_val_dice:.4f}"
        )

        if patience_counter >= args.early_stopping_patience:
            print(
                f"Early stopping after {args.early_stopping_patience} "
                "epochs without improvement."
            )
            break


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--train_ids",
        default="splits/train_ids.txt",
    )
    parser.add_argument(
        "--val_ids",
        default="splits/val_ids.txt",
    )
    parser.add_argument(
        "--output_dir",
        default="runs",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--val_batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1.5e-2)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=35,
    )
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

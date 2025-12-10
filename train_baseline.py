import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import os
from dotenv import load_dotenv

from baseline_model import CLIPNetwork
import utils


def train(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = utils.HatefulMemeDataset(
        h5_path="data/resized/train.h5",
        json_path="data/train.jsonl",
        do_augment=True,
    )
    val_dataset = utils.HatefulMemeDataset(
        h5_path="data/resized/val.h5",
        json_path="data/val.jsonl",
        do_augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.train.batch, shuffle=False, num_workers=4
    )

    model = instantiate(config.model).to(device)
    processor = CLIPNetwork().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.decay
    )
    scaler = GradScaler(device=str(device), enabled=config.train.use_fp16)

    total_steps = len(train_loader) * config.train.epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_val_acc = 0
    os.makedirs("checkpoint", exist_ok=True)

    for epoch in range(config.train.epoch):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.train.epoch}")
        for batch in loop:
            images = batch["pixel_values"]
            labels = batch["label"].float().to(device).unsqueeze(1)
            texts = batch["text"]

            text_features, image_features = processor(texts=texts, images=images)

            optimizer.zero_grad()
            with autocast(
                device_type=str(device),
                enabled=config.train.use_fp16,
                dtype=torch.float16,
            ):
                logits = model(text_features, image_features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            preds = torch.sigmoid(logits).detach().cpu().numpy() > 0.5
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item())

        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)

        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, processor, device, config.train.use_fp16
        )

        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "lr": scheduler.get_last_lr()[0],
            }
        )

        # Checkpoint Saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoint/best_model.pth")


@torch.no_grad()
def validate(model, loader, criterion, processor, device, use_fp16=False):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        images = batch["pixel_values"]
        labels = batch["label"].float().to(device).unsqueeze(1)
        texts = batch["text"]

        text_features, image_features = processor(texts=texts, images=images)
        with autocast(device_type=str(device), enabled=use_fp16, dtype=torch.float16):
            logits = model(text_features, image_features)
            loss = criterion(logits, labels)

        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = probs > 0.5

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return avg_loss, acc, auc


@torch.no_grad()
def evaluate(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = utils.HatefulMemeDataset(
        h5_path="data/resized/test.h5",
        json_path="data/test.jsonl",
        do_augment=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.train.batch, shuffle=False, num_workers=4
    )
    model = instantiate(config.model).to(device)
    state_dict = torch.load("checkpoint/best_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    processor = CLIPNetwork().to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(test_loader):
        images = batch["pixel_values"]
        labels = batch["label"].float().to(device).unsqueeze(1)
        texts = batch["text"]

        text_features, image_features = processor(texts=texts, images=images)
        logits = model(text_features.float(), image_features.float())

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = probs > 0.5

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return {"Accuracy": acc, "F1": f1, "AUC": auc}


@hydra.main(config_path="config", version_base=None)
def main(config: DictConfig):
    load_dotenv()
    utils.ignore_warnings()
    utils.fix_random_seed(config.train.seed)
    utils.init_wandb(
        name=os.getenv("WANDB_PROJECT"),
        configs=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )

    train(config=config)

    metrics = evaluate(config=config)
    print(metrics)
    wandb.log(metrics)

    wandb.finish()


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from dotenv import load_dotenv

from model import CLIPNetwork
from trm_model import *
import utils


def train(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset Setup
    train_dataset = utils.HatefulMemeDataset(
        h5_path="data/resized/train.h5", json_path="data/train.jsonl", do_augment=True
    )
    val_dataset = utils.HatefulMemeDataset(
        h5_path="data/resized/val.h5", json_path="data/val.jsonl", do_augment=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.train.batch, shuffle=False, num_workers=4
    )

    processor = CLIPNetwork().to(device)

    model = TRMModel(
        hidden_size=config.model.hidden_size,
        expansion=config.model.expansion,
        num_layers=config.model.layers,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.decay
    )
    criterion = nn.BCEWithLogitsLoss()

    N_sup = config.train.n_supervision
    total_steps = len(train_loader) * config.train.epoch * N_sup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_val_auc = 0.0
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

            with torch.no_grad():
                text_feats, img_feats = processor(texts=texts, images=images)
                combined_feats = torch.cat([text_feats, img_feats], dim=-1).float()

            # TRM Initialization
            actual_batch_size = combined_feats.size(0)
            y = model.H_init.expand(actual_batch_size, 1, -1).clone()
            z = model.L_init.expand(actual_batch_size, 1, -1).clone()

            # Deep Supervision Loop
            for step in range(N_sup):
                optimizer.zero_grad()
                x_input = model.projection(combined_feats.to(device)).unsqueeze(1)
                (y_next, z_next), logits = model.deep_recursion(
                    x_input,
                    y,
                    z,
                    n=config.train.recursion_n,
                    T=config.train.recursion_t,
                )

                pred_logits = logits.squeeze(1)
                loss_task = criterion(pred_logits, labels)
                loss_task.backward()

                optimizer.step()
                scheduler.step()

                y = y_next.detach()
                z = z_next.detach()

                if step == N_sup - 1:
                    train_loss += loss_task.item()
                    preds = torch.sigmoid(pred_logits).detach().cpu().numpy() > 0.5
                    train_preds.extend(preds)
                    train_labels.extend(labels.cpu().numpy())
                    loop.set_postfix(loss=loss_task.item())

        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc, val_auc = validate(
            model, processor, val_loader, criterion, device, config
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

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "checkpoint/best_model.pth")


@torch.no_grad()
def validate(model, processor, loader, criterion, device, config):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    N_sup = config.train.n_supervision

    for batch in loader:
        images = batch["pixel_values"]
        labels = batch["label"].float().to(device).unsqueeze(1)
        texts = batch["text"]

        text_feats, img_feats = processor(texts=texts, images=images)
        combined_feats = torch.cat([text_feats, img_feats], dim=-1).float()

        x_input = model.projection(combined_feats).unsqueeze(1)

        batch_size = x_input.size(0)
        y = model.H_init.expand(batch_size, 1, -1).clone()
        z = model.L_init.expand(batch_size, 1, -1).clone()

        final_logits = None

        for step in range(N_sup):
            (y, z), logits = model.deep_recursion(
                x_input,
                y,
                z,
                n=config.train.recursion_n,
                T=config.train.recursion_t,
            )

            if step == N_sup - 1:
                final_logits = logits

        loss = criterion(final_logits.squeeze(1), labels)
        total_loss += loss.item()
        probs = torch.sigmoid(final_logits.squeeze(1)).cpu().numpy()
        preds = probs > 0.5

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

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

    processor = CLIPNetwork().to(device)

    model = TRMModel(
        hidden_size=config.model.hidden_size,
        expansion=config.model.expansion,
        num_layers=config.model.layers,
    ).to(device)

    checkpoint_path = "checkpoint/best_model.pth"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Evaluating with random weights."
        )

    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    N_sup = config.train.n_supervision

    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch["pixel_values"]
        labels = batch["label"].float().to(device).unsqueeze(1)
        texts = batch["text"]

        text_feats, img_feats = processor(texts=texts, images=images)
        combined_feats = torch.cat([text_feats, img_feats], dim=-1).float()

        x_input = model.projection(combined_feats).unsqueeze(1)

        # TRM Initialization
        actual_batch_size = combined_feats.size(0)
        y = model.H_init.expand(actual_batch_size, 1, -1).clone()
        z = model.L_init.expand(actual_batch_size, 1, -1).clone()

        final_logits = None

        # Deep Supervision Loop (Inference)
        for step in range(N_sup):
            (y, z), logits = model.deep_recursion(
                x_input,
                y,
                z,
                n=config.train.recursion_n,
                T=config.train.recursion_t,
            )

            if step == N_sup - 1:
                final_logits = logits

        probs = torch.sigmoid(final_logits.squeeze(1)).cpu().numpy()
        preds = probs > 0.5

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    metrics = {"Accuracy": acc, "F1": f1, "AUC": auc}
    return metrics


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
    wandb.log(metrics)

    wandb.finish()


if __name__ == "__main__":
    main()

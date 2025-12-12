import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from dotenv import load_dotenv

from baseline_model import CLIPNetwork
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
    halt_criterion = nn.BCEWithLogitsLoss()

    N_sup = config.train.n_supervision
    total_steps = len(train_loader) * config.train.epoch * N_sup
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.train.min_lr
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

            with torch.no_grad():
                text_feats, img_feats = processor(texts=texts, images=images)
                combined_feats = torch.cat([text_feats, img_feats], dim=-1).float()

            # TRM Initialization
            actual_batch_size = combined_feats.size(0)
            y = model.H_init.expand(actual_batch_size, 1, -1).clone()
            z = model.L_init.expand(actual_batch_size, 1, -1).clone()

            halted_mask = torch.zeros(
                actual_batch_size, dtype=torch.bool, device=device
            )
            final_pred_logits = torch.empty_like(labels)
            last_loss = 0
            last_pred_logits = None

            # Deep Supervision Loop
            for _ in range(N_sup):
                if halted_mask.all():
                    break

                y_old, z_old = y, z

                x_input = model.projection(combined_feats.to(device)).unsqueeze(1)
                (y_next, z_next), logits, q_logits = model.deep_recursion(
                    x_input,
                    y,
                    z,
                    n=config.train.recursion_n,
                    T=config.train.recursion_t,
                )

                pred_logits = logits.squeeze(1)
                last_pred_logits = pred_logits

                loss_task = criterion(pred_logits, labels)

                # Halting Loss
                is_correct = (
                    (torch.sigmoid(pred_logits) > 0.5).float() == labels
                ).float()
                loss_halt = halt_criterion(q_logits.squeeze(1), is_correct)
                loss = loss_task + loss_halt
                last_loss = loss.item()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                halt_decision = (torch.sigmoid(q_logits.squeeze(1)) > 0).squeeze(-1)
                newly_halting_mask = halt_decision & ~halted_mask

                final_pred_logits[newly_halting_mask] = pred_logits.detach()[
                    newly_halting_mask
                ]
                halted_mask.logical_or_(newly_halting_mask)

                y = torch.where(halted_mask.view(-1, 1, 1), y_old, y_next.detach())
                z = torch.where(halted_mask.view(-1, 1, 1), z_old, z_next.detach())

            final_pred_logits[~halted_mask] = last_pred_logits.detach()[~halted_mask]

            train_loss += last_loss
            preds = torch.sigmoid(final_pred_logits).cpu().numpy() > 0.5
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            loop.set_postfix(loss=last_loss)

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
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

        actual_batch_size = combined_feats.size(0)
        y = model.H_init.expand(actual_batch_size, 1, -1).clone()
        z = model.L_init.expand(actual_batch_size, 1, -1).clone()

        final_logits = torch.empty_like(labels)
        halted_mask = torch.zeros(actual_batch_size, dtype=torch.bool, device=device)
        last_step_logits = None

        for _ in range(N_sup):
            if halted_mask.all():
                break

            y_old, z_old = y, z

            (y, z), logits, q_logits = model.deep_recursion(
                x_input,
                y,
                z,
                n=config.train.recursion_n,
                T=config.train.recursion_t,
            )
            last_step_logits = logits

            halt_decision = (torch.sigmoid(q_logits.squeeze(1)) > 0.5).squeeze(-1)
            newly_halting_mask = halt_decision & ~halted_mask

            final_logits[newly_halting_mask] = logits.squeeze(1)[newly_halting_mask]
            halted_mask.logical_or_(newly_halting_mask)

            y = torch.where(halted_mask.view(-1, 1, 1), y_old, y)
            z = torch.where(halted_mask.view(-1, 1, 1), z_old, z)

        final_logits[~halted_mask] = last_step_logits.squeeze(1)[~halted_mask]

        loss = criterion(final_logits, labels)
        total_loss += loss.item()
        probs = torch.sigmoid(final_logits).cpu().numpy()
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

        actual_batch_size = combined_feats.size(0)
        y = model.H_init.expand(actual_batch_size, 1, -1).clone()
        z = model.L_init.expand(actual_batch_size, 1, -1).clone()

        final_logits = torch.empty_like(labels)
        halted_mask = torch.zeros(actual_batch_size, dtype=torch.bool, device=device)
        last_step_logits = None

        # Deep Supervision Loop (Inference)
        for _ in range(N_sup):
            if halted_mask.all():
                break

            y_old, z_old = y, z

            (y, z), logits, q_logits = model.deep_recursion(
                x_input,
                y,
                z,
                n=config.train.recursion_n,
                T=config.train.recursion_t,
            )
            last_step_logits = logits

            halt_decision = (torch.sigmoid(q_logits.squeeze(1)) > 0.5).squeeze(-1)
            newly_halting_mask = halt_decision & ~halted_mask

            final_logits[newly_halting_mask] = logits.squeeze(1)[newly_halting_mask]
            halted_mask.logical_or_(newly_halting_mask)

            y = torch.where(halted_mask.view(-1, 1, 1), y_old, y)
            z = torch.where(halted_mask.view(-1, 1, 1), z_old, z)

        final_logits[~halted_mask] = last_step_logits.squeeze(1)[~halted_mask]

        probs = torch.sigmoid(final_logits).cpu().numpy()
        preds = probs > 0.5

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

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
    wandb.log(metrics)

    wandb.finish()


if __name__ == "__main__":
    main()

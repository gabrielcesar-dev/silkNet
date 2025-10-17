import json
from pathlib import Path
import random
from time import strftime, time

from loguru import logger
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from silknet.config import (
    DATASET_NAME,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_EPOCHS,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SEED,
)
from silknet.modeling.early_stopping import EarlyStopping
from silknet.modeling.models import ModelFactory, ModelNames
from silknet.modeling.train_loader import train_val_test_split

app = typer.Typer()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if not torch.isfinite(loss):
            logger.warning("Non-finite loss encountered (train). Skipping batch.")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # statistics
        running_loss += float(loss.item()) * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else float('nan')
    epoch_acc = float(correct_predictions) / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                try:
                    logger.warning("Non-finite loss encountered (val). Skipping batch in metrics.")
                except Exception:
                    pass
                continue

            running_loss += float(loss.item()) * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = float(correct_predictions) / total_samples

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    metrics = {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    return metrics


def train_history_report(history: dict, model_name: str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    history_filename = Path(model_name).stem + "_history.json"
    history_path = REPORTS_DIR / history_filename

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to: {history_path}")


def setup_environment(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            logger.info(f"Using device: {torch.cuda.get_device_name(device)}")
        except Exception:
            logger.info("Using CUDA device")
    else:
        logger.info("Using device: cpu")
    return device
def free_gpu_memory():
    import gc

    gc.collect()
    torch.cuda.empty_cache()


@app.command()
def main(input_path: Path = PROCESSED_DATA_DIR / DATASET_NAME, model_name: str = "resnetpretrained", fine_tune: bool = True):
    free_gpu_memory()
    device = setup_environment(SEED)

    train_loader, val_loader, _ = train_val_test_split(input_path, use_albumentations=True)

    if train_loader is None or val_loader is None:
        logger.error("Failed to create data loaders.")
        return

    NUM_CLASSES = len(train_loader.dataset.dataset.classes)  # type: ignore
    model = ModelFactory.create_model(ModelNames[model_name.upper()], NUM_CLASSES, fine_tune=fine_tune)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
    scaler = None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    timestamp = strftime("%Y%m%d_%H%M%S")
    model_class_name = type(model).__name__ if model is not None else "model"
    model_name = f"{DATASET_NAME}_{model_class_name}_{timestamp}_e{NUM_EPOCHS}_s{SEED}.pt"

    early_stopping = EarlyStopping(
        patience=7,
        verbose=True,
        delta=1e-4,
        path=MODELS_DIR / (Path(model_name).stem + ".pt"),
    )
    

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        start_time = time()


        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1_score"])

        end_time = time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        logger.info(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | Time: {int(epoch_mins)}m {int(epoch_secs)}s"
        )
        logger.info(f"\tTrain -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        logger.info(
            f"\tValid -> Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_score']:.4f}"
        )

        early_stopping(val_metrics["loss"], model)
        if early_stopping.early_stop:
            logger.info("EarlyStopping triggered. Stopping training loop.")
            break

        
        free_gpu_memory()

    logger.success("Training complete.")
    logger.info(
        f"Best model (by val loss) saved to: {MODELS_DIR / (Path(model_name).stem + '_best_by_val_loss.pt')}"
    )

    train_history_report(history, model_name)


if __name__ == "__main__":
    app()

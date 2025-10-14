import json
from pathlib import Path

from loguru import logger
import re
import torch
from tqdm import tqdm
import typer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from silknet.config import DATASET_NAME, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from silknet.modeling.models import ResNet18
from silknet.modeling.train_loader import train_val_test_split
from silknet.modeling.train import setup_environment
from silknet.config import SEED


def parse_model_filename(model_path: Path):
    """Parse dataset and seed from a model filename.

    Expected filename pattern:
        {dataset}_{ModelClass}_{YYYYMMDD_HHMMSS}_e{NUM_EPOCHS}_s{SEED}

    Returns a dict with keys: dataset (str) and seed (int) when parsing
    succeeds. If parsing fails, returns an empty dict.
    """
    stem = model_path.stem
    pattern = r"(?P<dataset>.+)_(?P<model>[^_]+)_(?P<timestamp>\d{8}_\d{6})_e(?P<epochs>\d+)_s(?P<seed>\d+)$"
    m = re.match(pattern, stem)
    if not m:
        return {}

    md = m.groupdict()
    dataset = md.get("dataset")
    seed_str = md.get("seed")
    try:
        seed = int(seed_str) if seed_str is not None else None
    except ValueError:
        seed = None

    result = {}
    if dataset:
        result["dataset"] = dataset
    if seed is not None:
        result["seed"] = seed
    return result

app = typer.Typer()


@app.command()
def main(
    model_path: Path = typer.Option(
        ..., "--model-path", "-p", help="Full path to the trained model file (state_dict)."
    ),
    input_path: Path = PROCESSED_DATA_DIR / DATASET_NAME,
):
    """
    Loads the best trained model and generates a predictions file
    (true labels vs. predicted labels) using the test set.
    """
    parsed = parse_model_filename(model_path)
    if parsed:
        parsed_dataset = parsed.get("dataset")
        parsed_seed = parsed.get("seed")
        logger.info(f"Parsed model info: dataset={parsed_dataset}, seed={parsed_seed}")

        default_input = PROCESSED_DATA_DIR / DATASET_NAME
        if input_path == default_input and parsed_dataset:
            input_path = PROCESSED_DATA_DIR / parsed_dataset

        seed_to_use = parsed_seed if parsed_seed is not None else SEED
        device = setup_environment(seed_to_use)
    else:
        logger.warning("Model filename did not match expected pattern; using default dataset and seed.")
        device = setup_environment(SEED)

    _, _, test_loader = train_val_test_split(input_path)
    if test_loader is None:
        logger.error("Failed to create the test data loader.")
        raise typer.Exit(code=1)

    class_names = test_loader.dataset.dataset.classes  # type: ignore
    NUM_CLASSES = len(class_names)

    if not model_path.exists():
        logger.error(f"Model not found at: {model_path}")
        raise typer.Exit(code=1)

    model = ResNet18(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path}")

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision (macro): {precision:.4f}")
    logger.info(f"Test Recall (macro): {recall:.4f}")
    logger.info(f"Test F1 (macro): {f1:.4f}")

    predictions_data = {
        "true_labels": all_labels,
        "predicted_labels": all_preds,
        "class_names": class_names,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_filename = model_path.stem + "_predictions.json"
    predictions_path = REPORTS_DIR / predictions_filename

    with open(predictions_path, "w") as f:
        json.dump(predictions_data, f, indent=4)

    logger.success(f"Final predictions saved to: {predictions_path}")

if __name__ == "__main__":
    app()

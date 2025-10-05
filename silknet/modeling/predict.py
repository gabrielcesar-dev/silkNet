import json
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
import typer

from silknet.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, DATASET_NAME
from silknet.modeling.models import SequentialCNN
from silknet.modeling.train_loader import train_val_test_split

app = typer.Typer()


@app.command()
def main(
    model_name: str = typer.Option(
        ..., "--model-name", "-n", help="Filename of the trained model in /models."
    ),
    input_path: Path = PROCESSED_DATA_DIR / DATASET_NAME,
):
    """
    Loads the best trained model and generates a predictions file
    (true labels vs. predicted labels) using the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    _, _, test_loader = train_val_test_split(input_path)
    if test_loader is None:
        logger.error("Failed to create the test data loader.")
        raise typer.Exit(code=1)

    class_names = test_loader.dataset.dataset.classes # type: ignore
    NUM_CLASSES = len(class_names)

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        logger.error(f"Model not found at: {model_path}")
        raise typer.Exit(code=1)

    model = SequentialCNN(num_classes=NUM_CLASSES)
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

    predictions_data = {
        "true_labels": all_labels,
        "predicted_labels": all_preds,
        "class_names": class_names,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    predictions_filename = Path(model_name).stem + "_predictions.json"
    predictions_path = REPORTS_DIR / predictions_filename
    
    with open(predictions_path, "w") as f:
        json.dump(predictions_data, f, indent=4)

    logger.success(f"Final predictions saved to: {predictions_path}")
    logger.info("You can now run `python -m silknet.plots` to generate the confusion matrix.")


if __name__ == "__main__":
    app()


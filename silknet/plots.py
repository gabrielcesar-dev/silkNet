import json
from pathlib import Path
from typing import Any, Dict

from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import typer

from silknet.config import FIGURES_DIR

app = typer.Typer()


def load_json_data(file_path: Path) -> Dict[str, Any]:
    """Loads and returns data from a JSON file."""
    if not file_path.exists():
        logger.error(f"Input file not found: {file_path}")
        logger.error("Please run the prerequisite script (train or predict) first.")
        raise typer.Exit(code=1)

    logger.info(f"Loading data from {file_path}...")
    with open(file_path) as f:
        return json.load(f)


def generate_confusion_matrix_plot(data: Dict[str, Any], output_path: Path):
    """Generates and saves a confusion matrix plot."""
    true_labels = data["true_labels"]
    predicted_labels = data["predicted_labels"]
    class_names = data["class_names"]

    logger.info("Calculating confusion matrix...")
    cm = confusion_matrix(true_labels, predicted_labels)

    logger.info("Generating plot...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    logger.success(f"Confusion matrix plot saved to {output_path}")


def generate_learning_curves_plot(data: Dict[str, Any], output_path: Path):
    """Generates and saves learning curve plots for metrics."""
    epochs = range(1, len(data["train_loss"]) + 1)

    logger.info("Generating plots...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    fig.suptitle("Learning Curves", fontsize=16)

    # Plot Loss
    axs[0].plot(epochs, data["train_loss"], "o-", label="Training Loss")
    axs[0].plot(epochs, data["val_loss"], "o-", label="Validation Loss")
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Plot Accuracy
    axs[1].plot(epochs, data["train_acc"], "o-", label="Training Accuracy")
    axs[1].plot(epochs, data["val_acc"], "o-", label="Validation Accuracy")
    axs[1].set_title("Training and Validation Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    # Plot F1-Score
    axs[2].plot(epochs, data["val_f1"], "o-", label="Validation F1-Score")
    axs[2].set_title("Validation F1-Score")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("F1-Score")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    logger.success(f"Learning curves plot saved to {output_path}")


@app.command()
def confusion_matrix_cmd(
    predictions_path: Path = typer.Option(
        ...,
        "--predictions-path",
        "-p",
        help="Full path to the predictions JSON file.",
    ),
):
    """
    CLI command to generate a confusion matrix plot from a predictions file.
    """
    input_path = Path(predictions_path)
    data = load_json_data(input_path)
    output_filename = input_path.stem + "_confusion_matrix.png"
    output_path = FIGURES_DIR / output_filename
    generate_confusion_matrix_plot(data, output_path)


@app.command()
def learning_curves_cmd(
    history_path: Path = typer.Option(
        ...,
        "--history-path",
        "-p",
        help="Full path to the training history JSON file.",
    ),
):
    """
    CLI command to generate learning curve plots from a history file.
    """
    input_path = Path(history_path)
    data = load_json_data(input_path)
    output_filename = input_path.stem + "_learning_curves.png"
    output_path = FIGURES_DIR / output_filename
    generate_learning_curves_plot(data, output_path)


if __name__ == "__main__":
    app()

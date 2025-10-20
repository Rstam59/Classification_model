import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from src.data import get_dataloaders


def evaluate(split="validation"):
    # Load config
    cfg = yaml.safe_load(open("configs/config.yaml"))
    batch_size = cfg["batch_size"]
    max_len = cfg["max_len"]
    model_ckpt = cfg["model_ckpt"]

    # Get dataloaders
    train_loader, val_loader, test_loader, dataset = get_dataloaders(
        model_ckpt, batch_size, max_len
    )

    loader = {"validation": val_loader, "test": test_loader}[split]

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple GPU
    else:
        device = torch.device("cpu")

    # Load model
    model_path = "saved_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating on {split}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    print(f"✅ {split.capitalize()} Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=dataset["train"].features["label"].names,
        )
    )

    # MLflow logging
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("emotion-classifier")

    with mlflow.start_run(run_name=f"evaluate-{split}"):
        mlflow.log_param("split", split)
        mlflow.log_metric("accuracy", acc)

        # Save & log confusion matrix as artifact
        cm = confusion_matrix(all_labels, all_preds)
        cm_path = f"confusion_matrix_{split}.csv"
        np.savetxt(cm_path, cm, delimiter=",", fmt="%d")
        mlflow.log_artifact(cm_path)
    return acc


# Placeholder function for code review purposes
def placeholder_function(data):
    """
    This function is a placeholder for demonstrating code review.
    It takes input data, processes it, and returns a result.
    """
    # Process the data
    processed_data = [item * 2 for item in data]

    # Perform some computation
    result = sum(processed_data) / len(processed_data)

    return result


# Example usage of the placeholder function
if __name__ == "__main__":
    sample_data = [1, 2, 3, 4, 5]
    output = placeholder_function(sample_data)
    print(f"Processed output: {output}")

    import sys

    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    evaluate(split)

import mlflow
import mlflow.pytorch
import torch
import yaml
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from src.data import get_dataloaders
from src.model import get_model


def train():
    cfg = yaml.safe_load(open("configs/config.yaml"))

    # now we also get dataset back
    train_loader, val_loader, test_loader, dataset = get_dataloaders(
        cfg["model_ckpt"], cfg["batch_size"], cfg["max_len"]
    )

    num_labels = len(dataset["train"].features["label"].names)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple GPU
    else:
        device = torch.device("cpu")

    model = get_model(cfg["model_ckpt"], num_labels).to(device)

    lr = float(cfg["lr"])
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_scheduler("linear", optimizer, 0, num_training_steps)

    criterion = CrossEntropyLoss()

    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("emotion-classifier")

    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params(
            {
                "model_ckpt": cfg["model_ckpt"],
                "batch_size": cfg["batch_size"],
                "epochs": cfg["epochs"],
                "lr": lr,
                "max_len": cfg["max_len"],
            }
        )

    # Training loop
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)  # <-- add labels!

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
                labels = batch["label"].to(device)
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

        # log metric for this epoch
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

    # Save model locally
    save_path = cfg.get("save_model", "saved_model")  # default = saved_model
    model.save_pretrained(save_path)

    # Define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_ckpt"])
    tokenizer.save_pretrained(save_path)
    print(f"✅ Model & tokenizer saved to {save_path}")

    # Log entire folder as MLflow artifact
    mlflow.log_artifacts(save_path, artifact_path="saved_model")


if __name__ == "__main__":
    train()

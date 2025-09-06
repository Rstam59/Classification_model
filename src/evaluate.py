import torch
from transformers import AutoModelForSequenceClassification
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import get_dataloaders
from src.model import get_model

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
    num_labels = len(dataset["train"].features["label"].names)

    # Load model from saved checkpoint
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple GPU
    else:
        device = torch.device("cpu")

    model_path = "saved_model"  # or "saved_model" depending on what you used in training
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Evaluation loop
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating on {split}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"✅ {split.capitalize()} Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    evaluate("validation")

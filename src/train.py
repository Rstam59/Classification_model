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

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Save model + tokenizer
    save_path = cfg.get("save_model", "saved_model")  # default = saved_model
    model.save_pretrained(save_path)

    AutoTokenizer.from_pretrained(cfg["model_ckpt"]).save_pretrained(save_path)
    print(f"✅ Model & tokenizer saved to {save_path}")


if __name__ == "__main__":
    train()

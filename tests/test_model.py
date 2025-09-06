import os

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model import get_model


def test_model_forward():
    """Check model builds and forward pass works."""
    num_labels = 6
    model = get_model("bert-base-uncased", num_labels=num_labels)
    model.eval()

    dummy_input_ids = torch.randint(0, 1000, (2, 10))
    dummy_attention_mask = torch.ones_like(dummy_input_ids)

    outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    assert outputs.logits.shape == (2, num_labels)
    assert torch.isfinite(outputs.logits).all()


@pytest.mark.skipif(not os.path.exists("saved_model"), reason="No trained model found")
def test_trained_model_loads():
    """Check trained model can be loaded from saved_model folder."""
    model = AutoModelForSequenceClassification.from_pretrained("saved_model")
    tokenizer = AutoTokenizer.from_pretrained("saved_model")
    inputs = tokenizer("I feel happy", return_tensors="pt")
    outputs = model(**inputs)
    assert outputs.logits.shape[1] == 6


@pytest.mark.skipif(not os.path.exists("saved_model"), reason="No trained model found")
def test_prediction_pipeline():
    """Run one prediction and check output is valid."""
    model = AutoModelForSequenceClassification.from_pretrained("saved_model")
    tokenizer = AutoTokenizer.from_pretrained("saved_model")
    inputs = tokenizer("This is a test sentence.", return_tensors="pt")
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=-1).item()
    assert isinstance(pred, int)
    assert 0 <= pred < 6

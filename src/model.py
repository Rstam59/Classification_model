import torch 
from transformers import AutoModelForSequenceClassification 

def get_model(model_ckpt, num_labels):
    return AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels = num_labels)
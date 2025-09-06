import torch
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

model_path = "saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]


@app.get("/")
def root():
    return {"message": "Emotion classifier API is running"}


@app.post("/predict")
def predict(payload: dict):
    text = payload["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    return {"label": label_names[pred]}

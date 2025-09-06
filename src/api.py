from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

app = FastAPI()

model_path = "saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

def predict_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    return label_names[pred]

# --- Gradio UI ---
demo = gr.Interface(fn=predict_text,
                    inputs="text",
                    outputs="label",
                    title="Emotion Classifier",
                    description="Type a sentence and get its predicted emotion.")

@app.on_event("startup")
async def startup_event():
    # Launch Gradio UI when FastAPI starts
    demo.launch(share=False, inbrowser=True)

@app.get("/")
def home():
    return {"message": "Emotion classifier API is running. Use /predict endpoint."}

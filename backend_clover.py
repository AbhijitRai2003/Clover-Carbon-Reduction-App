# backend_clover.py
# CLOVER Backend - Carbon-Aware Model Switching
# Author: Abhijit Rai

import random
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st

@st.cache_resource
def load_models():
    print("🔋 Loading models (first run only)...")

    bert_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    distilbert_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    quantized_model = AutoModelForSequenceClassification.from_pretrained(model_id)
    quantized_model = torch.quantization.quantize_dynamic(quantized_model, {torch.nn.Linear}, dtype=torch.qint8)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return bert_model, distilbert_model, quantized_model, tokenizer

bert_model, distilbert_model, quantized_model, tokenizer = load_models()

def quantized_pipeline(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = quantized_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "POSITIVE" if prediction == 1 else "NEGATIVE"

def get_carbon_intensity():
    return random.choice(["LOW", "MEDIUM", "HIGH"])

def clover_inference(text):
    carbon_level = get_carbon_intensity()

    if carbon_level == "HIGH":
        result = quantized_pipeline(text)
        model_used = "Quantized DistilBERT (Low Carbon)"
    elif carbon_level == "MEDIUM":
        result = distilbert_model(text)[0]['label']
        model_used = "DistilBERT (Balanced)"
    else:
        result = bert_model(text)[0]['label']
        model_used = "BERT (High Accuracy, High Carbon)"

    return model_used, result, carbon_level

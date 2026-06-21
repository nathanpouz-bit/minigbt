import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)

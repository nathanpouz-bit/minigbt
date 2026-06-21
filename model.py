
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

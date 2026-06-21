from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

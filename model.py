from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "llama3"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

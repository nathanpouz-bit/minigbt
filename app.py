import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.title("DEBUG LLM")

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@st.cache_resource
def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    return tokenizer, model

tokenizer, model = load()

msg = st.text_input("Message")

if st.button("Send") and msg:

    st.write("Model loaded OK")

    inputs = tokenizer(msg, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=50
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    st.write(text)

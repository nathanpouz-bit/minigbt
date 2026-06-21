import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

st.title("Test modèle")

MODEL_NAME = "distilgpt2"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

msg = st.text_input("Message")

if st.button("Envoyer") and msg:

    inputs = tokenizer(msg, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=30
    )

    response = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    st.write(response)

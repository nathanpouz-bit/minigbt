from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        low_cpu_mem_usage=True
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

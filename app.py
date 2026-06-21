import streamlit as st

from model import model, tokenizer
from chat import generate_response

st.title("🧠 Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for r, m in st.session_state.messages:
    st.write(("🧑 " if r=="user" else "🤖 ") + m)

user_input = st.text_input("Message")

if st.button("Envoyer") and user_input:

    response = generate_response(model, tokenizer, user_input)

    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("bot", response))

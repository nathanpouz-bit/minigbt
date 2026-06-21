import streamlit as st

from model import model, tokenizer
from chat import generate_response

st.title("🧠 Mon Chatbot IA")

if "messages" not in st.session_state:
    st.session_state.messages = []

# affichage historique
for role, msg in st.session_state.messages:
    st.write(("🧑 " if role == "user" else "🤖 ") + msg)

# input utilisateur
user_input = st.text_input("Écris ton message")

if st.button("Envoyer") and user_input:

    response = generate_response(model, tokenizer, user_input)

    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("bot", response))

    st.rerun()

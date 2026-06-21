import streamlit as st
from model import model, tokenizer
from chat import generate_response

st.set_page_config(page_title="Mini Chatbot IA", page_icon="🤖")

st.title("🤖 Mini ChatGPT (Streamlit)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# affichage historique
for role, msg in st.session_state.messages:
    if role == "user":
        st.write(f"🧑 {msg}")
    else:
        st.write(f"🤖 {msg}")

# input utilisateur
user_input = st.text_input("Toi :", "")

if user_input:
    response, st.session_state.chat_history = generate_response(
        model,
        tokenizer,
        user_input,
        st.session_state.chat_history
    )

    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("bot", response))

    st.rerun()

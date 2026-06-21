import json
import os

LOG_FILE = "logs/conversations.json"

def save_log(user, bot):
    if not os.path.exists("logs"):
        os.makedirs("logs")

    if not os.path.exists(LOG_FILE):
        data = []
    else:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)

    data.append({"user": user, "bot": bot})

    with open(LOG_FILE, "w") as f:
        json.dump(data, f)
        
    import streamlit as st

from model import model, tokenizer
from chat import generate_response
from vector_store import search, add_memory, rebuild_index


st.set_page_config(page_title="IA mémoire intelligente", page_icon="🧠")
st.title("🧠 Chatbot IA avec mémoire FAISS")


# init index
rebuild_index()


if "messages" not in st.session_state:
    st.session_state.messages = []


# afficher chat
for role, msg in st.session_state.messages:
    st.write(("🧑 " if role == "user" else "🤖 ") + msg)


user_input = st.text_input("Toi :", "")


if user_input:

    # 🧠 1. mémoire sémantique (FAISS)
    memory = search(user_input)

    if memory:
        response = memory["bot"]
    else:
        # 🤖 2. IA fallback
        response = generate_response(model, tokenizer, user_input)

    # 💾 3. apprentissage automatique
    add_memory(user_input, response)

    # UI
    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("bot", response))

    st.rerun()

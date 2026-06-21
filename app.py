import streamlit as st

from model import model, tokenizer
from chat import generate_response
from vector_store import search, add_memory, rebuild_index


st.set_page_config(page_title="IA", page_icon="🧠")
st.title("🧠 Chatbot stable")

rebuild_index()

# -------------------------
# SESSION STATE SAFE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_input" not in st.session_state:
    st.session_state.last_input = None


# -------------------------
# CHAT DISPLAY
# -------------------------
for role, msg in st.session_state.messages:
    st.write(("🧑 " if role == "user" else "🤖 ") + msg)


# -------------------------
# INPUT + BUTTON (IMPORTANT)
# -------------------------
user_input = st.text_input("Ton message :", "")
send = st.button("Envoyer")


# -------------------------
# LOGIQUE ANTI-BOUCLE
# -------------------------
if send and user_input:

    # 🔒 bloque répétition exacte
    if user_input != st.session_state.last_input:

        st.session_state.last_input = user_input

        # 🧠 mémoire
        memory = search(user_input)

        if memory:
            response = memory["bot"]
        else:
            response = generate_response(model, tokenizer, user_input)

        # 💾 save
        add_memory(user_input, response)

        # 🪟 UI
        st.session_state.messages.append(("user", user_input))
        st.session_state.messages.append(("bot", response))

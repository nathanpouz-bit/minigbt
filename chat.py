import streamlit as st

from model import model, tokenizer
from chat import generate_response
from vector_store import search, add_memory, rebuild_index


st.set_page_config(page_title="IA", page_icon="🧠")
st.title("🧠 Chatbot IA stable")

rebuild_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_input" not in st.session_state:
    st.session_state.last_input = ""


# -------------------
# AFFICHAGE CHAT
# -------------------
for role, msg in st.session_state.messages:
    st.write(("🧑 " if role == "user" else "🤖 ") + msg)


# -------------------
# INPUT + BOUTON
# -------------------
user_input = st.text_input("Écris ton message :", "")

send = st.button("Envoyer")


# -------------------
# LOGIQUE ANTI-BOUCLE
# -------------------
if send and user_input:

    # ❌ empêche double exécution
    if user_input != st.session_state.last_input:

        st.session_state.last_input = user_input

        memory = search(user_input)

        if memory:
            response = memory["bot"]
        else:
            response = generate_response(model, tokenizer, user_input)

        add_memory(user_input, response)

        st.session_state.messages.append(("user", user_input))
        st.session_state.messages.append(("bot", response))

        st.rerun()

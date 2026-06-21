import streamlit as st

# ⚠️ DOIT ÊTRE LA PREMIÈRE commande Streamlit
st.set_page_config(
    page_title="IA mémoire intelligente",
    page_icon="🧠"
)

# -------------------------
# IMPORTS
# -------------------------
from model import model, tokenizer
from chat import generate_response
from vector_store import search, add_memory, rebuild_index


# -------------------------
# INIT
# -------------------------
st.title("🧠 Chatbot IA avec mémoire intelligente")

rebuild_index()


# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------
# DISPLAY CHAT
# -------------------------
for role, msg in st.session_state.messages:
    if role == "user":
        st.write(f"🧑 {msg}")
    else:
        st.write(f"🤖 {msg}")


# -------------------------
# INPUT USER
# -------------------------
user_input = st.text_input("Écris ton message :", "")


# -------------------------
# CHAT LOGIC
# -------------------------
if user_input:

    # 🧠 1. mémoire intelligente (vector search)
    memory = search(user_input)

    if memory:
        response = memory["bot"]
    else:
        # 🤖 2. IA fallback
        response = generate_response(model, tokenizer, user_input)

    # 💾 3. sauvegarde mémoire automatique
    add_memory(user_input, response)

    # 🪟 4. update UI
    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("bot", response))

    st.rerun()


# -------------------------
# 🧠 APPRENTISSAGE MANUEL (OPTIONNEL)
# -------------------------
st.divider()
st.subheader("🧠 Enseigner le bot")

q = st.text_input("Question à apprendre")
a = st.text_input("Bonne réponse")

if st.button("Ajouter à la mémoire"):
    if q and a:
        add_memory(q, a)
        st.success("Ajouté ✔️")

import streamlit as st

from model import model, tokenizer
from chat import generate_response
from memory import load_data, add_example, find_best_match


# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Chatbot IA", page_icon="🤖")
st.title("🤖 Mini Chatbot IA qui apprend")


# -------------------------
# LOAD MEMORY
# -------------------------
memory = load_data()


# -------------------------
# MEMORY TEXT (context)
# -------------------------
def build_memory_text():
    text = ""
    for item in memory[-20:]:
        text += f"User: {item['user']}\nBot: {item['bot']}\n"
    return text


# -------------------------
# SESSION STATE (chat UI)
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------
for role, msg in st.session_state.messages:
    if role == "user":
        st.write(f"🧑 {msg}")
    else:
        st.write(f"🤖 {msg}")


# -------------------------
# USER INPUT
# -------------------------
user_input = st.text_input("Écris ton message :", "")


# -------------------------
# CHAT LOGIC
# -------------------------
if user_input:

    # 1. vérifier mémoire (réponse déjà apprise)
    cached_response = find_best_match(user_input)

    if cached_response:
        response = cached_response
    else:
        # 2. sinon IA
        memory_text = build_memory_text()
        response = generate_response(model, tokenizer, user_input, memory_text)

    # 3. sauvegarde automatique (apprentissage)
    add_example(user_input, response)

    # 4. ajouter à l'affichage
    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("bot", response))

    # 5. refresh UI
    st.rerun()


# -------------------------
# OPTIONAL: TRAINING PANEL
# -------------------------
st.divider()
st.subheader("🧠 Apprentissage manuel")

q = st.text_input("Question à apprendre")
a = st.text_input("Bonne réponse")

if st.button("Ajouter à la mémoire"):
    if q and a:
        add_example(q, a)
        st.success("Ajouté ✔️")

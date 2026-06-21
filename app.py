import streamlit as st
from chat import generate_response

st.title("🧠 Ollama Chatbot")

msg = st.text_input("Message")

if st.button("Envoyer") and msg:
    response = generate_response(msg)
    st.write(response)

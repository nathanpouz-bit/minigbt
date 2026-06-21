import streamlit as st

st.title("TEST STREAMLIT")

msg = st.text_input("Message")

if st.button("Envoyer"):
    st.write("Tu as écrit :", msg)

import streamlit as st

st.title("TEST STREAMLIT OK")

st.write("Si tu vois ça → Streamlit fonctionne")

msg = st.text_input("Message")

if st.button("Test"):
    st.write("Tu as écrit :", msg)

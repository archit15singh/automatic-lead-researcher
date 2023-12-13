from app import initiate_chat

import streamlit as st

with st.chat_message("assistant"):
    st.write("Hello human")

prompt = st.chat_input("Say something")
if prompt:
    with st.chat_message("human"):
        st.write(prompt)
        initiate_chat(prompt)
        st.write("done")

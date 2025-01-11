from io import BytesIO

import streamlit as st
from src import ner


st.title('Demo')
uploaded_file = st.file_uploader("", type=["wav", "mp3"])

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = BytesIO(uploaded_file.getvalue())
    output = ner(bytes_data)

    st.code(output["json"], language="json")
    st.text(output["transcription"])


import streamlit as st
from chain import run_agent


st.header("Query the Scopus")
q = st.text_input("Refer to the nodes and relations and ask some questions")
if q:
    response = run_agent(q)
    st.write(response)
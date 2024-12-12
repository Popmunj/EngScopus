import streamlit as st
from llm.chain import run_agent



st.header("Query the Scopus")
st.write("(Please refer to the video presentation. This will not work since it requires my database connection.)")
q = st.text_input("Refer to the nodes and relations and ask some questions")
if q:
    response = run_agent(q)
    st.write(response)
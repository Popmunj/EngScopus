import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
import numpy as np

def load_model():
    arch = nn.Sequential(
    nn.Linear(1649, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
    arch.load_state_dict(torch.load("model.pth", weights_only=True))
    return arch.eval()

def make_tensor(key, classification):
    from langchain_openai import OpenAIEmbeddings
    from dotenv import load_dotenv
    load_dotenv()
    i = pd.read_csv("ts_index.csv").set_index("Unnamed: 0") # adjust
   
    embedder = OpenAIEmbeddings()
    emb = embedder.embed_query(key)
    emb = np.stack(emb)

    one_hot = [i.loc[classification]["index"] == n for n in range(0,113)]
    cols = [c for c in i.index]
    oh = pd.DataFrame([one_hot], columns=cols)
    oh = oh.to_numpy().flatten()

    comb = np.hstack([emb, oh])
    x = torch.tensor(comb, dtype=torch.float32)
    return x


st.set_page_config(layout="wide")
nodes = pd.read_csv("csv/nodes.csv").set_index("Entities")
lit_key = pd.read_csv("csv/lit_key.csv")
lit_class = pd.read_csv("csv/lit_class.csv")
key_group = lit_key.groupby("k_name").size()
key_group = pd.DataFrame(key_group).rename(columns={"k_name":"Keyword",
                                                                     0:"Count"})

key_group = key_group.sort_values(by="Count", ascending=False).head(20)
lit_class = pd.DataFrame(lit_class.groupby("Classification").size()).sort_values(
    by="Classification", ascending=False
).rename(columns={0:"Count"}).head(20)

views = pd.read_csv("../addition/pageviews.csv")
class_date = pd.read_csv("csv/class_date.csv")
i = pd.read_csv("ts_index.csv").set_index("Unnamed: 0")

lit_auth_fund = pd.read_csv("csv/lit_ref_auth_fund.csv").drop(columns=["ReferenceCount"])
fund_score = lit_auth_fund.assign(FundingScore = lambda x : (x["FundingAgencyCount"]/ x["AuthorCount"])).drop(
    columns=["FundingAgencyCount", "AuthorCount"]
)
example = pd.read_csv("csv/example_ts.csv").set_index("LiteratureID")

st.header("Classification and Keyword")

left, right = st.columns([5,5])
with left:
    with st.container(border=True):
        st.metric(label=" ASJC Classification",
                  value=nodes.loc["Classification"])
        # st.write("Of 2025 literatures, 717 are classed using the ASJC code")
    with st.container(border=True):
        fig = px.bar(lit_class.sort_values(by="Count"),
                     x="Count",
                     y=lit_class.sort_values(by="Count").index,
                     )
        fig.update_layout(
            title="Top Classification Frequency",
            yaxis_title="Classification"
        
        )
        st.plotly_chart(fig, use_container_width=True)
  

       
        
with right:
    with st.container(border=True):
        st.metric(label="Keyword",
                  value=nodes.loc["Keyword"])
        # st.write("Of 2025 literatures, 1638 used keywords")
    with st.container(border=True):
        fig = px.bar(key_group.sort_values(by="Count"), x="Count", y=key_group.sort_values(by="Count").index)
        fig.update_layout(
            title="Top Keyword Frequency",
            yaxis_title="Keyword"
        )
        st.plotly_chart(fig, use_container_width=True)

    
with st.container(border=True):
    st.header("Trends of Classifications")
    c = st.selectbox("Select Classification",
                    ("Multidisciplinary", "Renewable Energy",
                    "Radiology", 
                    "Plant Science",
                    "Public Health", "Transplantation",
                    "Physical Therapy"
                
                    
                    ))
    
    
    class_date = pd.DataFrame(class_date[class_date["Classification"] == c].groupby("Date").size()
                        .reset_index(name="Literature"))
    
    c_dash = c.lower().replace(" ", "_")
    c_dash = c_dash[0].upper() + c_dash[1:] 
    comb = pd.merge(views[["date", c_dash]], class_date, left_on="date",
            right_on="Date").drop(columns=["date"]).set_index("Date")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(comb)
    

    fig = px.line(pd.DataFrame(scaled, index=comb.index).rename(columns={0:"Views", 1: "Literature"}))
    fig.update_layout(
        title="Wikipedia Page Views",
        xaxis_title="Date",
        yaxis_title="Normalized Count"
    )
    st.plotly_chart(fig, use_container_width=True)
    

with st.container(border=True):
    st.header("Predict Funding Score Based on Classification and Keyword")
    fig = px.histogram(fund_score, x="FundingScore", nbins=30)
   

    fig.update_layout(
        xaxis_title="Literature",
        yaxis_title="Funding Score",
        bargap=0.2,
        margin=dict(l=20, r=20, t=50, b=20),  
        showlegend=False,
        title_text="Funding Score (Funder count divided Author count) Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)


    key = st.text_input("Write a keyword here")
    cl = st.selectbox("Choose Classification",
                      (i.index))
    
    
    
    model = load_model()
    if key:
        ts = make_tensor(key, cl)
        st.write(f"Score: {model(ts).item()}")

    with st.expander("About NN"):
        st.header("Multimodal Model")

        st.write("""Below is an example training set which combines keywords' semantic sum and Classification; the MSE loss for
                   the training set is 0.2427758276462555 after 15 epochs and that of the test set is 0.3307""")
        st.table(example)

        


        
        
import pandas as pd
from loguru import logger
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
# cd to dashboard first
def execute_query(query: str,
                  params: dict):
    url = os.getenv("NEO4J_URL")
    usr = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
        
    with GraphDatabase.driver(url, auth=(usr, pwd)) as driver:
        try:
            recs, summary, keys = driver.execute_query(query, params)
            return recs, summary, keys
        except Exception as e :
            logger.error(e)

def to_csv(query, file):
    rs,_,_ = execute_query(query, {})
    recs = [r.data() for r in rs]

    if recs:
        df = pd.DataFrame(recs)
        df.to_csv(f"csv/{file}", index=False, encoding="utf-8")
        logger.info("Wrote to file.")
    
def fill_missing_date(file):
    df = pd.read_csv(f"csv/{file}")
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.set_index("Date")
    df_resampled = df.resample("D").asfreq().fillna(0)

    df_resampled.to_csv(f"csv/{file}")


def fetch_embedding():
    url = os.getenv("NEO4J_URL")
    usr = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    id = []
    emb = []
    with GraphDatabase.driver(url, auth=(usr, pwd)).session() as session:
        r = session.run(
            """
            MATCH (k:Keyword) RETURN k.id AS id, k.embedding AS embedding   
            """
        )
        for rec in r:
            id.append(rec["id"])
            emb.append(rec["embedding"])
       
        d = {"id": id, "embedding": emb}
        df = pd.DataFrame(d)
        df.to_pickle("csv/k_emb.pkl")
    
    

import requests
import pandas as pd
import requests
from dotenv import load_dotenv
import os
from loguru import logger
import json
from bs4 import BeautifulSoup


logger.add("log/caller.log")
load_dotenv()
BASE = "https://wikimedia.org/api/rest_v1/metrics"
TOKEN = os.getenv("WIKI")
 
TARGETS = ["bytes-difference", "edits", "pageviews"]
def get_pageviews(page, start_date="20180101", end_date="20231231"):
   ARGS =  "en.wikipedia.org/all-access/all-agents/{page}/daily/{start_date}/{end_date}"

   ARGS = ARGS.format(page=page, 
                      start_date=start_date
                      ,end_date=end_date)
   json_data = api_call("pageviews/per-article", ARGS)
   if json_data is not None:
      to_pageviews_csv(json_data)
   else:
      logger.warning(f"No data found for page: {page}")
   return json_data
  
   

def api_call(end, args):
   url = "/".join([BASE, end, args])
   res = requests.get(url, headers={
      "Authorization": f"Bearer {TOKEN}",
      "User-Agent": "Bot/0.0 (popepidope@gmail.com) generic-library/0.0"
   })
   if res.status_code == 200:
      return res.json()
   else:
      logger.error(f"BODY: {res.text}")

def to_pageviews_csv(json_obj):
   ents = json_obj["items"]
   page = ents[0]["article"]
   try:
      df = pd.read_csv("addition/pageviews.csv", index_col="date"
                       ,parse_dates=True)
   except:
      df = pd.DataFrame()

   dates = [pd.to_datetime(e["timestamp"][:-2]) for e in ents]
   views = [e["views"] for e in ents]
   d = {
      "date": dates,
      page: views}
   added = pd.DataFrame.from_dict(d).set_index("date")
   df = pd.concat([df, added], axis=1)
   df.to_csv("addition/pageviews.csv")


def fill_missing_date(file):
    df = pd.read_csv(f"{file}")
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.set_index("Date")
    df_resampled = df.resample("D").asfreq().fillna(0)

    df_resampled.to_csv(f"{file}")




def main():
   asjc = pd.read_csv("addition/ASJC.csv")
   for c in asjc["Description"]:
      page = c.lower().replace(" ", "_")
      page = page[0].upper() + page[1:] 
      get_pageviews(page)







   
   



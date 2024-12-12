from dotenv import load_dotenv

import os
from langchain.chains import GraphCypherQAChain
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_openai import ChatOpenAI
from langchain.graphs import Neo4jGraph
load_dotenv()


def get_cypher():
    url = os.getenv("NEO4J_URL")
    usr = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")

    graph = Neo4jGraph(
    url=url,
    username=usr,
    password=pwd)

    cypher_runnable = GraphCypherQAChain.from_llm(
        cypher_llm = ChatOpenAI(model="gpt-4"),
        qa_llm = ChatOpenAI(model="gpt-4"),
        graph=graph,
        verbose=True
    )
    return cypher_runnable


def run_agent(query):
    tools = [
    Tool(name="Graph Query",
         func=get_cypher().run,
         description="Useful for finding relations in a graph"
         )
]   
    agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(model="gpt-4o"),
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)
    return agent.run(query)

#run_agent("How many papers are published by Chulalongkorn University?")
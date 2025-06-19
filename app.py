import gradio, os
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.tools import tool
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from transformers import AutoTokenizer, TFAutoModel
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
import tensorflow as tf



#######################################
##           VECTORDB Qdrant         ##
#######################################
# Thresolhd similarity vector db:
THRESHOLD = 0.85

load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = TFAutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def embedding(text):
    tokens = tokenizer(text, return_tensors = "tf", truncation = True, padding = True)
    output = model(**tokens)
    embedding = tf.reduce_mean(output.last_hidden_state, axis = 1)
    
    # embedding is (1,384), need to put it in 384 dim
    embedding = tf.squeeze(embedding, axis= 0)
    return embedding.numpy().tolist()


class State(TypedDict):
    query: str
    not_in_db: bool

    data_to_used: Dict[str, Any]
    # Message:
    messages: Annotated[list[AnyMessage], add_messages]

    answer: str

#########################################
##               Nodes:                ##
#########################################

def search_in_vectordb(state: State, collection_name = "document_only", top_k = 5):
    query = state["query"]
    vector = embedding(query)
    result = qdrant_client.search(
        collection_name= collection_name,
        query_vector= vector,
        limit = top_k
    )

    if result["score"] > THRESHOLD : 
        return {
            "data_to_used" : result,
            "not_in_db" : False
        }
    else: 
        return {
            "not_in_db" : True
        }

#@tool
def search_on_web():
    pass

def assistant_answer(state: State):
    prompt = f"""
  As a helpful agent, which provide answer to the user 's question: {state["question"]}. 
  Use this data to answer correctly :{state["data_to_used"]}
  YOUR FINAL ANSWER MUST STRICTLY FOLLW THOSE RULES:
    - be the most ACCURATE as possible
    """

    message = [HumanMessage(content=prompt)]
    result = llm.invoke(message)

    return {
        "answer" : result
    }


# Edge:
def route_answer(state: State) -> str:
    if state["not_in_db"]:
        return "web_research"
    else: 
        return "in_vectordb"
##########################################
##                 LLM                  ##
##########################################

llm = ChatOllama(model="qwen2.5")

rag_graph = StateGraph(State)

rag_graph.add_node("search_in_vectordb", search_in_vectordb )
rag_graph.add_node("web_search", search_on_web)
rag_graph.add_node("agent_answer", assistant_answer)


rag_graph.add_edge(START, "search_in_vectordb")
rag_graph.add_conditional_edges("search_in_vectordb", route_answer,
                                  {
                                      "web_research": "web_search",
                                      "in_vectordb": "agent_answer"
                                  })
rag_graph.add_edge("web_search", "agent_answer")
rag_graph.add_edge("agent_answer", END)


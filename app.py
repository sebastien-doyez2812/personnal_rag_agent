import gradio, os
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.tools import tool
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf

#######################################
##           VECTORDB Qdrant         ##
#######################################


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
    # Question:
    question: str

    # data:
    data_from_vectordb: Optional[str]
    data_from_web : Optional[str]

    # Message:
    messages: Annotated[list[AnyMessage], add_messages]


#########################################
##               Tools:                ##
#########################################
@tool
def search_in_vectordb(query, collection_name = "document_only", top_k = 5):
    vector = embedding(query)
    result = qdrant_client.search(
        collection_name= collection_name,
        query_vector= vector,
        limit = top_k
    )
    return result

@tool
def search_on_web():
    pass

tools = [search_in_vectordb,
         search_on_web]

##########################################
##                 LLM                  ##
##########################################

llm = ChatOllama(model="qwen2.5")
llm_with_tools = llm.bind_tools(tools= tools)

def assistant():
    system_prompt = SystemPrompt(content="System prompt:" \
    "You are a helpful agent, which provide answer to the user 's question." \
    "   Answer to this question: {state["question"]}. "
    "YOUR FINAL ANSWER MUST STRICTLY FOLLW THOSE RULES:" \
    "- firstly, use the tool search_in_vectordb to ask the vectordatabse if there are elements about the topics of the question in the vectordatabase." \
    "- If there are nothing interresting in the vectordatabase, use the tool search_on_web to search inforsmation on the web" \
    "- Be the most accurate as possible to the question" \
    )
    answer = llm_with_tools.invoke([system_prompt] + state["messages"])


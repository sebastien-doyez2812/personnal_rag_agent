import langgraph
import gradio
from langchain_ollama import ollama
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Optional
from typing_extensions import TypedDict

class State(TypedDict):
    # Question:
    question: str

    # data:
    data_from_vectordb: Optional[str]
    data_from_web : Optional[str]

    # Message:
    messages: Annotated[list[AnyMessage], add_messages]




import os, gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, TFAutoModel
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langfuse.callback import CallbackHandler
import tensorflow as tf
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import time
import re 


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

    if result and result[0].score > THRESHOLD : 
        return {
            "data_to_used" : result,
            "not_in_db" : False
        }
    else: 
        return {
            "not_in_db" : True
        }

#TODO
def search_on_web(state: State, nb_url: int = 3, timeout_loading: int = 5) -> Dict[str, Any]:
    """
    Recherche sur DuckDuckGo, visite les URLs trouvées et extrait le contenu textuel des pages.

    Args:
        state (State): L'état actuel de l'agent, contenant la requête (query).
        nb_url (int): Le nombre d'URLs à récupérer de DuckDuckGo.
        timeout_loading (int): Le temps maximum en secondes pour charger une page web.

    Returns:
        Dict[str, Any]: Un dictionnaire contenant les données extraites du web.
                        Si rien n'est trouvé, 'data_to_used' contiendra un message d'erreur.
                        Sinon, il contiendra une liste de dictionnaires avec 'url', 'title' et 'content'.
    """
    query = state["query"]
    found_urls_with_snippets = []
    web_content_results = []

    print(f"[{time.strftime('%H:%M:%S')}] Démarrage de la recherche web pour : '{query}'")

    try:
        # Get some Urls:
        with DDGS() as ddgs:
            # nb_url is the limits of the url
            for r in ddgs.text(keywords=query, region='fr-fr', safesearch='moderate', max_results=nb_url):
                found_urls_with_snippets.append(r)
        
        if not found_urls_with_snippets:
            print(f"[{time.strftime('%H:%M:%S')}] Nothing Found on DuckDuckGo for : '{query}'")
            return {
                "data_to_used": "Nothing found on the web. Use what you know to answer the question."
            }
        
        print(f"[{time.strftime('%H:%M:%S')}] {len(found_urls_with_snippets)} URLs found... Start to extract.")

        for i, item in enumerate(found_urls_with_snippets):
            url = item.get('href')
            ddg_title = item.get('title', 'Title DuckDuckGo unavailable')
            ddg_snippet = item.get('body', 'Snippet DuckDuckGo unavailable')

            if not url:
                print(f"[{time.strftime('%H:%M:%S')}] URL empty found, skipping.")
                continue

            print(f"[{time.strftime('%H:%M:%S')}] Working on the URL n° {i+1}/{len(found_urls_with_snippets)}: {url}")
            page_content = "Unvailable content."
            page_title = ddg_title 

            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(url, timeout=timeout_loading, headers=headers)
                response.raise_for_status() # Handle Exception

                # Analysis of the html content:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Title extraction
                if soup.title and soup.title.string:
                    page_title = soup.title.string.strip()

                # Text extraction, using html basics balisis:
                paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span'])
                
                page_text_elements = []
                for tag in paragraphs:
                    # Exclure les scripts, styles, et les balises vides ou purement décoratives
                    if tag.name not in ['script', 'style'] and tag.get_text(strip=True):
                        # Simple nettoyage pour éviter les multiples espaces et retours à la ligne
                        text_content = re.sub(r'\s+', ' ', tag.get_text(separator=' ', strip=True)).strip()
                        if text_content: # Assurez-vous que le texte n'est pas vide après nettoyage
                            page_text_elements.append(text_content)
                
                page_content = '\n'.join(page_text_elements)

                web_content_results.append({
                    "url": url,
                    "title": page_title,
                    "content": page_content
                })

            except requests.exceptions.Timeout:
                print(f"[{time.strftime('%H:%M:%S')}] Timeout during the loading of {url}")
                web_content_results.append({
                    "url": url,
                    "title": ddg_title,
                    "content": "Error : Tiemout."
                })
            except requests.exceptions.RequestException as req_err:
                print(f"[{time.strftime('%H:%M:%S')}] Error Network or HTTP for {url}: {req_err}")
                web_content_results.append({
                    "url": url,
                    "title": ddg_title,
                    "content": f"Error: {req_err}"
                })
            except Exception as general_err:
                print(f"[{time.strftime('%H:%M:%S')}] Unknown error: with {url}: {general_err}")
                web_content_results.append({
                    "url": url,
                    "title": ddg_title,
                    "content": f"Error: {general_err}"
                })
            finally:
                time.sleep(1) 

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Critic Error : {e}")
        return {
            "data_to_used": f"an error occurs during the web research: {e}"
        }
    
    if web_content_results:
        print(f"[{time.strftime('%H:%M:%S')}] Web content ewtracted for {len(web_content_results)} pages.")
        return {
            "data_to_used": web_content_results
        }
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Nothing found on the web...")
        return {
            "data_to_used": "Nothing found on the web or in the vector database."
        }
    
def assistant_answer(state: State):
    prompt = f"""
  As a helpful agent, which provide answer to the user 's question: {state["query"]}. 
  Use this data to answer correctly :{state["data_to_used"]}
  YOUR FINAL ANSWER MUST STRICTLY FOLLW THOSE RULES:
    - be the most ACCURATE as possible
    """

    message = [HumanMessage(content=prompt)]
    result = llm.invoke(message)
    print(result)
    return {
        "answer" : result.content
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


############################################
##                Graph                   ##
############################################

# Nodes:
rag_graph = StateGraph(State)
rag_graph.add_node("search_in_vectordb", search_in_vectordb )
rag_graph.add_node("web_search", search_on_web)
rag_graph.add_node("agent_answer", assistant_answer)

# Edges:
rag_graph.add_edge(START, "search_in_vectordb")
rag_graph.add_conditional_edges("search_in_vectordb", route_answer,
                                  {
                                      "web_research": "web_search",
                                      "in_vectordb": "agent_answer"
                                  })
rag_graph.add_edge("web_search", "agent_answer")
rag_graph.add_edge("agent_answer", END)


compiled_rag_agent = rag_graph.compile()



#################################################
##               LANGFUSE DEBUG                ##
#################################################
langfuse_handler = CallbackHandler()
langfuse_handler.auth_check()

if __name__ == "__main__":
    # with gr.Blocks() as demo:
    #     gr.Markdown("Seb Doyez Personnal Rag Agent")
    #     with gr.Row(equal_height= True):
    #         text_box = gr.Textbox(lines = 5)
    #         button = gr.Button(text = "Ask")

    # demo.launch()

#########################################
##               Gradio                ##
#########################################

    def chat_interface(query, history):
        init_state = {
        "query" : query,
        "not_in_db": None,
        "data_to_used": None, 
        "messages": [],
        "answer": None
        }
        try:
            rag_result = compiled_rag_agent.invoke(input= init_state, config= {"callbacks": [langfuse_handler]})
            assistant_answer = rag_result["answer"]

        except Exception as e:
            assistant_answer = f"Error: {e}"
        return assistant_answer

    interface = gr.ChatInterface(
        fn = chat_interface,
        title = " Seb Doyez Rag Agent",
        description= "Ask me anything, and I will search my knowledge base and on the web!",
        theme = "soft"
    )

    interface.launch()

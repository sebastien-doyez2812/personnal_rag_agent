import gradio, os
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
        # 1. Récupérer les URLs et snippets de DuckDuckGo
        with DDGS() as ddgs:
            # Utilisez max_results=nb_url pour limiter le nombre de résultats de DDG
            for r in ddgs.text(keywords=query, region='fr-fr', safesearch='moderate', max_results=nb_url):
                found_urls_with_snippets.append(r)
                # Pas besoin de sleep ici, DDGS gère implicitement une certaine prudence.
        
        if not found_urls_with_snippets:
            print(f"[{time.strftime('%H:%M:%S')}] Aucune URL trouvée sur DuckDuckGo pour : '{query}'")
            return {
                "data_to_used": "Aucun résultat pertinent trouvé sur le web."
            }
        
        print(f"[{time.strftime('%H:%M:%S')}] {len(found_urls_with_snippets)} URLs trouvées. Début de l'extraction de contenu.")

        # 2. Visiter chaque URL et extraire le contenu
        for i, item in enumerate(found_urls_with_snippets):
            url = item.get('href')
            ddg_title = item.get('title', 'Titre DuckDuckGo indisponible')
            ddg_snippet = item.get('body', 'Snippet DuckDuckGo indisponible')

            if not url:
                print(f"[{time.strftime('%H:%M:%S')}] URL vide trouvée, skipping.")
                continue

            print(f"[{time.strftime('%H:%M:%S')}] Traitement de l'URL {i+1}/{len(found_urls_with_snippets)}: {url}")
            page_content = "Contenu non disponible."
            page_title = ddg_title # Utiliser le titre de DDG par défaut

            try:
                # Effectuer une requête HTTP GET pour télécharger le contenu de la page
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(url, timeout=timeout_loading, headers=headers)
                response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)

                # Analyser le contenu HTML de la page
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extraire le titre réel de la page
                if soup.title and soup.title.string:
                    page_title = soup.title.string.strip()

                # Extraire le texte visible de la page
                # On cible les balises courantes pour le texte principal
                paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span'])
                
                # Joindre le texte de ces balises
                # Filtrer les balises avec des attributs de style ou de script pour éviter le désordre
                # Utiliser un ensemble pour éviter les doublons si une balise est imbriquée
                page_text_elements = []
                for tag in paragraphs:
                    # Exclure les scripts, styles, et les balises vides ou purement décoratives
                    if tag.name not in ['script', 'style'] and tag.get_text(strip=True):
                        # Simple nettoyage pour éviter les multiples espaces et retours à la ligne
                        text_content = re.sub(r'\s+', ' ', tag.get_text(separator=' ', strip=True)).strip()
                        if text_content: # Assurez-vous que le texte n'est pas vide après nettoyage
                            page_text_elements.append(text_content)
                
                page_content = '\n'.join(page_text_elements)
                
                # Optionnel: Limiter la taille du contenu pour le LLM si les pages sont très longues
                # max_content_length = 2000 # Par exemple, limiter à 2000 caractères
                # if len(page_content) > max_content_length:
                #     page_content = page_content[:max_content_length] + "..."

                web_content_results.append({
                    "url": url,
                    "title": page_title,
                    "content": page_content
                })

            except requests.exceptions.Timeout:
                print(f"[{time.strftime('%H:%M:%S')}] Timeout lors du chargement de la page : {url}")
                web_content_results.append({
                    "url": url,
                    "title": ddg_title,
                    "content": "Erreur : Temps de chargement de la page dépassé."
                })
            except requests.exceptions.RequestException as req_err:
                print(f"[{time.strftime('%H:%M:%S')}] Erreur réseau ou HTTP lors de la récupération de la page {url}: {req_err}")
                web_content_results.append({
                    "url": url,
                    "title": ddg_title,
                    "content": f"Erreur lors de la récupération de la page: {req_err}"
                })
            except Exception as general_err:
                print(f"[{time.strftime('%H:%M:%S')}] Une erreur inattendue est survenue lors du traitement de la page {url}: {general_err}")
                web_content_results.append({
                    "url": url,
                    "title": ddg_title,
                    "content": f"Erreur lors du traitement du contenu de la page: {general_err}"
                })
            finally:
                # Ajouter un délai entre chaque requête de scraping pour être poli avec les serveurs
                time.sleep(1) # Délai d'une seconde

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Erreur critique lors de la recherche DuckDuckGo : {e}")
        return {
            "data_to_used": f"Une erreur générale est survenue lors de la recherche web: {e}"
        }
    
    if web_content_results:
        print(f"[{time.strftime('%H:%M:%S')}] Contenu web extrait pour {len(web_content_results)} pages.")
        return {
            "data_to_used": web_content_results
        }
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Aucun contenu pertinent extrait du web.")
        return {
            "data_to_used": "Aucun contenu pertinent n'a pu être extrait des pages web trouvées."
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
    rag_result = compiled_rag_agent.invoke(
        {
    "query" : "How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?",
    "not_in_db": None,
    "data_to_used": None, 
    "messages": [],
    "answer": None
    },
    config={"callbacks": [langfuse_handler]})
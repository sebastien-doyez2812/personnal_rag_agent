from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from dotenv import load_dotenv
from transformers import AutoTokenizer, TFAutoModel
from tqdm import tqdm
import fitz
from fitz import open as fitz_open
import os
import tensorflow as tf

load_dotenv()

FOLDER_DOCS = "data/text_desc_imgs_only" #"data\docs_only"
COLLECTIONS = ["document_only", "image_only"]
WHAT_COLLECTION = "image_only"

# Get the Url and API for qdrant, from the .env file:
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)


#qdrant_client.delete_collection(collection_name="document_only")
#qdrant_client.delete_collection(collection_name="image_only")
                                
# Tokenizer and Embedding:
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = TFAutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def embedding(text):
    tokens = tokenizer(text, return_tensors = "tf", truncation = True, padding = True)
    output = model(**tokens)
    embedding = tf.reduce_mean(output.last_hidden_state, axis = 1)
    
    # embedding is (1,384), need to put it in 384 dim
    embedding = tf.squeeze(embedding, axis= 0)
    return embedding.numpy().tolist()

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz_open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


try:
    # Create the Collections:
    for collection in tqdm(COLLECTIONS) :
        if not qdrant_client.collection_exists(collection_name = collection):
            print(f"\033[93m Create collection {collection} \033[0m")
            qdrant_client.create_collection(
                collection_name= collection,
                vectors_config = models.VectorParams(size=384, distance = models.Distance.COSINE)
            )
            print(f"\033[92m Collection {collection} created! \033[0m")
        else:
            print(f"\033[92m Collection {collection} already exists! \033[0m")

    print(f"COLLECTIONS n Qdrant are {qdrant_client.get_collections()}")

    # Embedding:

    points_list = []

    for idx, pdf_file in tqdm(enumerate(os.listdir(FOLDER_DOCS), start=1 )):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(FOLDER_DOCS, pdf_file)

            # Extraction = 
            text = extract_text_from_pdf(pdf_path)

            if not text.strip():
                raise Exception (f"File {pdf_path} is empty...")
            
            vector = embedding(text)

            point = PointStruct(
                id = idx,
                vector = vector,
                payload = {
                    "file_name": pdf_file,
                    "content": text[:500]
                }
            )
            points_list.append(point)
    qdrant_client.upsert(
        collection_name= WHAT_COLLECTION,
        points= points_list
    )
    print("\033[92m Embedding Done! \033[0m")
except Exception as e:
    print(f"\033[91m {e} \033[0m")
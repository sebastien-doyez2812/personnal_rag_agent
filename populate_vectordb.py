from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from transformers import AutoTokenizer, TFAutoModel

import os
import tensorflow as tf

load_dotenv()

FOLDER_DOCS = "data\docs_only"

# Get the Url and API for qdrant, from the .env file:
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)

# Tokenizer and Embedding:
tokenizer = AutoTokenizer("sentence-transformers/all-MiniLM-L6-v2")
model = TFAutoModel("sentence-transformers/all-MiniLM-L6-v2")



collections = ["document_only", "image_only"]

for collection in collections:
    if not qdrant_client.collection_exists(collection_name = "document_only"):
        print(f"\033[93m Create collection {collection} \033[0m")
        qdrant_client.create_collection(
            collection_name="document_only",
            vectors_config = models.VectorParams(size=1024, distance = models.Distance.COSINE)
        )
        print(f"\033[92m Collection {collection} created! \033[0m")
    else:
        print(f"\033[92m Collection {collection} already exists! \033[0m")

print(qdrant_client.get_collections())


def embedding(text):
    tokens = tokenizer(text, return_tensor = "pt", truncation = True, padding = True)
    output = model(**tokens)
    embedding = tf.reduce_mean(output.last_hidden_state, axis = 1)
    return embedding.numpy().tolist()


for root, dirs, files in os.walk(FOLDER)
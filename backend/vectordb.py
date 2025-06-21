import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf


#######################################
##           VECTORDB Qdrant         ##
#######################################


class VectorDB:
    def __init__(self):
        # Getting API keys keys loacl in the constructor:
        load_dotenv()
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        self.__qdrant_client = QdrantClient(
            url= qdrant_url,
            api_key= qdrant_api_key
        )
        self.__tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.__model = TFAutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")



    def __embedding(self, text):
        tokens = self.__tokenizer(text, return_tensors = "tf", truncation = True, padding = True)
        output = self.__model(**tokens)
        embedding = tf.reduce_mean(output.last_hidden_state, axis = 1)
        
        # embedding is (1,384), need to put it in 384 dim
        embedding = tf.squeeze(embedding, axis= 0)
        return embedding.numpy().tolist()

    def search(self, query, collection_name: str = "document_only", top_k: int = 5):
        vector = self.__embedding(query)
        result = self.__qdrant_client.search(
            collection_name= collection_name,
            query_vector= vector,
            limit = top_k
        )
        return result

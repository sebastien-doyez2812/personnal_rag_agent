from qdrant_client import QdrantClient, models



qdrant_client = QdrantClient()
qdrant_client.create_collection(
    collection_name="document_only",
    vectors_config = models.VectorParams(size=100, distance = models.Distance.COSINE)
)

qdrant_client.create_collection(
    collection_name = "image_only",
    vectors_config = models.VectorParams(size = 100, distance = models.Distance.COSINE)
)
print(qdrant_client.get_collections())
import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec


class PineconeClient:
    def __init__(self, index_name: str):
        self.client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name

    # Control Plane
    def create_index(self, dimension: int = 1536, metric: str = "cosine"):
        if self.index_name not in self.client.list_indexes().names():
            self.client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                )
            )

    def delete_index(self):
        self.client.delete_index(self.index_name)

    # Data Plane
    def upsert_vectors(self, namespace: str, vectors: List[List[float]]):
        """https://docs.pinecone.io/reference/api/data-plane/upsert"""
        index = self.client.Index(self.index_name)
        index.upsert(
            vectors=vectors,
            namespace=namespace
        )

    def delete_vectors(self, namespace: str, delete_all: bool = False, ids: List[str] = None):
        """https://docs.pinecone.io/reference/api/data-plane/delete"""
        index = self.client.Index(self.index_name)
        index.delete(
            namespace=namespace,
            ids=ids,
            deleteAll=delete_all
        )

    def query_index(self, namespace: str, vector: List[float], top_k: int, include_values: bool = False):
        index = self.client.Index(self.index_name)
        query_results = index.query(
            namespace=namespace,
            vector=vector,
            top_k=top_k,
            include_values=include_values
        )
        return query_results

    def describe_index(self):
        index = self.client.Index(self.index_name)
        print(index.describe_index_stats())

    @staticmethod
    def vectors_from_embeds(texts: List[str], embeds: List[List[float]]) -> List[Dict[str, Any]]:
        return [{"id": text, "values": embed} for text, embed in zip(texts, embeds)]

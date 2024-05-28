import os
from typing import List
from pinecone import Pinecone, ServerlessSpec


class PineconeClient:
    def __init__(self, index_name: str):
        self.client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name

    def create_index(self):
        if self.index_name not in self.client.list_indexes().names():
            self.client.create_index(
                name=self.index_name,
                dimension=2,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                )
            )

    def insert_vectors(self, namespace: str, vectors: List[List[float]]):
        index = self.client.Index(self.index_name)
        index.upsert(
            vectors=vectors,
            namespace=namespace
        )

    def query_index(self, namespace: str, vector: List[float], top_k: int):
        index = self.client.Index(self.index_name)
        query_results = index.query(
            namespace=namespace,
            vector=vector,
            top_k=top_k,
            include_values=True
        )
        return query_results

    def describe_index(self):
        index = self.client.Index(self.index_name)
        print(index.describe_index_stats())

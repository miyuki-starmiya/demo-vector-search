import os
from typing import List
from openai import OpenAI


class Embedding:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        res = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [d.embedding for d in res.data]

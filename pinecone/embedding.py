import os
from typing import List
from openai import OpenAI


class Embedding:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @classmethod
    def get_embeddings(cls, texts: List[str]) -> List[List[float]]:
        res = cls.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [d.embedding for d in res.data]

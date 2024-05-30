import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from db import PineconeClient
from embedding import Embedding
from data.kyoani import anime_title_dicts
from util.csv import create_df_from_csv

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client
index_name = "kyoani-title-index"
# namespace = "anime_titles"
namespace = "apparels"
pc = PineconeClient(index_name=index_name)

# Control Plane
# pc.create_index()
# pc.delete_index()
# pc.delete_vectors(namespace=namespace, delete_all=True)

# Data Plane
# dicts = anime_title_dicts
# ja_titles = [dict["ja"] for dict in dicts]
# en_titles = [dict["en"] for dict in dicts]
csv_path = os.path.abspath("./data/apparels.csv")
df = create_df_from_csv(csv_path)
df = df[:10]
embeds = Embedding().get_embeddings(df["jp"].tolist())
vectors = PineconeClient.vectors_from_embeds(df["en"].tolist(), embeds)
pc.upsert_vectors(namespace=namespace, vectors=vectors)

# Query
# query = "ぼっち・ざ・ろっく！"
# query_embed = Embedding().get_embeddings([query])[0]
# query_result = pc.query_index(namespace=namespace, vector=query_embed, top_k=5)
# print(query_result)

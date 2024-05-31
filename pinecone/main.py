import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from db import PineconeClient
from embedding import Embedding
from data.kyoani import anime_title_dicts
from util.csv import create_df_from_csv
from embed import embed

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client
index_name = "image-index"
namespace = "kyoani_key_visuals"
pc = PineconeClient(index_name=index_name)

# Control Plane
# pc.create_index(dimension=1024)
# pc.delete_index()
# pc.delete_vectors(namespace=namespace, delete_all=True)

# Data Plane
# dicts = anime_title_dicts
# ja_titles = [dict["ja"] for dict in dicts]
# en_titles = [dict["en"] for dict in dicts]
# csv_path = os.path.abspath("./data/upto_270000_apparels.csv")
# df = create_df_from_csv(csv_path)
# df = df[900:]
# embeds = Embedding().get_embeddings(df["jp"].tolist())
# vectors = PineconeClient.vectors_from_embeds(["Border"], [embed])
# pc.upsert_vectors(namespace=namespace, vectors=vectors)
# pc.delete_vectors(namespace=namespace, delete_all=True)

# Query
# query = "湊 あくあ フーディー"
# query_embed = Embedding().get_embeddings([query])[0]
query_result = pc.query_index(namespace=namespace, vector=embed, top_k=5)
print(query_result)

import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from embedding import embedding

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

embedding()

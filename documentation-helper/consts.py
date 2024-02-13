import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

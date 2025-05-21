import os

CHROMA_STORAGE_PATH = os.getenv("CHROMA_STORAGE_PATH", "./backend/data/chroma_storage")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
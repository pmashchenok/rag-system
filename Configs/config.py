import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    APP_PORT = int(os.getenv("APP_PORT", 8000))
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./.chromadb")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 200))
    MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 1024))
    TOP_K = int(os.getenv("TOP_K", 1))
    MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", 100))
    BATCH_SIZE_FOR_INDEXING = int(os.getenv("BATCH_SIZE_FOR_INDEXING", 500))


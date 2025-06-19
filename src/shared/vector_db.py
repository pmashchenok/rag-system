import chromadb
from Configs.config import Config


client = chromadb.Client(settings=chromadb.Settings(
    persist_directory=Config.VECTOR_DB_PATH,
    is_persistent=True
))

collection = client.get_or_create_collection(name=Config.COLLECTION_NAME)




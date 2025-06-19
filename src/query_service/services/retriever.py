from src.utils.logger import logging
from src.shared.vector_db import collection


def search_similar(embedding, top_k=1):
    """"Поиск релевантных документов в БД"""
    logging.info("Выполняем поиск похожих фрагментов в векторной БД...")
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents"]
    )
    docs = results["documents"][0] if results["documents"] else []
    logging.info(f"Найдено {len(docs)} фрагментов")
    return docs

from src.shared.vector_db import collection


def save_embedding(doc):
    """Сохранение оригинального текста и эмбеддинга очищенного текста в БД"""
    collection.add(
        ids=[doc['id']],
        documents=[doc['text']],
        embeddings=[doc['embedding']]
    )


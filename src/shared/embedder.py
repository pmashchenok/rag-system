from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def get_embedding(text):
    """Получаем эмбеддинг по тексту"""
    vector = embedding_model.encode(text)
    return vector

import logging
from fastapi import FastAPI, HTTPException
from src.shared.preprocess import preprocess_text
from src.shared.embedder import get_embedding
from src.models.models import QueryRequest
from src.query_service.services.generate_answer import generate_answer
from src.query_service.services.retriever import search_similar


app = FastAPI()


@app.post("/query")
def query(request: QueryRequest):
    """Запрос к LLM"""
    logging.info(f"Получен запрос: {request.text}")
    processed = preprocess_text(request.text)

    if not processed:
        logging.warning("Не удалось обработать текст запроса")
        raise HTTPException(status_code=400, detail="Невозможно обработать текст")

    emb = get_embedding(processed['cleaned_text'])
    similar_chunks = search_similar(emb, top_k=request.top_k)
    answer = generate_answer(similar_chunks, request.text)

    logging.info("Запрос обработан успешно")
    return {"answer": answer, "context": similar_chunks}

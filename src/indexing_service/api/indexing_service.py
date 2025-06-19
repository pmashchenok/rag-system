from fastapi import FastAPI, HTTPException
from src.models.models import Document
from typing import List
from src.shared.preprocess import preprocess_text
from src.shared.embedder import get_embedding
from src.shared.vector_db import collection
from src.indexing_service.services.save_to_db import save_embedding
from Configs.config import Config

app = FastAPI()


@app.post("/indexing")
def indexing(documents: List[Document]):
    """" Индексация документов"""
    try:
        input_ids = [str(item.uid) for item in documents]
        existing_ids = set()
        for i in range(0, len(input_ids), Config.BATCH_SIZE_FOR_INDEXING):
            batch = input_ids[i:i + Config.BATCH_SIZE_FOR_INDEXING]
            existing = collection.get(ids=batch)
            if existing and 'ids' in existing:
                existing_ids.update(existing['ids'])

        count = 0
        for item in documents:
            uid = str(item.uid)

            if uid in existing_ids:
                continue

            if len(item.text) < Config.MIN_TEXT_LENGTH:
                continue

            doc = preprocess_text(item.text)
            if not doc:
                continue

            vector = get_embedding(doc['cleaned_text'])
            embedded_doc = {
                'id': uid,
                'text': doc['original_text'],
                'embedding': vector
            }
            save_embedding(embedded_doc)
            count += 1

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": f"Indexed {count} documents"}


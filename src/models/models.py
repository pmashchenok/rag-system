from pydantic import BaseModel
from typing import Any
from Configs.config import Config


class Document(BaseModel):
    """Модель документа пользователя"""
    uid: Any
    text: str


class QueryRequest(BaseModel):
    """Модель запроса пользователя"""
    text: str
    top_k: int = Config.TOP_K


from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(BaseModel):
    """A class to hold the state of the RAG process."""
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
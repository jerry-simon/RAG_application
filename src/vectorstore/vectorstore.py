from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    """A class to handle vector store operations using FAISS."""

    def __init__(self):
        self.embedding = OpenAIEmbeddings()
        self.vector_store = None
        self.retriever = None

    def create_retriever(self, documents: List[Document]):
        """Create a FAISS vector store and retriever from the provided documents."""
        self.vector_store = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vector_store.as_retriever()

    def get_retriever(self):
        """Get the retriever for querying the vector store."""
        if self.retriever is None:
            raise ValueError("Retriever has not been created. Call create_retriever first.")
        return self.retriever
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve documents relevant to the query using the retriever."""
        if self.retriever is None:
            raise ValueError("Retriever has not been created. Call create_retriever first.")
        return self.retriever.get_relevant_documents(query)
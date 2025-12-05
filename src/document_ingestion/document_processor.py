from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
    )

class DocumentProcessor:
    """A class to process and split documents from various sources."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load and split document from a web URL."""
        loader = WebBaseLoader(url)
        return loader.load()
    
    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load and split document from a PDF file."""
        loader = PyPDFDirectoryLoader(str(str(directory)))
        return loader.load()
        
    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load and split document from a text file."""
        loader = TextLoader(str(file_path), encoding='utf8')
        return loader.load()
    
    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load and split document from a PDF file."""
        loader = PyPDFDirectoryLoader(str("data"))
        return loader.load()
    
    def load_documents(self, source: str) -> List[Document]:
        """Load and split documents based on the source type."""
        docs: List[Document] = []

        for src in source:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))

            path = Path("data")

            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))

            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))

            else:
                raise ValueError(f"Unsupported source type: {src}. Use URL or pdf directory.")
            
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        
        return self.text_splitter.split_documents(documents)
    
    def process_url(self, urls: List[str]) -> List[Document]:
        """Load and split documents from the given source."""
        documents = self.load_documents(urls)
        return self.split_documents(documents)


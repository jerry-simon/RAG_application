import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

class Config:
    """Configuration class to initialize LLM and other settings."""

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    LLM_MODEL = "openai:gpt-4o"

    DOC_CHUNK_SIZE: int = 500
    DOC_CHUNK_OVERLAP: int = 50

    DEFAULT_URLS = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning"]
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the chat model."""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:

    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    VOYAGEAI_API_KEY = os.getenv("VOYAGEAI_API_KEY")

    # Pincone Settings
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us_west1-gcp")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "research-assistant")

    # Model Settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "senetence_transformers/all-MiniLM-L6_v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001-v1:0")

    # Chunking Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Database SEttings
    DATABASE_PATH = os.getenv("DATABASE_PATH", "data/research_assistant.db")

    #Retrieval Settings
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

    #Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"

    @classmethod
    def validate(cls):
        """Validae that required environment variables are set."""
        required_vars = [
            "ANTHROPIC_API_KEY",
            "PINECONE_API_KEY"
        ]

        missing = []
        for var in required_vars:
            if not getattr(cls, var):
                missing.append(var)

            if missing:
                raise ValueError(
                    f"Missing required environemtn variables: {', '.join(missing)}\n"
                    f"Please et the min your .env file or environment"
                )
        return True
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exits."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        return True
    
config = Config()
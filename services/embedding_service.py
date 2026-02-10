from typing import List, Union
import numpy as np
import openai
import voyageai
from sentence_transformers import SentenceTransformer
from utils.config import config
from utils.text_processing import chunk_text

class EmbeddingService:
    def __init__(self):
        self.model_name = config.EMBEDDING_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.use_openai = False
            print(f"Loading embedding model: {self.model_name}")
            self.model = voyageai.Client(api_key=config.VOYAGEAI_API_KEY)
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        try:
            result = self.model.embed(
                text,
                self.model_name
            )
            return result.embeddings
        except Exception as e:
            print(f"Error generating embedding:{e}")
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            result = self.model.embed(
                texts,
                self.model_name
            )
            return result.embeddings
        except Exception as e:
            print(f"Error generating embedding:{e}")
    
    def _generate_local_embeddings(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _generate_openai_embedding(self, text: str) -> List[float]:
        try:
            openai.api_key = config.OPENAI_API_KEY

            response = openai.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embedding: {e}")
            raise
    
    def chunk_and_embed(
            self,
            text: str,
            chunk_size: int = None,
            chunk_overlap: int = None,
    ) -> List[dict]:
        chunk_size = chunk_size or config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        chunks = chunk_text(text, chunk_size, chunk_overlap)

        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_embeddings(chunk_texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding

        return chunks
    
    @property
    def embedding_dimension(self) -> int:
        return 1024 ##voyage-4

embedding_service = EmbeddingService()
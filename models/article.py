from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, HttpUrl, Field
from uuid import uuid4

class Article(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    url: str
    title: str
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    date_added: datetime = Field(default_factory=datetime.now)
    content: str
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=List)
    read_status: bool = False
    word_count: int = 0

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ArticleChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    article_id: str
    chunk_index: int
    text: str
    embedding: Optional[List[float]] = None
    article_title: str
    article_url: str
    article_author: Optional[str] = None
    article_date: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ArticleMetadata(BaseModel):
    title: str
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    description: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SearchResult(BaseModel):
    article_id: str
    article_title: str
    article_url: str
    chunk_text: str
    score: float
    metadata: dict = Field(default_factory=dict)

class QueryResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    query: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


import sqlite3
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from models import Article, ArticleChunk, SearchResult
from services.embedding_service import embedding_service
from utils.config import config

class DatabaseService:
    
    def __init__(self):
        self.pc = None
        self.index = None
        self.conn = None
        self._init_pinecone()
        self._init_sqlite()
    
    def _init_pinecone(self):
        try:
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            
            # Check if index exists, create if not
            index_name = config.PINECONE_INDEX_NAME
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
                print(f"Creating Pinecone index: {index_name}")

                dimension = embedding_service.embedding_dimension
                
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Index created successfully")
            
            self.index = self.pc.Index(index_name)
            print(f"Connected to Pinecone index: {index_name}")
            
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise
    
    def _init_sqlite(self):
        try:
            config.ensure_directories()
            db_path = config.DATA_DIR / "research_assistant.db"
            
            self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            # Create tables
            self._create_tables()
            print(f"Connected to SQLite database: {db_path}")
            
        except Exception as e:
            print(f"Error initializing SQLite: {e}")
            raise
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # Articles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                author TEXT,
                publish_date TEXT,
                date_added TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                tags TEXT,
                read_status INTEGER DEFAULT 0,
                word_count INTEGER DEFAULT 0
            )
        """)
        
        # Chunks table (for reference)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                article_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
            )
        """)
        
        self.conn.commit()
    
    def save_article(self, article: Article) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO articles 
                (id, url, title, author, publish_date, date_added, content, 
                 summary, tags, read_status, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article.id,
                article.url,
                article.title,
                article.author,
                article.publish_date.isoformat() if article.publish_date else None,
                article.date_added.isoformat(),
                article.content,
                article.summary,
                json.dumps(article.tags),
                1 if article.read_status else 0,
                article.word_count
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving article to SQLite: {e}")
            return False
    
    def save_chunks_to_pinecone(self, chunks: List[ArticleChunk]) -> bool:
        try:
            vectors = []
            for chunk in chunks:
                metadata = {
                    "article_id": chunk.article_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "article_title": chunk.article_title,
                    "article_url": chunk.article_url,
                    "article_author": chunk.article_author or "",
                    "article_date": chunk.article_date.isoformat() if chunk.article_date else "",
                }
                
                vectors.append({
                    "id": chunk.id,
                    "values": chunk.embedding,
                    "metadata": metadata
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)
            
            # Also save chunk references to SQLite
            cursor = self.conn.cursor()
            for chunk in chunks:
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks (id, article_id, chunk_index, text)
                    VALUES (?, ?, ?, ?)
                """, (chunk.id, chunk.article_id, chunk.chunk_index, chunk.text))
            self.conn.commit()
            
            return True
        except Exception as e:
            print(f"Error saving chunks to Pinecone: {e}")
            return False
    
    def query_pinecone(
        self, 
        query_embedding: List[float], 
        top_k: int = None,
        filter_dict: Dict[str, Any] = None
    ) -> List[SearchResult]:
        try:
            top_k = top_k or config.TOP_K_RESULTS
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            search_results = []
            for match in results.matches:
                search_results.append(SearchResult(
                    article_id=match.metadata.get("article_id", ""),
                    article_title=match.metadata.get("article_title", ""),
                    article_url=match.metadata.get("article_url", ""),
                    chunk_text=match.metadata.get("text", ""),
                    score=match.score,
                    metadata=match.metadata
                ))
            
            return search_results
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []
    
    def get_article_by_id(self, article_id: str) -> Optional[Article]:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
            row = cursor.fetchone()
            
            if row:
                return Article(
                    id=row['id'],
                    url=row['url'],
                    title=row['title'],
                    author=row['author'],
                    publish_date=datetime.fromisoformat(row['publish_date']) if row['publish_date'] else None,
                    date_added=datetime.fromisoformat(row['date_added']),
                    content=row['content'],
                    summary=row['summary'],
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    read_status=bool(row['read_status']),
                    word_count=row['word_count']
                )
            return None
        except Exception as e:
            print(f"Error getting article: {e}")
            return None
    
    def get_all_articles(self, limit: int = 100, offset: int = 0) -> List[Article]:
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM articles 
                ORDER BY date_added DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            articles = []
            for row in cursor.fetchall():
                articles.append(Article(
                    id=row['id'],
                    url=row['url'],
                    title=row['title'],
                    author=row['author'],
                    publish_date=datetime.fromisoformat(row['publish_date']) if row['publish_date'] else None,
                    date_added=datetime.fromisoformat(row['date_added']),
                    content=row['content'],
                    summary=row['summary'],
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    read_status=bool(row['read_status']),
                    word_count=row['word_count']
                ))
            
            return articles
        except Exception as e:
            print(f"Error getting articles: {e}")
            return []
    
    def delete_article(self, article_id: str) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM articles WHERE id = ?", (article_id,))
            cursor.execute("DELETE FROM chunks WHERE article_id = ?", (article_id,))
            self.conn.commit()
            
            # Delete from Pinecone (delete by metadata filter)
            # Note: This requires fetching chunk IDs first
            cursor.execute("SELECT id FROM chunks WHERE article_id = ?", (article_id,))
            chunk_ids = [row['id'] for row in cursor.fetchall()]
            
            if chunk_ids:
                self.index.delete(ids=chunk_ids)
            
            return True
        except Exception as e:
            print(f"Error deleting article: {e}")
            return False
    
    def article_exists(self, url: str) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM articles WHERE url = ?", (url,))
            return cursor.fetchone() is not None
        except Exception as e:
            print(f"Error checking article existence: {e}")
            return False
database_service = DatabaseService()

"""Service for ingesting articles from URLs using Playwright for better compatibility."""
from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
import trafilatura
from datetime import datetime
from typing import Optional, Tuple
from urllib.parse import urlparse
import validators
import time

from models import Article, ArticleChunk, ArticleMetadata
from services.embedding_service import embedding_service
from services.database_service import database_service
from utils.text_processing import clean_text, extract_keywords
from utils.config import config

class IngestionService:
    def __init__(self):
      #Initialize the ingestion service with Playwright
        self.playwright = None
        self.browser = None
        self._init_browser()
    
    def _init_browser(self):
        #Initialize Playwright browser (lazy loading)

        pass
    
    def _get_browser(self) -> Browser:
        #Get or create browser instance
        if self.browser is None:
            print("Initializing Playwright browser...")
            self.playwright = sync_playwright().start()

            self.browser = self.playwright.chromium.launch(
                headless=True,  # Run without opening window
                args=[
                    '--disable-blink-features=AutomationControlled',  # Hide automation
                    '--no-sandbox',
                    '--disable-dev-shm-usage'
                ]
            )
            print("Browser ready!")
        return self.browser
    
    def _fetch_with_playwright(self, url: str) -> Tuple[bool, str, str]:
        try:
            browser = self._get_browser()
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York'
            )
            
            page = context.new_page()
            
            # Navigate to the page
            print(f"Loading page: {url}")
            response = page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            # Check response status
            if response and response.status >= 400:
                context.close()
                return False, "", f"HTTP {response.status} error"
            
            # Wait a bit for JavaScript to load content
            page.wait_for_timeout(2000)
            
            # Get the HTML
            html = page.content()
            
            # Close context
            context.close()
            
            return True, html, ""
            
        except PlaywrightTimeout:
            return False, "", "Page load timeout (30 seconds)"
        except Exception as e:
            return False, "", f"Browser error: {str(e)}"
    
    def fetch_article(self, url: str) -> Tuple[bool, str, Optional[Article]]:

        if not validators.url(url):
            return False, "Invalid URL format", None
        
        if database_service.article_exists(url):
            return False, "Article already exists in your library", None
        
        try:
            print(f"Fetching URL with Playwright: {url}")
            success, html, error_msg = self._fetch_with_playwright(url)
            
            if not success:
                return False, f"Error fetching URL: {error_msg}", None
            
            # Extract content
            print("Extracting content...")
            content, metadata = self._extract_content(html, url)
            
            if not content or len(content.strip()) < 100:
                return False, "Could not extract meaningful content from the article", None
            
            content = clean_text(content)
            
            print("Generating summary...")
            summary = self._generate_summary(content)
            
            keywords = extract_keywords(content, top_n=5)
            
            article = Article(
                url=url,
                title=metadata.title,
                author=metadata.author,
                publish_date=metadata.publish_date,
                content=content,
                summary=summary,
                tags=keywords,
                word_count=len(content.split())
            )
            
            return True, "Article fetched successfully", article
            
        except Exception as e:
            return False, f"Error processing article: {str(e)}", None
    
    def _extract_content(self, html: str, url: str) -> Tuple[str, ArticleMetadata]:
        content = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False
        )
        

        try:
            soup = BeautifulSoup(html, 'lxml')
        except:
            soup = BeautifulSoup(html, 'html5lib')
        
        # Extract title
        title = None
        if soup.title:
            title = soup.title.string
        elif soup.find('meta', property='og:title'):
            title = soup.find('meta', property='og:title')['content']
        elif soup.find('h1'):
            title = soup.find('h1').get_text()
        
        if not title:
            title = urlparse(url).path.split('/')[-1] or "Untitled"
        
        # Extract author
        author = None
        author_meta = soup.find('meta', {'name': 'author'}) or \
                     soup.find('meta', property='article:author')
        if author_meta:
            author = author_meta.get('content')
        
        # Extract publish date
        publish_date = None
        date_meta = soup.find('meta', property='article:published_time') or \
                   soup.find('meta', {'name': 'publish_date'})
        if date_meta:
            try:
                date_str = date_meta.get('content')
                publish_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                pass
        
        # Extract description
        description = None
        desc_meta = soup.find('meta', {'name': 'description'}) or \
                   soup.find('meta', property='og:description')
        if desc_meta:
            description = desc_meta.get('content')
        
        metadata = ArticleMetadata(
            title=title.strip() if title else "Untitled",
            author=author,
            publish_date=publish_date,
            description=description
        )
        
        return content or "", metadata
    
    def _generate_summary(self, content: str, max_length: int = 300) -> str:
        try:
            from anthropic import Anthropic
            
            client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
            
            # Truncate content if too long (to fit in context)
            max_content_length = 4000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            prompt = f"""Provide a concise summary of the following article in 3-4 sentences. 
Focus on the main points and key takeaways.

Article:
{content}

Summary:"""
            
            message = client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = message.content[0].text.strip()
            return summary
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            # Fallback: return first few sentences
            sentences = content.split('.')[:3]
            return '. '.join(sentences) + '.'
    
    def process_and_store_article(self, article: Article) -> Tuple[bool, str]:
        try:
            print("Chunking and embedding article...")
            
            chunks_with_embeddings = embedding_service.chunk_and_embed(
                article.content,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            
            article_chunks = []
            for chunk_data in chunks_with_embeddings:
                chunk = ArticleChunk(
                    article_id=article.id,
                    chunk_index=chunk_data['chunk_index'],
                    text=chunk_data['text'],
                    embedding=chunk_data['embedding'],
                    article_title=article.title,
                    article_url=article.url,
                    article_author=article.author,
                    article_date=article.publish_date
                )
                article_chunks.append(chunk)
            
            print(f"Generated {len(article_chunks)} chunks")
            
            # Save to databases
            print("Saving to databases...")
            
            # Save article metadata to SQLite
            if not database_service.save_article(article):
                return False, "Error saving article metadata"
            
            # Save chunks to Pinecone
            if not database_service.save_chunks_to_pinecone(article_chunks):
                return False, "Error saving article chunks"
            
            print("Article saved successfully!")
            return True, f"Article processed and saved: {len(article_chunks)} chunks created"
            
        except Exception as e:
            print(f"Error processing article: {e}")
            return False, f"Error processing article: {str(e)}"
    
    def ingest_article(self, url: str) -> Tuple[bool, str, Optional[Article]]:
        success, message, article = self.fetch_article(url)
        if not success:
            return False, message, None
        
        success, process_message = self.process_and_store_article(article)
        if not success:
            return False, process_message, None
        
        return True, f"Successfully added article: {article.title}", article
    
    def __del__(self):
        try:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
        except:
            pass 

ingestion_service = IngestionService()

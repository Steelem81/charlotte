from typing import List, Optional, Tuple
from anthropic import Anthropic

from models import QueryResponse, SearchResult
from services.embedding_service import embedding_service
from services.database_service import database_service
from utils.config import config

class RetrievalService:
    def __init__(self):
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    
    def search_articles(
            self,
            query: str,
            top_k: int = None,
            filters: dict = None
    ) -> List[SearchResult]:
        try:
            print(f" Searching for: {query}")
            query_embedding = embedding_service.generate_embedding(query)

            results = database_service.query_pinecone(
                query_embedding=query_embedding,
                top_k=top_k or config.TOP_K_RESULTS,
                filter_dict=filters
            )

            print(f"Found {len(results)} relevant chunks")
            return results
        
        except Exception as e:
            print(f"Error searching articles: {e}")
            return []
    
    def generate_answer(
            self,
            question: str,
            context_results: List[SearchResult],
            max_tokens: int=1000
    ) -> str:
        try:
            context_parts = []
            for i, result in enumerate(context_results, 1):
                context_parts.append(
                    f"[Source {i}: {result.article_title}]\n{result.chunk_text}\n"
                )
            context = "\n".join(context_parts)

            prompt = f"""You are a helpful research assistant. Answer the user's question based on the provided context from their saved articles.
            Use only the invormation from the provided context. 
            If the context doesn't contain enough information to answer fully, request more context. 
            Cite sources by mentioning the article titles.
            Be concise but thorough.
            If multiple sources say different things, mention both perspectives.
            Context from saved articles: {context}
            Question: {question}
            Answer:"""

            message = self.client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = message.content[0].text.strip()
            return answer
        
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I'm sorry, but I encountered an error generating an answer: {str(e)}"
        
    def answer_question(
            self,
            question: str,
            top_k: int=None,
    ) -> QueryResponse:
        try:
            results = self.search_articles(question, top_k=top_k)

            if not results:
                return QueryResponse(
                    answer="I couldn't find any relevant information in your saved articles to answer this question. Try addming more articles on this topic",
                    sources=[],
                    query=question
                )
            
            print("Generating answer...")
            answer = self.generate_answer(question, results)

            return QueryResponse(
                answer=answer,
                sources=results,
                query=question
            )
        
        except Exception as e:
            print(f"Error answer question: {e}")
            return QueryResponse(
                answer=f"Error processing your question: {str(e)}",
                sources=[],
                query=question
            )
        
    def find_related_articles(
            self,
            article_id: str,
            top_k: int=5
    ) -> List[SearchResult]:
        try:
            article = database_service.get_article_by_id(article_id)
            if not article:
                return []
            
            query = f"{article.title} {article.summary or ''}"

            results = self.search_articles(query, top_k=top_k * 2)

            related = [r for r in results if r.article_id != article_id]

            seen_ids = set()
            unique_related = []
            for result in related:
                if result.article_id not in seen_ids:
                    seen_ids.add(result.article_id)
                    unique_related.append(result)
                    if len(unique_related) >= top_k:
                        break

            return unique_related
        
        except Exception as e:
            print(f"Error finding related articles: {e}")
            return []
        
    def synthesize_topic(
            self,
            topic: str,
            max_articles: int=10
    ) -> str:
        try:
            results = self.search_articles(topic, top_k=max_articles)

            if not results:
                return "No articles found on this topic."
            
            articles_content = {}
            for result in results:
                if result.article_id not in articles_content:
                    articles_content[result.article_id] = {
                        'title': result.article_title,
                        'url': result.article_url,
                        'chunks': []
                    }
                articles_content[result.article_id]['chunks'].append(result.chunk_text)

            context_parts = []
            for article_data in articles_content.values():
                combined_text = ' '.join(article_data['chunks'])
                context_parts.append(
                    f"From: {article_data['title']}\n{combined_text}\n"
                )

            context = "\n".join(context_parts)

            prompt = f"""Based on the following articles from the user's research library, provide a comprehensive synthesis about: {topic} using the following instructions:
            - Idnetify common themese and key points across articles
            -Note any disagreements or different perspectives
            -Organize the information cohernently
            -Cite which articles support each point

            Articles: {context}
            Synthesis:"""
            
            message = self.client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            synthesis = message.content[0].text.strip()
            return synthesis
        
        except Exception as e:
            print(f"Error synthesizing topic: {e}")
            return f"Error generating synthesis: {str(e)}"

retrieval_service = RetrievalService()
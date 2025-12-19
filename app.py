import gradio as gr
from datetime import datetime
from typing import List, Tuple

from services import ingestion_service, retrieval_service, database_service
from models import Article
from utils.config import config

# Validate configuration on startup
try:
    config.validate()
    config.ensure_directories()
    print("✓ Configuration validated")
except Exception as e:
    print(f"✗ Configuration error: {e}")
    print("Please check your .env file and ensure all required variables are set.")

# ============================================================================
# UI Helper Functions
# ============================================================================

def format_article_for_display(article: Article) -> str:
    """Format an article for display in the UI."""
    author_str = f"by {article.author}" if article.author else ""
    date_str = article.publish_date.strftime("%Y-%m-%d") if article.publish_date else "Date unknown"
    tags_str = ", ".join(article.tags) if article.tags else "No tags"
    
    return f"""
### {article.title}
**URL:** {article.url}  
**Author:** {author_str} | **Published:** {date_str}  
**Added:** {article.date_added.strftime("%Y-%m-%d %H:%M")}  
**Words:** {article.word_count} | **Tags:** {tags_str}

**Summary:**  
{article.summary or "No summary available"}

---
"""

def format_search_results(results) -> str:
    """Format search results for display."""
    if not results:
        return "No results found."
    
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"""
**{i}. {result.article_title}**  
*Relevance: {result.score:.2f}*  
{result.chunk_text[:300]}...  
[Read full article]({result.article_url})

---
""")
    
    return "\n".join(output)

# ============================================================================
# Tab 1: Add Article
# ============================================================================

def add_article(url: str, progress=gr.Progress()) -> Tuple[str, str]:
    """
    Add an article from URL.
    
    Returns:
        Tuple of (status_message, article_preview)
    """
    if not url or not url.strip():
        return "❌ Please enter a URL", ""
    
    url = url.strip()
    
    try:
        progress(0.2, desc="Fetching article...")
        
        # Ingest the article
        success, message, article = ingestion_service.ingest_article(url)
        
        if not success:
            return f"❌ {message}", ""
        
        progress(1.0, desc="Complete!")
        
        # Format article for display
        article_display = format_article_for_display(article)
        
        return f"✅ {message}", article_display
        
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

# ============================================================================
# Tab 2: Library
# ============================================================================

def load_library() -> str:
    """Load and display all articles in the library."""
    try:
        articles = database_service.get_all_articles(limit=50)
        
        if not articles:
            return "Your library is empty. Add some articles to get started!"
        
        output = [f"# Your Research Library ({len(articles)} articles)\n"]
        
        for article in articles:
            output.append(format_article_for_display(article))
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error loading library: {str(e)}"

def search_library(query: str) -> str:
    """Search the library for articles."""
    if not query or not query.strip():
        return "Please enter a search query."
    
    try:
        results = retrieval_service.search_articles(query, top_k=10)
        
        if not results:
            return "No articles found matching your query."
        
        return format_search_results(results)
        
    except Exception as e:
        return f"Error searching library: {str(e)}"

# ============================================================================
# Tab 3: Query
# ============================================================================

def answer_question(question: str, history: List = None) -> Tuple[str, List]:
    """
    Answer a question based on the research library.
    
    Returns:
        Tuple of (cleared input, updated history in messages format)
    """
    if not question or not question.strip():
        return "", history or []
    
    try:
        # Get answer
        response = retrieval_service.answer_question(question)
        
        # Format answer with sources
        answer = f"{response.answer}\n\n"
        
        if response.sources:
            answer += "**Sources:**\n"
            seen_articles = set()
            for i, source in enumerate(response.sources, 1):
                if source.article_id not in seen_articles:
                    seen_articles.add(source.article_id)
                    answer += f"{len(seen_articles)}. [{source.article_title}]({source.article_url})\n"
        
        # Update history - using messages format for Gradio 5.x
        if history is None:
            history = []
        
        # Append user message
        history.append({
            "role": "user",
            "content": question
        })
        
        # Append assistant message
        history.append({
            "role": "assistant", 
            "content": answer
        })
        
        return "", history
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if history is None:
            history = []
        
        # Add error in messages format
        history.append({
            "role": "user",
            "content": question
        })
        history.append({
            "role": "assistant",
            "content": error_msg
        })
        
        return "", history

def synthesize_topic(topic: str) -> str:
    """Synthesize information about a topic from multiple articles."""
    if not topic or not topic.strip():
        return "Please enter a topic to synthesize."
    
    try:
        synthesis = retrieval_service.synthesize_topic(topic, max_articles=10)
        return synthesis
    except Exception as e:
        return f"Error generating synthesis: {str(e)}"

# ============================================================================
# Create Gradio Interface
# ============================================================================

with gr.Blocks(theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🔬 Personal Research Assistant
    
    Save, organize, and query your research articles with AI-powered search and insights.
    """)
    
    with gr.Tabs():
        
        # ========== Tab 1: Add Article ==========
        with gr.Tab("➕ Add Article"):
            gr.Markdown("### Add a new article to your research library")
            
            with gr.Row():
                with gr.Column(scale=3):
                    url_input = gr.Textbox(
                        label="Article URL",
                        placeholder="https://example.com/article",
                        lines=1
                    )
                with gr.Column(scale=1):
                    add_btn = gr.Button("Add Article", variant="primary", size="lg")
            
            status_output = gr.Markdown(label="Status")
            article_preview = gr.Markdown(label="Article Preview")
            
            add_btn.click(
                fn=add_article,
                inputs=[url_input],
                outputs=[status_output, article_preview]
            )
            
            gr.Markdown("""
            **Tips:**
            - Paste any article URL from blogs, news sites, or research papers
            - The system will automatically extract content, generate a summary, and index it for search
            - Processing typically takes 10-30 seconds
            """)
        
        # ========== Tab 2: Library ==========
        with gr.Tab("📚 Library"):
            gr.Markdown("### Browse and search your research library")
            
            with gr.Row():
                with gr.Column():
                    search_input = gr.Textbox(
                        label="Search your library",
                        placeholder="Enter keywords or topics...",
                        lines=1
                    )
                    search_btn = gr.Button("Search", variant="primary")
                
            library_output = gr.Markdown(label="Library Contents")
            
            # Load library on tab open
            load_btn = gr.Button("Refresh Library", variant="secondary")
            load_btn.click(
                fn=load_library,
                outputs=[library_output]
            )
            
            search_btn.click(
                fn=search_library,
                inputs=[search_input],
                outputs=[library_output]
            )
            
            # Auto-load on startup
            app.load(fn=load_library, outputs=[library_output])
        
        # ========== Tab 3: Query ==========
        with gr.Tab("💬 Ask Questions"):
            gr.Markdown("### Ask questions about your research")
            
            chatbot = gr.Chatbot(
                label="Research Assistant",
                height=400,
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What does the research say about...?",
                    lines=2,
                    scale=4
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1)
            
            clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            # Handle question submission
            ask_btn.click(
                fn=answer_question,
                inputs=[question_input, chatbot],
                outputs=[question_input, chatbot]
            )
            
            question_input.submit(
                fn=answer_question,
                inputs=[question_input, chatbot],
                outputs=[question_input, chatbot]
            )
            
            clear_btn.click(
                fn=lambda: ([], ""),
                outputs=[chatbot, question_input]
            )
            
            gr.Markdown("""
            **Example questions:**
            - "What are the main arguments about X?"
            - "Compare the perspectives on Y from different articles"
            - "Summarize what I've saved about Z"
            """)
        
        # ========== Tab 4: Synthesize ==========
        with gr.Tab("🔍 Synthesize Topics"):
            gr.Markdown("### Generate comprehensive syntheses across multiple articles")
            
            topic_input = gr.Textbox(
                label="Topic to Synthesize",
                placeholder="e.g., 'machine learning interpretability' or 'climate change mitigation'",
                lines=2
            )
            
            synthesize_btn = gr.Button("Generate Synthesis", variant="primary")
            
            synthesis_output = gr.Markdown(label="Synthesis")
            
            synthesize_btn.click(
                fn=synthesize_topic,
                inputs=[topic_input],
                outputs=[synthesis_output]
            )
            
            gr.Markdown("""
            **What is synthesis?**  
            The assistant will analyze multiple articles on your topic and provide:
            - Common themes and patterns
            - Different perspectives and debates
            - Key insights and takeaways
            - Organized overview of the research landscape
            """)
    
    gr.Markdown("""
    ---
    **About:** This research assistant helps you save articles, search semantically across your library, 
    and get AI-powered answers to your questions based on your saved research.
    """)

# ============================================================================
# Launch App
# ============================================================================

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",  # Changed from 0.0.0.0 to localhost IP
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
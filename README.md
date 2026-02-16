# Charlotte - AI Research Assistant

**Intelligent research library with AI-powered chat and summarization**

Charlotte helps researchers and learners save, organize, and interact with web articles. Instead of bookmarking and forgetting, you can ask questions across your entire research library.

## Problem

Researching online means:
- Dozens of browser tabs you'll never read
- Bookmarks that pile up unused
- No way to search *across* multiple articles
- Forgetting where you read something important

## Solution

Charlotte scrapes, cleans, and indexes web articles into a vector database. You can then:
- **Ask questions** across your entire library
- **Get summaries** of long articles instantly
- **Find relevant sources** automatically with semantic search

## Features

- 🌐 **URL-based ingestion** - Paste a URL, Charlotte handles the rest
- 🧹 **Intelligent content extraction** - Removes ads, menus, and junk
- 🧠 **Vector database storage** - Semantic search powered by embeddings
- 💬 **AI chatbot** - Ask questions, get answers with source citations
- 📝 **Automatic summarization** - Get the key points instantly
- 🔍 **Semantic search** - Find relevant content even without exact keywords

## Tech Stack

- **Web Scraping:** Playwright (handles JavaScript-heavy sites)
- **Vector Database:** Pinecone (scalable semantic search)
- **LLM:** Anthropic Claude (reasoning + summarization)
- **Language:** Python
- **Embeddings:** OpenAI
- **Gradio:** FrontEnd UI

## Architecture


URL Input → Playwright Scraper → Content Cleaner → Embedding Generator → Pinecone
↓
User Query → Embedding → Semantic Search ← Pinecone Vector DB
↓
Relevant Chunks → Claude LLM → Answer + Citations


## Installation

### Prerequisites
- Python 3.8+
- Pinecone API key
- Anthropic API key

### Setup

```bash
# Clone the repository
git clone https://github.com/Steelem81/charlotte.git
cd charlotte

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# PINECONE_API_KEY=your_pinecone_key
# ANTHROPIC_API_KEY=your_anthropic_key
```

Usage

Add an Article
```bash
python main.py add "https://example.com/article"
```
Chat with Your Library
```bash
python main.py chat
> What are the main security risks of LLMs?
```
Summarize an Article
```bash
python main.py summarize "https://example.com/article"
```
Example Use Cases

For Students:

• Build a knowledge base from course readings
• Ask questions when studying for exams
• Quickly review key concepts before class
For Researchers:

• Organize papers and articles by topic
• Find connections across multiple sources
• Generate literature review summaries
For Professionals:

• Track industry news and trends
• Quickly reference past research
• Share knowledge with team members
Security Considerations

• API keys stored in .env (never committed to git)
• Input validation on URLs to prevent malicious sites
• Rate limiting on scraping to respect website policies
• Private Pinecone index (data not shared publicly)
What I Learned

• Web scraping challenges: Handling dynamic JavaScript content with Playwright vs simple requests
• Embeddings: How semantic search differs from keyword search, and choosing chunk sizes
• RAG architecture: Balancing retrieval accuracy with context window limits
• Production considerations: Error handling when scraping fails, managing API rate limits
Challenges & Solutions

Challenge: Some websites block automated scrapers
Solution: Playwright with realistic browser headers and delays

Challenge: Articles too long for context window
Solution: Chunking with overlap + retrieve only top-k relevant chunks

Challenge: LLM sometimes "hallucinates" beyond retrieved context
Solution: Explicit prompt instructions to cite sources and stay grounded

Future Improvements
- [x] Web UI instead of CLI
- [x] Update to use VoyageAI
- [x] Simplify Embedding service
- [x] Update => Gradio6
- [ ] Support for PDFs and academic papers
- [ ] Multi-language support
- [ ] Export notes/summaries to Markdown
- [ ] Share libraries with collaborators
- [ ] Browser extension for one-click saves
- [ ] Cost optimization (local embeddings vs API)

Performance
- Scraping: ~2-5 seconds per article
- Embedding: ~1 second per article chunk
- Query response: ~2-3 seconds with source retrieval
- Vector DB: Scales to 10,000+ articles
Demo

(Coming soon - screenshots and video walkthrough)

Contributing
This is a personal learning project, but suggestions welcome! Open an issue if you find bugs or have ideas.

License
MIT License - feel free to use for your own research!


Built by Max Steele
Learning AI systems architecture | Security-first development

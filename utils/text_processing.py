"""Text processing utilities for chunking and cleaning."""
import re
from typing import List, Dict
import tiktoken

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough estimate
        return len(text) // 4

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"]', '', text)
    
    # Remove multiple periods
    text = re.sub(r'\.{2,}', '.', text)
    
    return text.strip()

def chunk_text(
    text: str, 
    chunk_size: int = 600, 
    chunk_overlap: int = 100,
    preserve_sentences: bool = True
) -> List[Dict[str, any]]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target size in tokens
        chunk_overlap: Overlap between chunks in tokens
        preserve_sentences: Try to split on sentence boundaries
    
    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    if preserve_sentences:
        return _chunk_by_sentences(text, chunk_size, chunk_overlap)
    else:
        return _chunk_by_tokens(text, chunk_size, chunk_overlap)

def _chunk_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Chunk text by sentences to preserve semantic boundaries."""
    # Split into sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = count_tokens(sentence)
        
        # If single sentence exceeds chunk size, split it
        if sentence_tokens > chunk_size:
            if current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'chunk_index': len(chunks),
                    'start_sentence': i - len(current_chunk),
                    'end_sentence': i - 1
                })
                current_chunk = []
                current_tokens = 0
            
            # Split long sentence by words
            words = sentence.split()
            temp_chunk = []
            temp_tokens = 0
            
            for word in words:
                word_tokens = count_tokens(word)
                if temp_tokens + word_tokens > chunk_size and temp_chunk:
                    chunks.append({
                        'text': ' '.join(temp_chunk),
                        'chunk_index': len(chunks),
                        'start_sentence': i,
                        'end_sentence': i
                    })
                    # Keep overlap
                    overlap_words = []
                    overlap_tokens = 0
                    for w in reversed(temp_chunk):
                        w_tokens = count_tokens(w)
                        if overlap_tokens + w_tokens <= chunk_overlap:
                            overlap_words.insert(0, w)
                            overlap_tokens += w_tokens
                        else:
                            break
                    temp_chunk = overlap_words
                    temp_tokens = overlap_tokens
                
                temp_chunk.append(word)
                temp_tokens += word_tokens
            
            if temp_chunk:
                chunks.append({
                    'text': ' '.join(temp_chunk),
                    'chunk_index': len(chunks),
                    'start_sentence': i,
                    'end_sentence': i
                })
            continue
        
        # Check if adding this sentence exceeds chunk size
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunks.append({
                'text': ' '.join(current_chunk),
                'chunk_index': len(chunks),
                'start_sentence': i - len(current_chunk),
                'end_sentence': i - 1
            })
            
            # Keep overlap sentences
            overlap_chunk = []
            overlap_tokens = 0
            for sent in reversed(current_chunk):
                sent_tokens = count_tokens(sent)
                if overlap_tokens + sent_tokens <= chunk_overlap:
                    overlap_chunk.insert(0, sent)
                    overlap_tokens += sent_tokens
                else:
                    break
            
            current_chunk = overlap_chunk
            current_tokens = overlap_tokens
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add remaining chunk
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'chunk_index': len(chunks),
            'start_sentence': len(sentences) - len(current_chunk),
            'end_sentence': len(sentences) - 1
        })
    
    return chunks

def _chunk_by_tokens(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Simple chunking by token count."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for word in words:
        word_tokens = count_tokens(word)
        
        if current_tokens + word_tokens > chunk_size and current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'chunk_index': len(chunks)
            })
            
            # Keep overlap
            overlap_chunk = []
            overlap_tokens = 0
            for w in reversed(current_chunk):
                w_tokens = count_tokens(w)
                if overlap_tokens + w_tokens <= chunk_overlap:
                    overlap_chunk.insert(0, w)
                    overlap_tokens += w_tokens
                else:
                    break
            
            current_chunk = overlap_chunk
            current_tokens = overlap_tokens
        
        current_chunk.append(word)
        current_tokens += word_tokens
    
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'chunk_index': len(chunks)
        })
    
    return chunks

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Extract simple keywords from text (basic implementation)."""
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                  'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
                  'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    
    # Get words, lowercase, filter
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    words = [w for w in words if w not in stop_words]
    
    # Count frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]
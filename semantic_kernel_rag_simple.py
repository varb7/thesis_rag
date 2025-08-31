#!/usr/bin/env python3
"""
Simplified Semantic Kernel Enhanced Agentic RAG System
Compatible with current Semantic Kernel versions
"""

import os
import uuid
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

# Semantic Kernel imports - Simplified for compatibility
try:
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    SEMANTIC_KERNEL_AVAILABLE = True
except ImportError:
    print("⚠️  Semantic Kernel not available, falling back to OpenAI-only mode")
    SEMANTIC_KERNEL_AVAILABLE = False

# OpenAI and Qdrant
import openai
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "agenticragdb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")

@dataclass
class ChatMessage:
    """Individual chat message with metadata"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: float
    message_id: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class QueryContext:
    """Context for a specific query"""
    query: str
    intent: Dict[str, float]
    retrieved_chunks: List[Dict]
    processing_time: float
    metadata: Dict[str, Any]

class ChatMemory:
    """Session-based chat memory management"""
    
    def __init__(self, session_id: str, max_messages: int = 100):
        self.session_id = session_id
        self.max_messages = max_messages
        self.messages: List[ChatMessage] = []
        self.context: Dict[str, Any] = {}
        self.query_history: List[QueryContext] = []
        self.active_documents: List[str] = []
        self.user_preferences: Dict[str, Any] = {}
        self.created_at = time.time()
        self.last_accessed = time.time()
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a new message to chat history"""
        message_id = str(uuid.uuid4())
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            message_id=message_id,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        self.last_accessed = time.time()
        
        # Maintain memory limits
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        return message_id
    
    def get_recent_context(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent messages for context"""
        return self.messages[-limit:] if self.messages else []
    
    def get_relevant_context(self, query: str, limit: int = 5) -> List[ChatMessage]:
        """Get contextually relevant messages for a query"""
        # Simple relevance scoring based on keyword overlap
        query_words = set(query.lower().split())
        
        scored_messages = []
        for message in self.messages:
            message_words = set(message.content.lower().split())
            relevance = len(query_words.intersection(message_words)) / len(query_words) if query_words else 0
            if relevance > 0.1:  # Minimum relevance threshold
                scored_messages.append((message, relevance))
        
        # Sort by relevance and return top results
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        return [msg for msg, score in scored_messages[:limit]]
    
    def add_query_context(self, query_context: QueryContext):
        """Store context for a query"""
        self.query_history.append(query_context)
        self.last_accessed = time.time()
        
        # Keep only recent query history
        if len(self.query_history) > 20:
            self.query_history = self.query_history[-20:]
    
    def get_query_context(self, query: str) -> Optional[QueryContext]:
        """Find similar previous queries for context"""
        query_words = set(query.lower().split())
        
        best_match = None
        best_score = 0
        
        for ctx in self.query_history:
            ctx_words = set(ctx.query.lower().split())
            score = len(query_words.intersection(ctx_words)) / len(query_words) if query_words else 0
            if score > best_score and score > 0.3:  # Minimum similarity threshold
                best_score = score
                best_match = ctx
        
        return best_match
    
    def update_context(self, key: str, value: Any):
        """Update session context"""
        self.context[key] = value
        self.last_accessed = time.time()
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get session context value"""
        return self.context.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'messages': [msg.to_dict() for msg in self.messages],
            'context': self.context,
            'query_history': [asdict(ctx) for ctx in self.query_history],
            'active_documents': self.active_documents,
            'user_preferences': self.user_preferences,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed
        }

class MemoryManager:
    """Global memory management for all sessions"""
    
    def __init__(self, max_sessions: int = 100, session_timeout_hours: int = 24):
        self.sessions: Dict[str, ChatMemory] = {}
        self.max_sessions = max_sessions
        self.session_timeout = timedelta(hours=session_timeout_hours)
    
    def get_session(self, session_id: str) -> ChatMemory:
        """Get or create a session"""
        if session_id not in self.sessions:
            # Clean up old sessions if we're at capacity
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_old_sessions()
            
            self.sessions[session_id] = ChatMemory(session_id)
        
        # Update last accessed time
        self.sessions[session_id].last_accessed = time.time()
        return self.sessions[session_id]
    
    def _cleanup_old_sessions(self):
        """Remove expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_accessed > self.session_timeout.total_seconds():
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    def store_context(self, session_id: str, context: Dict[str, Any]):
        """Store context in a session"""
        session = self.get_session(session_id)
        session.context.update(context)
    
    def retrieve_relevant_context(self, session_id: str, query: str) -> List[Dict[str, Any]]:
        """Get relevant context from a session"""
        session = self.get_session(session_id)
        relevant_messages = session.get_relevant_context(query)
        return [msg.to_dict() for msg in relevant_messages]
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a session"""
        session = self.get_session(session_id)
        return {
            'session_id': session_id,
            'message_count': len(session.messages),
            'query_count': len(session.query_history),
            'active_documents': len(session.active_documents),
            'created_at': datetime.fromtimestamp(session.created_at).isoformat(),
            'last_accessed': datetime.fromtimestamp(session.last_accessed).isoformat()
        }

class EnhancedAgenticRAG:
    """Enhanced RAG system with memory management and Semantic Kernel (if available)"""
    
    def __init__(self, base_path: str = "mineru_out"):
        self.base_path = base_path
        self.memory_manager = MemoryManager()
        
        # Initialize Semantic Kernel if available
        if SEMANTIC_KERNEL_AVAILABLE:
            self.kernel = Kernel()
            self.kernel.add_chat_service(
                "chat-gpt",
                OpenAIChatCompletion(GPT_MODEL, OPENAI_API_KEY)
            )
            print("✅ Semantic Kernel initialized successfully")
        else:
            self.kernel = None
            print("⚠️  Running in OpenAI-only mode")
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        
        # Initialize OpenAI client for embeddings and chat
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        print(f"🚀 Enhanced Agentic RAG System initialized")
        print(f"📁 Base path: {base_path}")
        print(f"🗄️  Qdrant collection: {QDRANT_COLLECTION}")
    
    def detect_intent(self, query: str) -> Dict[str, float]:
        """Detect user intent using OpenAI or Semantic Kernel"""
        system_prompt = """Analyze the user's query and determine their intent with confidence scores (0-1):

INTENT TYPES:
- text_content: Looking for general text content, explanations, descriptions
- table_data: Looking for numerical data, comparisons, statistics, tables
- figure_caption: Looking for figure descriptions, visual content, charts
- methodology: Looking for methods, algorithms, procedures
- results: Looking for experimental results, findings, outcomes

Return JSON with intent and confidence scores."""

        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            # Fallback to default
            return {"text_content": 0.8, "table_data": 0.6, "figure_caption": 0.5}
    
    def expand_query(self, query: str, num_variants: int = 3) -> List[str]:
        """Expand query using OpenAI"""
        system_prompt = f"""Generate {num_variants} focused query variants that would help find relevant information. 
        Each variant should focus on different aspects or use different terminology.
        Return only the queries, one per line."""

        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original query: {query}"}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            variants = [query]  # Include original query
            variants.extend([line.strip() for line in response.choices[0].message.content.split('\n') if line.strip()])
            return variants[:num_variants + 1]
            
        except Exception as e:
            return [query]
    
    def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        """Generate answer using OpenAI"""
        system_prompt = """You are an expert research assistant. Answer questions based on the provided document context.

IMPORTANT RULES:
1. Only use information from the provided context
2. ALWAYS cite specific sources: "According to Source X (PDF Page Y, Section Z)..."
3. Reference the section and page clearly
4. If context is insufficient, acknowledge this clearly
5. Provide comprehensive, well-structured answers
6. Use exact terminology from the documents"""

        context_text = self._format_context(chunks)
        user_prompt = f"""Context from research documents:

{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""

        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            pdf_page = chunk['page'] + 1
            context_parts.append(
                f"Source {i}:\n"
                f"  - Section: {chunk['section']}\n"
                f"  - PDF Page: {pdf_page}\n"
                f"  - Paper: {chunk['doc_id']}\n"
                f"  - Content:\n{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def hyde_retrieval(self, query: str, limit: int = 5) -> List[Dict]:
        """HyDE retrieval using OpenAI"""
        print("🧠 Generating hypothetical answer for HyDE retrieval...")
        
        try:
            # Generate hypothetical answer
            hyde_prompt = f"""Based on the question, write a hypothetical answer that might be found in research documents.
            Make it realistic and detailed, using the kind of language and content you'd expect to find.
            
            Question: {query}
            
            Hypothetical answer:"""
            
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": hyde_prompt}],
                temperature=0.1,  # Lower temperature for better grounding
                max_tokens=100
            )
            
            hypothetical_answer = response.choices[0].message.content
            
            # Search with hypothetical answer
            hyde_embeddings = self.openai_client.embeddings.create(
                model=EMBED_MODEL,
                input=hypothetical_answer
            )
            hyde_vector = hyde_embeddings.data[0].embedding
            
            # Retrieve with HyDE
            hyde_hits = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=hyde_vector,
                limit=limit * 2,
                with_payload=True
            )
            
            # Filter for text blocks only
            text_hyde_hits = [hit for hit in hyde_hits if hit.payload.get('block_type') == 'text']
            
            return self._process_hits(text_hyde_hits)
            
        except Exception as e:
            print(f"HyDE failed: {e}")
            return []
    
    def _process_hits(self, hits) -> List[Dict]:
        """Process Qdrant hits into standardized format"""
        chunks = []
        for hit in hits:
            chunks.append({
                'text': hit.payload['text'],
                'page': hit.payload.get('page', 0),
                'doc_id': hit.payload['doc_id'],
                'block_type': hit.payload['block_type'],
                'section': hit.payload.get('section', 'Unknown'),
                'score': hit.score,
                'vector': hit.vector
            })
        return chunks
    
    def retrieve_with_strategy(self, query: str, limit: int = 8) -> List[Dict]:
        """Multi-strategy retrieval"""
        print(f"🔍 Orchestrating retrieval for: {query}")
        
        # 1. Intent Detection
        intent = self.detect_intent(query)
        print(f"🎯 Detected intent: {max(intent, key=intent.get)} (confidence: {max(intent.values()):.2f})")
        
        # 2. Query Expansion
        query_variants = self.expand_query(query)
        print(f"🔄 Generated {len(query_variants)} query variants")
        
        # 3. Multi-Strategy Retrieval
        all_chunks = []
        
        # Original query retrieval
        original_embeddings = self.openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=query
        )
        original_vector = original_embeddings.data[0].embedding
        
        original_hits = self.qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=original_vector,
            limit=limit * 3,
            with_payload=True
        )
        
        text_hits = [hit for hit in original_hits if hit.payload.get('block_type') == 'text']
        all_chunks.extend(self._process_hits(text_hits))
        
        # HyDE retrieval
        hyde_chunks = self.hyde_retrieval(query, limit=limit)
        all_chunks.extend(hyde_chunks)
        
        # Query variants retrieval
        for variant in query_variants[1:]:
            try:
                variant_embeddings = self.openai_client.embeddings.create(
                    model=EMBED_MODEL,
                    input=variant
                )
                variant_vector = variant_embeddings.data[0].embedding
                
                variant_hits = self.qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=variant_vector,
                    limit=limit,
                    with_payload=True
                )
                
                text_variant_hits = [hit for hit in variant_hits if hit.payload.get('block_type') == 'text']
                all_chunks.extend(self._process_hits(text_variant_hits))
            except Exception as e:
                print(f"Variant retrieval failed: {e}")
        
        # 4. Remove duplicates and apply MMR
        unique_chunks = self._remove_duplicates(all_chunks)
        print(f"📚 Total unique TEXT chunks found: {len(unique_chunks)}")
        
        # 5. MMR diversification
        diversified_chunks = self.mmr_diversification(unique_chunks, original_vector, limit=limit)
        
        return diversified_chunks
    
    def _remove_duplicates(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks based on content similarity"""
        seen = set()
        unique = []
        for chunk in chunks:
            text_hash = hash(chunk['text'][:100])
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(chunk)
        return unique
    
    def mmr_diversification(self, all_hits: List[Dict], query_vector: List[float], 
                          lambda_param: float = 0.5, limit: int = 8) -> List[Dict]:
        """MMR diversification for result selection"""
        print("🔄 Applying MMR diversification...")
        
        if len(all_hits) <= limit:
            return all_hits
        
        import numpy as np
        
        # Convert query vector to numpy array
        query_vec = np.array(query_vector)
        
        # Calculate relevance scores
        relevance_scores = []
        for hit in all_hits:
            if hit.get('vector') is None:
                relevance_scores.append(hit.get('score', 0.0))
                continue
                
            hit_vector = np.array(hit['vector'])
            cos_sim = np.dot(query_vec, hit_vector) / (np.linalg.norm(query_vec) * np.linalg.norm(hit_vector))
            relevance_scores.append(cos_sim)
        
        # MMR selection
        selected = []
        remaining = list(range(len(all_hits)))
        
        # Select first item (highest relevance)
        first_idx = np.argmax(relevance_scores)
        selected.append(all_hits[first_idx])
        remaining.remove(first_idx)
        
        # Select remaining items using MMR
        for _ in range(min(limit - 1, len(remaining))):
            mmr_scores = []
            
            for idx in remaining:
                relevance = relevance_scores[idx]
                
                max_similarity = 0
                for sel_idx in selected:
                    if sel_idx.get('vector') is None or all_hits[idx].get('vector') is None:
                        continue
                    sel_vector = np.array(sel_idx['vector'])
                    curr_vector = np.array(all_hits[idx]['vector'])
                    similarity = np.dot(sel_vector, curr_vector) / (np.linalg.norm(sel_vector) * np.linalg.norm(curr_vector))
                    max_similarity = max(max_similarity, similarity)
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)
            
            best_idx = remaining[np.argmax(mmr_scores)]
            selected.append(all_hits[best_idx])
            remaining.remove(best_idx)
        
        return selected
    
    def enhanced_retrieve_and_answer(self, query: str, session_id: str) -> Tuple[str, List[Dict]]:
        """Enhanced retrieval with memory integration"""
        print(f"🚀 Enhanced retrieval for: {query}")
        
        # Get session memory
        session = self.memory_manager.get_session(session_id)
        
        # Store user query
        session.add_message("user", query, {"type": "query"})
        
        # Check for relevant previous context
        previous_context = session.get_query_context(query)
        if previous_context:
            print(f"📚 Found relevant previous context from {previous_context.query}")
        
        # Get relevant chat history
        relevant_history = session.get_relevant_context(query)
        if relevant_history:
            print(f"💬 Found {len(relevant_history)} relevant chat history items")
        
        # Standard text-based RAG
        print("📚 Using standard text-based RAG...")
        chunks = self.retrieve_with_strategy(query)
        answer = self.generate_answer(query, chunks)
        
        # Store system response
        session.add_message("assistant", answer, {
            "type": "answer",
            "chunks_count": len(chunks),
            "query": query
        })
        
        # Store query context
        query_context = QueryContext(
            query=query,
            intent=self.detect_intent(query),
            retrieved_chunks=chunks,
            processing_time=time.time(),
            metadata={"session_id": session_id}
        )
        session.add_query_context(query_context)
        
        return answer, chunks
    
    def chat(self, session_id: str = None):
        """Enhanced chat loop with memory management"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        print(f"🚀 Enhanced Agentic RAG System Ready! (Session: {session_id[:8]}...)")
        print("Advanced features: Memory Management, Multi-query expansion, HyDE, MMR")
        if SEMANTIC_KERNEL_AVAILABLE:
            print("✅ Semantic Kernel integration enabled")
        else:
            print("⚠️  Running in OpenAI-only mode")
        print("Ask questions about your research documents. Type 'quit' to exit.")
        print("-" * 60)
        
        while True:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            try:
                # Enhanced retrieval with memory support
                answer, sources = self.enhanced_retrieve_and_answer(query, session_id)
                
                # Display results
                print("\n" + "="*70)
                print("🚀 ENHANCED AGENTIC RAG ANSWER:")
                print("="*70)
                print(answer)
                print("="*70)
                
                # Show sources
                if sources:
                    print("\n📚 Sources:")
                    for i, source in enumerate(sources[:3], 1):
                        print(f"  {i}. 📄 Text | Page {source.get('page', 'N/A')} | Section: {source.get('section', 'N/A')}")
                        print(f"     Content: {source.get('text', 'N/A')[:100]}...")
                
                # Show memory info
                session = self.memory_manager.get_session(session_id)
                print(f"\n💾 Memory: {len(session.messages)} messages, {len(session.query_history)} queries")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session"""
        return self.memory_manager.get_session_summary(session_id)
    
    def export_session(self, session_id: str) -> Dict[str, Any]:
        """Export session data"""
        session = self.memory_manager.get_session(session_id)
        return session.to_dict()

if __name__ == "__main__":
    # Initialize Enhanced RAG system
    rag = EnhancedAgenticRAG(base_path="mineru_out")
    
    # Start chat with new session
    rag.chat()







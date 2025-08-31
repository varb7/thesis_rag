#!/usr/bin/env python3
"""
Modular Agentic RAG System
Main system that orchestrates all modular components
"""

import uuid
import time
import asyncio
from typing import List, Dict, Any, Tuple

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# OpenAI and Qdrant
import openai
from qdrant_client import QdrantClient

# Local modules
from config import (
    OPENAI_API_KEY, QDRANT_URL, QDRANT_COLLECTION, EMBED_MODEL, 
    GPT_MODEL, validate_config, DEFAULT_BASE_PATH
)
from memory_manager import MemoryManager, QueryContext
from retrieval_strategies import RetrievalStrategies
from semantic_kernel_functions import SemanticKernelFunctions


class ModularRAGSystem:
    """Modular RAG system with separated concerns"""
    
    def __init__(self, base_path: str = DEFAULT_BASE_PATH):
        # Validate configuration
        validate_config()
        
        self.base_path = base_path
        self.memory_manager = MemoryManager()
        
        # Initialize Semantic Kernel
        self.kernel = Kernel()
        self.kernel.add_service(OpenAIChatCompletion(GPT_MODEL, OPENAI_API_KEY))
        
        # Initialize Semantic Kernel Functions
        self.sk_functions = SemanticKernelFunctions(self.kernel, GPT_MODEL, OPENAI_API_KEY)
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=OPENAI_API_KEY)
        
        # Initialize OpenAI client for embeddings
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize Retrieval Strategies
        self.retrieval_strategies = RetrievalStrategies(
            self.qdrant_client, 
            QDRANT_COLLECTION, 
            EMBED_MODEL, 
            self.openai_client
        )
    
    async def enhanced_retrieve_and_answer(self, query: str, session_id: str) -> Tuple[str, List[Dict]]:
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
        chunks = await self.retrieve_with_strategy(query)
        answer = await self.sk_functions.generate_answer(query, chunks)
        
        # Store system response
        session.add_message("assistant", answer, {
            "type": "answer",
            "chunks_count": len(chunks),
            "query": query
        })
        
        # Store query context
        query_context = QueryContext(
            query=query,
            intent=await self.sk_functions.detect_intent(query),
            retrieved_chunks=chunks,
            processing_time=time.time(),
            metadata={"session_id": session_id}
        )
        session.add_query_context(query_context)
        
        return answer, chunks
    
    async def retrieve_with_strategy(self, query: str, limit: int = 8) -> List[Dict]:
        """Multi-strategy retrieval using Semantic Kernel"""
        print(f"🔍 Orchestrating retrieval for: {query}")
        
        # 1. Intent Detection using Semantic Kernel
        intent = await self.sk_functions.detect_intent(query)
        print(f"🎯 Detected intent: {max(intent, key=intent.get)} (confidence: {max(intent.values()):.2f})")
        
        # 2. Query Expansion using Semantic Kernel
        query_variants = await self.sk_functions.expand_query(query)
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
        all_chunks.extend(self.retrieval_strategies._process_hits(text_hits))
        
        # HyDE retrieval
        hyde_chunks = self.retrieval_strategies.hyde_retrieval(query, limit=limit)
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
                all_chunks.extend(self.retrieval_strategies._process_hits(text_variant_hits))
            except Exception as e:
                print(f"Variant retrieval failed: {e}")
        
        # 4. Remove duplicates and apply MMR
        unique_chunks = self.retrieval_strategies._remove_duplicates(all_chunks)
        print(f"📚 Total unique TEXT chunks found: {len(unique_chunks)}")
        
        # 5. MMR diversification
        diversified_chunks = self.retrieval_strategies.mmr_diversification(unique_chunks, original_vector, limit=limit)
        
        return diversified_chunks
    
    async def chat(self, session_id: str = None):
        """Enhanced chat loop with memory management"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        print(f"🚀 Modular Agentic RAG System Ready! (Session: {session_id[:8]}...)")
        print("Advanced features: Semantic Kernel, Memory Management, Multi-query expansion, HyDE, MMR")
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
                answer, sources = await self.enhanced_retrieve_and_answer(query, session_id)
                
                # Display results
                print("\n" + "="*70)
                print("🚀 MODULAR AGENTIC RAG ANSWER:")
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
    # Initialize Modular RAG system
    rag = ModularRAGSystem(base_path=DEFAULT_BASE_PATH)
    
    # Start chat with new session
    asyncio.run(rag.chat())

#!/usr/bin/env python3
"""
Retrieval Strategies Module for Agentic RAG System
Handles HyDE, MMR diversification, and query expansion
"""

import numpy as np
from typing import List, Dict, Any
from qdrant_client import QdrantClient


class RetrievalStrategies:
    """Collection of retrieval strategies for enhanced RAG"""
    
    def __init__(self, qdrant_client: QdrantClient, collection_name: str, embed_model: str, openai_client):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.openai_client = openai_client
    
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
    
    def hyde_retrieval(self, query: str, limit: int = 5) -> List[Dict]:
        """HyDE retrieval using hypothetical answer generation"""
        print("🧠 Generating hypothetical answer for HyDE retrieval...")
        
        try:
            # Generate hypothetical answer using OpenAI directly
            hyde_prompt = f"""Based on the question, write a hypothetical answer that might be found in research documents.
            Make it realistic and detailed, using the kind of language and content you'd expect to find.
            
            Question: {query}
            
            Hypothetical answer:"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a research assistant that generates hypothetical answers for document retrieval."},
                    {"role": "user", "content": hyde_prompt}
                ]
            )
            hypothetical_answer = response.choices[0].message.content
            
            # Search with hypothetical answer
            hyde_embeddings = self.openai_client.embeddings.create(
                model=self.embed_model,
                input=hypothetical_answer
            )
            hyde_vector = hyde_embeddings.data[0].embedding
            
            # Retrieve with HyDE
            hyde_hits = self.qdrant_client.search(
                collection_name=self.collection_name,
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
    
    def retrieve_with_strategy(self, query: str, limit: int = 8) -> List[Dict]:
        """Multi-strategy retrieval combining multiple approaches"""
        print(f"🔍 Orchestrating retrieval for: {query}")
        
        # 3. Multi-Strategy Retrieval
        all_chunks = []
        
        # Original query retrieval
        original_embeddings = self.openai_client.embeddings.create(
            model=self.embed_model,
            input=query
        )
        original_vector = original_embeddings.data[0].embedding
        
        original_hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=original_vector,
            limit=limit * 3,
            with_payload=True
        )
        
        text_hits = [hit for hit in original_hits if hit.payload.get('block_type') == 'text']
        all_chunks.extend(self._process_hits(text_hits))
        
        # HyDE retrieval
        hyde_chunks = self.hyde_retrieval(query, limit=limit)
        all_chunks.extend(hyde_chunks)
        
        # 4. Remove duplicates and apply MMR
        unique_chunks = self._remove_duplicates(all_chunks)
        print(f"📚 Total unique TEXT chunks found: {len(unique_chunks)}")
        
        # 5. MMR diversification
        diversified_chunks = self.mmr_diversification(unique_chunks, original_vector, limit=limit)
        
        return diversified_chunks

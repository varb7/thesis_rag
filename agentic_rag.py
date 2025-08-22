#!/usr/bin/env python3
"""
Enhanced Agentic RAG System with On-Demand VLM Image Analysis
Advanced retrieval orchestration with multi-query expansion, HyDE, MMR, and intelligent image handling
"""

import os
import base64
import json
from qdrant_client import QdrantClient
import openai
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Tuple
import re
from pathlib import Path
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "agenticragdb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")
VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4-vision-preview")  # For image analysis

# Initialize clients
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

class EnhancedAgenticRAG:
    def __init__(self, base_path: str = None):
        self.openai_client = openai_client
        self.qdrant_client = qdrant_client
        self.base_path = base_path or "mineru_out"
        
    def detect_intent(self, query: str) -> Dict[str, float]:
        """Detect user intent and confidence scores"""
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
            import json
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            # Fallback to default
            return {"text_content": 0.8, "table_data": 0.6, "figure_caption": 0.5}

    def multi_query_expansion(self, query: str, num_variants: int = 3) -> List[str]:
        """Generate multiple query variants for better coverage"""
        system_prompt = """Generate {num_variants} focused query variants that would help find relevant information. 
        Each variant should focus on different aspects or use different terminology.
        Return only the queries, one per line."""

        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt.format(num_variants=num_variants)},
                    {"role": "user", "content": f"Original query: {query}"}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            variants = [query]  # Include original query
            variants.extend([line.strip() for line in response.choices[0].message.content.split('\n') if line.strip()])
            return variants[:num_variants + 1]  # +1 for original query
            
        except Exception as e:
            return [query]  # Fallback to original query

    def hyde_retrieval(self, query: str, limit: int = 5) -> List[Dict]:
        """HyDE: Generate hypothetical answer and use it for retrieval (TEXT ONLY)"""
        print("🧠 Generating hypothetical answer for HyDE retrieval...")
        
        # Generate hypothetical answer
        hyde_prompt = f"""Based on the question, write a hypothetical answer that might be found in research documents.
        Make it realistic and detailed, using the kind of language and content you'd expect to find.
        
        Question: {query}
        
        Hypothetical answer:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": hyde_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            hypothetical_answer = response.choices[0].message.content
            
            # Search with hypothetical answer
            hyde_embeddings = self.openai_client.embeddings.create(
                model=EMBED_MODEL,
                input=hypothetical_answer
            )
            hyde_vector = hyde_embeddings.data[0].embedding
            
            # Retrieve with HyDE - TEXT ONLY
            hyde_hits = self.qdrant_client.search(
                collection_name=COLLECTION,
                query_vector=hyde_vector,
                limit=limit * 2,  # Get more since we're filtering
                with_payload=True
            )
            
            # Filter for text blocks only
            text_hyde_hits = [hit for hit in hyde_hits if hit.payload.get('block_type') == 'text']
            
            return self._process_hits(text_hyde_hits)
            
        except Exception as e:
            print(f"HyDE failed: {e}")
            return []

    def mmr_diversification(self, all_hits: List[Dict], query_vector: List[float], 
                          lambda_param: float = 0.5, limit: int = 8) -> List[Dict]:
        """MMR: Reduce redundancy and ensure diversity in results"""
        print("🔄 Applying MMR diversification...")
        
        if len(all_hits) <= limit:
            return all_hits
        
        # Convert query vector to numpy array
        query_vec = np.array(query_vector)
        
        # Calculate relevance scores (cosine similarity)
        relevance_scores = []
        for hit in all_hits:
            if hit.get('vector') is None:
                # Skip hits without vectors (fallback to score-based selection)
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
                # Relevance to query
                relevance = relevance_scores[idx]
                
                # Max similarity to already selected
                max_similarity = 0
                for sel_idx in selected:
                    if sel_idx.get('vector') is None or all_hits[idx].get('vector') is None:
                        continue
                    sel_vector = np.array(sel_idx['vector'])
                    curr_vector = np.array(all_hits[idx]['vector'])
                    similarity = np.dot(sel_vector, curr_vector) / (np.linalg.norm(sel_vector) * np.linalg.norm(curr_vector))
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)
            
            # Select item with highest MMR score
            best_idx = remaining[np.argmax(mmr_scores)]
            selected.append(all_hits[best_idx])
            remaining.remove(best_idx)
        
        return selected

    def _process_hits(self, hits) -> List[Dict]:
        """Process Qdrant hits into standardized format"""
        chunks = []
        for hit in hits:
            chunks.append({
                'text': hit.payload['text'],
                'page': hit.payload.get('page', 0),  # Default to 0 if not present
                'doc_id': hit.payload['doc_id'],
                'block_type': hit.payload['block_type'],
                'section': hit.payload.get('section', 'Unknown'),  # Add section info
                'score': hit.score,
                'vector': hit.vector  # For MMR
            })
        return chunks

    def retrieve_with_strategy(self, query: str, limit: int = 8) -> List[Dict]:
        """Main retrieval orchestration with multiple strategies"""
        print(f"🔍 Orchestrating retrieval for: {query}")
        
        # 1. Intent Detection
        intent = self.detect_intent(query)
        print(f"🎯 Detected intent: {max(intent, key=intent.get)} (confidence: {max(intent.values()):.2f})")
        
        # 2. Multi-Query Expansion
        query_variants = self.multi_query_expansion(query)
        print(f"🔄 Generated {len(query_variants)} query variants")
        
        # 3. Multi-Strategy Retrieval (TEXT ONLY)
        all_chunks = []
        
        # Original query retrieval - TEXT ONLY
        original_embeddings = self.openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=query
        )
        original_vector = original_embeddings.data[0].embedding
        
        original_hits = self.qdrant_client.search(
            collection_name=COLLECTION,
            query_vector=original_vector,
            limit=limit * 3,  # Get more since we're filtering
            with_payload=True
        )
        # Filter for text blocks only
        text_hits = [hit for hit in original_hits if hit.payload.get('block_type') == 'text']
        all_chunks.extend(self._process_hits(text_hits))
        
        # HyDE retrieval - TEXT ONLY
        hyde_chunks = self.hyde_retrieval(query, limit=limit)
        all_chunks.extend(hyde_chunks)
        
        # Query variants retrieval - TEXT ONLY
        for variant in query_variants[1:]:  # Skip original
            try:
                variant_embeddings = self.openai_client.embeddings.create(
                    model=EMBED_MODEL,
                    input=variant
                )
                variant_vector = variant_embeddings.data[0].embedding
                
                variant_hits = self.qdrant_client.search(
                    collection_name=COLLECTION,
                    query_vector=variant_vector,
                    limit=limit,  # Get more since we're filtering
                    with_payload=True
                )
                # Filter for text blocks only
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
        """Remove duplicate chunks based on text content"""
        seen = set()
        unique = []
        for chunk in chunks:
            text_hash = hash(chunk['text'][:100])  # Hash first 100 chars
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(chunk)
        return unique

    def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        """Generate comprehensive answer using retrieved chunks"""
        print(f"🤖 Generating answer with {GPT_MODEL}...")
        
        # Format context
        context = self._format_context(chunks)
        
        system_prompt = """You are an expert research assistant. Answer questions based on the provided document context.

IMPORTANT RULES:
1. Only use information from the provided context
2. ALWAYS cite specific sources: "According to Source X (PDF Page Y, Section Z)..."
3. Reference the section and page clearly (e.g., "The text on PDF page X in Section Y states...")
4. If context is insufficient, acknowledge this clearly
5. Provide comprehensive, well-structured answers
6. Use exact terminology from the documents
7. Page numbers refer to actual PDF pages (1-indexed)
8. "Source X" refers to different text chunks from the same paper, not different papers"""

        user_prompt = f"""Context from research documents:

{context}

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
        """Format retrieved chunks into context for GPT"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Convert 0-indexed page to 1-indexed PDF page
            pdf_page = chunk['page'] + 1
            context_parts.append(
                f"Source {i}:\n"
                f"  - Section: {chunk['section']}\n"
                f"  - PDF Page: {pdf_page}\n"
                f"  - Paper: {chunk['doc_id']}\n"
                f"  - Content:\n{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)

    def classify_query_intent(self, query: str) -> Dict[str, float]:
        """Classify if query is about images, text, or both"""
        system_prompt = """Analyze the user's query and determine their intent with confidence scores (0-1):

INTENT TYPES:
- image_query: Looking for visual content, figures, diagrams, charts, graphs, images
- text_query: Looking for textual explanations, definitions, descriptions
- mixed_query: Looking for both visual and textual content
- equation_query: Looking for mathematical expressions, formulas, equations

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
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            # Fallback to keyword-based classification
            query_lower = query.lower()
            if any(word in query_lower for word in ['figure', 'image', 'diagram', 'chart', 'graph', 'plot', 'visual']):
                return {"image_query": 0.8, "text_query": 0.3, "mixed_query": 0.2, "equation_query": 0.1}
            elif any(word in query_lower for word in ['equation', 'formula', 'math', 'mathematical']):
                return {"equation_query": 0.8, "text_query": 0.6, "image_query": 0.4, "mixed_query": 0.3}
            else:
                return {"text_query": 0.8, "image_query": 0.2, "mixed_query": 0.3, "equation_query": 0.1}

    def find_image_context(self, query: str) -> List[Dict]:
        """Find relevant images and their context from the database"""
        print("🔍 Searching for relevant images...")
        
        # Generate query embedding
        query_embedding = self.openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=query
        )
        query_vector = query_embedding.data[0].embedding
        
        # Search for image-related content
        try:
            image_results = self.qdrant_client.search(
                collection_name=COLLECTION,
                query_vector=query_vector,
                limit=5,
                with_payload=True,
                query_filter=None  # Remove filter to search across all content types
            )
            
            # Filter for image-related results
            image_contexts = []
            for result in image_results:
                payload = result.payload
                
                # Check if this result is image-related
                if (payload.get('block_type') == 'image' or 
                    'image' in payload.get('text', '').lower() or
                    'figure' in payload.get('text', '').lower() or
                    'caption' in payload.get('text', '').lower()):
                    
                    # Get surrounding context
                    surrounding_context = self.get_surrounding_context(
                        payload.get('doc_id', ''),
                        payload.get('page', 0)
                    )
                    
                    image_contexts.append({
                        'image_path': payload.get('img_path', ''),
                        'caption': payload.get('image_caption', payload.get('text', '')),
                        'page': payload.get('page', 0),
                        'section': payload.get('section', ''),
                        'surrounding_context': surrounding_context,
                        'relevance_score': result.score,
                        'doc_id': payload.get('doc_id', ''),
                        'text': payload.get('text', '')
                    })
            
            print(f"📸 Found {len(image_contexts)} relevant image contexts")
            return image_contexts
            
        except Exception as e:
            print(f"❌ Error searching for images: {e}")
            return []

    def get_surrounding_context(self, doc_id: str, page: int) -> str:
        """Get surrounding text context for a given page"""
        try:
            # Search for text content around the same page
            context_results = self.qdrant_client.search(
                collection_name=COLLECTION,
                query_vector=[0.1] * 1536,  # Dummy vector to get all results
                limit=100,
                with_payload=True,
                query_filter=None
            )
            
            # Filter for same document and nearby pages
            relevant_contexts = []
            for result in context_results:
                payload = result.payload
                if (payload.get('doc_id') == doc_id and 
                    payload.get('block_type') == 'text' and
                    abs(payload.get('page', 0) - page) <= 1):  # Within 1 page
                    
                    relevant_contexts.append({
                        'text': payload.get('text', ''),
                        'page': payload.get('page', 0),
                        'section': payload.get('section', ''),
                        'score': result.score
                    })
            
            # Sort by page proximity and relevance
            relevant_contexts.sort(key=lambda x: (abs(x['page'] - page), -x['score']))
            
            # Combine context (limit to avoid too long context)
            context_parts = []
            for ctx in relevant_contexts[:5]:  # Top 5 most relevant
                context_parts.append(f"Page {ctx['page']} ({ctx['section']}): {ctx['text'][:200]}...")
            
            return "\n".join(context_parts) if context_parts else "No surrounding context found."
            
        except Exception as e:
            print(f"❌ Error getting surrounding context: {e}")
            return "Context retrieval failed."

    def query_vlm_agent(self, query: str, image_context: Dict) -> str:
        """Send query to VLM agent with rich context"""
        print("🤖 Activating VLM agent for image analysis...")
        
        # Build comprehensive prompt
        vlm_prompt = f"""
You are analyzing a research paper image to answer a specific question.

USER QUESTION: {query}

IMAGE CONTEXT:
- Caption: {image_context.get('caption', 'N/A')}
- Section: {image_context.get('section', 'N/A')}
- Page: {image_context.get('page', 'N/A')}
- Document: {image_context.get('doc_id', 'N/A')}
- Surrounding Context: {image_context.get('surrounding_context', 'N/A')}

Please analyze the image and provide a comprehensive answer that:
1. Directly addresses the user's question
2. References the image caption and context
3. Explains how the image relates to the surrounding text
4. Provides specific details visible in the image
5. Connects the visual content to the research concepts

Focus on being helpful and specific to the user's query. If you cannot see the image clearly, acknowledge this and provide what you can infer from the context.
"""

        try:
            # Try to find the image file
            image_path = self.locate_image_file(image_context.get('image_path', ''))
            
            if image_path and os.path.exists(image_path):
                # Use GPT-4V for image analysis
                response = self.analyze_image_with_gpt4v(image_path, vlm_prompt)
                return response
            else:
                # Fallback: analyze based on context only
                return self.analyze_image_context_only(query, image_context, vlm_prompt)
                
        except Exception as e:
            print(f"❌ VLM analysis failed: {e}")
            return f"Image analysis failed: {str(e)}"

    def locate_image_file(self, image_path: str) -> str:
        """Locate the actual image file from the path"""
        if not image_path:
            return None
            
        # Try different possible locations
        possible_paths = [
            os.path.join(self.base_path, image_path),
            image_path,  # Try as absolute path
            os.path.join("mineru_out", image_path)  # Fallback
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📁 Found image at: {path}")
                return path
                
        print(f"❌ Image not found at any of these paths: {possible_paths}")
        return None

    def analyze_image_with_gpt4v(self, image_path: str, prompt: str) -> str:
        """Analyze image using GPT-4V"""
        try:
            # Encode image to base64
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model=VLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ GPT-4V analysis failed: {e}")
            return f"Image analysis failed: {str(e)}"

    def analyze_image_context_only(self, query: str, image_context: Dict, prompt: str) -> str:
        """Fallback analysis when image file is not available"""
        print(" Performing context-only analysis...")
        
        context_prompt = f"""
Since the actual image file is not available, please analyze the user's question based on the available context:

USER QUESTION: {query}

AVAILABLE CONTEXT:
- Caption: {image_context.get('caption', 'N/A')}
- Section: {image_context.get('section', 'N/A')}
- Page: {image_context.get('page', 'N/A')}
- Surrounding Text: {image_context.get('surrounding_context', 'N/A')}

Please provide the best possible answer based on this context, acknowledging that you cannot see the actual image.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a research assistant analyzing academic content."},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=600,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Context analysis failed: {str(e)}"

    def enhanced_retrieve_and_answer(self, query: str) -> Tuple[str, List[Dict]]:
        """Main method for enhanced image-aware RAG"""
        print(f" Enhanced retrieval for: {query}")
        
        # 1. Classify query intent
        intent = self.classify_query_intent(query)
        print(f" Detected intent: {max(intent, key=intent.get)} (confidence: {max(intent.values()):.2f})")
        
        if intent.get('image_query', 0) > 0.6:
            # Image-focused query
            print("🖼️ Activating image-aware retrieval...")
            
            # 2. Find relevant images and context
            image_contexts = self.find_image_context(query)
            
            if not image_contexts:
                return "No relevant images found for your query. Falling back to text-based search.", []
            
            # 3. Use VLM agent for the most relevant image
            best_image = image_contexts[0]  # Highest relevance score
            
            vlm_answer = self.query_vlm_agent(query, best_image)
            
            # 4. Format response with image metadata
            response = f"""
🤖 VLM Analysis for: {best_image.get('caption', 'Image')}

📄 Location: Page {best_image.get('page', 'N/A')}, Section: {best_image.get('section', 'N/A')}
📚 Document: {best_image.get('doc_id', 'N/A')}

🔍 Analysis:
{vlm_answer}

 Tip: This analysis is based on the image content and surrounding context from the research paper.
            """
            
            return response.strip(), image_contexts
            
        else:
            # Standard text-based RAG (existing functionality)
            print("📚 Using standard text-based RAG...")
            chunks = self.retrieve_with_strategy(query)
            answer = self.generate_answer(query, chunks)
            return answer, chunks

    def chat(self):
        """Enhanced chat loop with image-aware capabilities"""
        print(" Enhanced Agentic RAG System Ready!")
        print("Advanced features: Multi-query expansion, HyDE, MMR, Intent routing, VLM Image Analysis")
        print("Ask questions about your research documents or specific images. Type 'quit' to exit.")
        print("-" * 60)
        
        while True:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            try:
                # Enhanced retrieval with image support
                answer, sources = self.enhanced_retrieve_and_answer(query)
                
                # Display results
                print("\n" + "="*70)
                print(" ENHANCED AGENTIC RAG ANSWER:")
                print("="*70)
                print(answer)
                print("="*70)
                
                # Show sources
                if sources:
                    print("\n📚 Sources:")
                    for i, source in enumerate(sources[:3], 1):  # Show top 3
                        if 'image_path' in source:
                            # Image source
                            print(f"  {i}. 📸 Image | Page {source.get('page', 'N/A')} | Section: {source.get('section', 'N/A')}")
                            print(f"     Caption: {source.get('caption', 'N/A')[:100]}...")
                        else:
                            # Text source
                            print(f"  {i}. 📄 Text | Page {source.get('page', 'N/A')} | Section: {source.get('section', 'N/A')}")
                            print(f"     Content: {source.get('text', 'N/A')[:100]}...")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Initialize with the path to your mineru_out directory
    rag = EnhancedAgenticRAG(base_path="mineru_out")
    rag.chat()

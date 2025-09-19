#!/usr/bin/env python3
"""
Semantic Kernel Enhanced Agentic RAG System with VLM Image Analysis
Modular architecture with built-in memory management, AI orchestration, and image analysis
"""

import os
import uuid
import time
import json
import hashlib
import base64
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio

# Semantic Kernel imports - Updated for version 1.35.3
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.functions import KernelFunctionFromPrompt
from semantic_kernel.kernel import KernelArguments

# OpenAI and Qdrant
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import qdrant_client.http.models as qm
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_LOCAL_PATH = "./qdrant_local"
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "agenticragdb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")
VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4-vision-preview")

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
    timestamp: float
    context_id: str
    metadata: Dict[str, Any]

class ChatMemory:
    """Session-based chat memory management"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[ChatMessage] = []
        self.query_history: List[QueryContext] = []
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.context: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the session"""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            message_id=str(uuid.uuid4()),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_accessed = time.time()
    
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
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "query_history": [ctx.__dict__ for ctx in self.query_history],
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "context": self.context
        }

class MemoryManager:
    """Centralized memory management for sessions"""
    
    def __init__(self, max_sessions: int = 100, session_timeout_hours: int = 24):
        self.sessions: Dict[str, ChatMemory] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout_hours * 3600  # Convert to seconds
    
    def get_session(self, session_id: str) -> ChatMemory:
        """Get or create a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMemory(session_id)
            self._cleanup_old_sessions()
        else:
            self.sessions[session_id].last_accessed = time.time()
        
        return self.sessions[session_id]
    
    def _cleanup_old_sessions(self):
        """Remove old sessions to manage memory"""
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_accessed > self.session_timeout:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        # If still over limit, remove oldest sessions
        if len(self.sessions) > self.max_sessions:
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].last_accessed
            )
            excess = len(self.sessions) - self.max_sessions
            for session_id, _ in sorted_sessions[:excess]:
                del self.sessions[session_id]
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a session"""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "message_count": len(session.messages),
            "query_count": len(session.query_history),
            "created_at": session.created_at,
            "last_accessed": session.last_accessed,
            "context_keys": list(session.context.keys())
        }

class SemanticKernelRAG:
    """Semantic Kernel enhanced RAG system with VLM image analysis"""
    
    def __init__(self, base_path: str = "mineru_out_german"):
        self.base_path = base_path
        self.memory_manager = MemoryManager()
        
        # Initialize Semantic Kernel
        self.kernel = Kernel()
        
        # Add OpenAI chat completion
        self.kernel.add_service(OpenAIChatCompletion(GPT_MODEL, OPENAI_API_KEY))
        
        # Initialize memory store - Simplified for current version
        self.memory_store = None  # We'll use our custom memory instead
        
        # Initialize Qdrant client with local storage
        self.qdrant_client = QdrantClient(path=QDRANT_LOCAL_PATH)
        
        # Initialize OpenAI client for embeddings
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Register plugins
        self._register_plugins()
    
    def _register_plugins(self):
        """Register all plugins with the kernel"""
        
        # Custom RAG Plugin
        self._register_rag_plugin()
    
    def _register_rag_plugin(self):
        """Register custom RAG functionality as a plugin"""
        
        # Store function references for later use
        self.intent_function = None
        self.hyde_function = None
        self.expand_function = None
        self.answer_function = None
        
        # Add functions directly to kernel with prompts
        self.intent_function = self.kernel.add_function(
            "rag_plugin",
            function_name="detect_intent",
            prompt="""Analysiere die Benutzeranfrage und bestimme die Absicht mit Konfidenzwerten (0-1):

INTENTIONSTYPEN:
- text_content: Suche nach allgemeinem Textinhalt, Erklärungen, Beschreibungen
- table_data: Suche nach numerischen Daten, Vergleichen, Statistiken, Tabellen
- figure_caption: Suche nach Abbildungsbeschreibungen, visuellen Inhalten, Diagrammen
- methodology: Suche nach Methoden, Algorithmen, Verfahren
- results: Suche nach experimentellen Ergebnissen, Befunden, Ergebnissen
- image_query: Suche nach Bildern, visuellen Analysen, Diagrammen, Charts

Gib JSON mit Absicht und Konfidenzwerten zurück.

Benutzeranfrage: {{$input}}""",
            description="Detect user intent from queries"
        )
        
        # Add the HyDE function
        self.hyde_function = self.kernel.add_function(
            "rag_plugin",
            function_name="hyde_generate",
            prompt="""Basierend auf der Frage, schreibe eine hypothetische Antwort, die in Forschungsdokumenten gefunden werden könnte.
            Mache sie realistisch und detailliert, verwende die Art von Sprache und Inhalt, die man erwarten würde.
            Antworte auf Deutsch.
            
            Frage: {{$input}}
            
            Hypothetische Antwort:""",
            description="Generate hypothetical answers for HyDE retrieval"
        )
        
        # Do the same for expand_query and generate_answer
        self.expand_function = self.kernel.add_function(
            "rag_plugin",
            function_name="expand_query",
            prompt="""Generiere {{$num_variants}} fokussierte Abfragevarianten, die helfen würden, relevante Informationen zu finden.
            Jede Variante sollte sich auf verschiedene Aspekte konzentrieren oder verschiedene Terminologie verwenden.
            Gib nur die Abfragen zurück, eine pro Zeile.

            Originale Abfrage: {{$input}}
            Anzahl der Varianten: {{$num_variants}}""",
            description="Generate query variants for better coverage"
        )
        
        self.answer_function = self.kernel.add_function(
            "rag_plugin",
            function_name="generate_answer",
            prompt="""Du bist ein Experte für Forschungshilfe. Beantworte Fragen basierend auf dem bereitgestellten Dokumentenkontext.

WICHTIGE REGELN:
1. Verwende nur Informationen aus dem bereitgestellten Kontext
2. ZITIERE IMMER spezifische Quellen: "Laut Quelle X (PDF-Seite Y, Abschnitt Z)..."
3. Referenziere Abschnitt und Seite klar
4. Wenn der Kontext unzureichend ist, gib dies klar zu erkennen
5. Gib umfassende, gut strukturierte Antworten
6. Verwende exakte Terminologie aus den Dokumenten
7. Antworte IMMER auf Deutsch

Kontext: {{$context}}
Frage: {{$input}}

Bitte gib eine umfassende Antwort basierend auf dem obigen Kontext.""",
            description="Generate comprehensive answers from context"
        )
    
    async def detect_intent(self, query: str) -> Dict[str, float]:
        """Detect user intent using Semantic Kernel"""
        try:
            args = KernelArguments(input=query)
            result = await self.kernel.invoke(self.intent_function, args)
            
            # Parse JSON response
            import json
            result_str = str(result)
            
            # Clean markdown formatting if present
            if result_str.startswith('```json'):
                result_str = result_str.replace('```json', '').replace('```', '').strip()
            elif result_str.startswith('```'):
                result_str = result_str.replace('```', '').strip()
            
            parsed_result = json.loads(result_str)
            
            # Handle different response formats
            if isinstance(parsed_result, dict):
                if "intent" in parsed_result and "confidence" in parsed_result:
                    # Format: {"intent": "methodology", "confidence": 0.85}
                    intent_type = parsed_result["intent"]
                    confidence = parsed_result["confidence"]
                    return {intent_type: confidence}
                elif "intent" in parsed_result and isinstance(parsed_result["intent"], dict):
                    # Format: {"intent": {"text_content": 0.7, "methodology": 0.8}}
                    return parsed_result["intent"]
                elif "intents" in parsed_result:
                    # Format: {"intents": {"text_content": 0.7, "methodology": 0.8}}
                    return parsed_result["intents"]
                else:
                    # Try to find any numeric values
                    numeric_values = {k: v for k, v in parsed_result.items() if isinstance(v, (int, float))}
                    if numeric_values:
                        return numeric_values
            
            # Fallback to default
            return {"text_content": 0.8, "table_data": 0.6, "figure_caption": 0.5}
        except Exception as e:
            print(f"⚠️ Intent detection error: {e}")
            # Fallback to default
            return {"text_content": 0.8, "table_data": 0.6, "figure_caption": 0.5}
    
    async def expand_query(self, query: str, num_variants: int = 3) -> List[str]:
        """Expand query using Semantic Kernel"""
        try:
            args = KernelArguments(input=query, num_variants=str(num_variants))
            result = await self.kernel.invoke(self.expand_function, args)
            
            # Parse response
            variants = [v.strip() for v in str(result).split('\n') if v.strip()]
            return variants[:num_variants]
        except Exception as e:
            print(f"Query expansion error: {e}")
            return [query]
    
    async def hyde_retrieval(self, query: str, limit: int = 5) -> List[Dict]:
        """HyDE retrieval with Semantic Kernel integration"""
        print("🧠 Generiere hypothetische Antwort für HyDE-Retrieval...")
        
        try:
            # Generate hypothetical answer using Semantic Kernel
            args = KernelArguments(input=query)
            result = await self.kernel.invoke(self.hyde_function, args)
            hyde_text = str(result)
            
            # Use HyDE text for retrieval
            embedding = self.openai_client.embeddings.create(
                model=EMBED_MODEL,
                input=hyde_text
            )
            
            results = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=embedding.data[0].embedding,
                limit=limit,
                with_payload=True
            )
            
            return self._process_hits(results)
            
        except Exception as e:
            print(f"HyDE fehlgeschlagen: {e}")
            return []
    
    def _process_hits(self, hits) -> List[Dict]:
        """Process Qdrant hits into standardized format"""
        chunks = []
        for hit in hits:
            chunks.append({
                'text': hit.payload['text'],
                'title': hit.payload.get('title', 'Unbekannter Titel'),
                'page': hit.payload.get('page', 0),
                'doc_id': hit.payload['doc_id'],
                'block_type': hit.payload['block_type'],
                'section': hit.payload.get('section', 'Unbekannt'),
                'has_image': hit.payload.get('has_image', False),
                'score': hit.score,
                'vector': hit.vector
            })
        return chunks
    
    # VLM Image Analysis Methods
    def locate_image_file(self, image_path: str) -> Optional[str]:
        """Locate image file in the system"""
        if not image_path:
            return None
        
        # Try direct path first
        if os.path.exists(image_path):
            return image_path
        
        # Try relative to base_path
        relative_path = os.path.join(self.base_path, image_path)
        if os.path.exists(relative_path):
            return relative_path
        
        # Try to find in mineru_out_german structure
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file == os.path.basename(image_path):
                    return os.path.join(root, file)
        
        return None

    def find_image_context(self, query: str, limit: int = 5) -> List[Dict]:
        """Find image-related chunks for VLM analysis"""
        try:
            # Generate embedding for the query
            query_embedding = self.openai_client.embeddings.create(
                model=EMBED_MODEL,
                input=query
            )
            
            # Search for image chunks
            hits = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_embedding.data[0].embedding,
                query_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="has_image",
                            match=qm.MatchValue(value=True)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True
            )
            
            image_contexts = []
            for hit in hits:
                context = {
                    'text': hit.payload.get('text', ''),
                    'title': hit.payload.get('title', 'Unbekannter Titel'),
                    'page': hit.payload.get('page', 0),
                    'doc_id': hit.payload.get('doc_id', ''),
                    'section': hit.payload.get('section', 'Unbekannt'),
                    'has_image': hit.payload.get('has_image', False),
                    'score': hit.score,
                    'image_caption': hit.payload.get('image_caption', []),
                    'img_path': hit.payload.get('img_path', ''),
                    'caption': hit.payload.get('text', '')  # Use text as caption
                }
                image_contexts.append(context)
            
            return image_contexts
            
        except Exception as e:
            print(f"❌ Fehler beim Finden von Bildkontext: {e}")
            return []

    def get_surrounding_context(self, image_context: Dict, window: int = 2) -> str:
        """Get surrounding text context for an image"""
        try:
            doc_id = image_context.get('doc_id', '')
            page = image_context.get('page', 0)
            
            # Search for text chunks around the same page
            hits = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=[0.0] * 1536,  # Dummy vector
                query_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="doc_id",
                            match=qm.MatchValue(value=doc_id)
                        ),
                        qm.FieldCondition(
                            key="page",
                            match=qm.MatchValue(value=page)
                        ),
                        qm.FieldCondition(
                            key="block_type",
                            match=qm.MatchValue(value="text")
                        )
                    ]
                ),
                limit=window * 2,
                with_payload=True
            )
            
            context_parts = []
            for hit in hits:
                context_parts.append(f"Seite {hit.payload.get('page', 'N/A')} ({hit.payload.get('section', 'N/A')}): {hit.payload.get('text', '')[:200]}...")
            
            return "\n".join(context_parts) if context_parts else "Kein umgebender Kontext gefunden."
            
        except Exception as e:
            print(f"❌ Fehler beim Abrufen des umgebenden Kontexts: {e}")
            return "Kontext-Abruf fehlgeschlagen."

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
            print(f"❌ GPT-4V Analyse fehlgeschlagen: {e}")
            return f"Bildanalyse fehlgeschlagen: {str(e)}"

    def analyze_image_context_only(self, query: str, image_context: Dict, prompt: str) -> str:
        """Fallback analysis when image file is not available"""
        print("🔄 Führe Kontext-basierte Analyse durch...")
        
        try:
            # Use GPT-4 for context-based analysis
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für Forschungsdokumente. Analysiere den bereitgestellten Kontext und beantworte die Frage auf Deutsch."},
                    {"role": "user", "content": f"{prompt}\n\nKontext: {image_context.get('text', '')}"}
                ],
                max_tokens=600,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Kontext-Analyse fehlgeschlagen: {e}")
            return f"Kontext-Analyse fehlgeschlagen: {str(e)}"

    def query_vlm_agent(self, query: str, image_context: Dict) -> str:
        """Send query to VLM agent with rich context"""
        print("🤖 Aktiviere VLM-Agent für Bildanalyse...")
        
        # Get surrounding context
        surrounding_context = self.get_surrounding_context(image_context)
        
        # Build comprehensive prompt
        vlm_prompt = f"""
Du analysierst ein Bild aus einem Forschungsdokument, um eine spezifische Frage zu beantworten.

BENUTZERFRAGE: {query}

BILDKONTEXT:
- Beschriftung: {image_context.get('caption', 'N/A')}
- Abschnitt: {image_context.get('section', 'N/A')}
- Seite: {image_context.get('page', 'N/A')}
- Dokument: {image_context.get('doc_id', 'N/A')}
- Umgebender Kontext: {surrounding_context}

Bitte analysiere das Bild und gib eine umfassende Antwort, die:
1. Die Frage des Benutzers direkt beantwortet
2. Die Bildbeschriftung und den Kontext referenziert
3. Erklärt, wie das Bild mit dem umgebenden Text zusammenhängt
4. Spezifische Details aus dem Bild liefert
5. Den visuellen Inhalt mit den Forschungskonzepten verbindet

Antworte auf Deutsch und sei hilfreich und spezifisch. Wenn du das Bild nicht klar sehen kannst, gib dies zu erkennen und liefere das, was du aus dem Kontext ableiten kannst.
"""

        try:
            # Try to find the image file
            image_path = self.locate_image_file(image_context.get('img_path', ''))
            
            if image_path and os.path.exists(image_path):
                # Use GPT-4V for image analysis
                response = self.analyze_image_with_gpt4v(image_path, vlm_prompt)
                return response
            else:
                # Fallback: analyze based on context only
                return self.analyze_image_context_only(query, image_context, vlm_prompt)
                
        except Exception as e:
            print(f"❌ VLM-Analyse fehlgeschlagen: {e}")
            return f"Bildanalyse fehlgeschlagen: {str(e)}"

    async def retrieve_with_strategy(self, query: str, limit: int = 8) -> List[Dict]:
        """Multi-strategy retrieval using Semantic Kernel"""
        print(f"🔍 Orchestriere Retrieval für: {query}")
        
        # 1. Intent Detection using Semantic Kernel
        intent = await self.detect_intent(query)
        
        # Safety check for empty intent dictionary
        if not intent or len(intent) == 0:
            print("⚠️ Intent detection failed, using default intent")
            intent = {"text_content": 0.8, "table_data": 0.6, "figure_caption": 0.5}
        
        print(f"🎯 Erkannte Absicht: {max(intent, key=intent.get)} (Konfidenz: {max(intent.values()):.2f})")
        
        # 2. Query Expansion using Semantic Kernel
        query_variants = await self.expand_query(query)
        print(f"🔄 Generiert {len(query_variants)} Abfragevarianten")
        
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
        hyde_chunks = await self.hyde_retrieval(query, limit=limit)
        all_chunks.extend(hyde_chunks)
        
        # Query variant retrieval
        for variant in query_variants:
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
                print(f"Varianten-Retrieval fehlgeschlagen: {e}")
        
        # 4. Remove duplicates and apply MMR
        unique_chunks = self._remove_duplicates(all_chunks)
        print(f"📚 Insgesamt {len(unique_chunks)} einzigartige TEXT-Abschnitte gefunden")
        
        # 5. MMR diversification
        diversified_chunks = self._mmr_diversify(unique_chunks, query, limit)
        
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

    def _mmr_diversify(self, all_hits: List[Dict], query: str, 
                      lambda_param: float = 0.5, limit: int = 8) -> List[Dict]:
        """MMR diversification for result selection"""
        print("🔄 Wende MMR-Diversifizierung an...")
        
        if len(all_hits) <= limit:
            return all_hits
        
        # Calculate relevance scores (using existing scores)
        relevance_scores = [hit['score'] for hit in all_hits]
        
        # Select first item (highest relevance)
        selected = [all_hits[0]]
        remaining = list(range(1, len(all_hits)))
        
        # Select remaining items using MMR
        for _ in range(min(limit - 1, len(remaining))):
            mmr_scores = []
            
            for idx in remaining:
                relevance = relevance_scores[idx]
                
                max_similarity = 0
                for sel_idx in selected:
                    if sel_idx.get('vector') is None or all_hits[idx].get('vector') is None:
                        continue
                    sel_vector = sel_idx['vector']
                    curr_vector = all_hits[idx]['vector']
                    similarity = sum(a * b for a, b in zip(sel_vector, curr_vector)) / (
                        (sum(a * a for a in sel_vector) ** 0.5) * 
                        (sum(b * b for b in curr_vector) ** 0.5)
                    )
                    max_similarity = max(max_similarity, similarity)
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)
            
            best_idx = remaining[np.argmax(mmr_scores)]
            selected.append(all_hits[best_idx])
            remaining.remove(best_idx)
        
        return selected

    async def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        """Generate answer using Semantic Kernel"""
        try:
            context_text = self._format_context(chunks)
            args = KernelArguments(input=query, context=context_text)
            result = await self.kernel.invoke(self.answer_function, args)
            return str(result)
        except Exception as e:
            return f"Fehler beim Generieren der Antwort: {str(e)}"
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context for GPT"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Convert 0-indexed page to 1-indexed PDF page
            pdf_page = chunk['page'] + 1
            context_parts.append(
                f"Quelle {i}:\n"
                f"  - Abschnitt: {chunk['section']}\n"
                f"  - PDF-Seite: {pdf_page}\n"
                f"  - Papier: {chunk['doc_id']}\n"
                f"  - Inhalt:\n{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)

    async def enhanced_retrieve_and_answer(self, query: str, session_id: str) -> Tuple[str, List[Dict]]:
        """Enhanced retrieval with memory integration and VLM support"""
        print(f"🚀 Erweiterte Suche für: {query}")
        
        # Get session memory
        session = self.memory_manager.get_session(session_id)
        
        # Store user query
        session.add_message("user", query, {"type": "query"})
        
        # Check for relevant previous context
        previous_context = session.get_query_context(query)
        if previous_context:
            print(f"📚 Relevanter vorheriger Kontext von {previous_context.query} gefunden")
        
        # Get relevant chat history
        relevant_history = session.get_relevant_context(query)
        if relevant_history:
            print(f"💬 {len(relevant_history)} relevante Chat-Historie-Elemente gefunden")
        
        # Check if this is an image query
        intent = await self.detect_intent(query)
        if intent.get('image_query', 0) > 0.6:
            print("🖼️ Aktiviere bildbewusste Suche...")
            
            # Find relevant images and context
            image_contexts = self.find_image_context(query)
            
            if not image_contexts:
                print("📚 Keine relevanten Bilder gefunden, verwende Standard-Text-basiertes RAG...")
                chunks = await self.retrieve_with_strategy(query)
                answer = await self.generate_answer(query, chunks)
            else:
                # Use VLM agent for the most relevant image
                best_image = image_contexts[0]  # Highest relevance score
                
                vlm_answer = self.query_vlm_agent(query, best_image)
                
                # Format response with image metadata
                answer = f"""
🤖 VLM-Analyse für: {best_image.get('caption', 'Bild')}

📄 Position: Seite {best_image.get('page', 'N/A')}, Abschnitt: {best_image.get('section', 'N/A')}
📚 Dokument: {best_image.get('doc_id', 'N/A')}

🔍 Analyse:
{vlm_answer}

💡 Tipp: Diese Analyse basiert auf dem Bildinhalt und dem umgebenden Kontext aus dem Forschungsdokument.
                """
                
                chunks = image_contexts
        else:
            # Standard text-based RAG
            print("📚 Verwende Standard-Text-basiertes RAG...")
            chunks = await self.retrieve_with_strategy(query)
            answer = await self.generate_answer(query, chunks)
        
        # Store system response
        session.add_message("assistant", answer, {
            "type": "answer",
            "chunks_count": len(chunks),
            "query": query
        })
        
        # Store query context
        query_context = QueryContext(
            query=query,
            timestamp=time.time(),
            context_id=str(uuid.uuid4()),
            metadata={"chunks_count": len(chunks)}
        )
        session.add_query_context(query_context)
        
        return answer, chunks

    def chat(self):
        """Enhanced chat loop with image-aware capabilities"""
        session_id = str(uuid.uuid4())
        
        print(f"🚀 Erweiterte Agentic RAG System bereit! (Session: {session_id[:8]}...)")
        print("Erweiterte Funktionen: Semantic Kernel, Speicherverwaltung, Multi-Query-Expansion, HyDE, MMR, VLM-Bildanalyse")
        print("Stellen Sie Fragen zu Ihren Forschungsdokumenten oder spezifischen Bildern. Tippen Sie 'quit' zum Beenden.")
        print("-" * 60)
        
        while True:
            try:
                query = input("\n❓ Ihre Frage: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Auf Wiedersehen!")
                    break
                
                if not query:
                    continue
                
                # Enhanced retrieval with image support
                answer, sources = await self.enhanced_retrieve_and_answer(query, session_id)
                
                # Display results
                print("\n" + "="*70)
                print("🚀 ERWEITERTE AGENTIC RAG ANTWORT:")
                print("="*70)
                print(answer)
                print("="*70)
                
                # Show sources
                if sources:
                    print("\n📚 Quellen:")
                    for i, source in enumerate(sources[:3], 1):
                        if 'img_path' in source:
                            # Image source
                            print(f"  {i}. 📸 Bild | Seite {source.get('page', 'N/A')} | Abschnitt: {source.get('section', 'N/A')}")
                            print(f"     Beschriftung: {source.get('caption', 'N/A')[:100]}...")
                        else:
                            # Text source
                            print(f"  {i}. 📄 Text | Seite {source.get('page', 'N/A')} | Abschnitt: {source.get('section', 'N/A')}")
                            print(f"     Inhalt: {source.get('text', 'N/A')[:100]}...")
                
                # Show memory info
                session = self.memory_manager.get_session(session_id)
                print(f"\n💾 Speicher: {len(session.messages)} Nachrichten, {len(session.query_history)} Abfragen")
                
            except Exception as e:
                print(f"❌ Fehler: {str(e)}")
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        return self.memory_manager.get_session_summary(session_id)
    
    def export_session(self, session_id: str) -> Dict[str, Any]:
        """Export session data"""
        session = self.memory_manager.get_session(session_id)
        return session.to_dict()

if __name__ == "__main__":
    import asyncio
    # Initialize Semantic Kernel RAG system with VLM
    rag = SemanticKernelRAG(base_path="mineru_out_german")
    
    # Start chat with new session
    asyncio.run(rag.chat())

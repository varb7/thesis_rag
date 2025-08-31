#!/usr/bin/env python3
"""
Memory Management Module for Agentic RAG System
Handles session-based chat memory and context management
"""

import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict


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
        self.semantic_memory = None  # For future persistent storage
    
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
        
        # Type checking and safety
        if not isinstance(context, dict):
            return
        
        if not isinstance(session.context, dict):
            session.context = {}  # Reset to empty dict
        
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

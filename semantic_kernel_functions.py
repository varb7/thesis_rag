#!/usr/bin/env python3
"""
Semantic Kernel Functions Module for Agentic RAG System
Handles function registration and management for the Semantic Kernel
"""

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.kernel import KernelArguments
from typing import Dict, List


class SemanticKernelFunctions:
    """Manages Semantic Kernel functions for RAG operations"""
    
    def __init__(self, kernel: Kernel, gpt_model: str, openai_api_key: str):
        self.kernel = kernel
        self.gpt_model = gpt_model
        self.openai_api_key = openai_api_key
        
        # Store function references
        self.intent_function = None
        self.hyde_function = None
        self.expand_function = None
        self.answer_function = None
        
        # Register all functions
        self._register_functions()
    
    def _register_functions(self):
        """Register all RAG functions with the kernel"""
        
        # Intent Detection Function
        self.intent_function = self.kernel.add_function(
            "rag_plugin",
            function_name="detect_intent",
            prompt="""Analyze the user's query and determine their intent with confidence scores (0-1):

INTENT TYPES:
- text_content: Looking for general text content, explanations, descriptions
- table_data: Looking for numerical data, comparisons, statistics, tables
- figure_caption: Looking for figure descriptions, visual content, charts
- methodology: Looking for methods, algorithms, procedures
- results: Looking for experimental results, findings, outcomes

Return JSON with intent and confidence scores.

User Query: {{$input}}""",
            description="Detect user intent from queries"
        )
        
        # HyDE Function
        self.hyde_function = self.kernel.add_function(
            "rag_plugin",
            function_name="hyde_generate",
            prompt="""Based on the question, write a hypothetical answer that might be found in research documents.
            Make it realistic and detailed, using the kind of language and content you'd expect to find.
            
            Question: {{$input}}
            
            Hypothetical answer:""",
            description="Generate hypothetical answers for HyDE retrieval"
        )
        
        # Query Expansion Function
        self.expand_function = self.kernel.add_function(
            "rag_plugin",
            function_name="expand_query",
            prompt="""Generate {{$num_variants}} focused query variants that would help find relevant information. 
            Each variant should focus on different aspects or use different terminology.
            Return only the queries, one per line.

            Original Query: {{$input}}
            Number of Variants: {{$num_variants}}""",
            description="Generate query variants for better coverage"
        )
        
        # Answer Generation Function
        self.answer_function = self.kernel.add_function(
            "rag_plugin",
            function_name="generate_answer",
            prompt="""You are an expert research assistant. Answer questions based on the provided document context.

IMPORTANT RULES:
1. Only use information from the provided context
2. ALWAYS cite specific sources: "According to Source X (PDF Page Y, Section Z)..."
3. Reference the section and page clearly
4. If context is insufficient, acknowledge this clearly
5. Provide comprehensive, well-structured answers
6. Use exact terminology from the documents

Context: {{$context}}
Question: {{$input}}

Please provide a comprehensive answer based on the context above.""",
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
                    return {k: v for k, v in parsed_result.items() if isinstance(v, (int, float))}
            
            # Fallback to default
            return {"text_content": 0.8, "table_data": 0.6, "figure_caption": 0.5}
        except Exception as e:
            # Fallback to default
            return {"text_content": 0.8, "table_data": 0.6, "figure_caption": 0.5}
    
    async def expand_query(self, query: str, num_variants: int = 3) -> List[str]:
        """Expand query using Semantic Kernel"""
        try:
            args = KernelArguments(input=query, num_variants=str(num_variants))
            result = await self.kernel.invoke(self.expand_function, args)
            
            variants = [query]  # Include original query
            variants.extend([line.strip() for line in str(result).split('\n') if line.strip()])
            return variants[:num_variants + 1]
        except Exception as e:
            return [query]
    
    async def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        """Generate answer using Semantic Kernel"""
        try:
            context_text = self._format_context(chunks)
            args = KernelArguments(input=query, context=context_text)
            result = await self.kernel.invoke(self.answer_function, args)
            return str(result)
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

#!/usr/bin/env python3
"""
Configuration Module for Agentic RAG System
Handles environment variables and configuration settings
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")
VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4-vision-preview")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "agenticragdb")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# System Configuration
DEFAULT_BASE_PATH = "mineru_out"
MAX_SESSIONS = 100
SESSION_TIMEOUT_HOURS = 24
MAX_MESSAGES_PER_SESSION = 100
DEFAULT_RETRIEVAL_LIMIT = 8
DEFAULT_MMR_LAMBDA = 0.5

# Validation
def validate_config():
    """Validate that required configuration is present"""
    required_vars = [
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("QDRANT_URL", QDRANT_URL)
    ]
    
    missing_vars = []
    for var_name, var_value in required_vars:
        if not var_value:
            missing_vars.append(var_name)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return True

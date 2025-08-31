#!/usr/bin/env python3
"""
Quick Database Clear
Simple command line script to clear the database
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import qdrant_client.models as qm

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "agenticragdb")

def quick_clear():
    """Quick clear of the entire collection"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        # Delete collection if it exists
        collections = client.get_collections()
        if COLLECTION in [c.name for c in collections.collections]:
            print(f"🗑️  Deleting collection: {COLLECTION}")
            client.delete_collection(COLLECTION)
            print(f"✅ Collection {COLLECTION} deleted")
        else:
            print(f"ℹ️  Collection {COLLECTION} doesn't exist")
        
        # Recreate collection
        print(f"🔧 Recreating collection: {COLLECTION}")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(size=1536, distance=qm.Distance.COSINE),
        )
        
        # Add title index
        client.create_payload_index(COLLECTION, "title", field_type=qm.PayloadSchemaType.TEXT)
        
        print(f"✅ Collection {COLLECTION} recreated")
        print("🎉 Database cleared! Ready for fresh ingestion.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        quick_clear()
    else:
        print("🗑️  Quick Database Clear")
        print("Usage: python quick_clear.py --force")
        print("⚠️  This will delete ALL data!")

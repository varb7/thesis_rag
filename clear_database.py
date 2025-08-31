#!/usr/bin/env python3
"""
Database Clear Utility
Clears ingested data from Qdrant database
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import qdrant_client.models as qm

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "agenticragdb")

def clear_entire_collection():
    """Clear entire collection and recreate it"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if COLLECTION in collection_names:
            print(f"🗑️  Deleting collection: {COLLECTION}")
            client.delete_collection(COLLECTION)
            print(f"✅ Collection {COLLECTION} deleted")
        else:
            print(f"ℹ️  Collection {COLLECTION} doesn't exist")
        
        # Recreate collection with proper schema
        print(f"🔧 Recreating collection: {COLLECTION}")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(size=1536, distance=qm.Distance.COSINE),
        )
        
        # Create payload indexes
        print("📊 Creating payload indexes...")
        client.create_payload_index(COLLECTION, "block_type", field_type=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "doc_id", field_type=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "title", field_type=qm.PayloadSchemaType.TEXT)
        client.create_payload_index(COLLECTION, "section", field_type=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "content_type", field_type=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "page", field_type=qm.PayloadSchemaType.INTEGER)
        
        print(f"✅ Collection {COLLECTION} recreated with fresh schema")
        print("🎉 Database cleared! Ready for fresh ingestion.")
        
    except Exception as e:
        print(f"❌ Error clearing collection: {e}")

def clear_specific_documents(doc_ids: list):
    """Clear specific documents by doc_id"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        total_deleted = 0
        
        for doc_id in doc_ids:
            print(f"🗑️  Deleting documents with doc_id: {doc_id}")
            
            # Delete points with specific doc_id
            deleted = client.delete(
                collection_name=COLLECTION,
                points_selector=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="doc_id",
                            match=qm.MatchValue(value=doc_id)
                        )
                    ]
                )
            )
            
            print(f"✅ Deleted {deleted.status.deleted_count} points for {doc_id}")
            total_deleted += deleted.status.deleted_count
        
        print(f"🎉 Total deleted: {total_deleted} points")
        
    except Exception as e:
        print(f"❌ Error deleting documents: {e}")

def list_collections():
    """List all collections and their sizes"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        collections = client.get_collections()
        
        if not collections.collections:
            print("📚 No collections found")
            return
        
        print("📚 Available collections:")
        print("=" * 60)
        
        for collection in collections.collections:
            try:
                info = client.get_collection(collection.name)
                count = info.points_count
                print(f"📊 {collection.name}: {count} points")
            except Exception as e:
                print(f"📊 {collection.name}: Error getting count - {e}")
                
    except Exception as e:
        print(f"❌ Error listing collections: {e}")

def main():
    """Main menu for database operations"""
    print("🗑️  Database Clear Utility")
    print("=" * 60)
    
    while True:
        print("\n📋 Available operations:")
        print("1. Clear entire collection (fresh start)")
        print("2. Clear specific documents by doc_id")
        print("3. List collections and sizes")
        print("4. Exit")
        
        choice = input("\n❓ Choose an option (1-4): ").strip()
        
        if choice == "1":
            confirm = input("⚠️  This will delete ALL data! Type 'YES' to confirm: ").strip()
            if confirm == "YES":
                clear_entire_collection()
            else:
                print("❌ Operation cancelled")
                
        elif choice == "2":
            doc_ids_input = input("📝 Enter doc_ids to delete (comma-separated): ").strip()
            if doc_ids_input:
                doc_ids = [doc_id.strip() for doc_id in doc_ids_input.split(",")]
                clear_specific_documents(doc_ids)
            else:
                print("❌ Please enter doc_ids")
                
        elif choice == "3":
            list_collections()
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()

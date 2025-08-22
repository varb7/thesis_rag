#!/usr/bin/env python3
"""
Enhanced Ingestion with Image Metadata Support
Processes both text content and image metadata for comprehensive RAG
"""

import os
import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple
import re
from qdrant_client import QdrantClient
import qdrant_client.models as qm
import openai
from dotenv import load_dotenv
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "agenticragdb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536

# Constants
MAX_CHARS = 2000
MIN_CHARS = 100

def load_content_list_with_images(content_list_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load content list and separate text and image entries"""
    print(f"📋 Loading content list from: {content_list_path}")
    
    with open(content_list_path, 'r', encoding='utf-8') as f:
        content_list = json.load(f)
    
    text_entries = []
    image_entries = []
    
    for item in content_list:
        if item.get('type') == 'text' and item.get('text', '').strip():
            text_entries.append(item)
        elif item.get('type') == 'image':
            image_entries.append(item)
        elif item.get('type') == 'equation':
            # Treat equations as special text with image reference
            text_entries.append({
                **item,
                'text': f"Equation: {item.get('text', '')} [Image: {item.get('img_path', '')}]",
                'has_equation_image': True,
                'equation_image_path': item.get('img_path', '')
            })
    
    print(f"📄 Found {len(text_entries)} text entries and {len(image_entries)} image entries")
    return text_entries, image_entries

def create_image_blocks(image_entries: List[Dict], text_to_page: Dict[str, int]) -> List[Dict]:
    """Create searchable blocks for images"""
    image_blocks = []
    
    for item in image_entries:
        # Create a searchable text representation
        searchable_text = f"Image: {item.get('image_caption', [''])[0] if item.get('image_caption') else 'No caption'}"
        
        # Add metadata to make it searchable
        if item.get('img_path'):
            searchable_text += f" File: {item.get('img_path')}"
        
        # Find page number
        page_number = item.get('page_idx', 0)
        
        image_blocks.append({
            'id': f"img_{hashlib.md5(item.get('img_path', '').encode()).hexdigest()[:8]}",
            'text': searchable_text,
            'section': 'Figures and Images',
            'page': page_number,
            'block_type': 'image',
            'content_type': 'figure',
            'img_path': item.get('img_path', ''),
            'image_caption': item.get('image_caption', []),
            'image_footnote': item.get('image_footnote', []),
            'has_image': True
        })
    
    return image_blocks

def main():
    ap = argparse.ArgumentParser(description="Enhanced ingestion with image metadata")
    ap.add_argument("folder", help="Path to folder containing MinerU output")
    ap.add_argument("--doc_id", help="Document ID (default: folder name)")
    
    args = ap.parse_args()
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        return
    
    # Find content_list.json and markdown files
    content_list_files = list(folder_path.rglob("*_content_list.json"))
    md_files = list(folder_path.rglob("*.md"))
    
    if not content_list_files:
        print(f"❌ No content_list.json files found in: {folder_path}")
        return
    
    print(f"�� Found {len(content_list_files)} content list files and {len(md_files)} markdown files")
    
    # Initialize Qdrant client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Ensure collection exists
    names = {c.name for c in client.get_collections().collections}
    if COLLECTION not in names:
        print(f"🔧 Creating collection: {COLLECTION}")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
        )
        # Create payload indexes
        client.create_payload_index(COLLECTION, "block_type", field_type=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "doc_id", field_type=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "section", field_type=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "content_type", field_type=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "has_image", field_type=qm.PayloadSchemaType.BOOL)
    else:
        print(f"✅ Collection {COLLECTION} already exists")
    
    total_blocks = 0
    
    for content_list_file in content_list_files:
        print(f"\n📖 Processing: {content_list_file.name}")
        
        # Extract doc_id from filename or use provided one
        doc_id = args.doc_id or content_list_file.parent.name
        
        # Load content list
        text_entries, image_entries = load_content_list_with_images(str(content_list_file))
        
        # Create text blocks (existing functionality)
        text_blocks = []
        for entry in text_entries:
            if len(entry.get('text', '')) >= MIN_CHARS:
                text_blocks.append({
                    'id': f"text_{len(text_blocks)}",
                    'text': entry.get('text', ''),
                    'section': 'Content',
                    'page': entry.get('page_idx', 0),
                    'block_type': 'text',
                    'content_type': 'paragraph',
                    'doc_id': doc_id,
                    'has_image': entry.get('has_equation_image', False),
                    'equation_image_path': entry.get('equation_image_path', '')
                })
        
        # Create image blocks
        image_blocks = create_image_blocks(image_entries, {})
        
        # Combine all blocks
        all_blocks = text_blocks + image_blocks
        
        if not all_blocks:
            print(f"⚠️  No content extracted from {content_list_file.name}")
            continue
        
        print(f"�� Created {len(text_blocks)} text blocks and {len(image_blocks)} image blocks")
        
        # Generate embeddings
        texts = [block['text'] for block in all_blocks]
        vectors = embed_texts(texts)
        
        # Prepare payloads
        payloads = []
        for block, vector in zip(all_blocks, vectors):
            payload = {
                **block,
                'doc_id': doc_id,
                'source_file': str(content_list_file)
            }
            payloads.append(payload)
        
        # Upsert to Qdrant
        upsert_to_qdrant(client, payloads, vectors)
        
        total_blocks += len(all_blocks)
    
    print(f"\n�� Enhanced ingestion complete!")
    print(f"📊 Total blocks ingested: {total_blocks}")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI"""
    print(f"🧠 Generating embeddings for {len(texts)} text chunks...")
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = []
    
    for text in texts:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )
        embeddings.append(response.data[0].embedding)
    
    return embeddings

def upsert_to_qdrant(client: QdrantClient, payloads: List[Dict], vectors: List[List[float]]):
    """Upsert points to Qdrant"""
    if not payloads:
        return
        
    points = []
    for payload, vector in zip(payloads, vectors):
        # Generate deterministic ID
        h = hashlib.sha1()
        h.update(payload['doc_id'].encode())
        h.update(str(payload['page']).encode())
        h.update(payload['text'][:100].encode())
        point_id = int(h.hexdigest()[:16], 16)
        
        points.append(qm.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        ))
    
    if points:
        print(f"💾 Upserting {len(points)} points to Qdrant...")
        client.upsert(collection_name=COLLECTION, points=points)

if __name__ == "__main__":
    main()

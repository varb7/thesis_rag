#!/usr/bin/env python3
"""
Qdrant Local Ingestion System
Full-featured vector database running locally
"""

import os
import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import qdrant_client.http.models as qm
load_dotenv()

# Configuration
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536  # OpenAI text-embedding-3-small
QDRANT_LOCAL_PATH = "./qdrant_local"  # Local storage directory
COLLECTION = "agenticragdb"

# Constants
MAX_CHARS = 2000  # Maximum characters per chunk
MIN_CHARS = 100   # Minimum characters per chunk

def deterministic_id(doc_id: str, page: int, text: str, prefix_len: int = 64) -> str:
    """Generate deterministic UUID for Qdrant local"""
    h = hashlib.sha1()
    h.update(doc_id.encode())
    h.update(str(page).encode())
    h.update(text[:prefix_len].encode())
    # Convert to UUID format
    hex_string = h.hexdigest()[:32]  # Need 32 chars for UUID
    return str(uuid.UUID(hex_string))

def extract_paper_title(content_list: List[Dict]) -> Optional[str]:
    """Extract paper title from content list"""
    print("🔍 Extracting paper title...")
    
    # Look for the first text entry with text_level == 1 (top-level header)
    for item in content_list:
        if (item.get('type') == 'text' and 
            item.get('text_level') == 1 and 
            item.get('page_idx') == 0 and
            item.get('text', '').strip()):
            
            title = item['text'].strip()
            print(f"📄 Found title: {title}")
            return title
    
    # Fallback: look for first substantial text entry on page 0
    for item in content_list:
        if (item.get('type') == 'text' and 
            item.get('page_idx') == 0 and
            item.get('text', '').strip() and
            len(item['text'].strip()) > 20):  # Must be substantial
            
            title = item['text'].strip()
            print(f"📄 Found fallback title: {title}")
            return title
    
    print("⚠️  No title found in content list")
    return None

def load_content_list_with_title(content_list_path: str) -> Tuple[Dict[str, int], Optional[str]]:
    """Load content list and extract title"""
    print(f"📋 Loading content list from: {content_list_path}")
    
    with open(content_list_path, 'r', encoding='utf-8') as f:
        content_list = json.load(f)
    
    # Extract title first
    title = extract_paper_title(content_list)
    
    # Create a mapping from text content to page number
    text_to_page = {}
    
    for item in content_list:
        if item.get('type') == 'text' and item.get('text', '').strip():
            # Clean the text for matching
            clean_text = item['text'].strip()
            if len(clean_text) > 10:  # Only map substantial text
                text_to_page[clean_text] = item.get('page_idx', 0)
    
    print(f"📄 Mapped {len(text_to_page)} text segments to page numbers")
    return text_to_page, title

def find_best_page_match(text: str, text_to_page: Dict[str, int]) -> int:
    """Find the best matching page number for a given text"""
    # Try exact match first
    if text in text_to_page:
        return text_to_page[text]
    
    # Try partial matches
    for stored_text, page in text_to_page.items():
        if text in stored_text or stored_text in text:
            return page
    
    # Try fuzzy matching for longer texts
    if len(text) > 50:
        text_words = set(text.lower().split())
        best_match = 0
        best_score = 0
        
        for stored_text, page in text_to_page.items():
            if len(stored_text) > 50:
                stored_words = set(stored_text.lower().split())
                overlap = len(text_words.intersection(stored_words))
                if overlap > best_score:
                    best_score = overlap
                    best_match = page
        
        if best_score > 3:  # At least 3 words overlap
            return best_match
    
    # Default to page 0 if no match found
    return 0

def load_markdown_with_titles(md_path: str, content_list_path: str) -> Tuple[List[Dict], Optional[str]]:
    """Load markdown content and extract title"""
    print(f"📖 Loading markdown with title extraction from: {md_path}")
    
    # Load content list for page mapping and title extraction
    text_to_page, title = load_content_list_with_title(content_list_path)
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = []
    
    # Split content into sections based on headers
    sections = re.split(r'(^#+\s+.+$)', content, flags=re.MULTILINE)
    
    current_section = "Introduction"
    current_content = ""
    
    for i, section in enumerate(sections):
        if section.strip() == "":
            continue
            
        # Check if this is a header
        if section.startswith('#'):
            # Save previous section content if it exists
            if current_content.strip():
                blocks.extend(create_blocks_with_pages(current_content.strip(), current_section, text_to_page))
            
            # Start new section
            current_section = section.strip('#').strip()
            current_content = ""
        else:
            # This is content for the current section
            current_content += section + "\n"
    
    # Don't forget the last section
    if current_content.strip():
        blocks.extend(create_blocks_with_pages(current_content.strip(), current_section, text_to_page))
    
    print(f"📚 Created {len(blocks)} text blocks with title extraction")
    return blocks, title

def create_blocks_with_pages(text: str, section: str, text_to_page: Dict[str, int]) -> List[Dict]:
    """Create text blocks with page mapping"""
    if not text.strip():
        return []
    
    # Split text into sentences or paragraphs
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    blocks = []
    current_block = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if adding this sentence would exceed max length
        if len(current_block) + len(sentence) > MAX_CHARS and current_block:
            # Save current block
            page = find_best_page_match(current_block, text_to_page)
            blocks.append({
                'id': f"block_{len(blocks)}",
                'text': current_block.strip(),
                'section': section,
                'page': page,
                'block_type': 'text',
                'content_type': 'paragraph'
            })
            current_block = sentence
        else:
            current_block += " " + sentence if current_block else sentence
    
    # Don't forget the last block
    if current_block.strip():
        page = find_best_page_match(current_block, text_to_page)
        blocks.append({
            'id': f"block_{len(blocks)}",
            'text': current_block.strip(),
            'section': section,
            'page': page,
            'block_type': 'text',
            'content_type': 'paragraph'
        })
    
    return blocks

def merge_blocks_with_pages(blocks: List[Dict]) -> List[Dict]:
    """Merge blocks while preserving page information"""
    if not blocks:
        return []
    
    # Sort blocks by page, then by section
    sorted_blocks = sorted(blocks, key=lambda x: (x['page'], x['section']))
    
    merged = []
    current = sorted_blocks[0].copy()
    
    for block in sorted_blocks[1:]:
        # Merge if same page, section, and combined length is reasonable
        if (block['page'] == current['page'] and 
            block['section'] == current['section'] and
            len(current['text']) + len(block['text']) <= MAX_CHARS):
            
            current['text'] += "\n\n" + block['text']
        else:
            merged.append(current)
            current = block.copy()
    
    merged.append(current)
    return merged

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI"""
    if not texts:
        return []
    
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return []

def build_payloads_with_titles(merged_blocks: List[Dict], doc_id: str, source_md: str, title: Optional[str]) -> List[Dict]:
    """Build payloads for Qdrant with title support"""
    out = []
    for b in merged_blocks:
        if not b["text"].strip():
            continue
        out.append({
            "doc_id": doc_id,
            "title": title or "Unknown Title",  # Include title in every payload
            "source_md": source_md,
            "section": b["section"],
            "page": b["page"],
            "block_type": b["block_type"],
            "content_type": b["content_type"],
            "block_id": b["id"],
            "text": b["text"]
        })
    return out

def get_or_create_qdrant_collection():
    """Get or create Qdrant collection"""
    # Initialize Qdrant client with local storage
    client = QdrantClient(path=QDRANT_LOCAL_PATH)
    
    # Check if collection exists
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]
    
    if COLLECTION in collection_names:
        print(f"✅ Loaded existing collection: {COLLECTION}")
        # Get collection info
        info = client.get_collection(COLLECTION)
        print(f"📊 Collection size: {info.points_count}")
    else:
        print(f"🔧 Creating new collection: {COLLECTION}")
        # Create collection with proper configuration
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        
        # Create payload indexes for efficient filtering
        print("🔍 Creating payload indexes...")
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="title",
            field_schema=qm.PayloadSchemaType.TEXT
        )
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="doc_id",
            field_schema=qm.PayloadSchemaType.TEXT
        )
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="section",
            field_schema=qm.PayloadSchemaType.TEXT
        )
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="page",
            field_schema=qm.PayloadSchemaType.INTEGER
        )
        print("✅ Payload indexes created")
    
    return client

def add_to_qdrant(client, vectors: List[List[float]], payloads: List[Dict]):
    """Add documents to Qdrant"""
    if not vectors or not payloads:
        return
    
    # Prepare points for Qdrant
    points = []
    for i, (vector, payload) in enumerate(zip(vectors, payloads)):
        # Generate unique ID
        point_id = deterministic_id(payload['doc_id'], payload['page'], payload['text'])
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
        points.append(point)
    
    # Add points to collection
    client.upsert(
        collection_name=COLLECTION,
        points=points
    )
    
    print(f"✅ Added {len(points)} documents to Qdrant")
    
    # Get updated collection info
    info = client.get_collection(COLLECTION)
    print(f"📊 Total documents in collection: {info.points_count}")

def main():
    ap = argparse.ArgumentParser(description="Qdrant Local ingestion with title extraction")
    ap.add_argument("folder", help="Path to folder containing markdown and content_list files")
    ap.add_argument("--doc_id", help="Document ID (default: folder name)")
    
    args = ap.parse_args()
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        return
    
    # Find markdown and content_list files
    md_files = list(folder_path.rglob("*.md"))
    content_list_files = list(folder_path.rglob("*_content_list.json"))
    
    if not md_files:
        print(f"❌ No markdown files found in: {folder_path}")
        return
    
    if not content_list_files:
        print(f"⚠️  No content_list.json files found. Will use default page 0 and no title.")
        content_list_files = []
    
    print(f"📁 Found {len(md_files)} markdown files and {len(content_list_files)} content list files")
    
    # Get or create Qdrant collection
    client = get_or_create_qdrant_collection()
    
    total_blocks = 0
    total_chunks = 0
    
    for md_file in md_files:
        print(f"\n📖 Processing: {md_file.name}")
        
        # Find corresponding content_list.json
        content_list_path = None
        for cl_file in content_list_files:
            if cl_file.parent == md_file.parent:
                content_list_path = str(cl_file)
                break
        
        if not content_list_path:
            print(f"⚠️  No content_list.json found for {md_file.name}, using default page 0 and no title")
            content_list_path = None
        
        # Extract doc_id from filename or use provided one
        doc_id = args.doc_id or md_file.stem
        
        # Load and process markdown with title extraction
        if content_list_path:
            blocks, title = load_markdown_with_titles(str(md_file), content_list_path)
        else:
            # Fallback to original method without page mapping and title
            from ingest_md import load_markdown_content
            blocks = load_markdown_content(str(md_file))
            title = None
            # Add default page 0
            for block in blocks:
                block['page'] = 0
        
        if not blocks:
            print(f"⚠️  No content extracted from {md_file.name}")
            continue
        
        # Merge blocks
        if content_list_path:
            merged_blocks = merge_blocks_with_pages(blocks)
        else:
            from ingest_md import merge_blocks
            merged_blocks = merge_blocks(blocks)
            # Add default page 0
            for block in merged_blocks:
                block['page'] = 0
        
        print(f"📝 Blocks: {len(blocks)} → Chunks: {len(merged_blocks)}")
        if title:
            print(f"📄 Title: {title}")
        
        # Build payloads with title support
        payloads = build_payloads_with_titles(merged_blocks, doc_id, str(md_file), title)
        if not payloads:
            print(f"⚠️  No payloads created for {md_file.name}")
            continue
        
        # Generate embeddings
        texts = [p["text"] for p in payloads]
        vectors = embed_texts(texts)
        
        # Add to Qdrant
        add_to_qdrant(client, vectors, payloads)
        
        total_blocks += len(blocks)
        total_chunks += len(merged_blocks)
    
    print(f"\n🎉 Qdrant Local ingestion with titles complete!")
    print(f"📊 Total blocks processed: {total_blocks}")
    print(f"📊 Total chunks created: {total_chunks}")
    print(f"🔍 Titles are now searchable in Qdrant!")
    print(f"💾 Data stored in: {QDRANT_LOCAL_PATH}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Enhanced Local Ingestion Script
Ingests markdown files with title extraction, image support, and local Qdrant storage
"""

import os
import sys
import json
import argparse
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import qdrant_client.http.models as qm
from dotenv import load_dotenv

# Import title extraction
from title_extractor import extract_title_from_markdown, extract_title_from_content

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_LOCAL_PATH = "./qdrant_local"  # Local storage directory
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "agenticragdb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536  # OpenAI text-embedding-3-small

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

def load_content_list_with_images(content_list_path: str) -> Tuple[Dict[str, int], List[Dict]]:
    """Load content list and separate text and image entries"""
    with open(content_list_path, 'r', encoding='utf-8') as f:
        content_list = json.load(f)
    
    text_to_page = {}
    image_entries = []
    
    for item in content_list:
        if item.get('type') == 'text':
            page_idx = item.get('page_idx', 0)
            page_number = page_idx + 1  # Convert 0-based to 1-based
            text_to_page[item['text']] = page_number
        elif item.get('type') == 'image':
            page_idx = item.get('page_idx', 0)
            page_number = page_idx + 1  # Convert 0-based to 1-based
            item['page'] = page_number
            image_entries.append(item)
        elif item.get('type') == 'equation':
            # Treat equations as special text with image reference
            page_idx = item.get('page_idx', 0)
            page_number = page_idx + 1  # Convert 0-based to 1-based
            text_to_page[item['text']] = page_number
            if 'image' in item:
                image_entries.append({
                    'type': 'image',
                    'text': item['text'],
                    'image': item['image'],
                    'page': page_number
                })
    
    return text_to_page, image_entries

def create_image_blocks(image_entries: List[Dict], text_to_page: Dict[str, int]) -> List[Dict]:
    """Create searchable blocks for images"""
    image_blocks = []
    
    for img_entry in image_entries:
        # Create a searchable text block for the image
        caption = img_entry.get('text', '')
        page = img_entry.get('page', 0)
        img_path = img_entry.get('image', '')
        
        # Create image block
        image_block = {
            'block_id': f"img_{len(image_blocks)}",
            'block_type': 'image',
            'content_type': 'image',
            'text': caption,
            'page': page,
            'has_image': True,
            'image_caption': [caption] if caption else [],
            'image_footnote': [],
            'img_path': img_path
        }
        
        image_blocks.append(image_block)
    
    return image_blocks

def load_markdown_with_pages(md_path: str, content_list_path: str) -> List[Dict]:
    """Load markdown content with page mapping"""
    from ingest_md import load_markdown_content
    
    # Load content list for page mapping
    with open(content_list_path, 'r', encoding='utf-8') as f:
        content_list = json.load(f)
    
    # Check if content_list has actual page information (using page_idx)
    pages = [item.get('page_idx', 0) for item in content_list if item.get('type') == 'text']
    has_page_info = max(pages) > 0 if pages else False
    
    if not has_page_info:
        print(f"   Warning: No page information in content_list.json, using section-based organization")
        # Load markdown blocks without page mapping
        blocks = load_markdown_content(md_path)
        
        # Use section-based page estimation
        current_page = 1
        chars_per_page = 2000  # Approximate characters per page
        current_chars = 0
        
        for i, block in enumerate(blocks):
            text = block.get('text', '')
            text_length = len(text)
            
            # Check if this looks like a new section (header, short text, etc.)
            is_header = (text_length < 200 and 
                        (text.startswith('#') or 
                         text.isupper() or 
                         text.startswith(('1 ', '2 ', '3 ', '4 ', '5 ', '6 ', '7 ', '8 ', '9 ')) or
                         'Einleitung' in text or 'Stand der Technik' in text or 'Methodik' in text or
                         'Ergebnisse' in text or 'Diskussion' in text or 'Fazit' in text or 'Literatur' in text))
            
            # If it's a header or we've accumulated enough content, move to next page
            if is_header and current_chars > 500:
                current_page += 1
                current_chars = text_length
            else:
                current_chars += text_length
                
                # If we've accumulated enough characters, move to next page
                if current_chars > chars_per_page:
                    current_page += 1
                    current_chars = text_length
            
            block['page'] = current_page
        return blocks
    
    # Create text to page mapping (convert page_idx to 1-based page numbers)
    text_to_page = {}
    for item in content_list:
        if item.get('type') == 'text':
            page_idx = item.get('page_idx', 0)
            page_number = page_idx + 1  # Convert 0-based to 1-based
            text_to_page[item['text']] = page_number
    
    # Load markdown blocks
    blocks = load_markdown_content(md_path)
    
    # Add page information
    for block in blocks:
        text = block.get('text', '')
        # Try to find matching page
        page = 0
        for content_text, content_page in text_to_page.items():
            if text.strip() in content_text.strip() or content_text.strip() in text.strip():
                page = content_page
                break
        block['page'] = page
    
    return blocks

def build_payloads(merged_blocks: List[Dict], doc_id: str, source_md: str, title: str = None) -> List[Dict]:
    """Build payloads for Qdrant with title and image metadata"""
    payloads = []
    
    for i, block in enumerate(merged_blocks):
        # Base payload
        payload = {
            "doc_id": doc_id,
            "title": title or "Unknown Title",
            "source_md": source_md,
            "section": block.get('section', ''),
            "page": block.get('page', 0),
            "block_type": block.get('block_type', 'text'),
            "content_type": block.get('content_type', 'text'),
            "block_id": block.get('block_id', f"block_{i}"),
            "text": block.get('text', ''),
            "has_image": block.get('has_image', False)
        }
        
        # Add image-specific metadata if it's an image block
        if block.get('block_type') == 'image':
            payload.update({
                "image_caption": block.get('image_caption', []),
                "image_footnote": block.get('image_footnote', []),
                "img_path": block.get('img_path', ''),
                "cloud_image_url": block.get('cloud_image_url', ''),
                "s3_key": block.get('s3_key', '')
            })
        
        payloads.append(payload)
    
    return payloads

def get_or_create_qdrant_collection():
    """Get or create Qdrant collection with local storage"""
    # Initialize Qdrant client with local storage
    client = QdrantClient(path=QDRANT_LOCAL_PATH)
    
    # Check if collection exists
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]
    
    if QDRANT_COLLECTION in collection_names:
        print(f"✅ Loaded existing collection: {QDRANT_COLLECTION}")
        # Get collection info
        info = client.get_collection(QDRANT_COLLECTION)
        print(f"📊 Collection size: {info.points_count}")
    else:
        print(f"🔧 Creating new collection: {QDRANT_COLLECTION}")
        # Create collection with proper configuration
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        
        # Create payload indexes for efficient filtering
        print("🔍 Creating payload indexes...")
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="title",
            field_schema=qm.PayloadSchemaType.TEXT
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="doc_id",
            field_schema=qm.PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="section",
            field_schema=qm.PayloadSchemaType.TEXT
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="page",
            field_schema=qm.PayloadSchemaType.INTEGER
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="block_type",
            field_schema=qm.PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="has_image",
            field_schema=qm.PayloadSchemaType.BOOL
        )
        print("✅ Payload indexes created")
    
    return client

def add_to_qdrant(client, vectors: List[List[float]], payloads: List[Dict]):
    """Add documents to Qdrant with deterministic IDs"""
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
        collection_name=QDRANT_COLLECTION,
        points=points
    )
    
    print(f"✅ Added {len(points)} documents to Qdrant")
    
    # Get updated collection info
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"📊 Total documents in collection: {info.points_count}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced local ingestion with title extraction and image support")
    parser.add_argument("input_path", help="Path to markdown files or single file")
    parser.add_argument("--doc_id", help="Document ID (if not provided, will use filename)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Error: Path {input_path} does not exist")
        return
    
    # Find markdown files
    if input_path.is_file() and input_path.suffix == '.md':
        md_files = [input_path]
    else:
        md_files = list(input_path.rglob("*.md"))
    
    # Find content list files
    content_list_files = list(input_path.rglob("*_content_list.json"))
    
    if not content_list_files:
        print(f"Warning: No content_list.json files found. Will use default page 0.")
        content_list_files = []
    
    print(f"Found {len(md_files)} markdown files and {len(content_list_files)} content list files")
    
    # Get or create Qdrant collection
    client = get_or_create_qdrant_collection()
    
    total_blocks = 0
    total_chunks = 0
    
    for md_file in md_files:
        print(f"\nProcessing: {md_file.name}")
        
        # Find corresponding content_list.json
        content_list_path = None
        for cl_file in content_list_files:
            if cl_file.parent == md_file.parent:
                content_list_path = str(cl_file)
                break
        
        if not content_list_path:
            print(f"Warning: No content_list.json found for {md_file.name}, using default page 0")
            content_list_path = None
        
        # Extract doc_id from filename or use provided one
        doc_id = args.doc_id or md_file.stem
        
        # Load and process markdown with page mapping and images
        image_blocks = []
        if content_list_path:
            try:
                # Load both text and image data
                text_to_page, image_entries = load_content_list_with_images(content_list_path)
                blocks = load_markdown_with_pages(str(md_file), content_list_path)
                # Create image blocks
                image_blocks = create_image_blocks(image_entries, text_to_page)
            except FileNotFoundError as e:
                print(f"Warning: Content list file not found (likely due to path length): {e}")
                print(f"   Falling back to method without page mapping")
                content_list_path = None  # Reset to None to use fallback
                try:
                    from ingest_md import load_markdown_content
                    blocks = load_markdown_content(str(md_file))
                    # Add default page 0
                    for block in blocks:
                        block['page'] = 0
                except FileNotFoundError as file_error:
                    print(f"Error: Markdown file also not found: {file_error}")
                    print(f"   Skipping this file due to path length issues")
                    continue
        else:
            # Fallback to original method without page mapping
            from ingest_md import load_markdown_content
            blocks = load_markdown_content(str(md_file))
            # Add default page 0
            for block in blocks:
                block['page'] = 0
        
        # Combine text and image blocks
        all_blocks = blocks + image_blocks
        
        print(f"Created {len(blocks)} text blocks with page mapping")
        print(f"Text blocks: {len(blocks)} -> Chunks: {len([b for b in blocks if b.get('block_type') == 'text'])}")
        print(f"Image blocks: {len(image_blocks)}")
        print(f"Total blocks: {len(all_blocks)}")
        
        # Extract title
        title = None
        try:
            if len(str(md_file)) > 260:
                print("Path too long for direct file access, using content-based extraction")
                # Try to read the file content directly
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    title = extract_title_from_content(content)
                except:
                    title = None
            else:
                title = extract_title_from_markdown(str(md_file))
        except Exception as e:
            print(f"Warning: Could not extract title: {e}")
            title = None
        
        if title:
            print(f"Extracted title: {title}")
        else:
            print("No title extracted")
        
        # Build payloads
        payloads = build_payloads(all_blocks, doc_id, str(md_file), title)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(payloads)} text chunks...")
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        texts = [p['text'] for p in payloads if p['text']]
        if texts:
            embeddings = openai_client.embeddings.create(
                model=EMBED_MODEL,
                input=texts
            )
            
            # Create vectors list for all payloads
            vectors = []
            for payload in payloads:
                if payload['text']:
                    # Find the corresponding embedding
                    text_idx = [p['text'] for p in payloads if p['text']].index(payload['text'])
                    vectors.append(embeddings.data[text_idx].embedding)
                else:
                    # For blocks without text, use zero vector
                    vectors.append([0.0] * EMBED_DIM)
            
            # Add to Qdrant
            add_to_qdrant(client, vectors, payloads)
            
            total_blocks += len(all_blocks)
            total_chunks += len([p for p in payloads if p.get('block_type') == 'text'])
    
    print(f"\n🎉 Enhanced local ingestion complete!")
    print(f"📊 Total blocks processed: {total_blocks}")
    print(f"📊 Total chunks ingested: {total_chunks}")
    print(f"📄 Page numbers mapped from content_list.json files")
    print(f"🖼️ Image blocks included for visual content")
    print(f"💾 Data stored in: {QDRANT_LOCAL_PATH}")

if __name__ == "__main__":
    main()

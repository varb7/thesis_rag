#!/usr/bin/env python3
"""
Enhanced Ingestion System
Combines markdown content with content_list.json for accurate page numbers
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
EMBED_DIM = 1536  # OpenAI text-embedding-3-small

# Constants
MAX_CHARS = 2000  # Maximum characters per chunk
MIN_CHARS = 100   # Minimum characters per chunk

def deterministic_id(doc_id: str, page: int, text: str, prefix_len: int = 64) -> int:
    """Generate deterministic integer ID for Qdrant"""
    h = hashlib.sha1()
    h.update(doc_id.encode())
    h.update(str(page).encode())
    h.update(text[:prefix_len].encode())
    return int(h.hexdigest()[:16], 16)

def load_content_list(content_list_path: str) -> Dict[str, int]:
    """Load content list and create text-to-page mapping"""
    print(f"📋 Loading content list from: {content_list_path}")
    
    with open(content_list_path, 'r', encoding='utf-8') as f:
        content_list = json.load(f)
    
    # Create a mapping from text content to page number
    text_to_page = {}
    
    for item in content_list:
        if item.get('type') == 'text' and item.get('text', '').strip():
            # Clean the text for matching
            clean_text = item['text'].strip()
            if len(clean_text) > 10:  # Only map substantial text
                text_to_page[clean_text] = item.get('page_idx', 0)
    
    print(f"📄 Mapped {len(text_to_page)} text segments to page numbers")
    return text_to_page

def find_best_page_match(text: str, text_to_page: Dict[str, int]) -> int:
    """Find the best matching page number for a given text"""
    if not text_to_page:
        return 0
    
    # Try exact match first
    if text in text_to_page:
        return text_to_page[text]
    
    # Try partial matches (text is contained in content_list text)
    for content_text, page in text_to_page.items():
        if text in content_text or content_text in text:
            return page
    
    # Try fuzzy matching with first few sentences
    first_sentences = '. '.join(text.split('.')[:2]).strip()
    for content_text, page in text_to_page.items():
        if first_sentences in content_text or content_text in first_sentences:
            return page
    
    # Default to page 0 if no match found
    return 0

def load_markdown_with_pages(md_path: str, content_list_path: str) -> List[Dict]:
    """Load markdown content and map to page numbers using content list"""
    print(f"📖 Loading markdown with page mapping from: {md_path}")
    
    # Load content list for page mapping
    text_to_page = load_content_list(content_list_path)
    
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
    
    print(f"📚 Created {len(blocks)} text blocks with page mapping")
    return blocks

def create_blocks_with_pages(text: str, section: str, text_to_page: Dict[str, int]) -> List[Dict]:
    """Create text blocks with page numbers from content list"""
    blocks = []
    
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())
    
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) < MIN_CHARS:
            continue
            
        # Find the best page match for this paragraph
        page_number = find_best_page_match(para, text_to_page)
        
        # If paragraph is too long, split it further
        if len(para) > MAX_CHARS:
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= MAX_CHARS:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        # Find page for this chunk
                        chunk_page = find_best_page_match(current_chunk.strip(), text_to_page)
                        blocks.append({
                            "id": f"{section}_{i}_{len(blocks)}",
                            "text": current_chunk.strip(),
                            "section": section,
                            "page": chunk_page,
                            "block_type": "text",
                            "content_type": "paragraph"
                        })
                    current_chunk = sentence + " "
            
            # Don't forget the last chunk
            if current_chunk.strip():
                chunk_page = find_best_page_match(current_chunk.strip(), text_to_page)
                blocks.append({
                    "id": f"{section}_{i}_{len(blocks)}",
                    "text": current_chunk.strip(),
                    "section": section,
                    "page": chunk_page,
                    "block_type": "text",
                    "content_type": "paragraph"
                })
        else:
            # Paragraph fits in one chunk
            blocks.append({
                "id": f"{section}_{i}",
                "text": para,
                "section": section,
                "page": page_number,
                "block_type": "text",
                "content_type": "paragraph"
            })
    
    return blocks

def merge_blocks_with_pages(blocks: List[Dict]) -> List[Dict]:
    """Merge small blocks while preserving page information"""
    blocks = sorted(blocks, key=lambda x: (x["section"], x["page"], x["id"]))
    merged, buf = [], []

    def flush():
        nonlocal buf
        if not buf:
            return
        if len(buf) == 1:
            merged.append(buf[0])
        else:
            text = "\n".join(x["text"] for x in buf).strip()
            b0 = buf[0]
            # Use the page number from the first block in the merged group
            merged.append({
                "id": f"{b0['id']}+{buf[-1]['id']}",
                "text": text,
                "section": b0["section"],
                "page": b0["page"],
                "block_type": b0["block_type"],
                "content_type": b0["content_type"]
            })
        buf = []

    for b in blocks:
        if not b["text"].strip():
            continue
        if not buf:
            buf = [b]
            continue
            
        cur = "\n".join(x["text"] for x in buf)
        same_section_page = (b["section"] == buf[-1]["section"] and b["page"] == buf[-1]["page"])
        
        if same_section_page and len(cur) < MAX_CHARS and len(cur) + len(b["text"]) <= MAX_CHARS:
            buf.append(b)
        else:
            flush()
            buf = [b]
    
    flush()
    
    # Ensure minimum size by merging with previous when possible
    final = []
    for b in merged:
        if final and len(b["text"]) < MIN_CHARS and final[-1]["section"] == b["section"] and final[-1]["page"] == b["page"]:
            last = final.pop()
            final.append({
                "id": f"{last['id']}+{b['id']}",
                "text": (last["text"] + "\n" + b["text"]).strip(),
                "section": last["section"],
                "page": last["page"],
                "block_type": last["block_type"],
                "content_type": last["content_type"]
            })
        else:
            final.append(b)
    
    return final

def ensure_collection(client: QdrantClient):
    """Ensure the collection exists with proper configuration"""
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
        client.create_payload_index(COLLECTION, "page", field_type=qm.PayloadSchemaType.INTEGER)
        client.create_payload_index(COLLECTION, "content_type", field_type=qm.PayloadSchemaType.KEYWORD)
    else:
        print(f"✅ Collection {COLLECTION} already exists")

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

def upsert(client: QdrantClient, payloads: List[Dict], vectors: List[List[float]]):
    """Upsert points to Qdrant"""
    if not payloads:
        return
        
    points = []
    for payload, vector in zip(payloads, vectors):
        points.append(qm.PointStruct(
            id=deterministic_id(payload["doc_id"], payload["page"], payload["text"]),
            vector=vector,
            payload=payload
        ))
    
    if points:
        print(f"💾 Upserting {len(points)} points to Qdrant...")
        client.upsert(collection_name=COLLECTION, points=points)

def build_payloads(merged_blocks: List[Dict], doc_id: str, source_md: str) -> List[Dict]:
    """Build payloads for Qdrant"""
    out = []
    for b in merged_blocks:
        if not b["text"].strip():
            continue
        out.append({
            "doc_id": doc_id,
            "source_md": source_md,
            "section": b["section"],
            "page": b["page"],
            "block_type": b["block_type"],
            "content_type": b["content_type"],
            "block_id": b["id"],
            "text": b["text"]
        })
    return out

def main():
    ap = argparse.ArgumentParser(description="Enhanced ingestion with page numbers from content_list.json")
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
        print(f"⚠️  No content_list.json files found. Will use default page 0.")
        content_list_files = []
    
    print(f"📁 Found {len(md_files)} markdown files and {len(content_list_files)} content list files")
    
    # Initialize Qdrant client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    ensure_collection(client)
    
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
            print(f"⚠️  No content_list.json found for {md_file.name}, using default page 0")
            content_list_path = None
        
        # Extract doc_id from filename or use provided one
        doc_id = args.doc_id or md_file.stem
        
        # Load and process markdown with page mapping
        if content_list_path:
            blocks = load_markdown_with_pages(str(md_file), content_list_path)
        else:
            # Fallback to original method without page mapping
            from ingest_md import load_markdown_content
            blocks = load_markdown_content(str(md_file))
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
        
        # Build payloads
        payloads = build_payloads(merged_blocks, doc_id, str(md_file))
        if not payloads:
            print(f"⚠️  No payloads created for {md_file.name}")
            continue
        
        # Generate embeddings
        texts = [p["text"] for p in payloads]
        vectors = embed_texts(texts)
        
        # Upsert to Qdrant
        upsert(client, payloads, vectors)
        
        total_blocks += len(blocks)
        total_chunks += len(merged_blocks)
    
    print(f"\n🎉 Enhanced ingestion complete!")
    print(f"📊 Total blocks processed: {total_blocks}")
    print(f"📊 Total chunks ingested: {total_chunks}")
    print(f"📄 Page numbers mapped from content_list.json files")

if __name__ == "__main__":
    main()

i#!/usr/bin/env python3
"""
Ingest Markdown Files for RAG
Extracts rich text content from MinerU markdown output
"""

import os
import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Dict
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

def deterministic_id(doc_id: str, section: str, text: str, prefix_len: int = 64) -> int:
    """Generate deterministic integer ID for Qdrant"""
    h = hashlib.sha1()
    h.update(doc_id.encode())
    h.update(section.encode())
    h.update(text[:prefix_len].encode())
    return int(h.hexdigest()[:16], 16)

def load_markdown_content(md_path: str) -> List[Dict]:
    """Load and parse markdown content into structured blocks"""
    print(f" Loading markdown from: {md_path}")
    
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
                blocks.extend(create_blocks_from_text(current_content.strip(), current_section))
            
            # Start new section
            current_section = section.strip('#').strip()
            current_content = ""
        else:
            # This is content for the current section
            current_content += section + "\n"
    
    # Don't forget the last section
    if current_content.strip():
        blocks.extend(create_blocks_from_text(current_content.strip(), current_section))
    
    print(f" Created {len(blocks)} text blocks from markdown")
    return blocks

def create_blocks_from_text(text: str, section: str) -> List[Dict]:
    """Create text blocks from a section's content"""
    blocks = []
    
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())
    
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) < MIN_CHARS:
            continue
            
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
                        blocks.append({
                            "id": f"{section}_{i}_{len(blocks)}",
                            "text": current_chunk.strip(),
                            "section": section,
                            "block_type": "text",
                            "content_type": "paragraph"
                        })
                    current_chunk = sentence + " "
            
            # Don't forget the last chunk
            if current_chunk.strip():
                blocks.append({
                    "id": f"{section}_{i}_{len(blocks)}",
                    "text": current_chunk.strip(),
                    "section": section,
                    "block_type": "text",
                    "content_type": "paragraph"
                })
        else:
            # Paragraph fits in one chunk
            blocks.append({
                "id": f"{section}_{i}",
                "text": para,
                "section": section,
                "block_type": "text",
                "content_type": "paragraph"
            })
    
    return blocks

def merge_blocks(blocks: List[Dict]) -> List[Dict]:
    """Merge small blocks with adjacent ones when possible"""
    blocks = sorted(blocks, key=lambda x: (x["section"], x["id"]))
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
            merged.append({
                "id": f"{b0['id']}+{buf[-1]['id']}",
                "text": text,
                "section": b0["section"],
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
        same_section = (b["section"] == buf[-1]["section"])
        
        if same_section and len(cur) < MAX_CHARS and len(cur) + len(b["text"]) <= MAX_CHARS:
            buf.append(b)
        else:
            flush()
            buf = [b]
    
    flush()
    
    # Ensure minimum size by merging with previous when possible
    final = []
    for b in merged:
        if final and len(b["text"]) < MIN_CHARS and final[-1]["section"] == b["section"]:
            last = final.pop()
            final.append({
                "id": f"{last['id']}+{b['id']}",
                "text": (last["text"] + "\n" + b["text"]).strip(),
                "section": last["section"],
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
        print(f" Creating collection: {COLLECTION}")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
        )
        # Create payload indexes
        client.create_payload_index(COLLECTION, "block_type", field_type=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "doc_id", field_type=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "section", field_type=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "content_type", field_type=qm.PayloadSchemaType.KEYWORD)
    else:
        print(f" Collection {COLLECTION} already exists")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI"""
    print(f" Generating embeddings for {len(texts)} text chunks...")
    
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
            id=deterministic_id(payload["doc_id"], payload["section"], payload["text"]),
            vector=vector,
            payload=payload
        ))
    
    if points:
        print(f" Upserting {len(points)} points to Qdrant...")
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
            "block_type": b["block_type"],
            "content_type": b["content_type"],
            "block_id": b["id"],
            "text": b["text"]
        })
    return out

def main():
    ap = argparse.ArgumentParser(description="Ingest markdown files into Qdrant")
    ap.add_argument("folder", help="Path to folder containing markdown files")
    ap.add_argument("--doc_id", help="Document ID (default: folder name)")
    
    args = ap.parse_args()
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f" Folder not found: {folder_path}")
        return
    
    # Find markdown files
    md_files = list(folder_path.rglob("*.md"))
    if not md_files:
        print(f" No markdown files found in: {folder_path}")
        return
    
    print(f" Found {len(md_files)} markdown files")
    
    # Initialize Qdrant client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    ensure_collection(client)
    
    total_blocks = 0
    total_chunks = 0
    
    for md_file in md_files:
        print(f"\n Processing: {md_file.name}")
        
        # Extract doc_id from filename or use provided one
        doc_id = args.doc_id or md_file.stem
        
        # Load and process markdown
        blocks = load_markdown_content(str(md_file))
        if not blocks:
            print(f"  No content extracted from {md_file.name}")
            continue
        
        # Merge blocks
        merged_blocks = merge_blocks(blocks)
        print(f" Blocks: {len(blocks)} → Chunks: {len(merged_blocks)}")
        
        # Build payloads
        payloads = build_payloads(merged_blocks, doc_id, str(md_file))
        if not payloads:
            print(f"  No payloads created for {md_file.name}")
            continue
        
        # Generate embeddings
        texts = [p["text"] for p in payloads]
        vectors = embed_texts(texts)
        
        # Upsert to Qdrant
        upsert(client, payloads, vectors)
        
        total_blocks += len(blocks)
        total_chunks += len(merged_blocks)
    
    print(f"\n Ingestion complete!")
    print(f" Total blocks processed: {total_blocks}")
    print(f" Total chunks ingested: {total_chunks}")

if __name__ == "__main__":
    main()

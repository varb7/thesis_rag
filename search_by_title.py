#!/usr/bin/env python3
"""
Search Papers by Title
Utility to search and display papers using the enhanced title field
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "agenticragdb")

def search_papers_by_title(query: str, limit: int = 10):
    """Search papers by title"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        # Search for papers with matching titles
        results = client.search(
            collection_name=COLLECTION,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="title",
                        match=MatchText(text=query)
                    )
                ]
            ),
            limit=limit,
            with_payload=True
        )
        
        return results
    except Exception as e:
        print(f"❌ Error searching: {e}")
        return []

def list_all_papers(limit: int = 50):
    """List all papers with their titles"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        # Get all documents, grouped by doc_id
        results = client.scroll(
            collection_name=COLLECTION,
            limit=limit,
            with_payload=True
        )
        
        # Group by doc_id and get unique titles
        papers = {}
        for point in results[0]:
            doc_id = point.payload.get('doc_id')
            title = point.payload.get('title', 'Unknown Title')
            
            if doc_id not in papers:
                papers[doc_id] = {
                    'title': title,
                    'chunks': 0,
                    'sections': set(),
                    'pages': set()
                }
            
            papers[doc_id]['chunks'] += 1
            papers[doc_id]['sections'].add(point.payload.get('section', 'Unknown'))
            papers[doc_id]['pages'].add(point.payload.get('page', 0))
        
        return papers
    except Exception as e:
        print(f"❌ Error listing papers: {e}")
        return {}

def search_papers_by_keyword(keyword: str, limit: int = 10):
    """Search papers by keyword in title or content"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        # Search for papers with keyword in title or text
        results = client.search(
            collection_name=COLLECTION,
            query_filter=Filter(
                should=[
                    FieldCondition(
                        key="title",
                        match=MatchText(text=keyword)
                    ),
                    FieldCondition(
                        key="text",
                        match=MatchText(text=keyword)
                    )
                ]
            ),
            limit=limit,
            with_payload=True
        )
        
        return results
    except Exception as e:
        print(f"❌ Error searching: {e}")
        return []

def display_paper_details(papers: dict):
    """Display paper details in a formatted way"""
    if not papers:
        print("📚 No papers found")
        return
    
    print(f"\n📚 Found {len(papers)} papers:")
    print("=" * 80)
    
    for i, (doc_id, paper_info) in enumerate(papers.items(), 1):
        print(f"\n{i}. 📄 Document ID: {doc_id}")
        print(f"   📖 Title: {paper_info['title']}")
        print(f"   📝 Chunks: {paper_info['chunks']}")
        print(f"   📚 Sections: {', '.join(sorted(paper_info['sections']))}")
        print(f"   📄 Pages: {', '.join(map(str, sorted(paper_info['pages'])))}")
        print("-" * 60)

def display_search_results(results, search_type: str):
    """Display search results"""
    if not results:
        print(f"🔍 No {search_type} found")
        return
    
    print(f"\n🔍 {search_type} Results ({len(results)} found):")
    print("=" * 80)
    
    # Group by doc_id to avoid duplicates
    papers = {}
    for result in results:
        doc_id = result.payload.get('doc_id')
        title = result.payload.get('title', 'Unknown Title')
        section = result.payload.get('section', 'Unknown')
        page = result.payload.get('page', 0)
        score = result.score
        
        if doc_id not in papers:
            papers[doc_id] = {
                'title': title,
                'best_score': score,
                'sections': set(),
                'pages': set()
            }
        
        papers[doc_id]['sections'].add(section)
        papers[doc_id]['pages'].add(page)
        papers[doc_id]['best_score'] = max(papers[doc_id]['best_score'], score)
    
    # Display grouped results
    for i, (doc_id, paper_info) in enumerate(papers.items(), 1):
        print(f"\n{i}. 📄 Document ID: {doc_id}")
        print(f"   📖 Title: {paper_info['title']}")
        print(f"   🎯 Relevance Score: {paper_info['best_score']:.3f}")
        print(f"   📚 Sections: {', '.join(sorted(paper_info['sections']))}")
        print(f"   📄 Pages: {', '.join(map(str, sorted(paper_info['pages'])))}")
        print("-" * 60)

def main():
    """Main interactive search interface"""
    print("🔍 Paper Title Search Utility")
    print("=" * 80)
    
    while True:
        print("\n📋 Available commands:")
        print("1. List all papers")
        print("2. Search by title")
        print("3. Search by keyword")
        print("4. Exit")
        
        choice = input("\n❓ Choose an option (1-4): ").strip()
        
        if choice == "1":
            print("\n📚 Listing all papers...")
            papers = list_all_papers()
            display_paper_details(papers)
            
        elif choice == "2":
            query = input("\n🔍 Enter title to search for: ").strip()
            if query:
                print(f"\n🔍 Searching for papers with title: '{query}'")
                results = search_papers_by_title(query)
                display_search_results(results, "Title Matches")
            else:
                print("❌ Please enter a search query")
                
        elif choice == "3":
            keyword = input("\n🔍 Enter keyword to search for: ").strip()
            if keyword:
                print(f"\n🔍 Searching for papers with keyword: '{keyword}'")
                results = search_papers_by_keyword(keyword)
                display_search_results(results, "Keyword Matches")
            else:
                print("❌ Please enter a keyword")
                
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()

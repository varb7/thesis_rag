#!/usr/bin/env python3
"""
Test Title Extraction
Demonstrates how the enhanced ingestion extracts paper titles
"""

import json
from pathlib import Path
from ingest_enhanced_with_titles import extract_paper_title, load_content_list_with_title

def test_title_extraction():
    """Test title extraction on existing papers"""
    
    # Find all content_list.json files
    mineru_dir = Path("mineru_out")
    content_list_files = list(mineru_dir.rglob("*_content_list.json"))
    
    print(f"🔍 Found {len(content_list_files)} content list files to test")
    print("=" * 80)
    
    for content_list_file in content_list_files:
        print(f"\n📖 Testing: {content_list_file.name}")
        print("-" * 60)
        
        try:
            # Load content list
            with open(content_list_file, 'r', encoding='utf-8') as f:
                content_list = json.load(f)
            
            # Extract title using our function
            title = extract_paper_title(content_list)
            
            if title:
                print(f"✅ Title extracted: {title}")
                
                # Show the first few items for context
                print("📋 First few content items:")
                for i, item in enumerate(content_list[:5]):
                    item_type = item.get('type', 'unknown')
                    text_level = item.get('text_level', 'N/A')
                    page_idx = item.get('page_idx', 'N/A')
                    text_preview = item.get('text', '')[:100] + "..." if len(item.get('text', '')) > 100 else item.get('text', '')
                    
                    print(f"  {i+1}. Type: {item_type}, Level: {text_level}, Page: {page_idx}")
                    print(f"     Text: {text_preview}")
            else:
                print("❌ No title found")
                
                # Show what we have instead
                print("📋 Available content items:")
                for i, item in enumerate(content_list[:3]):
                    item_type = item.get('type', 'unknown')
                    text_level = item.get('text_level', 'N/A')
                    page_idx = item.get('page_idx', 'N/A')
                    text_preview = item.get('text', '')[:100] + "..." if len(item.get('text', '')) > 100 else item.get('text', '')
                    
                    print(f"  {i+1}. Type: {item_type}, Level: {text_level}, Page: {page_idx}")
                    print(f"     Text: {text_preview}")
        
        except Exception as e:
            print(f"❌ Error processing {content_list_file.name}: {e}")
        
        print("-" * 60)

def test_content_list_loading():
    """Test the full content list loading with title extraction"""
    
    print("\n🧪 Testing Full Content List Loading")
    print("=" * 80)
    
    # Test with one specific file
    test_file = Path("mineru_out/2401.02930v1-f951b4dc/2401.02930v1/auto/2401.02930v1_content_list.json")
    
    if test_file.exists():
        print(f"📖 Testing with: {test_file.name}")
        
        try:
            text_to_page, title = load_content_list_with_title(str(test_file))
            
            print(f"✅ Title extracted: {title}")
            print(f"📄 Page mappings created: {len(text_to_page)}")
            
            # Show some page mappings
            print("\n📋 Sample page mappings:")
            for i, (text, page) in enumerate(list(text_to_page.items())[:3]):
                text_preview = text[:80] + "..." if len(text) > 80 else text
                print(f"  {i+1}. Page {page}: {text_preview}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"❌ Test file not found: {test_file}")

if __name__ == "__main__":
    print("🚀 Testing Enhanced Title Extraction")
    print("=" * 80)
    
    # Test basic title extraction
    test_title_extraction()
    
    # Test full content list loading
    test_content_list_loading()
    
    print("\n🎉 Title extraction testing complete!")
    print("\n💡 To use the enhanced ingestion:")
    print("   python ingest_enhanced_with_titles.py mineru_out/2401.02930v1-f951b4dc/2401.02930v1/auto/")

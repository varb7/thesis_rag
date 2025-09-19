#!/usr/bin/env python3
"""
Title Extraction Utility
Extracts paper titles from the first line of markdown files
"""

import re
from typing import Optional

def extract_title_from_markdown(md_path: str) -> Optional[str]:
    """
    Extract the paper title from the first line of a markdown file.
    
    Args:
        md_path: Path to the markdown file
        
    Returns:
        The extracted title or None if not found
    """
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # Remove markdown header syntax (# ## ### etc.)
        title = re.sub(r'^#+\s*', '', first_line)
        
        # Clean up any extra whitespace
        title = title.strip()
        
        # Return title if it's substantial (more than 10 characters)
        if len(title) > 10:
            print(f"Extracted title from markdown: {title}")
            return title
        else:
            print(f"Title too short or empty: '{title}'")
            return None
            
    except Exception as e:
        print(f"Error extracting title from {md_path}: {e}")
        return None

def extract_title_from_content(content: str) -> Optional[str]:
    """
    Extract the paper title from markdown content string.
    
    Args:
        content: The markdown content as a string
        
    Returns:
        The extracted title or None if not found
    """
    try:
        # Get the first line
        first_line = content.split('\n')[0].strip()
        
        # Remove markdown header syntax (# ## ### etc.)
        title = re.sub(r'^#+\s*', '', first_line)
        
        # Clean up any extra whitespace
        title = title.strip()
        
        # Return title if it's substantial (more than 10 characters)
        if len(title) > 10:
            return title
        else:
            return None
            
    except Exception as e:
        print(f"Error extracting title from content: {e}")
        return None

if __name__ == "__main__":
    # Test with the sample file
    test_file = "mineru_out/NeurIPS-2018-causal-discovery-from-discrete-data-using-hidden-compact-representation-Paper(1)-c917b2ef/NeurIPS-2018-causal-discovery-from-discrete-data-using-hidden-compact-representation-Paper(1)/auto/NeurIPS-2018-causal-discovery-from-discrete-data-using-hidden-compact-representation-Paper(1).md"
    
    title = extract_title_from_markdown(test_file)
    if title:
        print(f"Successfully extracted title: {title}")
    else:
        print("Failed to extract title")

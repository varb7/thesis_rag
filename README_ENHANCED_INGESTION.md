# Enhanced Ingestion System with Title Support

A comprehensive upgrade to the ingestion system that properly extracts, stores, and makes paper titles easily searchable.

## 🚀 **What's New**

### **✅ Title Extraction**
- **Automatic title detection** from `content_list.json`
- **Smart fallback** for papers without clear title structure
- **Title validation** to ensure quality extraction

### **✅ Enhanced Storage**
- **Dedicated `title` field** in Qdrant payloads
- **Title-specific indexing** for fast searches
- **Title in every chunk** for consistent access

### **✅ Improved Search**
- **Title-based queries** for finding specific papers
- **Keyword search** across titles and content
- **Paper listing** with full metadata

## 📁 **Files Overview**

```
ingest_enhanced_with_titles.py    # 🆕 Main enhanced ingestion script
test_title_extraction.py         # 🧪 Test title extraction functionality
search_by_title.py               # 🔍 Search papers by title
README_ENHANCED_INGESTION.md     # 📚 This documentation
```

## 🔧 **How It Works**

### **1. Title Extraction Process**
```python
def extract_paper_title(content_list):
    # Primary: Look for text_level == 1 (top-level header)
    for item in content_list:
        if (item.get('type') == 'text' and 
            item.get('text_level') == 1 and 
            item.get('page_idx') == 0):
            return item['text'].strip()
    
    # Fallback: First substantial text on page 0
    for item in content_list:
        if (item.get('type') == 'text' and 
            item.get('page_idx') == 0 and
            len(item['text'].strip()) > 20):
            return item['text'].strip()
```

### **2. Enhanced Storage Schema**
```json
{
    "doc_id": "2401.02930v1",
    "title": "DAGMA-DCE: Interpretable, Non-Parametric Differentiable Causal Discovery",
    "section": "I. INTRODUCTION",
    "page": 0,
    "block_type": "text",
    "text": "The discovery of causal relationships...",
    "source_md": "path/to/markdown",
    "content_type": "paragraph"
}
```

### **3. Title Indexing**
```python
# Creates dedicated title index for fast searches
client.create_payload_index(COLLECTION, "title", field_type=qm.PayloadSchemaType.TEXT)
```

## 🚀 **Usage**

### **Enhanced Ingestion**
```bash
# Process a single paper folder
python ingest_enhanced_with_titles.py mineru_out/2401.02930v1-f951b4dc/2401.02930v1/auto/

# Process with custom document ID
python ingest_enhanced_with_titles.py mineru_out/paper_folder/ --doc_id "my_paper"
```

### **Test Title Extraction**
```bash
# Test title extraction on existing papers
python test_title_extraction.py
```

### **Search Papers by Title**
```bash
# Interactive search interface
python search_by_title.py
```

## 🔍 **Search Capabilities**

### **Title-Based Search**
```python
# Search for papers with specific title
results = search_papers_by_title("DAGMA-DCE")
```

### **Keyword Search**
```python
# Search for papers containing keywords
results = search_papers_by_keyword("causal discovery")
```

### **List All Papers**
```python
# Get overview of all ingested papers
papers = list_all_papers()
```

## 📊 **Before vs After**

| Feature | Old System | Enhanced System |
|---------|------------|-----------------|
| **Title Storage** | ❌ Mixed with content | ✅ Dedicated `title` field |
| **Title Search** | ❌ Difficult/impossible | ✅ Fast title-based queries |
| **Title Display** | ❌ Not available | ✅ Always accessible |
| **Title Indexing** | ❌ No index | ✅ Optimized TEXT index |
| **Paper Discovery** | ❌ Content only | ✅ Title + content search |

## 🧪 **Testing**

### **Test Title Extraction**
```bash
python test_title_extraction.py
```

**Expected Output:**
```
🔍 Found 2 content list files to test
=====================================

📖 Testing: 2401.02930v1_content_list.json
------------------------------------------------------------
✅ Title extracted: DAGMA-DCE: Interpretable, Non-Parametric Differentiable Causal Discovery
📋 First few content items:
  1. Type: text, Level: 1, Page: 0
     Text: DAGMA-DCE: Interpretable, Non-Parametric Differentiable Causal Discovery
```

### **Test Enhanced Ingestion**
```bash
python ingest_enhanced_with_titles.py mineru_out/2401.02930v1-f951b4dc/2401.02930v1/auto/
```

**Expected Output:**
```
🔍 Extracting paper title...
📄 Found title: DAGMA-DCE: Interpretable, Non-Parametric Differentiable Causal Discovery
📚 Created 45 text blocks with title extraction
📝 Blocks: 45 → Chunks: 23
📄 Title: DAGMA-DCE: Interpretable, Non-Parametric Differentiable Causal Discovery
💾 Upserted 23 documents with titles
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Required
OPENAI_API_KEY=your_openai_key
QDRANT_URL=http://localhost:6333

# Optional
QDRANT_COLLECTION=agenticragdb
EMBED_MODEL=text-embedding-3-small
```

### **Collection Schema**
The enhanced system automatically creates:
- **Vector field**: 1536-dimensional embeddings
- **Title index**: TEXT type for fast title searches
- **Other indexes**: doc_id, section, page, etc.

## 🚨 **Migration Notes**

### **Existing Collections**
- **Safe to run**: Will add title index to existing collections
- **No data loss**: All existing data preserved
- **Backward compatible**: Old queries still work

### **Re-ingestion**
- **Recommended**: Re-ingest papers to get title support
- **Incremental**: Can process papers one by one
- **No duplicates**: Uses deterministic IDs

## 🎯 **Future Enhancements**

### **Planned Features**
- **Title validation**: AI-powered title quality checking
- **Title normalization**: Consistent formatting across papers
- **Title clustering**: Group similar papers by title patterns
- **Title suggestions**: Auto-complete for title searches

### **Integration Points**
- **RAG System**: Enhanced answer generation with title context
- **Paper Browser**: Web interface for title-based navigation
- **Citation System**: Automatic title-based paper references

## 🔍 **Troubleshooting**

### **Common Issues**

#### **No Title Found**
```
⚠️  No title found in content list
```
**Solution**: Check if `content_list.json` has `text_level: 1` entries on page 0

#### **Title Index Creation Failed**
```
ℹ️  Title index already exists or couldn't be created
```
**Solution**: This is normal for existing collections. Title support will still work.

#### **Qdrant Connection Error**
```
❌ Error searching: Connection refused
```
**Solution**: Ensure Qdrant is running and `QDRANT_URL` is correct

### **Debug Mode**
```python
# Add debug prints to see extraction process
print(f"DEBUG: Content list items: {len(content_list)}")
print(f"DEBUG: First item: {content_list[0]}")
```

## 📚 **Examples**

### **Example 1: Find Paper by Title**
```bash
python search_by_title.py
# Choose option 2: Search by title
# Enter: "DAGMA-DCE"
```

### **Example 2: Find Papers by Keyword**
```bash
python search_by_title.py
# Choose option 3: Search by keyword
# Enter: "causal"
```

### **Example 3: List All Papers**
```bash
python search_by_title.py
# Choose option 1: List all papers
```

## 🤝 **Contributing**

### **Adding New Features**
1. **Follow the pattern**: Use existing functions as templates
2. **Add tests**: Include test cases for new functionality
3. **Update docs**: Document new features in this README
4. **Backward compatibility**: Ensure existing functionality still works

### **Testing New Features**
```bash
# Run all tests
python test_title_extraction.py
python -c "from ingest_enhanced_with_titles import *; print('Import successful')"
```

## 📝 **License**

Same as the main project. This is an enhancement to the existing ingestion system.

---

**🎉 Enhanced ingestion is ready to use! Your papers will now have proper title support for easy discovery and search.**

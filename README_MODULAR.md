 # Modular Agentic RAG System

A clean, modular implementation of the Agentic RAG system with separated concerns and improved maintainability.

## 🏗️ Architecture Overview

The system is split into focused modules, each handling a specific responsibility:

```
modular_rag_system.py          # Main orchestrator
├── config.py                  # Configuration & environment variables
├── memory_manager.py          # Session memory & context management
├── retrieval_strategies.py    # HyDE, MMR, and retrieval logic
└── semantic_kernel_functions.py # SK function definitions & management
```

## 📁 Module Breakdown

### 1. `config.py`
- **Purpose**: Centralized configuration management
- **Features**: Environment variables, validation, default values
- **Benefits**: Easy to modify settings, environment-specific configs

### 2. `memory_manager.py`
- **Purpose**: Session-based memory and context management
- **Features**: Chat history, query context, session management
- **Benefits**: Persistent conversations, context awareness

### 3. `retrieval_strategies.py`
- **Purpose**: Advanced retrieval techniques
- **Features**: HyDE, MMR diversification, duplicate removal
- **Benefits**: Better search results, diversity in responses

### 4. `semantic_kernel_functions.py`
- **Purpose**: Semantic Kernel function management
- **Features**: Intent detection, query expansion, answer generation
- **Benefits**: AI-powered query understanding and response generation

### 5. `modular_rag_system.py`
- **Purpose**: Main system orchestrator
- **Features**: Coordinates all modules, provides unified interface
- **Benefits**: Clean separation of concerns, easy to extend

## 🚀 Key Benefits

### **Maintainability**
- Each module has a single responsibility
- Easy to locate and fix issues
- Clear interfaces between components

### **Extensibility**
- Add new retrieval strategies without touching other modules
- Swap out memory management systems
- Add new Semantic Kernel functions easily

### **Testing**
- Test each module independently
- Mock dependencies for unit tests
- Clear separation makes testing straightforward

### **Reusability**
- Use memory manager in other projects
- Reuse retrieval strategies in different contexts
- Configuration module can be shared across projects

## 🔧 Usage

### Basic Usage
```python
from modular_rag_system import ModularRAGSystem

# Initialize the system
rag = ModularRAGSystem()

# Start chatting
await rag.chat()
```

### Advanced Usage
```python
# Custom configuration
rag = ModularRAGSystem(base_path="custom_path")

# Get session info
session_info = rag.get_session_info("session_id")

# Export session data
exported_data = rag.export_session("session_id")
```

## 📦 Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements-modular.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp azure_env_example.env .env
   # Edit .env with your API keys
   ```

3. **Run the system**:
   ```bash
   python modular_rag_system.py
   ```

## 🔄 Migration from Monolithic

The original `semantic_kernel_rag.py` remains unchanged. To migrate:

1. **Keep both systems** during transition
2. **Gradually move functionality** to modular system
3. **Test thoroughly** before switching
4. **Update imports** in dependent code

## 🎯 Future Enhancements

### **Planned Modules**
- `vector_store_manager.py` - Abstract vector store operations
- `embedding_manager.py` - Handle different embedding models
- `query_optimizer.py` - Advanced query optimization
- `response_formatter.py` - Customizable output formatting

### **Integration Points**
- **LangChain**: Replace Semantic Kernel if needed
- **LlamaIndex**: Alternative RAG framework
- **Custom Models**: Local LLM integration
- **Database Backends**: PostgreSQL, MongoDB support

## 🧪 Testing

```bash
# Test individual modules
python -m pytest tests/test_memory_manager.py
python -m pytest tests/test_retrieval_strategies.py

# Test full system
python -m pytest tests/test_modular_system.py
```

## 📊 Performance

The modular system maintains the same performance as the monolithic version while providing:
- **Better memory usage** through focused modules
- **Faster development** through clear interfaces
- **Easier debugging** through isolated components

## 🤝 Contributing

1. **Follow the modular pattern** for new features
2. **Add tests** for new modules
3. **Update documentation** for changes
4. **Maintain backward compatibility** where possible

## 📝 License

Same as the original project - this is a refactored version for better maintainability.

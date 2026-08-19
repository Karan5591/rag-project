# Streamlit UI Setup & Usage Guide

## Installation

### Install Streamlit
```bash
pip install streamlit
```

Or add to your requirements.txt and install:
```bash
pip install -r requirements.txt
```

## Running the UI

### From the project root directory:
```bash
streamlit run app/streamlit_app.py
```

### Or using Python directly:
```bash
python -m streamlit run app/streamlit_app.py
```

This will:
1. Start a local Streamlit server (usually at `http://localhost:8501`)
2. Open the UI in your default browser
3. Keep the server running while you use the app

## Features

### 📄 Document Upload
- Upload single or multiple PDF files
- Process documents with automatic:
  - PDF conversion
  - Text chunking (800-token chunks with 100-token overlap)
  - Embedding generation (384-dimensional)
  - Vector storage (PGVector)

### 💬 Chat Interface
- Natural language questions about uploaded documents
- Retrieval-augmented generation (RAG) pipeline:
  - Vector similarity search
  - BGE reranking
  - LLM-powered answer generation (Llama 2)
- View source chunks for each answer

### 📊 Status Dashboard
- Track number of documents loaded
- Monitor total chunks ingested
- See retrieval and answer quality

### 🔄 Database Management
- Clear database to start fresh
- Automatic table creation on startup

## Workflow

1. **Upload Documents**
   - Click "Select PDF files" in sidebar
   - Choose one or multiple PDFs
   - Click "Process Documents"
   - Wait for processing to complete (shows progress)

2. **Chat with Documents**
   - Type a question in the chat input box
   - Press Enter or click Send
   - View AI-generated answer based on document content
   - Expand "Source Chunks" to see relevant excerpts

3. **Clear & Start Over**
   - Click "Clear Database" button to remove all documents
   - Upload new documents to continue

## Requirements

### System Requirements
- Python 3.10+
- 4GB+ RAM (8GB recommended)
- PostgreSQL 15 with pgvector extension running

### Python Dependencies
- streamlit >= 1.28.0
- pytorch (with torch.compile disabled for Windows)
- sentence-transformers (for embeddings)
- pgvector (for vector storage)
- docling (for PDF processing)
- ollama (for LLM inference)

### Running Services
Make sure these are running before starting the UI:
```bash
# PostgreSQL with pgvector
docker-compose up -d

# Ollama with Llama 2
ollama run llama2
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
Solution: `pip install streamlit`

### "Connection refused" for database
Solution: Ensure PostgreSQL is running via `docker-compose up -d`

### "No module named 'app.services'"
Solution: Run Streamlit from the project root directory:
```bash
cd d:\NIC\rag-project
streamlit run app/streamlit_app.py
```

### UI doesn't open in browser
Solution: Manually open `http://localhost:8501` in your browser

### Slow document processing
- This is normal for PDF parsing and embedding generation
- Processing time depends on PDF size and number of pages
- Large PDFs may take 30-60 seconds

## Architecture

```
Streamlit UI
    ↓
Document Upload (Sidebar)
    ↓
DocumentConverter (Docling)
    ↓
ChunkingService
    ↓
EmbeddingService (sentence-transformers)
    ↓
VectorStore (PostgreSQL + pgvector)
    ↓
Chat Input (Main Area)
    ↓
RetrievalService (cosine similarity)
    ↓
RerankerService (BGE reranker)
    ↓
ContextBuilder
    ↓
RAGPromptBuilder
    ↓
LLMService (Llama 2)
    ↓
Chat Output (Streamed response)
```

## Advanced Usage

### Modify UI Appearance
Edit colors, layout, and components in `streamlit_app.py`:
```python
st.set_page_config(page_title="RAG Document Chat", layout="wide")
```

### Adjust Chunking Parameters
In `streamlit_app.py`, modify ChunkingService initialization:
```python
"chunker": ChunkingService(chunk_size=800, chunk_overlap=100)
```

### Change LLM Model
In `services/llm.py`, modify the model name:
```python
response = self.client.generate(model="llama2", prompt=prompt)
```

## Tips for Best Results

1. **Clear Documents** between different topics to avoid confusion
2. **Use Specific Questions** for better retrieval accuracy
3. **Review Source Chunks** to verify answer quality
4. **Upload Multiple PDFs** if questions span multiple documents
5. **Ask Follow-up Questions** in the same session for context

## Performance Notes

- First query takes longer due to model initialization
- Subsequent queries are faster (models cached in memory)
- Large documents (50+ pages) may take 1-2 minutes to process
- LLM generation typically takes 2-10 seconds per query
- Vector search is fast (<100ms)

## Support

For issues or improvements:
1. Check the troubleshooting section above
2. Verify all services are running (DB, Ollama)
3. Check Python and Streamlit versions are compatible
4. Review terminal output for specific error messages

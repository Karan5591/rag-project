# Quick Start: RAG Document Chat UI

## 🚀 Run the UI

```bash
# Navigate to project directory
cd d:\NIC\rag-project

# Start the Streamlit app
streamlit run app/streamlit_app.py
```

The app will open at: **http://localhost:8501**

---

## 📋 Usage Steps

### 1. Upload Documents
- Click "Select PDF files" in the left sidebar
- Choose one or multiple PDF documents
- Click "Process Documents" button
- Wait for processing (shows progress bar)
- You'll see the document and chunk count update

### 2. Chat with Your Documents
- Type any question about your uploaded documents
- Press Enter to submit
- AI generates an answer based on your documents
- Click "📖 Source Chunks" to see which parts were used

### 3. Clear & Start Over
- Click "Clear Database" to remove all documents
- Upload new documents to continue

---

## 🎯 Example Questions

For your test documents:

**From test_document.pdf:**
- "How many unused PTO days can an employee roll over?"
- "What was the adjusted EBITDA for FY2025?"
- "What is the maximum PTO rollover policy?"

**From hybrid_test_document.pdf:**
- "What are the main Python operators?"
- "What is Python?"

---

## 🔧 Before Starting

Make sure these services are running:

### 1. PostgreSQL + pgvector
```bash
docker-compose up -d
```

### 2. Ollama with Llama 2
```bash
ollama run llama2
```

Open a new terminal and run this command once, then keep it running in background.

---

## ✨ Features

✅ **Multi-document upload** - Upload multiple PDFs at once  
✅ **Automatic chunking** - 800-token chunks with overlap  
✅ **Smart embeddings** - 384-dimensional vectors  
✅ **Vector search** - Fast similarity matching  
✅ **Intelligent reranking** - BGE reranker for quality  
✅ **Chat history** - Keep conversation context  
✅ **Source tracking** - See which chunks were used  
✅ **Clean UI** - Simple, intuitive interface  

---

## 📊 What Happens Behind the Scenes

```
User uploads PDF
    ↓
Docling converts PDF → text
    ↓
ChunkingService splits into 800-token chunks
    ↓
EmbeddingService creates 384-dim vectors
    ↓
VectorStore saves to PostgreSQL+pgvector
    ↓
User asks question
    ↓
RetrievalService finds similar chunks (top 10)
    ↓
RerankerService scores relevance (top 5)
    ↓
ContextBuilder assembles context
    ↓
RAGPromptBuilder creates prompt
    ↓
LLMService (Llama 2) generates answer
    ↓
Response shown in chat
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | Run from `d:\NIC\rag-project` directory |
| "Connection refused" | Start PostgreSQL: `docker-compose up -d` |
| "No module ollama" | Start Ollama: `ollama run llama2` |
| Slow processing | Large PDFs take 30-60 seconds - this is normal |
| UI won't open | Manually go to `http://localhost:8501` |

---

## 📁 Files Created

- **[streamlit_app.py](d:\NIC\rag-project\app\streamlit_app.py)** - Main UI application
- **[STREAMLIT_SETUP.md](d:\NIC\rag-project\STREAMLIT_SETUP.md)** - Detailed setup guide

---

## 🎓 Tips for Best Results

1. **Upload both test documents** to see full functionality
2. **Clear database** between different document sets to avoid confusion
3. **Ask specific questions** for better retrieval
4. **Check source chunks** to verify answer quality
5. **Use follow-up questions** in the same chat session

---

Ready? Let's go! 🚀

```bash
streamlit run app/streamlit_app.py
```

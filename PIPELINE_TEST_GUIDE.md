# Complete RAG Pipeline Test Guide

## Overview
This guide walks you through testing the complete RAG (Retrieval-Augmented Generation) pipeline with 2 PDF files in your `data/raw` directory:
- `test_document.pdf` (text-native)
- `hybrid_test_document.pdf` (scanned/hybrid)

---

## Pipeline Architecture

```
PDF Files (2)
    ↓
[STEP 2] Document Chunking Service
    ↓
[STEP 3] Embedding Generation (sentence-transformers, 384-dim)
    ↓
[STEP 4] PGVector Database Storage (PostgreSQL + pgvector)
    ↓
[STEP 5] Retrieval Service (broad vector search)
    ↓
[STEP 6] Reranking (BGE-Reranker-v2-m3)
    ↓
[STEP 7] Context Builder (filtering & aggregation)
    ↓
[STEP 8] RAG Prompt Constructor
    ↓
[STEP 9] LLM Generation (Llama)
    ↓
Answer
```

---

## Prerequisites
- Python 3.12 virtual environment activated
- PostgreSQL with pgvector running (Docker via docker-compose.yml)
- All dependencies installed

---

## STEP 1: Verify Prerequisites & Dependencies

### 1.1 Check Python environment
```bash
d:/NIC/rag-project/.venv/Scripts/python.exe --version
# Expected: Python 3.12.x
```

### 1.2 Verify database is running
```bash
# Check if PostgreSQL is running in Docker
docker ps | findstr postgres
# You should see a postgres container running
```

### 1.3 Verify data directory has 2 PDFs
```bash
ls -la data/raw/
# Expected output:
# - test_document.pdf
# - hybrid_test_document.pdf
```

---

## STEP 2: Clean the PGvector Database

This removes all existing documents to start fresh.

```bash
cd d:\NIC\rag-project

d:/NIC/rag-project/.venv/Scripts/python.exe -m app.clean_database
```

**Expected Output:**
```
============================================================
PGvector Database Cleaner
============================================================

[STEP 1] Checking current database state...
[TABLE] Documents Table Schema:
   - id: integer
   - content: text
   - embedding: vector(384)
   - metadata: jsonb
   - created_at: timestamp
[INFO] Total Documents: 26

[STEP 2] Cleaning database...
[OK] Dropped existing documents table
[OK] Database cleaned successfully

[STEP 3] Verifying database is empty...
[OK] Database is EMPTY - ready for fresh insertion

============================================================
[OK] Database is ready for fresh insertion!
============================================================
```

✅ **Database is now clean and ready for fresh ingestion**

---

## STEP 3: Ingest & Process Both PDF Files

This step:
1. Converts PDFs to structured documents using Docling
2. Chunks documents with overlap (chunk_size=800, overlap=100)
3. Generates embeddings for each chunk (384-dimensional)
4. Stores chunks + embeddings in PGvector database

### Command:
```bash
cd d:\NIC\rag-project

d:/NIC/rag-project/.venv/Scripts/python.exe -m app.test_chunk_vector_insert
```

### What It Does:
- Processes `test_document.pdf` → generates chunks
- Converts chunks to embeddings (384-dim via sentence-transformers)
- Inserts into pgvector database

**Expected Output Example:**
```
======================================================================
CHUNK -> EMBEDDING -> VECTOR STORE TEST
======================================================================

Total PDF files: 1

STEP 0: VALIDATING PDF FILES
File: data\raw\test_document.pdf
Exists: True

STEP 1: CONFIGURING DOCLING
Using lightweight PDF processing configuration.
Docling converter initialized.

STEP 2: INITIALIZING CHUNKING SERVICE
Chunking service initialized.

STEP 3: INITIALIZING EMBEDDING SERVICE
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Embedding model loaded successfully (384 dimensions)

STEP 4: INITIALIZING VECTOR DATABASE
Documents table created successfully with 384-dimensional embeddings

======================================================================
PROCESSING PDF 1/1: test_document.pdf
======================================================================

DOCUMENT CONVERSION
File: data\raw\test_document.pdf
Converting document...
[OK] Document processed successfully.

STRUCTURE-AWARE CHUNKING
[OK] Total chunks generated: 26

EMBEDDING + INSERTING CHUNKS
Chunk 00 → text  → embedding 384 dimensions → DB ID 1
Chunk 01 → table → embedding 384 dimensions → DB ID 2
...
Chunk 25 → text  → embedding 384 dimensions → DB ID 26

PDF VALIDATION
PDF                : test_document.pdf
Total chunks       : 26
Successfully added : 26
Failed             : 0

[OK] All chunks from this PDF were embedded and stored successfully.

======================================================================
FINAL VECTOR STORE VALIDATION
======================================================================
PDF files processed : 1
Total chunks        : 26
Successfully added  : 26
Failed              : 0

[OK] SUCCESS
All chunks were successfully embedded and stored in the vector database.
```

### ⚠️ Important: Ingesting Both PDFs

The current `test_chunk_vector_insert.py` only processes `test_document.pdf`. To ingest BOTH PDFs, you need to:

**Option A: Modify the test file** (Quick approach)
```python
# In test_chunk_vector_insert.py, change:
PDF_PATHS = [
    Path("data/raw/test_document.pdf"),
    Path("data/raw/hybrid_test_document.pdf"),  # ADD THIS
]
```

**Option B: Run ingestion twice**
```bash
# 1st run (test_document.pdf is already in PDF_PATHS)
d:/NIC/rag-project/.venv/Scripts/python.exe -m app.test_chunk_vector_insert

# 2nd run (modify PDF_PATHS temporarily to process hybrid_test_document.pdf)
```

✅ **After this step: Database contains all chunks + embeddings from both PDFs**

---

## STEP 4: Test Retrieval Service

This tests the retrieval pipeline: embedding generation → vector search → reranking

### Command:
```bash
cd d:\NIC\rag-project

d:/NIC/rag-project/.venv/Scripts/python.exe -m app.test_retrieval_service
```

### What It Tests:
- Embeds test queries
- Performs vector similarity search
- Reranks results using BGE-Reranker
- Returns top-k most relevant chunks

**Expected Output Example:**
```
======================================================================
RETRIEVAL SERVICE TEST
======================================================================

======================================================================
QUERY 1
======================================================================
Type     : RELEVANT
Question : How many unused PTO days can an employee roll over?
Retrieved: 5

----------------------------------------------------------------------
Rank       : 1
Database ID: 23
Distance   : 0.1234
Section    : HR Policies
Document   : test_document.pdf
Content Type: text
Content:
Unused PTO days carry-over policy states that employees may roll over...

----------------------------------------------------------------------
Rank       : 2
Database ID: 15
Distance   : 0.2156
...
```

✅ **Retrieval pipeline is working correctly**

---

## STEP 5: Test Full RAG Pipeline (Question Answering)

This tests the complete end-to-end pipeline: retrieval → context building → LLM generation

### Command:
```bash
cd d:\NIC\rag-project

d:/NIC/rag-project/.venv/Scripts/python.exe -m app.test_rag_service
```

### What It Tests:
1. Asks 5 test questions (3 relevant, 2 irrelevant)
2. For each question:
   - Retrieves relevant chunks
   - Builds context
   - Generates LLM response
   - Validates answer quality

**Expected Output Example:**
```
======================================================================
END-TO-END RAG SERVICE TEST
======================================================================

======================================================================
TEST 1: RELEVANT
======================================================================
Question:
How many unused PTO days can an employee roll over?

----------------------------------------------------------------------
ANSWER
----------------------------------------------------------------------
According to the company PTO policy, employees may roll over up to 5
unused days per year. This carries forward to the next fiscal year.

======================================================================
TEST 2: RELEVANT
======================================================================
...

======================================================================
TEST 5: IRRELEVANT
======================================================================
Question:
What is the capital of France?

----------------------------------------------------------------------
ANSWER
----------------------------------------------------------------------
I could not find the answer in the provided documents.
```

✅ **Full RAG pipeline is working correctly**

---

## STEP 6: Run RAG Evaluation Framework (Optional)

This runs comprehensive metrics on all test cases

### Command:
```bash
cd d:\NIC\rag-project

d:/NIC/rag-project/.venv/Scripts/python.exe -m app.test_evaluation_framework
```

### What It Tests:
- 12 test cases across 4 categories
- Metrics: accuracy, precision, recall, F1-score
- Category breakdown: DIRECT, SEMANTIC, IRRELEVANT, UNSUPPORTED

**Expected Output:**
```
======================================================================
EVALUATION SUMMARY
======================================================================
Total Tests        : 12
Passed             : 11
Failed             : 1
Pass Rate          : 91.67%

RETRIEVAL / RELEVANCE
----------------------------------------------------------------------
Relevant Accuracy  : 100.00% (5/5)
Rejection Accuracy : 83.33% (5/6)

BY CATEGORY
----------------------------------------------------------------------
DIRECT RELEVANT       4/4 (100.00%)
SEMANTIC RELEVANT     3/4 (75.00%)
IRRELEVANT            3/3 (100.00%)
UNSUPPORTED           1/1 (100.00%)
```

✅ **System evaluation complete**

---

## STEP 7: Test Hard Cases (Optional)

Tests 4 buried-detail cases that require precise retrieval

### Command:
```bash
cd d:\NIC\rag-project

d:/NIC/rag-project/.venv/Scripts/python.exe -m app.test_rag_hard_cases
```

### Test Cases:
1. **T013**: PTO rollover exception (deadline + approver)
2. **T014**: Tier 2 international travel (flight class + budget)
3. **T015**: Highest-growth region (region + revenue, from table)
4. **T016**: Router thermal threshold

---

## Complete Test Execution Checklist

```
[  ] STEP 1: Prerequisites verified
       - Python 3.12 environment ready
       - PostgreSQL/pgvector running
       - 2 PDFs in data/raw/

[  ] STEP 2: Database cleaned
       $ d:/NIC/rag-project/.venv/Scripts/python.exe -m app.clean_database
       Expected: [OK] Database is ready for fresh insertion!

[  ] STEP 3: PDFs ingested
       $ d:/NIC/rag-project/.venv/Scripts/python.exe -m app.test_chunk_vector_insert
       Expected: [OK] SUCCESS - All chunks embedded and stored

[  ] STEP 4: Retrieval tested
       $ d:/NIC/rag-project/.venv/Scripts/python.exe -m app.test_retrieval_service
       Expected: All queries return relevant results

[  ] STEP 5: RAG pipeline tested
       $ d:/NIC/rag-project/.venv/Scripts/python.exe -m app.test_rag_service
       Expected: Relevant questions answered, irrelevant rejected

[  ] STEP 6: Evaluation framework (optional)
       $ d:/NIC/rag-project/.venv/Scripts/python.exe -m app.test_evaluation_framework
       Expected: Pass rate >= 90%

[  ] STEP 7: Hard cases (optional)
       $ d:/NIC/rag-project/.venv/Scripts/python.exe -m app.test_rag_hard_cases
       Expected: 3/4 or 4/4 tests pass
```

---

## Troubleshooting

### Issue: "Compiler: cl is not found"
**Solution**: Already fixed in `test_chunk_vector_insert.py`
- We disabled torch compilation with `torch.compile = lambda x: x`

### Issue: "UnicodeEncodeError" on Windows
**Solution**: Already fixed in all files
- Replaced Unicode characters (✅, ❌, etc.) with ASCII ([OK], [ERROR], etc.)

### Issue: "Database connection refused"
**Solution**: Ensure PostgreSQL is running
```bash
docker-compose up -d
docker ps | findstr postgres
```

### Issue: "Module not found"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

---

## Services Involved

| Service | Purpose | Model |
|---------|---------|-------|
| **ChunkingService** | Splits documents into chunks | RecursiveCharacterTextSplitter |
| **EmbeddingService** | Generates vector embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| **VectorStore** | Stores + searches embeddings | PGvector (PostgreSQL) |
| **RetrievalService** | Retrieves relevant chunks | Vector similarity + BGE-Reranker |
| **ContextBuilder** | Filters + aggregates context | Custom filtering |
| **RAGPromptBuilder** | Constructs LLM prompt | Template-based |
| **LLMService** | Generates answers | Llama 2 (via Ollama) |
| **RAGEvaluator** | Evaluates system performance | Metrics framework |

---

## Expected Final Status

✅ **Complete Pipeline Success Indicators:**
1. Database cleaned successfully
2. 52+ chunks ingested from both PDFs
3. All chunks embedded (384-dim)
4. Retrieval returns relevant results (top chunks match query intent)
5. RAG answers relevant questions correctly
6. RAG rejects irrelevant questions
7. Evaluation pass rate >= 85%
8. Hard cases solved >= 75%

---

## Next Steps After Testing
- Adjust chunk size/overlap if retrieval quality is poor
- Fine-tune reranker threshold if getting too many false positives
- Test with custom questions specific to your documents
- Monitor LLM response quality and adjust prompt templates

from pathlib import Path
import os
import sys

# ===================================================================
# DISABLE TORCH COMPILATION TO AVOID C++ COMPILER REQUIREMENT
# ===================================================================
# PyTorch tries to use torch.compile which requires a C++ compiler (cl.exe).
# We disable it to avoid compilation errors on systems without MSVC installed.
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHINDUCTOR_SKIP_AUTOGRAD_FALLBACK"] = "1"
os.environ["TORCH_INDUCTOR_DISABLE_TRITON"] = "1"

import torch
# Disable dynamo compilation - this should prevent torch from trying to compile
torch._dynamo.config.suppress_errors = True
# More aggressive: override the compile function to be a no-op
torch.compile = lambda x: x

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from app.services.chunking_service import ChunkingService
from app.services.embedding import EmbeddingService
from app.services.vector_store import (
    create_documents_table,
    insert_document,
)


# ===================================================================
# CONFIGURATION
# ===================================================================

# Process both PDF files:
# 1. test_document.pdf - simple text-based PDF
# 2. hybrid_test_document.pdf - scanned/hybrid PDF with OCR + tables
#
PDF_PATHS = [
    Path("data/raw/test_document.pdf"),
    Path("data/raw/hybrid_test_document.pdf"),
]


# ===================================================================
# START
# ===================================================================

print("=" * 70)
print("CHUNK -> EMBEDDING -> VECTOR STORE TEST")
print("=" * 70)

print(f"\nTotal PDF files: {len(PDF_PATHS)}")


# ===================================================================
# STEP 0: VALIDATE PDF FILES
# ===================================================================

print("\nSTEP 0: VALIDATING PDF FILES")
print("-" * 70)

for pdf_path in PDF_PATHS:

    print(f"File: {pdf_path}")
    print(f"Exists: {pdf_path.exists()}")

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"File not found: {pdf_path}"
        )


# ===================================================================
# STEP 1: CONFIGURE DOCLING
# ===================================================================

print("\nSTEP 1: CONFIGURING DOCLING")
print("-" * 70)

print("Using lightweight PDF processing configuration.")

pipeline_options = PdfPipelineOptions(

    # ---------------------------------------------------------------
    # Use the PDF's embedded/native text instead of relying on the
    # layout model for text detection.
    # ---------------------------------------------------------------

    force_backend_text=True,

    # ---------------------------------------------------------------
    # Disable OCR for this basic text-PDF test.
    # ---------------------------------------------------------------

    do_ocr=False,

    # ---------------------------------------------------------------
    # Disable table extraction for this first test.
    # ---------------------------------------------------------------

    do_table_structure=False,

    # ---------------------------------------------------------------
    # Disable layout detection to avoid torch compilation errors
    # on systems without C++ compiler.
    # ---------------------------------------------------------------

    do_layout=False,
)


converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

print("Docling converter initialized.")


# ===================================================================
# STEP 2: INITIALIZE CHUNKING SERVICE
# ===================================================================

print("\nSTEP 2: INITIALIZING CHUNKING SERVICE")
print("-" * 70)

chunking_service = ChunkingService(
    chunk_size=800,
    chunk_overlap=100,
)

print("Chunking service initialized.")


# ===================================================================
# STEP 3: INITIALIZE EMBEDDING SERVICE
# ===================================================================

print("\nSTEP 3: INITIALIZING EMBEDDING SERVICE")
print("-" * 70)

embedding_service = EmbeddingService()

print("Embedding service initialized.")


# ===================================================================
# STEP 4: INITIALIZE VECTOR DATABASE
# ===================================================================

print("\nSTEP 4: INITIALIZING VECTOR DATABASE")
print("-" * 70)

create_documents_table()

print("Vector database ready.")


# ===================================================================
# GLOBAL STATISTICS
# ===================================================================

total_chunks = 0
total_successful = 0
total_failed = 0

all_inserted_ids = []


# ===================================================================
# PROCESS EACH PDF
# ===================================================================

for pdf_index, pdf_path in enumerate(PDF_PATHS, start=1):

    print("\n")
    print("=" * 70)
    print(
        f"PROCESSING PDF {pdf_index}/{len(PDF_PATHS)}: "
        f"{pdf_path.name}"
    )
    print("=" * 70)


    # ===============================================================
    # DOCUMENT CONVERSION
    # ===============================================================

    print("\nDOCUMENT CONVERSION")
    print("-" * 70)

    print(f"File: {pdf_path}")

    try:

        print("Converting document...")

        result = converter.convert(pdf_path)

        document = result.document

        print("[OK] Document processed successfully.")

    except Exception as e:

        print("\n[ERROR] DOCUMENT CONVERSION FAILED")
        print("-" * 70)
        print(f"Error: {e}")

        total_failed += 1

        continue


    # ===============================================================
    # STRUCTURE-AWARE CHUNKING
    # ===============================================================

    print("\nSTRUCTURE-AWARE CHUNKING")
    print("-" * 70)

    try:

        chunks = chunking_service.chunk_docling_document(
            document=document,
            document_name=pdf_path.stem,
        )

        print(
            f"[OK] Total chunks generated: {len(chunks)}"
        )

    except Exception as e:

        print("\n[ERROR] CHUNKING FAILED")
        print("-" * 70)
        print(f"Error: {e}")

        total_failed += 1

        continue


    total_chunks += len(chunks)


    # ===============================================================
    # SHOW CHUNK PREVIEW
    # ===============================================================

    print("\nCHUNK PREVIEW")
    print("-" * 70)

    preview_count = min(3, len(chunks))

    for i in range(preview_count):

        chunk = chunks[i]

        print(
            f"\nChunk {chunk.chunk_id}"
        )

        print(
            f"Content type: "
            f"{'table' if chunk.content.lstrip().startswith('|') else 'text'}"
        )

        print(
            f"Content preview: "
            f"{chunk.content[:300].replace(chr(10), ' ')}"
        )


    # ===============================================================
    # EMBEDDING + INSERTING
    # ===============================================================

    print("\nEMBEDDING + INSERTING CHUNKS")
    print("-" * 70)

    successful = 0
    failed = 0
    inserted_ids = []


    for chunk in chunks:

        try:

            # -------------------------------------------------------
            # Generate embedding
            # -------------------------------------------------------

            embedding = embedding_service.embed(
                chunk.content
            )


            # -------------------------------------------------------
            # Determine content type
            # -------------------------------------------------------

            content_type = (
                "table"
                if chunk.content.lstrip().startswith("|")
                else "text"
            )


            # -------------------------------------------------------
            # Prepare metadata
            # -------------------------------------------------------

            metadata = {
                **chunk.metadata,

                "content_type": content_type,

                "source_file": pdf_path.name,
            }


            # -------------------------------------------------------
            # Insert into vector database
            # -------------------------------------------------------

            document_id = insert_document(
                content=chunk.content,
                embedding=embedding,
                metadata=metadata,
            )


            inserted_ids.append(document_id)

            all_inserted_ids.append(document_id)

            successful += 1

            total_successful += 1


            print(
                f"Chunk {chunk.chunk_id:02d} "
                f"→ {content_type:<5} "
                f"→ embedding {len(embedding)} dimensions "
                f"→ DB ID {document_id}"
            )


        except Exception as e:

            failed += 1

            total_failed += 1

            print(
                f"[ERROR] Chunk {chunk.chunk_id:02d} failed: {e}"
            )


    # ===============================================================
    # PDF VALIDATION
    # ===============================================================

    print("\nPDF VALIDATION")
    print("-" * 70)

    print(
        f"PDF                : {pdf_path.name}"
    )

    print(
        f"Total chunks       : {len(chunks)}"
    )

    print(
        f"Successfully added : {successful}"
    )

    print(
        f"Failed             : {failed}"
    )

    print(
        f"Inserted IDs       : {inserted_ids}"
    )


    if successful == len(chunks) and failed == 0:

        print(
            "\n[OK] All chunks from this PDF "
            "were embedded and stored successfully."
        )

    else:

        print(
            "\n[WARN] Some chunks from this PDF "
            "failed to store."
        )


# ===================================================================
# FINAL VALIDATION
# ===================================================================

print("\n")

print("=" * 70)
print("FINAL VECTOR STORE VALIDATION")
print("=" * 70)

print(
    f"PDF files processed : {len(PDF_PATHS)}"
)

print(
    f"Total chunks        : {total_chunks}"
)

print(
    f"Successfully added  : {total_successful}"
)

print(
    f"Failed              : {total_failed}"
)

print(
    f"Inserted IDs        : {all_inserted_ids}"
)


# ===================================================================
# FINAL RESULT
# ===================================================================

if (
    total_chunks > 0
    and total_successful == total_chunks
    and total_failed == 0
):

    print("\n" + "=" * 70)
    print("[OK] SUCCESS")
    print("=" * 70)

    print(
        "All chunks were successfully "
        "embedded and stored in the vector database."
    )

else:

    print("\n" + "=" * 70)
    print("[WARN] COMPLETED WITH ERRORS")
    print("=" * 70)

    print(
        "The pipeline completed, but some "
        "documents or chunks failed."
    )


# ===================================================================
# COMPLETE
# ===================================================================

print("\n" + "=" * 70)
print("CHUNK → EMBEDDING → VECTOR STORE TEST COMPLETE")
print("=" * 70)

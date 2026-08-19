"""
Diagnostic script -- run this locally against the Cisco IAD2801 PDF to
confirm/refute whether Table A-1 / Table A-2 are being split across the
page break into separate, context-less chunks.

Usage:
    python debug_table_chunks.py path/to/07_hw.pdf

Prints every chunk Docling+ChunkingService produced, filtered to just
table chunks and any chunk mentioning "SYS PWR", "Amber", or "LED" --
so you can see exactly what content (and what section metadata) each
piece of Table A-1 / A-2 actually landed in.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.document_converter import HybridDocumentConverter
from app.services.chunking_service import ChunkingService


def main(pdf_path: str):
    converter = HybridDocumentConverter()
    chunker = ChunkingService(chunk_size=800, chunk_overlap=100)

    result = converter.convert(Path(pdf_path))
    print(f"Pages needing OCR: {result.ocr_page_numbers or 'none'}\n")

    chunks = chunker.chunk_docling_document(
        result.docling_document,
        document_name=Path(pdf_path).name,
    )

    print(f"Total chunks: {len(chunks)}\n")
    print("=" * 80)

    keywords = ("SYS PWR", "Amber", "amber", "LED", "FE port", "ACT", "FDX")

    for chunk in chunks:
        is_table = chunk.metadata.get("content_type") == "table"
        has_keyword = any(k in chunk.content for k in keywords)

        if is_table or has_keyword:
            print(f"[chunk_id={chunk.chunk_id}] "
                  f"type={chunk.metadata.get('content_type')} "
                  f"section={chunk.metadata.get('section')!r} "
                  f"page={chunk.metadata.get('page_number')}")
            print("-" * 80)
            print(chunk.content)
            print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_table_chunks.py path/to/07_hw.pdf")
        sys.exit(1)

    main(sys.argv[1])

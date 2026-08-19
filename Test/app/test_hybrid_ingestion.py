"""
Sanity check for the hybrid (text-native / OCR-needed) ingestion path.

test_document.pdf is a fully text-native PDF, so the expected result
here is boring: every page classified as text-native, zero pages
routed to OCR, and chunk output unchanged from the existing
chunk_docling_document behavior (now just carrying page_number too).

This is the regression check that matters before testing against an
actual scanned/hybrid PDF tomorrow: if this breaks, the OCR routing
work has nothing solid to build on.

Run with:
    python -m app.test_hybrid_ingestion
"""

from pathlib import Path

from app.services.document_converter import HybridDocumentConverter
from app.services.chunking_service import ChunkingService

PDF_PATH = Path("data/raw/test_document.pdf")


def main():
    print("=" * 70)
    print("HYBRID INGESTION TEST")
    print("=" * 70)

    converter = HybridDocumentConverter()

    print(f"\nClassifying pages: {PDF_PATH}")
    result = converter.convert(PDF_PATH)

    print(f"\nTotal pages       : {len(result.page_classifications)}")
    print(f"Pages needing OCR : {sorted(result.ocr_page_numbers) or 'none'}")

    for page in result.page_classifications:
        flag = "NEEDS_OCR" if page.needs_ocr else "text-native"
        print(
            f"  page {page.page_number:2d} | "
            f"chars={page.char_count:5d} | "
            f"words={page.word_count:4d} | {flag}"
        )

    print("\nChunking (with page-number tracking)...")
    chunker = ChunkingService(chunk_size=800, chunk_overlap=100)

    chunks = chunker.chunk_docling_document(
        result.docling_document,
        document_id="test-doc-hybrid",
        document_name="test_document.pdf",
    )

    chunks = chunker.apply_ocr_routing(
        chunks,
        ocr_page_numbers=result.ocr_page_numbers,
        document_id="test-doc-hybrid",
        document_name="test_document.pdf",
    )

    print(f"Total chunks: {len(chunks)}")

    missing_page_number = [c for c in chunks if c.metadata.get("page_number") is None]
    ocr_pending = [c for c in chunks if c.metadata.get("content_type") == "ocr_pending"]

    print(f"Chunks missing page_number : {len(missing_page_number)}")
    print(f"OCR-pending placeholder chunks : {len(ocr_pending)}")

    print("\n" + "=" * 70)
    if result.has_scanned_pages:
        print(f"RESULT: {len(result.ocr_page_numbers)} page(s) flagged for OCR "
              f"(expected once OCR is wired up tomorrow; not implemented yet).")
    elif ocr_pending or missing_page_number:
        print("RESULT: UNEXPECTED — this doc should be fully text-native. "
              "Check page classification thresholds or provenance extraction.")
    else:
        print("RESULT: PASS — fully text-native doc, all pages classified "
              "correctly, page numbers attached, nothing routed to OCR.")
    print("=" * 70)


if __name__ == "__main__":
    main()

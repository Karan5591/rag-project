"""
Hybrid PDF ingestion — per-page text-native vs. OCR-needed routing.

Design (per discussion, OCR implementation to follow separately):

    1. Classify EACH page independently: does it have a usable native
       text layer, or is it image-based / has too little extractable
       text to be reliable?
    2. Run Docling on the whole document with do_ocr=False — this is
       fast and gives us clean structured extraction for every
       text-native page. (do_ocr=False also means Docling won't waste
       time running its own OCR pass on the image pages — we're
       routing those ourselves.)
    3. For pages classified as needing OCR, chunking_service.
       apply_ocr_routing() drops Docling's unreliable output for those
       pages and inserts a clearly-flagged placeholder chunk instead.
    4. THIS FILE DOES NOT IMPLEMENT OCR ITSELF. `_ocr_page()` below is
       the explicit integration point — replace its body with the real
       OCR call when that's ready, and route its output back into
       chunking (see the docstring on apply_ocr_routing in
       chunking_service.py for how the pieces fit together).

This handles hybrid PDFs where some pages are native text and others
are scanned, without waiting on a slow full-document OCR pass for
pages that don't need it.

Requires: pip install pypdf  (docling is already a dependency)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption


@dataclass
class PageClassification:
    page_number: int          # 1-indexed, matches Docling's item.prov[].page_no
    char_count: int            # non-whitespace characters extracted natively
    word_count: int
    needs_ocr: bool


@dataclass
class ConversionResult:
    docling_document: object   # the Docling DoclingDocument, for chunk_docling_document()
    page_classifications: List[PageClassification]

    @property
    def ocr_page_numbers(self) -> set:
        return {
            p.page_number for p in self.page_classifications if p.needs_ocr
        }

    @property
    def has_scanned_pages(self) -> bool:
        return len(self.ocr_page_numbers) > 0


@dataclass
class CoverageReport:
    raw_chars: int
    chunked_chars: int
    coverage_ratio: float
    low_coverage: bool


def check_chunking_coverage(
    pdf_path: Path,
    chunks: list,
    min_coverage_ratio: float = 0.70,
) -> CoverageReport:
    """
    Document-agnostic completeness check, not a per-document rule.

    Compares total characters actually chunked against a raw pypdf
    extraction of the same file. If chunking captured meaningfully
    less than what pypdf can read directly, something was silently
    dropped along the way (e.g. Docling labeling content as
    page_header/page_footer and excluding it from the body tree —
    the exact issue found in Karan_Singh_Appointment_Letter — or any
    other future extraction gap). Runs identically for every PDF;
    nothing here is tuned to a specific document's content.

    A ratio below 1.0 is normal and expected — boilerplate line
    breaks, whitespace normalization, and table reformatting all
    shift character counts somewhat. min_coverage_ratio=0.70 is a
    starting point flagging genuinely large gaps, not exact parity;
    tune it against real low/false-positive cases as they show up.
    """

    reader = PdfReader(str(pdf_path))

    raw_chars = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        raw_chars += len(text.replace(" ", "").replace("\n", ""))

    chunked_chars = sum(
        len(chunk.content.replace(" ", "").replace("\n", ""))
        for chunk in chunks
    )

    coverage_ratio = (
        chunked_chars / raw_chars if raw_chars > 0 else 1.0
    )

    return CoverageReport(
        raw_chars=raw_chars,
        chunked_chars=chunked_chars,
        coverage_ratio=coverage_ratio,
        low_coverage=coverage_ratio < min_coverage_ratio,
    )


class HybridDocumentConverter:
    """
    Classifies pages, then runs Docling (text-layer only) once.

    min_chars_per_page / min_words_per_page are the "does this page have
    a usable native text layer" thresholds. Defaults are deliberately
    conservative (a page with only a header/footer/watermark should
    still be routed to OCR) — tune against real scanned samples once
    OCR is wired up.
    """

    def __init__(
        self,
        min_chars_per_page: int = 40,
        min_words_per_page: int = 8,
    ):
        self.min_chars_per_page = min_chars_per_page
        self.min_words_per_page = min_words_per_page

        # do_ocr=False: we handle OCR routing ourselves (step 3/4 above)
        # rather than letting Docling silently run its own OCR pass.
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def classify_pages(self, pdf_path: Path) -> List[PageClassification]:
        """
        Fast, Docling-independent pass: use pypdf to check each page's
        native text layer. This runs BEFORE the (heavier) Docling
        conversion so we know up front which pages will need OCR.
        """

        reader = PdfReader(str(pdf_path))
        classifications = []

        for index, page in enumerate(reader.pages):
            page_number = index + 1

            raw_text = page.extract_text() or ""
            cleaned = raw_text.strip()

            char_count = len(cleaned.replace(" ", "").replace("\n", ""))
            word_count = len(cleaned.split())

            needs_ocr = (
                char_count < self.min_chars_per_page
                or word_count < self.min_words_per_page
            )

            classifications.append(
                PageClassification(
                    page_number=page_number,
                    char_count=char_count,
                    word_count=word_count,
                    needs_ocr=needs_ocr,
                )
            )

        return classifications

    def convert(self, pdf_path: Path) -> ConversionResult:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        page_classifications = self.classify_pages(pdf_path)

        result = self._converter.convert(pdf_path)
        docling_document = result.document

        return ConversionResult(
            docling_document=docling_document,
            page_classifications=page_classifications,
        )

    def _ocr_page(self, pdf_path: Path, page_number: int) -> str:
        """
        INTEGRATION POINT — not yet implemented.

        Replace this with the real OCR call (e.g. Tesseract/EasyOCR/
        Docling's own do_ocr=True path scoped to a single page) and
        return the extracted text for `page_number`.

        Once implemented, the caller should:
          1. Call this for each page in ConversionResult.ocr_page_numbers
          2. Build DocumentChunk objects from the returned text
             (content_type="text", page_number=page_number)
          3. Use those in place of the "[OCR_PENDING...]" placeholder
             chunks apply_ocr_routing() currently inserts.
        """

        raise NotImplementedError(
            f"OCR not yet implemented for page {page_number} of {pdf_path}. "
            f"See HybridDocumentConverter._ocr_page docstring."
        )

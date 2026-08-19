import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DocumentChunk:
    chunk_id: int
    content: str
    metadata: dict


class ChunkingService:
    """
    Structure-aware chunking service for Docling documents.

    Document-agnostic:
    - Works with different document types
    - Preserves section/subsection context
    - Handles normal text and lists
    - Keeps tables as independent chunks
    - Produces metadata suitable for embeddings/vector storage
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _normalize_contract_text(text: str) -> str:
        """Clean contract-style text extracted from PDFs."""
        if not text:
            return ""

        text = text.replace("\r", "\n")
        text = text.replace("www.netbooks.i n", "www.netbooks.in")
        text = text.replace("www.netbooks. i n", "www.netbooks.in")
        text = re.sub(r"(?<=\w)\s+(?=\.)", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,;:.])", r"\1", text)
        text = re.sub(r"(?<=\d)\s+(?=th|st|nd|rd)\b", "", text, flags=re.I)
        return text.strip()

    @staticmethod
    def _detect_contract_field(text: str) -> Optional[str]:
        """Add field hints for common HR/legal values."""
        lowered = text.lower()

        if re.search(r"\b(date of joining|joining date|effective .* join)", lowered):
            return "joining_date"
        if re.search(r"\b(remuneration|salary|monthly pay|per month)\b", lowered):
            return "remuneration"
        if re.search(r"\b(working hours|working hour|leaves|leave policy)\b", lowered):
            return "working_hours_leave_policy"
        if re.search(r"\b(website|email|contact|cin|company identification)\b", lowered):
            return "contact_info"
        if re.search(r"\b(employee code|employee id|candidate acceptance)\b", lowered):
            return "employee_identity"

        return None

    def _get_item_text(self, item) -> str:
        """Safely extract text from a Docling document item."""

        text = getattr(item, "text", None)

        if text:
            return text.strip()

        return ""

    def _get_item_type(self, item) -> str:
        """Return a readable Docling item type."""

        return type(item).__name__

    def _get_item_page(self, item) -> Optional[int]:
        """
        Best-effort extraction of the source page number from a Docling
        item's provenance data. Returns None if unavailable rather than
        raising — page tracking is additive metadata, not a hard
        requirement for chunking to work.
        """

        prov = getattr(item, "prov", None)

        if prov:
            first = prov[0]
            page_no = getattr(first, "page_no", None)

            if page_no is not None:
                return page_no

        return None

    def _create_metadata(
        self,
        document_id: Optional[str],
        document_name: Optional[str],
        section: Optional[str],
        subsection: Optional[str],
        content_type: str,
        page_number: Optional[int] = None,
    ) -> dict:
        """
        Create consistent metadata for every chunk.
        """

        return {
            "document_id": document_id,
            "document_name": document_name,
            "section": section,
            "subsection": subsection,
            "content_type": content_type,
            "page_number": page_number,
        }

    def chunk_docling_document(
        self,
        document,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Convert a DoclingDocument into structure-aware chunks.

        Handles:
        - Section headers
        - Subsections
        - Text
        - Lists
        - Tables

        Tables are kept as independent chunks because
        splitting a table across normal text chunks can
        damage its semantic meaning.
        """

        chunks = []

        current_section = None
        current_subsection = None

        buffer = []
        buffer_size = 0
        buffer_page = None

        chunk_id = 0

        def create_chunk(
            content: str,
            content_type: str = "text",
            page_number: Optional[int] = None,
        ):
            nonlocal chunk_id

            content = self._normalize_contract_text(content)

            if not content:
                return

            metadata = self._create_metadata(
                document_id=document_id,
                document_name=document_name,
                section=current_section,
                subsection=current_subsection,
                content_type=content_type,
                page_number=page_number,
            )
            metadata["field_name"] = self._detect_contract_field(content)

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata,
                )
            )

            chunk_id += 1

        def flush_buffer():
            nonlocal buffer
            nonlocal buffer_size
            nonlocal buffer_page

            if not buffer:
                return

            content = "\n\n".join(buffer)

            create_chunk(
                content=content,
                content_type="text",
                page_number=buffer_page,
            )

            buffer = []
            buffer_size = 0
            buffer_page = None

        # -------------------------------------------------
        # Iterate through Docling document structure
        # -------------------------------------------------

        for item, level in document.iterate_items():

            item_type = self._get_item_type(item)

            # -------------------------------------------------
            # Section headers
            # -------------------------------------------------

            if item_type == "SectionHeaderItem":

                flush_buffer()

                text = self._get_item_text(item)

                if not text:
                    continue

                if level == 1:
                    current_section = text
                    current_subsection = None

                else:
                    current_subsection = text

                continue

            # -------------------------------------------------
            # Tables
            # -------------------------------------------------

            if item_type == "TableItem":

                # Never mix a table with surrounding text.
                flush_buffer()

                table_text = self._extract_table_text(item, document)

                if table_text:
                    create_chunk(
                        content=table_text,
                        content_type="table",
                        page_number=self._get_item_page(item),
                    )

                continue

            # -------------------------------------------------
            # Normal text / lists
            # -------------------------------------------------

            text = self._get_item_text(item)

            if not text:
                continue

            # If adding this item would exceed chunk size,
            # create the current chunk first.
            if buffer_size + len(text) > self.chunk_size:

                flush_buffer()

            if buffer_page is None:
                buffer_page = self._get_item_page(item)

            buffer.append(text)
            buffer_size += len(text)

        # -------------------------------------------------
        # Flush remaining content
        # -------------------------------------------------

        flush_buffer()

        # -------------------------------------------------
        # Running headers/footers (page_header / page_footer)
        #
        # Docling's layout model correctly extracts letterhead content
        # (contact info, registration/CIN lines, "Regd. Office"
        # footers, repeated branding, etc.) into document.texts — but
        # tags it page_header/page_footer, and iterate_items() above
        # only walks the body tree, so this content is silently
        # dropped for ANY document that has a running header/footer,
        # not just one PDF. Pull it back in explicitly here, deduped
        # (the same line repeats once per page) into a single chunk.
        # -------------------------------------------------

        header_footer_lines = self._collect_header_footer_lines(document)

        if header_footer_lines:
            current_section = "Document Header/Footer"
            current_subsection = None

            create_chunk(
                content="\n".join(header_footer_lines),
                content_type="header_footer",
                page_number=None,
            )

        # -------------------------------------------------
        # Apply overlap between consecutive TEXT chunks only.
        #
        # IMPORTANT:
        # Overlap is allowed only when consecutive text chunks
        # belong to the SAME section. This prevents the tail of
        # one section from being injected into the beginning of
        # the next section.
        #
        # Table chunks stay atomic.
        # -------------------------------------------------

        chunks = self._apply_overlap(chunks)

        return chunks

    def _collect_header_footer_lines(self, document) -> List[str]:
        """
        Pull every page_header/page_footer-labeled text item out of
        document.texts (Docling's full internal text registry —
        includes items iterate_items() excludes because they sit
        outside the body tree), deduping repeated lines that recur
        once per page (e.g. the same letterhead on every page of a
        multi-page document). General-purpose: applies to any
        document with a running header/footer.
        """

        seen = set()
        lines = []

        for item in getattr(document, "texts", []) or []:

            label = str(getattr(item, "label", ""))

            if label not in ("page_header", "page_footer"):
                continue

            text = self._normalize_contract_text(self._get_item_text(item))

            if not text or text in seen:
                continue

            seen.add(text)
            lines.append(text)

        return lines

    def _apply_overlap(
        self,
        chunks: List["DocumentChunk"],
    ) -> List["DocumentChunk"]:
        """
        Apply overlap only between adjacent text chunks that belong
        to the same section.

        This prevents content from one section being copied into
        the beginning of the next section.

        Example:

            Section A
                Chunk 1
                Chunk 2  <-- can receive overlap from Chunk 1

            Section B
                Chunk 3  <-- MUST NOT receive overlap from Chunk 2
        """

        if self.chunk_overlap <= 0 or len(chunks) < 2:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):

            content = chunk.content

            if i > 0:

                previous_chunk = chunks[i - 1]

                # Both chunks must be normal text chunks.
                is_text_to_text = (
                    chunk.metadata.get("content_type") == "text"
                    and previous_chunk.metadata.get("content_type") == "text"
                )

                # Overlap must stay inside the same logical section.
                same_section = (
                    chunk.metadata.get("section")
                    == previous_chunk.metadata.get("section")
                )

                if is_text_to_text and same_section:

                    overlap_text = previous_chunk.content[
                        -self.chunk_overlap:
                    ]

                    content = (
                        overlap_text
                        + "\n\n"
                        + content
                    )

            overlapped_chunks.append(
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    content=content,
                    metadata=chunk.metadata.copy(),
                )
            )

        return overlapped_chunks

    def apply_ocr_routing(
        self,
        chunks: List[DocumentChunk],
        ocr_page_numbers: set,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Reconcile Docling's structured output with page-level OCR routing.

        For pages classified as needing OCR (see document_converter.py),
        Docling's own extraction is unreliable — it was run with
        do_ocr=False, so an image-only page produces empty or near-empty
        items. Rather than let those thin/empty chunks silently pollute
        retrieval, this:

        1. Drops any chunk whose page_number is in ocr_page_numbers.
        2. Inserts one clearly-flagged placeholder chunk per such page,
           so the gap is traceable instead of silently missing, and so
           there's a single obvious place to slot in real OCR output
           once it's implemented (search "OCR_PENDING").

        Call this AFTER chunk_docling_document(). Once OCR is wired up,
        replace the placeholder-insertion loop below with real OCR text
        run through the same create_chunk-style logic.
        """

        if not ocr_page_numbers:
            return chunks

        filtered = [
            chunk for chunk in chunks
            if chunk.metadata.get("page_number") not in ocr_page_numbers
        ]

        next_id = max((c.chunk_id for c in filtered), default=-1) + 1

        for page_number in sorted(ocr_page_numbers):
            metadata = self._create_metadata(
                document_id=document_id,
                document_name=document_name,
                section=None,
                subsection=None,
                content_type="ocr_pending",
                page_number=page_number,
            )

            filtered.append(
                DocumentChunk(
                    chunk_id=next_id,
                    content=(
                        f"[OCR_PENDING: page {page_number} has "
                        f"insufficient extractable text and requires "
                        f"OCR processing, not yet implemented.]"
                    ),
                    metadata=metadata,
                )
            )

            next_id += 1

        return filtered

    def _extract_table_text(self, table, document) -> str:
        """
        Convert a Docling table into embedding-friendly text.

        Preferred:
            Docling -> pandas DataFrame -> Markdown

        Fallback:
            Docling table text representation
        """

        # ---------------------------------------------
        # Method 1: DataFrame -> Markdown
        # ---------------------------------------------

        try:
            dataframe = table.export_to_dataframe(doc=document)

            if dataframe is not None:

                # Remove completely empty rows/columns.
                dataframe = dataframe.dropna(
                    axis=0,
                    how="all",
                ).dropna(
                    axis=1,
                    how="all",
                )

                if not dataframe.empty:
                    return dataframe.to_markdown(
                        index=False
                    )

        except Exception:
            pass

        # ---------------------------------------------
        # Method 2: Direct text fallback
        # ---------------------------------------------

        text = getattr(table, "text", None)

        if text:
            return text.strip()

        # ---------------------------------------------
        # Method 3: Generic representation
        # ---------------------------------------------

        try:
            return str(table).strip()
        except Exception:
            return "[Table content could not be extracted]"

    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> List[DocumentChunk]:
        """
        Boundary-aware text chunking fallback.

        Attempts to preserve:
        1. Paragraph boundaries
        2. Sentence boundaries
        3. Word boundaries

        Hard character splitting is used only when a single
        unit is larger than chunk_size.
        """

        if not text or not text.strip():
            return []

        metadata = metadata or {}

        # Normalize excessive whitespace while preserving paragraphs.
        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks = []
        chunk_id = 0

        current_parts = []
        current_length = 0

        def add_chunk(content: str):
            nonlocal chunk_id

            content = content.strip()

            if not content:
                return

            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = chunk_id
            chunk_metadata["content_type"] = "text"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=chunk_metadata,
                )
            )

            chunk_id += 1

        def flush_current():
            nonlocal current_parts
            nonlocal current_length

            if not current_parts:
                return

            content = "\n\n".join(current_parts)
            add_chunk(content)

            current_parts = []
            current_length = 0

        for paragraph in paragraphs:

            # -------------------------------------------------
            # Paragraph fits into current chunk
            # -------------------------------------------------

            if current_length + len(paragraph) + 2 <= self.chunk_size:
                current_parts.append(paragraph)
                current_length += len(paragraph) + 2
                continue

            # -------------------------------------------------
            # Flush existing content first
            # -------------------------------------------------

            flush_current()

            # -------------------------------------------------
            # Paragraph itself fits into one chunk
            # -------------------------------------------------

            if len(paragraph) <= self.chunk_size:
                current_parts.append(paragraph)
                current_length = len(paragraph)
                continue

            # -------------------------------------------------
            # Paragraph is too large.
            # Split using sentence boundaries.
            # -------------------------------------------------

            sentences = [
                sentence.strip()
                for sentence in paragraph.replace("!", ".").replace("?", ".").split(".")
                if sentence.strip()
            ]

            sentence_buffer = []
            sentence_length = 0

            for sentence in sentences:

                sentence_with_space = len(sentence) + 1

                # Sentence fits in current sentence chunk.
                if (
                    sentence_length + sentence_with_space
                    <= self.chunk_size
                ):
                    sentence_buffer.append(sentence)
                    sentence_length += sentence_with_space
                    continue

                # Flush sentence buffer.
                if sentence_buffer:
                    add_chunk(" ".join(sentence_buffer))

                    sentence_buffer = []
                    sentence_length = 0

                # -------------------------------------------------
                # Individual sentence is too large.
                # Fall back to word boundaries.
                # -------------------------------------------------

                if len(sentence) > self.chunk_size:

                    words = sentence.split()

                    word_buffer = []
                    word_length = 0

                    for word in words:

                        word_with_space = len(word) + 1

                        if (
                            word_length + word_with_space
                            <= self.chunk_size
                        ):
                            word_buffer.append(word)
                            word_length += word_with_space
                        else:
                            if word_buffer:
                                add_chunk(" ".join(word_buffer))

                            word_buffer = [word]
                            word_length = len(word) + 1

                    if word_buffer:
                        add_chunk(" ".join(word_buffer))

                else:
                    sentence_buffer = [sentence]
                    sentence_length = len(sentence) + 1

            if sentence_buffer:
                add_chunk(" ".join(sentence_buffer))

        # Flush any remaining parts left in current_parts
        flush_current()

        # -------------------------------------------------
        # Add overlap between chunks
        # -------------------------------------------------

        if self.chunk_overlap > 0 and len(chunks) > 1:

            overlapped_chunks = []

            for i, chunk in enumerate(chunks):

                content = chunk.content

                if i > 0:

                    previous_content = chunks[i - 1].content

                    overlap_text = previous_content[
                        -self.chunk_overlap:
                    ]

                    content = (
                        overlap_text
                        + "\n\n"
                        + content
                    )

                overlapped_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk.chunk_id,
                        content=content,
                        metadata=chunk.metadata.copy(),
                    )
                )

            chunks = overlapped_chunks

        return chunks
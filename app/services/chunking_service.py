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

    Features:
    - Preserves section/subsection context
    - Handles normal text and lists
    - Keeps tables as independent chunks
    - Merges tables split across page boundaries
    - Reconstructs table row context for retrieval
    - Preserves page metadata
    - Recovers Docling page headers/footers
    - Supports OCR routing
    - Applies overlap only between compatible text chunks
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

    # ============================================================
    # TEXT NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_contract_text(text: str) -> str:
        """
        Clean extracted PDF text.

        This is intentionally conservative. It should improve
        extraction quality without changing the document meaning.
        """

        if not text:
            return ""

        text = text.replace("\r", "\n")

        # Known PDF extraction artifacts.
        text = text.replace(
            "www.netbooks.i n",
            "www.netbooks.in",
        )

        text = text.replace(
            "www.netbooks. i n",
            "www.netbooks.in",
        )

        # Remove spaces before punctuation.
        text = re.sub(
            r"(?<=\w)\s+(?=\.)",
            "",
            text,
        )

        # Collapse excessive whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        # Normalize spaces before punctuation.
        text = re.sub(
            r"\s+([,;:.])",
            r"\1",
            text,
        )

        # Normalize ordinal extraction.
        text = re.sub(
            r"(?<=\d)\s+(?=th|st|nd|rd)\b",
            "",
            text,
            flags=re.I,
        )

        return text.strip()

    @staticmethod
    def _detect_contract_field(text: str) -> Optional[str]:
        """
        Add field hints for common HR/legal values.
        """

        lowered = text.lower()
 
        if re.search(
            r"\b(date of joining|joining date|effective .* join)",
            lowered,
        ):
            return "joining_date"

        if re.search(
            r"\b(remuneration|salary|monthly pay|per month)\b",
            lowered,
        ):
            return "remuneration"

        if re.search(
            r"\b(working hours|working hour|leaves|leave policy)\b",
            lowered,
        ):
            return "working_hours_leave_policy"

        if re.search(
            r"\b(website|email|contact|cin|company identification)\b",
            lowered,
        ):
            return "contact_info"

        if re.search(
            r"\b(employee code|employee id|candidate acceptance)\b",
            lowered,
        ):
            return "employee_identity"

        return None

    # ============================================================
    # DOCLING HELPERS
    # ============================================================

    def _get_item_text(self, item) -> str:
        """
        Safely extract text from a Docling document item.
        """

        text = getattr(item, "text", None)

        if text:
            return str(text).strip()

        return ""

    def _get_item_type(self, item) -> str:
        """
        Return a readable Docling item type.
        """

        return type(item).__name__

    def _get_item_page(self, item) -> Optional[int]:
        """
        Best-effort extraction of source page number.
        """

        prov = getattr(item, "prov", None)

        if prov:
            first = prov[0]

            page_no = getattr(
                first,
                "page_no",
                None,
            )

            if page_no is not None:
                return page_no

        return None

    # ============================================================
    # METADATA
    # ============================================================

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

    # ============================================================
    # MAIN DOCLING CHUNKER
    # ============================================================

    def chunk_docling_document(
        self,
        document,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Convert a DoclingDocument into structure-aware chunks.

        Important table behavior:

        Docling can split a PDF table across page boundaries.

        Example:

            Page 4:
                SYS PWR | Off
                SYS PWR | Solid green
                SYS PWR | Blinking green

            Page 5:
                [repeated table header]
                Amber | System error

        Docling may expose those as two separate TableItem objects.

        This implementation detects likely continuation tables and
        merges them before final chunk creation.

        Tables are then converted into retrieval-friendly text so that
        relationships such as:

            SYS PWR -> Amber -> System error

        remain explicit to embedding/search systems.
        """

        chunks = []

        current_section = None
        current_subsection = None

        buffer = []
        buffer_size = 0
        buffer_page = None

        chunk_id = 0

        # ------------------------------------------------------------
        # Pending table state
        #
        # Instead of immediately emitting a table, we keep the most
        # recent table pending. If the next item is a continuation,
        # the two tables can be merged.
        # ------------------------------------------------------------

        pending_table = None

        def create_chunk(
            content: str,
            content_type: str = "text",
            page_number: Optional[int] = None,
            section_override: Optional[str] = None,
            subsection_override: Optional[str] = None,
        ):
            nonlocal chunk_id

            content = self._normalize_contract_text(
                content
            )

            if not content:
                return

            metadata = self._create_metadata(
                document_id=document_id,
                document_name=document_name,
                section=(
                    section_override
                    if section_override is not None
                    else current_section
                ),
                subsection=(
                    subsection_override
                    if subsection_override is not None
                    else current_subsection
                ),
                content_type=content_type,
                page_number=page_number,
            )

            metadata["field_name"] = (
                self._detect_contract_field(content)
            )

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

        def flush_pending_table():
            nonlocal pending_table

            if pending_table is None:
                return

            table_text = self._render_table_for_retrieval(
                pending_table["table"],
                document,
                table_context=pending_table.get("context"),
                continuation_tables=pending_table.get(
                    "continuations",
                    [],
                ),
            )

            if table_text:
                create_chunk(
                    content=table_text,
                    content_type="table",
                    page_number=pending_table.get("page"),
                    section_override=pending_table.get(
                        "section"
                    ),
                    subsection_override=pending_table.get(
                        "subsection"
                    ),
                )

            pending_table = None

        # ============================================================
        # ITERATE DOCLING DOCUMENT
        # ============================================================

        for item, level in document.iterate_items():

            item_type = self._get_item_type(item)

            # --------------------------------------------------------
            # Section headers
            # --------------------------------------------------------

            if item_type == "SectionHeaderItem":

                flush_buffer()
                flush_pending_table()

                text = self._get_item_text(item)

                if not text:
                    continue

                if level == 1:
                    current_section = text
                    current_subsection = None
                else:
                    current_subsection = text

                continue

            # --------------------------------------------------------
            # TABLE
            # --------------------------------------------------------

            if item_type == "TableItem":

                flush_buffer()

                table_page = self._get_item_page(item)

                # If there is no pending table, hold this one.
                if pending_table is None:

                    pending_table = {
                        "table": item,
                        "page": table_page,
                        "section": current_section,
                        "subsection": current_subsection,
                        "context": self._get_table_context(
                            item,
                            document,
                        ),
                        "continuations": [],
                    }

                    continue

                # ----------------------------------------------------
                # See whether this table is a continuation of the
                # previous table.
                # ----------------------------------------------------

                should_merge = (
                    self._looks_like_table_continuation(
                        previous_table=pending_table["table"],
                        current_table=item,
                        previous_page=pending_table.get("page"),
                        current_page=table_page,
                        previous_section=pending_table.get(
                            "section"
                        ),
                        current_section=current_section,
                        document=document,
                    )
                )

                if should_merge:

                    pending_table["continuations"].append(
                        item
                    )

                    continue

                # ----------------------------------------------------
                # Not a continuation.
                # Emit previous table and start a new one.
                # ----------------------------------------------------

                flush_pending_table()

                pending_table = {
                    "table": item,
                    "page": table_page,
                    "section": current_section,
                    "subsection": current_subsection,
                    "context": self._get_table_context(
                        item,
                        document,
                    ),
                    "continuations": [],
                }

                continue

            # --------------------------------------------------------
            # Table caption / label lines (e.g. "Table A-1 System LEDs
            # on Cisco IAD2801...") are emitted by Docling as their own
            # plain-text item, often sitting between two fragments of
            # the SAME table that Docling split across a page boundary.
            # If we treat this caption as ordinary text here, it force-
            # flushes pending_table before the next TableItem arrives,
            # so _looks_like_table_continuation() never gets a chance
            # to run. Skip caption-only lines while a table is pending
            # so the merge logic above can actually do its job.
            # --------------------------------------------------------

            if pending_table is not None:
                caption_text = self._get_item_text(item)
                if caption_text and self._is_table_caption_text(caption_text):
                    continue

            # --------------------------------------------------------
            # Any non-table item means a pending table has reached its
            # boundary.
            # --------------------------------------------------------

            flush_pending_table()

            # --------------------------------------------------------
            # NORMAL TEXT / LISTS
            # --------------------------------------------------------

            text = self._get_item_text(item)

            if not text:
                continue

            if buffer_size + len(text) > self.chunk_size:

                flush_buffer()

            if buffer_page is None:
                buffer_page = self._get_item_page(item)

            buffer.append(text)

            buffer_size += len(text)

        # ============================================================
        # FINAL FLUSH
        # ============================================================

        flush_pending_table()
        flush_buffer()

        # ============================================================
        # HEADER / FOOTER RECOVERY
        # ============================================================

        header_footer_lines = (
            self._collect_header_footer_lines(
                document
            )
        )

        if header_footer_lines:

            create_chunk(
                content="\n".join(
                    header_footer_lines
                ),
                content_type="header_footer",
                page_number=None,
                section_override="Document Header/Footer",
                subsection_override=None,
            )

        # ============================================================
        # APPLY TEXT OVERLAP
        # ============================================================

        chunks = self._apply_overlap(chunks)

        # Reassign IDs sequentially after all processing.
        chunks = self._reindex_chunks(chunks)

        return chunks

    # ============================================================
    # TABLE CONTINUATION DETECTION
    # ============================================================

    def _is_table_caption_text(self, text: str) -> bool:
        """
        True if `text` looks like a table caption/title line (e.g.
        "Table A-1 System LEDs on Cisco IAD2801...", "Table 3.2 Fees")
        rather than real body content.

        Docling repeats a table's caption as its own plain-text item
        on the continuation page when a table spans a page break. The
        caption carries no retrievable information of its own, but if
        it is treated as ordinary text in chunk_docling_document(), it
        incorrectly force-flushes a pending table before the real
        continuation TableItem is reached, so
        _looks_like_table_continuation() never runs. See the caption
        check right before flush_pending_table() above.
        """
        if not text:
            return False

        stripped = text.strip()

        # Captions are short, single-line labels, not paragraphs.
        if "\n" in stripped or len(stripped) > 150:
            return False

        return bool(
            re.match(r"^Table\s+[A-Za-z0-9](-?[A-Za-z0-9]+)?\b", stripped)
        )

    def _looks_like_table_continuation(
        self,
        previous_table,
        current_table,
        previous_page: Optional[int],
        current_page: Optional[int],
        previous_section: Optional[str],
        current_section: Optional[str],
        document,
    ) -> bool:
        """
        Determine whether current_table is likely a continuation of
        previous_table.

        The detection is deliberately conservative.

        Requirements:

        1. Both objects must be TableItems.
        2. Current table should normally be on the next page.
        3. Section should remain the same.
        4. The current table should resemble the previous table
           structurally or contain a repeated header.
        5. The current table should not clearly represent a new table.

        This prevents unrelated adjacent tables from being merged.
        """

        if previous_table is None or current_table is None:
            return False

        if self._get_item_type(previous_table) != "TableItem":
            return False

        if self._get_item_type(current_table) != "TableItem":
            return False

        # ------------------------------------------------------------
        # Page continuity.
        # ------------------------------------------------------------

        if (
            previous_page is not None
            and current_page is not None
        ):
            if current_page != previous_page + 1:
                return False

        # ------------------------------------------------------------
        # Section continuity.
        # ------------------------------------------------------------

        if (
            previous_section
            and current_section
            and previous_section != current_section
        ):
            return False

        previous_text = self._get_raw_table_text(
            previous_table,
            document,
        )

        current_text = self._get_raw_table_text(
            current_table,
            document,
        )

        if not previous_text or not current_text:
            return False

        previous_normalized = self._normalize_table_for_matching(
            previous_text
        )

        current_normalized = self._normalize_table_for_matching(
            current_text
        )

        # ------------------------------------------------------------
        # Strong signal:
        #
        # Same table header appears in both fragments.
        # ------------------------------------------------------------

        previous_header = (
            self._extract_table_header_signature(
                previous_text
            )
        )

        current_header = (
            self._extract_table_header_signature(
                current_text
            )
        )

        if (
            previous_header
            and current_header
            and previous_header == current_header
        ):
            return True

        # ------------------------------------------------------------
        # Strong signal:
        #
        # Current fragment contains a standard Markdown table header.
        # ------------------------------------------------------------

        if self._contains_generic_table_header(
            current_text
        ):

            # If the previous table looks structurally compatible,
            # treat it as a continuation.
            if self._table_column_count(
                previous_text
            ) == self._table_column_count(
                current_text
            ):
                return True

        # ------------------------------------------------------------
        # Cisco/manual-specific but still generic enough:
        #
        # Repeated table labels/captions often appear when a PDF
        # table crosses a page.
        # ------------------------------------------------------------

        if (
            "LED INDICATOR" in current_normalized
            and "LED INDICATOR" in previous_normalized
        ):
            return True

        # ------------------------------------------------------------
        # Structural fallback:
        #
        # If both tables have the same number of columns and the
        # current table begins with what looks like a data row rather
        # than a new title, it is likely a continuation.
        # ------------------------------------------------------------

        previous_columns = self._table_column_count(
            previous_text
        )

        current_columns = self._table_column_count(
            current_text
        )

        if (
            previous_columns >= 2
            and current_columns == previous_columns
            and self._looks_like_data_table(current_text)
        ):
            return True

        return False

    # ============================================================
    # TABLE RAW EXTRACTION
    # ============================================================

    def _get_raw_table_text(
        self,
        table,
        document,
    ) -> str:
        """
        Extract raw table text.

        First preference is DataFrame -> Markdown because this gives
        us actual row/column structure.

        Then fall back to table.text.
        """

        # ------------------------------------------------------------
        # DataFrame
        # ------------------------------------------------------------

        try:

            dataframe = table.export_to_dataframe(
                doc=document
            )

            if dataframe is not None:

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

        # ------------------------------------------------------------
        # Direct table text
        # ------------------------------------------------------------

        text = getattr(
            table,
            "text",
            None,
        )

        if text:
            return str(text).strip()

        # ------------------------------------------------------------
        # Generic fallback
        # ------------------------------------------------------------

        try:
            return str(table).strip()
        except Exception:
            return ""

    # ============================================================
    # TABLE CONTEXT
    # ============================================================

    def _get_table_context(
        self,
        table,
        document,
    ) -> Optional[str]:
        """
        Try to determine useful contextual information for a table.

        For example:

            Table A-1 System LEDs on Cisco IAD2801...

        becomes useful context for rows that appear later on the
        continuation page.
        """

        raw = self._get_raw_table_text(
            table,
            document,
        )

        if not raw:
            return None

        # Look for common table captions in the surrounding raw text.
        # The table itself may not contain the caption, so this helper
        # mainly provides a safe placeholder for future document-model
        # specific improvements.
        return None

    # ============================================================
    # TABLE HEADER DETECTION
    # ============================================================

    def _extract_table_header_signature(
        self,
        text: str,
    ) -> Optional[str]:
        """
        Extract a normalized representation of a table header.

        Example:

            | LED Indicator | State | Meaning | Corrective Action |

        becomes:

            led indicator|state|meaning|corrective action
        """

        if not text:
            return None

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return None

        for line in lines[:5]:

            if "|" not in line:
                continue

            cells = self._parse_markdown_row(
                line
            )

            if not cells:
                continue

            normalized = [
                self._normalize_table_cell(cell)
                for cell in cells
            ]

            # Skip separator rows.
            if all(
                self._is_markdown_separator(cell)
                for cell in normalized
            ):
                continue

            # Header-like row.
            joined = "|".join(normalized)

            if any(
                keyword in joined
                for keyword in (
                    "led indicator",
                    "state",
                    "meaning",
                    "corrective action",
                    "possible causes",
                )
            ):
                return joined

        return None

    def _contains_generic_table_header(
        self,
        text: str,
    ) -> bool:
        """
        Detect common table headers.
        """

        normalized = self._normalize_table_for_matching(
            text
        )

        header_patterns = (
            "led indicator",
            "state",
            "meaning",
            "corrective action",
            "possible causes",
        )

        matches = sum(
            1
            for pattern in header_patterns
            if pattern in normalized
        )

        return matches >= 2

    # ============================================================
    # TABLE STRUCTURE HELPERS
    # ============================================================

    def _normalize_table_for_matching(
        self,
        text: str,
    ) -> str:
        """
        Normalize table text for continuation comparisons.
        """

        text = text.upper()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = text.replace(
            "\u00a0",
            " ",
        )

        return text.strip()

    def _parse_markdown_row(
        self,
        line: str,
    ) -> List[str]:
        """
        Parse a Markdown table row.
        """

        if "|" not in line:
            return []

        parts = line.split("|")

        # Remove empty outer cells caused by leading/trailing '|'.
        if parts and not parts[0].strip():
            parts = parts[1:]

        if parts and not parts[-1].strip():
            parts = parts[:-1]

        return [
            part.strip()
            for part in parts
        ]

    @staticmethod
    def _normalize_table_cell(
        cell: str,
    ) -> str:
        """
        Normalize an individual table cell.
        """

        cell = str(cell or "")

        cell = re.sub(
            r"\s+",
            " ",
            cell,
        )

        cell = cell.strip()

        return cell.lower()

    @staticmethod
    def _is_markdown_separator(
        cell: str,
    ) -> bool:
        """
        Return True for Markdown separator cells such as:

            ---
            :---
            ---:
        """

        return bool(
            re.fullmatch(
                r":?-{2,}:?",
                cell.strip(),
            )
        )

    def _table_column_count(
        self,
        text: str,
    ) -> int:
        """
        Estimate table column count from Markdown.
        """

        if not text:
            return 0

        for line in text.splitlines():

            line = line.strip()

            if "|" not in line:
                continue

            cells = self._parse_markdown_row(
                line
            )

            if cells:
                return len(cells)

        return 0

    def _looks_like_data_table(
        self,
        text: str,
    ) -> bool:
        """
        Determine whether a table fragment looks like actual data
        rather than a standalone title.
        """

        if not text:
            return False

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if len(lines) < 2:
            return False

        for line in lines[:4]:

            cells = self._parse_markdown_row(
                line
            )

            if len(cells) >= 2:
                return True

        return False

    # ============================================================
    # TABLE RETRIEVAL RENDERING
    # ============================================================

    def _render_table_for_retrieval(
        self,
        table,
        document,
        table_context: Optional[str] = None,
        continuation_tables: Optional[List] = None,
    ) -> str:
        """
        Convert a Docling table into retrieval-friendly text.

        The representation intentionally makes relationships explicit.

        Example:

            Table: System LEDs

            LED Indicator: SYS PWR
            State: Amber
            Meaning: System error
            Possible Causes and Corrective Actions:
            Contact Cisco technical support.

        This is considerably easier for embedding/retrieval systems
        to match than an orphaned Markdown row such as:

            Amber | System error | Contact Cisco technical support
        """

        continuation_tables = (
            continuation_tables or []
        )

        all_tables = [
            table,
            *continuation_tables,
        ]

        rendered_parts = []

        for table_index, current_table in enumerate(
            all_tables
        ):

            raw_text = self._get_raw_table_text(
                current_table,
                document,
            )

            if not raw_text:
                continue

            # --------------------------------------------------------
            # Parse Markdown rows.
            # --------------------------------------------------------

            rows = self._extract_markdown_rows(
                raw_text
            )

            if not rows:
                # Fallback to normalized plain text.
                normalized = (
                    self._normalize_contract_text(
                        raw_text
                    )
                )

                if normalized:
                    rendered_parts.append(
                        normalized
                    )

                continue

            # --------------------------------------------------------
            # Remove repeated headers from continuation tables.
            # --------------------------------------------------------

            if table_index > 0:
                rows = self._remove_repeated_header_rows(
                    rows
                )

            if not rows:
                continue

            # --------------------------------------------------------
            # First table establishes column names.
            # --------------------------------------------------------

            if table_index == 0:

                headers = self._find_table_headers(
                    rows
                )

                data_rows = self._remove_header_rows(
                    rows,
                    headers,
                )

            else:

                headers = (
                    self._find_table_headers(
                        self._extract_markdown_rows(
                            self._get_raw_table_text(
                                table,
                                document,
                            )
                        )
                    )
                )

                data_rows = rows

            # --------------------------------------------------------
            # Render structured rows.
            # --------------------------------------------------------

            for row in data_rows:

                structured = self._render_table_row(
                    row=row,
                    headers=headers,
                )

                if structured:
                    rendered_parts.append(
                        structured
                    )

        # ------------------------------------------------------------
        # Add raw Markdown after structured rows when useful.
        #
        # This gives retrieval both semantic row relationships and
        # the original tabular representation.
        # ------------------------------------------------------------

        markdown_parts = []

        for current_table in all_tables:

            raw = self._get_raw_table_text(
                current_table,
                document,
            )

            if raw:
                cleaned = self._clean_table_markdown(
                    raw,
                    remove_header=(
                        current_table is not table
                    ),
                )

                if cleaned:
                    markdown_parts.append(
                        cleaned
                    )

        final_parts = []

        if table_context:
            final_parts.append(
                f"Table context: {table_context}"
            )

        if rendered_parts:
            final_parts.append(
                "\n\n".join(rendered_parts)
            )

        if markdown_parts:
            final_parts.append(
                "Original table representation:\n"
                + "\n\n".join(markdown_parts)
            )

        return "\n\n".join(
            part
            for part in final_parts
            if part
        ).strip()

    # ============================================================
    # MARKDOWN TABLE PARSING
    # ============================================================

    def _extract_markdown_rows(
        self,
        text: str,
    ) -> List[List[str]]:
        """
        Extract rows from Markdown-like table output.
        """

        rows = []

        for line in text.splitlines():

            line = line.strip()

            if "|" not in line:
                continue

            cells = self._parse_markdown_row(
                line
            )

            if not cells:
                continue

            # Skip Markdown separator rows.
            if all(
                self._is_markdown_separator(
                    cell
                )
                for cell in cells
            ):
                continue

            rows.append(cells)

        return rows

    def _find_table_headers(
        self,
        rows: List[List[str]],
    ) -> List[str]:
        """
        Find the most likely table header row.
        """

        if not rows:
            return []

        for row in rows[:3]:

            normalized = [
                self._normalize_table_cell(
                    cell
                )
                for cell in row
            ]

            joined = " ".join(
                normalized
            )

            if (
                "led indicator" in joined
                or "state" in joined
                or "meaning" in joined
                or "corrective action" in joined
                or "possible causes" in joined
            ):
                return row

        # Generic fallback: first row.
        return rows[0]

    def _remove_header_rows(
        self,
        rows: List[List[str]],
        headers: List[str],
    ) -> List[List[str]]:
        """
        Remove the header row and separator-like rows.
        """

        result = []

        normalized_headers = [
            self._normalize_table_cell(
                cell
            )
            for cell in headers
        ]

        for row in rows:

            normalized_row = [
                self._normalize_table_cell(
                    cell
                )
                for cell in row
            ]

            if normalized_row == normalized_headers:
                continue

            if all(
                self._is_markdown_separator(
                    cell
                )
                for cell in normalized_row
            ):
                continue

            result.append(row)

        return result

    def _remove_repeated_header_rows(
        self,
        rows: List[List[str]],
    ) -> List[List[str]]:
        """
        Remove header rows that Docling repeats on a continuation page.
        """

        if not rows:
            return []

        result = []

        for row in rows:

            normalized = " ".join(
                self._normalize_table_cell(
                    cell
                )
                for cell in row
            )

            header_words = (
                "led indicator",
                "state",
                "meaning",
                "corrective action",
                "possible causes",
            )

            header_match_count = sum(
                1
                for word in header_words
                if word in normalized
            )

            if header_match_count >= 2:
                continue

            result.append(row)

        return result

    # ============================================================
    # TABLE ROW RENDERING
    # ============================================================

    def _render_table_row(
        self,
        row: List[str],
        headers: List[str],
    ) -> str:
        """
        Convert one table row into explicit key/value text.

        This is the most important retrieval-quality change.

        Example:

            ["SYS PWR", "Amber", "System error", "Contact support"]

        becomes:

            LED Indicator: SYS PWR
            State: Amber
            Meaning: System error
            Possible Causes and Corrective Actions: Contact support
        """

        if not row:
            return ""

        # ------------------------------------------------------------
        # Normalize column count.
        # ------------------------------------------------------------

        values = list(row)

        if headers:

            if len(values) < len(headers):

                values.extend(
                    [""] * (
                        len(headers)
                        - len(values)
                    )
                )

            elif len(values) > len(headers):

                # Preserve extra information rather than dropping it.
                values = values[
                    :len(headers) - 1
                ] + [
                    " ".join(
                        values[
                            len(headers) - 1:
                        ]
                    )
                ]

        # ------------------------------------------------------------
        # Special handling for common Cisco LED tables.
        # ------------------------------------------------------------

        normalized_headers = [
            self._normalize_table_cell(
                header
            )
            for header in headers
        ]

        if (
            "led indicator" in normalized_headers
            and "state" in normalized_headers
        ):

            parts = []

            for index, value in enumerate(
                values
            ):

                if index >= len(headers):
                    break

                value = self._normalize_contract_text(
                    value
                )

                if not value:
                    continue

                header = (
                    headers[index]
                    .strip()
                )

                parts.append(
                    f"{header}: {value}"
                )

            if parts:
                return "\n".join(parts)

        # ------------------------------------------------------------
        # Generic table rendering.
        # ------------------------------------------------------------

        parts = []

        for index, value in enumerate(
            values
        ):

            value = self._normalize_contract_text(
                value
            )

            if not value:
                continue

            if index < len(headers):
                key = headers[index].strip()
            else:
                key = f"Column {index + 1}"

            parts.append(
                f"{key}: {value}"
            )

        if parts:
            return "\n".join(parts)

        return ""

    # ============================================================
    # TABLE MARKDOWN CLEANING
    # ============================================================

    def _clean_table_markdown(
        self,
        text: str,
        remove_header: bool = False,
    ) -> str:
        """
        Clean Markdown table representation.
        """

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return ""

        if remove_header:
            rows = self._extract_markdown_rows(
                text
            )

            rows = self._remove_repeated_header_rows(
                rows
            )

            if not rows:
                return ""

            rendered = []

            for row in rows:
                rendered.append(
                    "| "
                    + " | ".join(
                        row
                    )
                    + " |"
                )

            return "\n".join(
                rendered
            )

        return "\n".join(
            lines
        )

    # ============================================================
    # HEADER / FOOTER RECOVERY
    # ============================================================

    def _collect_header_footer_lines(
        self,
        document,
    ) -> List[str]:
        """
        Pull page_header/page_footer text from Docling's full text
        registry.

        Deduplicates repeated lines across pages.
        """

        seen = set()
        lines = []

        for item in getattr(
            document,
            "texts",
            [],
        ) or []:

            label = str(
                getattr(
                    item,
                    "label",
                    "",
                )
            )

            if label not in (
                "page_header",
                "page_footer",
            ):
                continue

            text = self._normalize_contract_text(
                self._get_item_text(item)
            )

            if not text:
                continue

            if text in seen:
                continue

            seen.add(text)

            lines.append(text)

        return lines

    # ============================================================
    # TEXT OVERLAP
    # ============================================================

    def _apply_overlap(
        self,
        chunks: List[DocumentChunk],
    ) -> List[DocumentChunk]:
        """
        Apply overlap only between adjacent normal text chunks
        belonging to the same logical section.

        Tables and header/footer chunks remain atomic.
        """

        if (
            self.chunk_overlap <= 0
            or len(chunks) < 2
        ):
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(
            chunks
        ):

            content = chunk.content

            if i > 0:

                previous_chunk = chunks[
                    i - 1
                ]

                is_text_to_text = (
                    chunk.metadata.get(
                        "content_type"
                    )
                    == "text"
                    and previous_chunk.metadata.get(
                        "content_type"
                    )
                    == "text"
                )

                same_section = (
                    chunk.metadata.get(
                        "section"
                    )
                    == previous_chunk.metadata.get(
                        "section"
                    )
                )

                if (
                    is_text_to_text
                    and same_section
                ):

                    overlap_text = (
                        previous_chunk.content[
                            -self.chunk_overlap:
                        ]
                    )

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

    # ============================================================
    # CHUNK ID REINDEXING
    # ============================================================

    def _reindex_chunks(
        self,
        chunks: List[DocumentChunk],
    ) -> List[DocumentChunk]:
        """
        Ensure chunk IDs are sequential.
        """

        result = []

        for index, chunk in enumerate(
            chunks
        ):

            metadata = chunk.metadata.copy()

            result.append(
                DocumentChunk(
                    chunk_id=index,
                    content=chunk.content,
                    metadata=metadata,
                )
            )

        return result

    # ============================================================
    # OCR ROUTING
    # ============================================================

    def apply_ocr_routing(
        self,
        chunks: List[DocumentChunk],
        ocr_page_numbers: set,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Reconcile Docling's structured output with page-level OCR
        routing.

        Pages classified as needing OCR have their Docling chunks
        removed and are replaced with an OCR_PENDING placeholder.

        Once HybridDocumentConverter._ocr_page() is implemented,
        replace the placeholder with real OCR-derived chunks.
        """

        if not ocr_page_numbers:
            return chunks

        filtered = [
            chunk
            for chunk in chunks
            if chunk.metadata.get(
                "page_number"
            ) not in ocr_page_numbers
        ]

        next_id = (
            max(
                (
                    c.chunk_id
                    for c in filtered
                ),
                default=-1,
            )
            + 1
        )

        for page_number in sorted(
            ocr_page_numbers
        ):

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
                        f"[OCR_PENDING: page "
                        f"{page_number} has insufficient "
                        f"extractable text and requires OCR "
                        f"processing, not yet implemented.]"
                    ),
                    metadata=metadata,
                )
            )

            next_id += 1

        return self._reindex_chunks(
            filtered
        )

    # ============================================================
    # LEGACY / DIRECT TABLE EXTRACTION
    # ============================================================

    def _extract_table_text(
        self,
        table,
        document,
    ) -> str:
        """
        Publicly-compatible table extraction helper.

        Converts a table into retrieval-friendly text.
        """

        return self._render_table_for_retrieval(
            table,
            document,
        )

    # ============================================================
    # GENERIC TEXT CHUNKING
    # ============================================================

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

        Hard character splitting is used only when a single unit
        exceeds chunk_size.
        """

        if not text or not text.strip():
            return []

        metadata = metadata or {}

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split(
                "\n\n"
            )
            if paragraph.strip()
        ]

        chunks = []

        chunk_id = 0

        current_parts = []
        current_length = 0

        def add_chunk(
            content: str,
        ):
            nonlocal chunk_id

            content = content.strip()

            if not content:
                return

            chunk_metadata = metadata.copy()

            chunk_metadata[
                "chunk_id"
            ] = chunk_id

            chunk_metadata[
                "content_type"
            ] = "text"

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

            content = "\n\n".join(
                current_parts
            )

            add_chunk(
                content
            )

            current_parts = []
            current_length = 0

        for paragraph in paragraphs:

            # --------------------------------------------------------
            # Paragraph fits.
            # --------------------------------------------------------

            if (
                current_length
                + len(paragraph)
                + 2
                <= self.chunk_size
            ):

                current_parts.append(
                    paragraph
                )

                current_length += (
                    len(paragraph)
                    + 2
                )

                continue

            # --------------------------------------------------------
            # Flush existing chunk.
            # --------------------------------------------------------

            flush_current()

            # --------------------------------------------------------
            # Paragraph itself fits.
            # --------------------------------------------------------

            if len(paragraph) <= self.chunk_size:

                current_parts.append(
                    paragraph
                )

                current_length = len(
                    paragraph
                )

                continue

            # --------------------------------------------------------
            # Paragraph too large.
            # Split using sentence boundaries.
            # --------------------------------------------------------

            sentences = [
                sentence.strip()
                for sentence in re.split(
                    r"(?<=[.!?])\s+",
                    paragraph,
                )
                if sentence.strip()
            ]

            sentence_buffer = []
            sentence_length = 0

            for sentence in sentences:

                sentence_with_space = (
                    len(sentence) + 1
                )

                if (
                    sentence_length
                    + sentence_with_space
                    <= self.chunk_size
                ):

                    sentence_buffer.append(
                        sentence
                    )

                    sentence_length += (
                        sentence_with_space
                    )

                    continue

                # ----------------------------------------------------
                # Flush sentence buffer.
                # ----------------------------------------------------

                if sentence_buffer:

                    add_chunk(
                        " ".join(
                            sentence_buffer
                        )
                    )

                    sentence_buffer = []
                    sentence_length = 0

                # ----------------------------------------------------
                # Individual sentence too large.
                # Split by words.
                # ----------------------------------------------------

                if (
                    len(sentence)
                    > self.chunk_size
                ):

                    words = sentence.split()

                    word_buffer = []
                    word_length = 0

                    for word in words:

                        word_with_space = (
                            len(word) + 1
                        )

                        if (
                            word_length
                            + word_with_space
                            <= self.chunk_size
                        ):

                            word_buffer.append(
                                word
                            )

                            word_length += (
                                word_with_space
                            )

                        else:

                            if word_buffer:

                                add_chunk(
                                    " ".join(
                                        word_buffer
                                    )
                                )

                            word_buffer = [
                                word
                            ]

                            word_length = (
                                len(word)
                                + 1
                            )

                    if word_buffer:

                        add_chunk(
                            " ".join(
                                word_buffer
                            )
                        )

                else:

                    sentence_buffer = [
                        sentence
                    ]

                    sentence_length = (
                        len(sentence) + 1
                    )

            if sentence_buffer:

                add_chunk(
                    " ".join(
                        sentence_buffer
                    )
                )

        flush_current()

        # ------------------------------------------------------------
        # Apply overlap.
        # ------------------------------------------------------------

        if (
            self.chunk_overlap > 0
            and len(chunks) > 1
        ):

            overlapped_chunks = []

            for i, chunk in enumerate(
                chunks
            ):

                content = chunk.content

                if i > 0:

                    previous_content = (
                        chunks[
                            i - 1
                        ].content
                    )

                    overlap_text = (
                        previous_content[
                            -self.chunk_overlap:
                        ]
                    )

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

        return self._reindex_chunks(
            chunks
        )

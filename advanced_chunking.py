from __future__ import annotations

import os
import re
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

class GlobalMetadata(BaseModel):
    global_summary: str = Field(description="Tóm tắt ngắn gọn toàn bộ tài liệu")
    target_audience: List[str] = Field(description="Đối tượng mục tiêu")
    doc_type: Optional[str] = Field(default=None, description="Loại tài liệu pháp lý")
    jurisdiction: Optional[str] = Field(default=None, description="Thẩm quyền tài phán/luật áp dụng")


class Hierarchy(BaseModel):
    section: Optional[str] = Field(default=None, description="Số/mã Section (ví dụ: '5', '5.2')")
    subsection: Optional[str] = Field(default=None, description="Ký hiệu Subsection (ví dụ: 'a', 'c')")
    header: Optional[str] = Field(default=None, description="Tiêu đề điều khoản/đầu mục gần nhất")


class Enrichment(BaseModel):
    keywords: List[str] = Field(description="Từ khóa quan trọng")
    summary: str = Field(description="Tóm tắt chunk 1-2 câu")
    hypothetical_questions: List[str] = Field(description="Câu hỏi giả định phục vụ truy hồi")


class LegalMetadata(BaseModel):
    risk_category: Optional[str] = Field(default=None, description="Nhóm rủi ro pháp lý")
    jurisdiction: Optional[str] = Field(default=None, description="Thẩm quyền tài phán/luật áp dụng")
    related_statutes: List[str] = Field(default_factory=list, description="Văn bản luật liên quan (nếu có căn cứ)")
    party_obligation: Optional[str] = Field(default=None, description="Bên chịu nghĩa vụ chính (nếu xác định được)")


class FinalChunkOutput(BaseModel):
    chunk_id: str
    source_doc: str
    doc_type: Optional[str] = None
    hierarchy: Hierarchy

    table_number: Optional[int] = None
    chunk_type: str  # 'text' | 'table'

    raw_text: str

    global_summary: str
    target_audience: List[str]

    enrichment: Enrichment
    legal_metadata: LegalMetadata

    offsets: Optional[Dict[str, Any]] = None


class LayoutAwareChunker:
    """
    Layout-aware chunking + LLM enrichment in one class.

    - Narrative text: semantic, sentence-boundary chunking; respects detected headings.
    - HTML tables: split by rows while preserving headers (Markdown per chunk).
    - Global metadata injected into each chunk.
    - Per-chunk enrichment + legal metadata produced via responses.parse.
    """

    _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])", re.UNICODE)
    _TABLE_PATTERN = re.compile(r"<table.*?>.*?</table>", re.DOTALL | re.IGNORECASE)
    _PLACEHOLDER_SPLIT = re.compile(r"(\{table_\d+\})")

    _HEADING_PATTERNS = [
        re.compile(r"^\s*(Section|SECTION)\s+(\d+(?:\.\d+)*)\s*(?:\((.*?)\))?\s*$"),
        re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$"),
        re.compile(r"^\s*\(?([a-zA-Z])\)\s+(.+?)\s*$"),
    ]

    class _ChunkEnrichmentFull(Enrichment):
        legal_metadata: LegalMetadata

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-nano-2025-08-07",
        chunk_word_limit: int = 384,
        sentence_overlap: int = 2,
        table_rows_per_chunk: int = 15,
        log_file: str = "chunking.log",
    ) -> None:
        load_dotenv()
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

        self.chunk_word_limit = chunk_word_limit
        self.sentence_overlap = sentence_overlap
        self.table_rows_per_chunk = table_rows_per_chunk

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            encoding="utf-8",
        )

    # ----------------------------
    # Public API
    # ----------------------------
    def process_document(
        self,
        original_file_path: Path,
        intermediate_dir: Optional[Path] = None,
    ) -> List[Dict]:
        """
        Process a single .md file that contains narrative text + embedded <table>...</table>.
        Returns list[dict] ready to be dumped to JSON.
        """
        source_doc_stem = original_file_path.stem
        source_doc_name = original_file_path.name

        if intermediate_dir is None:
            intermediate_dir = original_file_path.parent / "intermediate_outputs"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        no_tables_path, table_file_map = self._separate_tables(original_file_path, intermediate_dir)
        text_with_placeholders = no_tables_path.read_text(encoding="utf-8")

        global_meta = self._generate_global_metadata(text_with_placeholders)

        segments = self._PLACEHOLDER_SPLIT.split(text_with_placeholders)

        out: List[Dict] = []
        seq = 1
        carry_h = Hierarchy(section=None, subsection=None, header=None)

        for segment in segments:
            seg = segment.strip()
            if not seg:
                continue

            m = re.fullmatch(r"\{table_(\d+)\}", seg)
            if m:
                table_idx = int(m.group(1))
                table_path = table_file_map.get(table_idx)
                if not table_path or not table_path.exists():
                    logging.warning(f"Table file not found for index {table_idx} in {source_doc_stem}")
                    continue

                table_md_chunks = self._table_chunks(table_path, self.table_rows_per_chunk)
                for part_i, md_chunk in enumerate(table_md_chunks, start=1):
                    enrichment_dict, legal_dict = self._enrich_chunk(
                        content=md_chunk,
                        global_summary=global_meta["global_summary"],
                        doc_type=global_meta.get("doc_type"),
                        hierarchy=carry_h,
                        is_table=True,
                    )
                    chunk_obj = FinalChunkOutput(
                        chunk_id=self._stable_chunk_id(source_doc_stem, part_i, carry_h, table_number=table_idx),
                        source_doc=source_doc_name,
                        doc_type=global_meta.get("doc_type"),
                        hierarchy=carry_h,
                        table_number=table_idx,
                        chunk_type="table",
                        raw_text=md_chunk,
                        global_summary=global_meta["global_summary"],
                        target_audience=global_meta["target_audience"],
                        enrichment=Enrichment(**enrichment_dict),
                        legal_metadata=LegalMetadata(**legal_dict),
                    )
                    out.append(chunk_obj.model_dump())
                    seq += 1
                continue

            # text segment
            blocks = self._semantic_text_blocks(seg)
            if blocks:
                carry_h = blocks[-1][0]  # nearest context for subsequent tables

            for h, block_text in blocks:
                for t in self._sentence_chunk(block_text, self.chunk_word_limit, self.sentence_overlap):
                    enrichment_dict, legal_dict = self._enrich_chunk(
                        content=t,
                        global_summary=global_meta["global_summary"],
                        doc_type=global_meta.get("doc_type"),
                        hierarchy=h,
                        is_table=False,
                    )
                    chunk_obj = FinalChunkOutput(
                        chunk_id=self._stable_chunk_id(source_doc_stem, seq, h, table_number=None),
                        source_doc=source_doc_name,
                        doc_type=global_meta.get("doc_type"),
                        hierarchy=h,
                        table_number=None,
                        chunk_type="text",
                        raw_text=t,
                        global_summary=global_meta["global_summary"],
                        target_audience=global_meta["target_audience"],
                        enrichment=Enrichment(**enrichment_dict),
                        legal_metadata=LegalMetadata(**legal_dict),
                    )
                    out.append(chunk_obj.model_dump())
                    seq += 1

        return out

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        intermediate_dir: Optional[Path] = None,
        skip_existing: bool = True,
    ) -> None:
        """
        Batch process *.md files: input_dir -> output_dir/*.json
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if intermediate_dir is None:
            intermediate_dir = output_dir.parent / "intermediate_outputs"
        Path(intermediate_dir).mkdir(parents=True, exist_ok=True)

        md_files = sorted(input_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=False)
        print(f"📂 Found {len(md_files)} input files.")

        for fp in md_files:
            out_json = output_dir / f"{fp.stem}.json"
            if skip_existing and out_json.exists():
                print(f"⏩ Skipped: {fp.name} (Output {out_json.name} already exists)")
                continue

            try:
                results = self.process_document(fp, intermediate_dir=Path(intermediate_dir))
                out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"✅ FINAL JSON SAVED: {out_json.name}")
            except Exception as e:
                logging.error(f"Failed file {fp.name}: {e}")
                print(f"❌ Error processing {fp.name}: {e}")

    # ----------------------------
    # Internals: global metadata + enrichment
    # ----------------------------
    def _generate_global_metadata(self, text_content: str) -> Dict:
        try:
            resp = self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là chuyên gia phân tích tài liệu pháp lý/contract.\n"
                            "Hãy trả về global_summary, target_audience, doc_type và jurisdiction (nếu suy ra được).\n"
                            "Ngắn gọn, đúng trọng tâm. Nếu không chắc doc_type/jurisdiction thì để null."
                        ),
                    },
                    {"role": "user", "content": text_content[:25000]},
                ],
                text_format=GlobalMetadata,
            )
            return resp.output_parsed.model_dump()
        except Exception as e:
            logging.error(f"Global metadata error: {e}")
            return {
                "global_summary": "N/A",
                "target_audience": [],
                "doc_type": None,
                "jurisdiction": None,
            }

    def _enrich_chunk(
        self,
        content: str,
        global_summary: str,
        doc_type: Optional[str],
        hierarchy: Hierarchy,
        is_table: bool,
    ) -> Tuple[Dict, Dict]:
        context_type = "HTML Table (Markdown table chunk; header must be preserved)" if is_table else "Narrative Text (legal clause)"
        hierarchy_str = {
            "section": hierarchy.section,
            "subsection": hierarchy.subsection,
            "header": hierarchy.header,
        }

        prompt = (
            "Bạn là hệ thống enrichment cho retrieval trong bối cảnh hợp đồng/pháp lý.\n"
            f"Context (global summary): {global_summary}\n"
            f"Doc type: {doc_type}\n"
            f"Hierarchy (nearest): {json.dumps(hierarchy_str, ensure_ascii=False)}\n"
            f"Content type: {context_type}\n\n"
            "Nhiệm vụ:\n"
            "1) summary: tóm tắt 1-2 câu, giữ thuật ngữ pháp lý.\n"
            "2) keywords: 4-6 từ khóa trọng tâm.\n"
            "3) hypothetical_questions: đúng 2-4 câu hỏi phục vụ truy hồi.\n"
            "4) legal_metadata:\n"
            "   - risk_category: chọn 1 nhóm phù hợp (Liability & Indemnity, Payment, Termination, Confidentiality, IP, SLA, Warranty, Compliance, Force Majeure, Governing Law).\n"
            "   - jurisdiction: nếu chunk thể hiện rõ.\n"
            "   - related_statutes: chỉ liệt kê nếu có căn cứ rõ trong text; nếu không để rỗng.\n"
            "   - party_obligation: Party A/Party B/Supplier/Customer… nếu xác định được.\n"
            "Nếu không chắc, để null hoặc list rỗng thay vì đoán."
        )

        try:
            resp = self.client.responses.parse(
                model=self.model,
                input=[{"role": "system", "content": prompt}, {"role": "user", "content": content}],
                text_format=self._ChunkEnrichmentFull,
            )
            parsed = resp.output_parsed.model_dump()
            legal = parsed.pop("legal_metadata", {}) or {}
            return parsed, legal
        except Exception as e:
            logging.error(f"Chunk enrichment error: {e}")
            return (
                {"summary": "Error", "keywords": [], "hypothetical_questions": []},
                {"risk_category": None, "jurisdiction": None, "related_statutes": [], "party_obligation": None},
            )

    # ----------------------------
    # Internals: table extraction + chunking
    # ----------------------------
    def _separate_tables(self, file_path: Path, intermediate_dir: Path) -> Tuple[Path, Dict[int, Path]]:
        source_name = file_path.stem
        full_text = file_path.read_text(encoding="utf-8")

        matches = list(self._TABLE_PATTERN.finditer(full_text))
        table_file_map: Dict[int, Path] = {}
        modified_text = full_text

        logging.info(f"--- Separating {file_path.name}: Found {len(matches)} tables ---")

        for i, match in enumerate(matches):
            table_idx = i + 1
            html = match.group(0)

            table_filename = f"{source_name}_table_{table_idx}.md"
            table_save_path = intermediate_dir / table_filename
            table_save_path.write_text(html, encoding="utf-8")
            table_file_map[table_idx] = table_save_path

        for i in range(len(matches) - 1, -1, -1):
            start, end = matches[i].span()
            table_idx = i + 1
            modified_text = modified_text[:start] + f"\n\n{{table_{table_idx}}}\n\n" + modified_text[end:]

        no_table_path = intermediate_dir / f"{source_name}_no_tables.md"
        no_table_path.write_text(modified_text, encoding="utf-8")
        return no_table_path, table_file_map

    def _table_chunks(self, table_file_path: Path, rows_per_chunk: int) -> List[str]:
        try:
            html = table_file_path.read_text(encoding="utf-8")
            dfs = pd.read_html(StringIO(html))
            if not dfs:
                return []

            df = dfs[0]
            if len(df) == 0:
                return []

            if len(df) <= rows_per_chunk:
                return [df.to_markdown(index=False, tablefmt="pipe")]

            chunks: List[str] = []
            for i in range(0, len(df), rows_per_chunk):
                sub_df = df.iloc[i : i + rows_per_chunk]
                chunks.append(sub_df.to_markdown(index=False, tablefmt="pipe"))
            return chunks
        except Exception as e:
            logging.error(f"Error processing table file {table_file_path}: {e}")
            return []

    # ----------------------------
    # Internals: semantic + hierarchy-aware text chunking
    # ----------------------------
    def _detect_heading(self, line: str) -> Optional[Hierarchy]:
        ln = line.strip()
        if not ln:
            return None

        m = self._HEADING_PATTERNS[0].match(ln)
        if m:
            sec = m.group(2)
            hdr = m.group(3) or None
            return Hierarchy(section=sec, subsection=None, header=hdr)

        m = self._HEADING_PATTERNS[1].match(ln)
        if m:
            sec = m.group(1)
            hdr = m.group(2).strip()
            if len(hdr.split()) <= 18:
                return Hierarchy(section=sec, subsection=None, header=hdr)
            return None

        m = self._HEADING_PATTERNS[2].match(ln)
        if m:
            sub = m.group(1).lower()
            hdr = m.group(2).strip()
            if len(hdr.split()) <= 24:
                return Hierarchy(section=None, subsection=sub, header=hdr)
            return None

        return None

    def _merge_hierarchy(self, current: Hierarchy, update: Hierarchy) -> Hierarchy:
        return Hierarchy(
            section=update.section or current.section,
            subsection=update.subsection or current.subsection,
            header=update.header or current.header,
        )

    def _semantic_text_blocks(self, text: str) -> List[Tuple[Hierarchy, str]]:
        lines = [ln.rstrip() for ln in text.splitlines()]
        blocks: List[Tuple[Hierarchy, List[str]]] = []

        cur_h = Hierarchy(section=None, subsection=None, header=None)
        buf: List[str] = []

        def flush():
            nonlocal buf
            chunk = "\n".join([x for x in buf if x.strip()]).strip()
            if chunk:
                blocks.append((cur_h, [chunk]))
            buf = []

        for ln in lines:
            maybe = self._detect_heading(ln)
            if maybe:
                flush()
                cur_h = self._merge_hierarchy(cur_h, maybe)
                buf.append(ln)  # keep heading in block
            else:
                buf.append(ln)

        flush()
        return [(h, "\n".join(parts).strip()) for h, parts in blocks]

    def _sentence_chunk(self, block_text: str, word_limit: int, sent_overlap: int) -> List[str]:
        clean = re.sub(r"\s+", " ", block_text).strip()
        if not clean:
            return []

        sents = self._SENT_SPLIT.split(clean)
        sents = [s.strip() for s in sents if s.strip()]
        if not sents:
            return []

        chunks: List[str] = []
        i = 0
        while i < len(sents):
            buf: List[str] = []
            w = 0
            j = i
            while j < len(sents):
                sw = len(sents[j].split())
                if buf and w + sw > word_limit:
                    break
                buf.append(sents[j])
                w += sw
                j += 1

            if buf:
                chunks.append(" ".join(buf).strip())

            if j >= len(sents):
                break

            i = max(j - sent_overlap, i + 1)

        return chunks

    # ----------------------------
    # Internals: chunk id
    # ----------------------------
    def _stable_chunk_id(self, source_doc_stem: str, seq: int, h: Hierarchy, table_number: Optional[int]) -> str:
        if table_number is not None:
            return f"{source_doc_stem}_table_{table_number}_part_{seq}"

        sec = (h.section or "").replace(".", "_")
        sub = (h.subsection or "")

        if sec and sub:
            return f"{source_doc_stem}_clause_{sec}{sub}"
        if sec:
            return f"{source_doc_stem}_clause_{sec}_part_{seq}"
        if sub:
            return f"{source_doc_stem}_clause_{sub}_part_{seq}"
        return f"{source_doc_stem}_chunk_{seq}"

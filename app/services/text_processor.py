"""Text cleaning, patent-content segmentation, and JSONL output.

Processing pipeline
--------------------
1. Clean raw OCR text using multi-stage pipeline:
   a. Replace \\f page separators → paragraph breaks
   b. Join OCR line breaks (mid-sentence \\n)
   c. Attach orphaned [0001] numbering
   d. Remove multi-line footer blocks (CN+说明书+页码)
   e. Remove "权利要求书" footer variants
   f. Single-line regex removals (page numbers, CN IDs, garbage)
   g. Remove noise fragments (single letters, serial numbers)
   h. Normalise whitespace
2. Segment into chunks respecting [50, 2000] character bounds.
3. Write JSONL in pretrain.json format.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Patent section headers (Chinese invention patents) ──────────────
_SECTION_HEADERS = [
    "技术领域",
    "背景技术",
    "发明内容",
    "附图说明",
    "具体实施方式",
    "权利要求书",
    "权利要求",
    "摘要",
]

_SECTION_PATTERN = re.compile(
    r"(?:^|\n)(" + "|".join(_SECTION_HEADERS) + r")\s*[:：]?\s*(?=\n|$)",
    re.MULTILINE,
)

# ═══════════════════════════════════════════════════════════════════
# Single-line cleaning patterns
# ═══════════════════════════════════════════════════════════════════

# Page numbers & metadata
_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
_CN_PAGE_NUMBER_RE = re.compile(r"^\s*第\s*\d+\s*页\s*$", re.MULTILINE)
_BIBLIOGRAPHIC_RE = re.compile(
    r"^\s*"
    r"(?:\(\d{2,3}\))?\s*"                        # optional (10) (21) (71) …
    r"(?:"
    r"申请公布号|申请公布日|申请号|申请日|"
    r"申请人|发明人|专利代理机构|代理人|"
    r"地址|优先权日|公开日|Int\.CI[\.]?|"
    r"专利号|专利代理师|专利权人|"
    r"证书号|授权公告日|授权公告号|优先权"
    r")"
    r"[^。\n]*[。]?\s*$",
    re.MULTILINE,
)

# Generic bibliographic header lines: (19) 国家知识产权局, (12) 发明专利申请, etc.
_BIBLIOGRAPHIC_HEADER_RE = re.compile(
    r"^\s*\(\d{2,3}\)\s*[^)\n]+\s*$",
    re.MULTILINE,
)

# IPC classification codes (bibliographic data): G06Q 40/04, G06F 21/60, etc.
_IPC_CLASS_RE = re.compile(
    r"^\s*[A-Z]\d{2}[A-Z]\s+\d{1,3}/\d{2,}\s*(?:\([^)]+\))?\s*$",
    re.MULTILINE,
)

# Company/institution name lines (standalone, with common suffixes)
_COMPANY_LINE_RE = re.compile(
    r"^\s*[一-鿿（）()]+"
    r"(?:有限公司|集团公司|有限责任公司|股份公司|"
    r"分公司|公司|"
    r"大学|学院|研究所|研究院|设计院|设计所|"
    r"分局|总队|支队|办公室|委员会|管理处|"
    r"集团|厂|局|院|所|校|社|部|处|室|中心)"
    r"\s*$",
    re.MULTILINE,
)

# Person-name-only lines: "孙玉华 郭建光 姚晓涛 刘伟光"
# Matches 3+ Chinese name groups separated by spaces (each 2-4 chars).
# Requires at least 3 names to avoid false positives on two-word phrases.
# Also rejects groups of single characters (e.g. "的 了 是") — each
# name component must be 2-4 CJK chars.
_PERSON_NAME_LINE_RE = re.compile(
    r"^\s*[一-鿿]{2,4}"                           # first name: "孙玉华"
    r"(?:\s+[一-鿿]{2,4}){2,6}"                   # at least 2 more names: " 郭建光 姚晓涛"
    r"\s*$",
    re.MULTILINE,
)

# CN patent IDs:  CN 118014700 A  (space between CN and digits, trailing letter)
_CN_ID_RE = re.compile(
    r"^\s*CN\s*\d{6,13}\s*[A-Za-z][A-Za-z0-9\.]?\s*$",
    re.MULTILINE,
)

# Garbage lines (punctuation-only OCR noise)
_GARBAGE_LINE_RE = re.compile(
    r"^\s*[．。、，；：、？！\-\*＝=╋\+\|/\\>\s]{3,}\s*$", re.MULTILINE
)

# ═══════════════════════════════════════════════════════════════════
# New: Footer / noise detection patterns
# ═══════════════════════════════════════════════════════════════════

# Detect a single line that is part of a page footer block.
# Matches: CN patent IDs, 说明书 (and garbled variants), N/N页
_PAGE_FOOTER_LINE_RE = re.compile(
    r"^\s*("
    r"(?:CN\s*\d{6,13}\s*[A-Za-z][A-Za-z0-9]?)"      # CN 102955986 A
    r"|"
    r"(?:[玩讥况机]明[书节]?)"                           # garbled 说明书: 机明节/玩明书等
    r"|"
    r"(?:说\s*明\s*书)"                                # 说明书
    r"|"
    r"(?:说\s*明\s*书\s*附\s*图)"                      # 说明书附图
    r"|"
    r"(?:\d+\s*/\s*\d+\s*页)"                          # 1/3页
    r")\s*$",
    re.MULTILINE,
)

# Garbled patent number fragments
_GARBLED_CN_LINE_RE = re.compile(
    r"^\s*(?:"
    r"NN\s+\d+\s+[A-Za-z0-9]+\s*"                       # NN 017 A
    r"|"
    r"N\s+\d+[A-Za-z0-9]*"                               # N 175A
    r"|"
    r"全N\s+\d+\s*[A-Za-z0-9]*"                          # 全N 1999191 A
    r")\s*$",
    re.MULTILINE,
)

# Isolated noise: single letters, serial numbers, partial CN numbers
_ISOLATED_NOISE_RE = re.compile(
    r"^\s*("
    r"[A-Za-z]"                                          # Single letter: X, A
    r"|"
    r"\d{4}-\d{6,}[A-Z0-9]*"                             # Serial: 1101-194T20200528
    r"|"
    r"N\d{6,}"                                           # Partial: N1110700
    r")\s*$",
    re.MULTILINE,
)

# Lines that are > 80% special characters (garbled figure text)
_GARBLED_LINE_CLEAN_RE = re.compile(
    r"^\s*[．。、，；：、？！\-＝=╋\+\|/\\>\s\d]{6,}\s*$",
    re.MULTILINE,
)

# Orphaned numbering: [0001] on its own line (to attach to next line)
_ORPHAN_NUMBERING_RE = re.compile(r"^(\[\d{4}\])\s*$", re.MULTILINE)

# Footer prefix to strip from claims section start
_FOOTER_PREFIX_RE = re.compile(
    r"^(?:CN\s*\d{6,13}\s*[A-Za-z][A-Za-z0-9]?\s*\n)?"
    r"\d+\s*/\s*\d+\s*页\s*\n?"
)

# Page count info inline: 权利要求书 1 页 说明书 3 页 附图 1 页
_PAGE_COUNT_INLINE_RE = re.compile(
    r"权利要求书\s*\d+\s*页\s*说明书\s*\d+\s*页\s*(?:附图\s*\d+\s*页)?"
)

# Agent info merged mid-line: 公司 11227代理人王宝筠, 所 37236代理人刘晓
_AGENT_INLINE_RE = re.compile(
    r"\d+\s*代理人\S+"
)

# Patent boilerplate endings (generic legal text, not bidding-specific).
# Each alternative matches from the key phrase through to the sentence end (。),
# so that ``.sub("", text)`` removes the entire boilerplate sentence, not just
# the leading characters.
_BOILERPLATE_STARTS = [
    "以上所述仅为本发明的较佳实施例",
    "本发明的保护范围由所附权利要求",
    "凡依本发明申请范围",
    "应当理解的是，本申请中所述实施例仅用以说明",
    "对于本领域技术人员而言，显然本发明不限于上述",
    "在不脱离本发明的精神或基本特征",
    "在不脱离本发明设计思想的前提下",
    "任何熟习相关技艺者",
    "因此，无论从哪一点来看",
    "本发明的范围由所附权利要求",
    "但均应涵盖在本发明的保护范围",
]

_BOILERPLATE_LINE_RE = re.compile(
    r"^(?:" + "|".join(re.escape(s) for s in _BOILERPLATE_STARTS) + r")[^。]*[。]?",
    re.MULTILINE,
)

# Standalone generic patent info lines (entire line removal)
_GENERIC_PATENT_LINE_RE = re.compile(
    r"^\s*(?:"
    r"权利要求书\s*\d+\s*页\s*说明书\s*\d+\s*页"   # page count on its own line
    r")\s*$",
    re.MULTILINE,
)


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════
def process_document(
    raw_text: str,
    min_len: int = 120,
    max_len: int = 2000,
    category: str = "专利文献",
    source: str = "local_pdf",
    source_url: str = "",
) -> list[dict[str, str]]:
    """Full pipeline: clean → segment → wrap into dicts, with quality filters.

    Args:
        source: Origin type — ``"local_pdf"``, ``"zip_pdf"``, ``"http_pdf"``,
                ``"http_html"``, or ``"url_batch"``.
        source_url: Original download URL (empty for local uploads).
    """
    cleaned = clean_text(raw_text)
    segments = segment_text(cleaned, min_len, max_len)
    records = [
        {"text": seg, "category": category}
        for seg in segments
    ]
    records = _filter_quality(records, min_len)
    return records


def write_jsonl(
    records: list[dict[str, str]],
    output_path: str | Path,
) -> Path:
    """Write records as JSONL (one JSON object per line)."""
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════════
# Main cleaning pipeline
# ═══════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """Multi-stage cleaning: line joining → footer removal → noise removal."""
    if not text.strip():
        return ""

    # Stage 1: Pre-processing
    text = text.replace("\f", "\n\n")           # page separators → paragraph breaks
    text = _COMPANY_LINE_RE.sub("", text)       # remove standalone company lines
    text = _PERSON_NAME_LINE_RE.sub("", text)   # remove standalone person name lines
    text = _join_ocr_line_breaks(text)          # mid-sentence \n → join
    text = _attach_orphaned_numbering(text)     # [0001]\n内容 → [0001] 内容

    # Stage 2: Remove multi-line footer blocks
    text = _remove_footer_blocks(text)          # CN+说明书+页码 blocks
    text = _remove_claims_footer_blocks(text)   # 权利要求书 footer variants

    # Stage 3: Single-line regex removals
    text = _PAGE_NUMBER_RE.sub("", text)
    text = _CN_PAGE_NUMBER_RE.sub("", text)
    text = _BIBLIOGRAPHIC_RE.sub("", text)
    text = _BIBLIOGRAPHIC_HEADER_RE.sub("", text)
    text = _IPC_CLASS_RE.sub("", text)
    text = _CN_ID_RE.sub("", text)
    text = _GARBLED_CN_LINE_RE.sub("", text)
    text = _GARBAGE_LINE_RE.sub("", text)
    text = _ISOLATED_NOISE_RE.sub("", text)
    text = _GARBLED_LINE_CLEAN_RE.sub("", text)

    # Stage 4: Inline pattern removal (page count, agent info, boilerplate)
    text = _PAGE_COUNT_INLINE_RE.sub("", text)
    text = _AGENT_INLINE_RE.sub("", text)
    text = _BOILERPLATE_LINE_RE.sub("", text)
    text = _GENERIC_PATENT_LINE_RE.sub("", text)

    # Stage 5: Standalone noise fragments
    text = _remove_noise_fragments(text)

    # Stage 6: Whitespace normalisation
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Shared mojibake (garbled text) detection
# ═══════════════════════════════════════════════════════════════════

# CP1252 / Latin-1 high bytes that appear when GBK is decoded as Latin-1
_GARBLED_CHAR_RE = re.compile(r"[\x80-\xbf\xc0-\xff]")
# Box-drawing / diacritic chars typical of encoding corruption
_BOX_GARBLED_RE = re.compile(r"[\x80-\x9f╠-╿═-╬─-╋]")


def _is_mojibake(text: str) -> bool:
    """Return True if *text* looks like GBK bytes decoded as Latin-1.

    Detects the typical pattern where a multi-byte Chinese encoding
    (GBK/GB2312/GB18030) is mistakenly interpreted as a single-byte
    Latin-1 / CP1252 encoding, producing lines like:
    ``ú¿╩╘╨╨ú⌐┬╔╩ª░∞└φ╣·╙╨╣½╦╛...``
    """
    if not text or len(text) < 6:
        return False
    garbled = len(_GARBLED_CHAR_RE.findall(text))
    total = len(text)
    if total == 0:
        return False
    # > 30% high bytes in Latin-1 range → likely garbled
    if garbled / total > 0.30:
        return True
    # 3+ box-drawing / diacritic characters → strong signal
    if len(_BOX_GARBLED_RE.findall(text)) >= 3:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════
# Quality filtering
# ═══════════════════════════════════════════════════════════════════
def _filter_quality(
    records: list[dict[str, str]],
    min_len: int = 120,
) -> list[dict[str, str]]:
    """Remove low-quality records: boilerplate-only, too-short, empty, or garbled.

    A record is removed if:
    - It contains only generic boilerplate text
    - It is shorter than ``min_len`` chars
    - It is empty or whitespace-only after trimming
    - It contains mojibake (encoding corruption)
    """
    result: list[dict[str, str]] = []
    for rec in records:
        text = rec.get("text", "").strip()
        if not text:
            continue

        # Remove garbled records (encoding corruption)
        if _is_mojibake(text):
            logger.warning(
                "Dropping mojibake record (len=%d): %s…",
                len(text), text[:80],
            )
            continue

        is_boilerplate = bool(_BOILERPLATE_LINE_RE.search(text))

        # Remove pure boilerplate (generic legal endings with no substance)
        if is_boilerplate:
            continue

        # Remove records shorter than min_len (too short for pre-training)
        if len(text) < min_len:
            continue

        result.append(rec)

    return result


# ═══════════════════════════════════════════════════════════════════
# Footer block removal
# ═══════════════════════════════════════════════════════════════════
def _is_footer_line(line: str) -> bool:
    """Check if a single line is part of a patent page footer."""
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(
        r"^(?:CN\s*\d{6,13}\s*[A-Za-z][A-Za-z0-9]?)$", stripped
    ):
        return True
    if re.match(
        r"^(?:[玩讥况机]明[书节]?|说\s*明\s*书|说\s*明\s*书\s*附\s*图)$", stripped
    ):
        return True
    if re.match(r"^\d+\s*/\s*\d+\s*页$", stripped):
        return True
    if re.match(
        r"^(?:NN\s+\d+\s+[A-Za-z0-9]+|N\s+\d+[A-Za-z0-9]*|全N\s+\d+\s*[A-Za-z0-9]*)$",
        stripped,
    ):
        return True
    return False


def _remove_footer_blocks(text: str) -> str:
    """Remove consecutive footer-line runs (blocks) of 2+ lines.

    A "footer block" is a contiguous run of lines where each line
    matches ``_is_footer_line``.  Only runs of length >= 2 are removed.

    Additionally, if a line containing only "权利要求书" immediately
    precedes the footer block, it is also removed (it is a page footer,
    not a section header).
    """
    lines = text.split("\n")
    if len(lines) < 2:
        return text

    is_footer = [_is_footer_line(ln) for ln in lines]
    keep = [True] * len(lines)
    i = 0
    while i < len(lines):
        if is_footer[i]:
            # Find the end of this contiguous footer run
            j = i + 1
            while j < len(lines) and is_footer[j]:
                j += 1
            run_len = j - i
            if run_len >= 2:
                # Remove the preceding line if it's 权利要求书 (footer prefix)
                if i > 0 and lines[i - 1].strip() == "权利要求书":
                    keep[i - 1] = False
                for k in range(i, j):
                    keep[k] = False
            i = j  # skip past the entire run
            continue
        i += 1

    kept = [ln for ln, k in zip(lines, keep) if k]
    return "\n".join(kept)


def _remove_claims_footer_blocks(text: str) -> str:
    """Remove '权利要求书' footer blocks (with or without CN+page).

    A "权利要求书" is treated as a page footer (and removed) if the
    immediately following 1-2 lines match page footer patterns (CN number
    and/or ``N/N页``).  If no footer pattern follows, it is treated as a
    legitimate section header and preserved.
    """
    lines = text.split("\n")
    if len(lines) < 2:
        return text

    keep = [True] * len(lines)
    i = 0
    pat_cn = re.compile(r"^CN\s*\d{6,13}\s*[A-Za-z]")
    pat_page = re.compile(r"^\d+\s*/\s*\d+\s*页$")

    while i < len(lines):
        if lines[i].strip() == "权利要求书":
            # Check the following lines for footer patterns
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()

                # Case 1: 权利要求书 + N/N页 (no CN)
                if pat_page.match(nxt):
                    keep[i] = False      # remove 权利要求书
                    keep[i + 1] = False  # remove N/N页
                    i += 2
                    continue

                # Case 2: 权利要求书 + CN + N/N页
                if i + 2 < len(lines) and pat_cn.match(nxt):
                    nxt2 = lines[i + 2].strip()
                    if pat_page.match(nxt2):
                        keep[i] = False      # remove 权利要求书
                        keep[i + 1] = False  # remove CN
                        keep[i + 2] = False  # remove N/N页
                        i += 3
                        continue

        i += 1

    kept = [ln for ln, k in zip(lines, keep) if k]
    return "\n".join(kept)


# ═══════════════════════════════════════════════════════════════════
# Noise fragment removal
# ═══════════════════════════════════════════════════════════════════
def _remove_noise_fragments(text: str) -> str:
    """Remove lines that are clearly meaningless noise."""
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue

        # Single Latin letter on its own line (seal/stamp noise)
        if re.match(r"^[A-Za-z]$", stripped):
            continue

        # Single CJK character on its own line (isolated garbage)
        if len(stripped) == 1 and '一' <= stripped <= '鿿':
            continue

        # Partial patent number pattern
        if re.match(r"^N\d{6,}$", stripped):
            continue

        # Serial number pattern: 1101-194T20200528
        if re.match(r"^\d{4}-[A-Za-z0-9]+$", stripped):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ═══════════════════════════════════════════════════════════════════
# Line joining (OCR mid-sentence break reconstruction)
# ═══════════════════════════════════════════════════════════════════
def _is_section_boundary(line: str) -> bool:
    """Check if a line represents a structural boundary (don't join across)."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped in _SECTION_HEADERS:
        return True
    if re.match(r"^\[\d{4}\]", stripped):
        return True
    if re.match(r"^\d+[\.．、]", stripped):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、．]", stripped):
        return True
    return False


def _is_continuation_of(line: str, prev_line: str) -> bool:
    """Determine if ``line`` is a mid-sentence continuation of ``prev_line``."""
    if not line or not prev_line:
        return False

    # Never join across structural boundaries
    if _is_section_boundary(line) or _is_section_boundary(prev_line):
        return False

    # Never join lines that are potential footers (CN IDs, 说明书, page numbers)
    if _is_footer_line(prev_line) or _is_footer_line(line):
        return False

    # Never join if current line starts with parenthetical patent markers
    if re.match(r"^\(\d{2}\)", line):
        return False

    # Never join if previous line is a parenthetical bibliographic header
    if re.match(r"^\(\d{2}\)", prev_line):
        return False

    # If previous line ends with sentence-ending punctuation, it's a boundary
    if re.search(r"[。！？；]\s*$", prev_line):
        return False

    # If previous line ends with closing bracket, allow if line is lowercase continuation
    if re.search(r"[\)）】]\s*$", prev_line):
        return bool(line and line[0].islower())

    # Line starts with lowercase letter → continuation
    if line and line[0].islower():
        return True

    # Line starts with Chinese punctuation → continuation
    if line and line[0] in "，、；：":
        return True

    # Both lines are Chinese; prev doesn't end with sentence end
    # → likely mid-sentence OCR break
    if (
        re.search(r"[一-鿿]", prev_line)
        and re.search(r"[一-鿿]", line)
        and not re.search(r"[。！？；]\s*$", prev_line)
        and not re.match(r"^[A-Z0-9\[一二三四五六七八九十\d\(]", line)
    ):
        return True

    return False


def _join_ocr_line_breaks(text: str) -> str:
    """Join lines broken mid-sentence by OCR text-block detection."""
    lines = text.split("\n")
    if len(lines) <= 1:
        return text

    result: list[str] = []
    i = 0
    while i < len(lines):
        current = lines[i]
        stripped = current.strip()
        if not stripped:
            result.append(current)
            i += 1
            continue

        # Check if this line continues the previous one
        if result and _is_continuation_of(stripped, result[-1].strip()):
            result[-1] += current
            i += 1
            continue

        # Check if following lines should be absorbed
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt:
                break
            if _is_continuation_of(nxt, current.strip()):
                current += lines[j]
                j += 1
                continue
            break

        result.append(current)
        i = j if j > i + 1 else i + 1

    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════
# Orphaned numbering attachment
# ═══════════════════════════════════════════════════════════════════
def _attach_orphaned_numbering(text: str) -> str:
    """Merge ``[0001]\\nContent`` into ``[0001] Content``."""
    return re.sub(
        r"^(\[\d{4}\])\s*\n(?=\S)",
        r"\1 ",
        text,
        flags=re.MULTILINE,
    )


# ═══════════════════════════════════════════════════════════════════
# Segmentation
# ═══════════════════════════════════════════════════════════════════
def segment_text(text: str, min_len: int = 50, max_len: int = 2000) -> list[str]:
    """Split cleaned patent text into chunks respecting length bounds."""
    if not text.strip():
        return []

    sections = _split_by_section_headers(text)

    groups: list[str] = []
    for section in sections:
        groups.extend(_sub_split_paragraphs(section))

    return _merge_and_split(groups, min_len, max_len)


def _strip_leading_footer(content: str) -> str:
    """Remove residual CN+page prefix from claims section content."""
    return _FOOTER_PREFIX_RE.sub("", content).strip()


def _split_by_section_headers(text: str) -> list[str]:
    """Split text at known patent section headers, keeping headers with content."""
    parts = _SECTION_PATTERN.split(text)
    if len(parts) <= 1:
        return [text.strip()] if text.strip() else []

    sections: list[str] = []

    # Preamble before first header
    preamble = parts[0].strip()
    if preamble:
        sections.append(preamble)

    # Pair headers (odd indices) with content (even indices)
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        # Strip leading footer remnants from claims sections
        if header in ("权利要求书", "权利要求"):
            content = _strip_leading_footer(content)

        combined = header + "\n" + content if content else header
        if combined.strip():
            sections.append(combined.strip())

    return sections


def _sub_split_paragraphs(section: str) -> list[str]:
    """Sub-divide a section into paragraph-sized groups.

    Tries, in order:
    1. Patent numbering: ``[0001]`` … ``[9999]``
    2. Chinese / decimal numbering: ``一、``, ``1.1``
    3. Double newline (paragraph break)
    """
    if not section.strip():
        return []

    parts = re.split(r"\n(?=\[\d{4}\])", section)
    if len(parts) >= 2:
        return _non_empty(parts)

    parts = re.split(r"\n(?=[一-鿿\d]+[、．.．])", section)
    if len(parts) >= 2:
        return _non_empty(parts)

    parts = re.split(r"\n\n+", section)
    if len(parts) >= 2:
        return _non_empty(parts)

    return [section.strip()]


def _non_empty(strings: list[str]) -> list[str]:
    return [s.strip() for s in strings if s.strip()]


def _merge_and_split(
    groups: list[str],
    min_len: int,
    max_len: int,
) -> list[str]:
    """Merge groups < ``min_len`` and split groups > ``max_len``."""
    result: list[str] = []
    buffer = ""

    for group in groups:
        g_len = len(group)

        if g_len > max_len:
            if buffer:
                result.append(buffer)
                buffer = ""
            result.extend(_split_long_text(group, max_len))
            continue

        if buffer and len(buffer) < min_len:
            candidate = buffer + "\n" + group
            if len(candidate) <= max_len:
                buffer = candidate
                continue
            else:
                result.append(buffer)
                buffer = group
        else:
            if buffer:
                result.append(buffer)
            buffer = group

    if buffer:
        if len(buffer) < min_len and result:
            result[-1] = result[-1] + "\n" + buffer
        else:
            result.append(buffer)

    return result


def _split_long_text(text: str, max_len: int) -> list[str]:
    """Split text at ``[0001]`` boundaries first, then sentence punctuation."""
    # Try [0001] boundaries first to keep numbered items atomic
    numbered_parts = re.split(r"\n(?=\[\d{4}\])", text)
    if len(numbered_parts) >= 2:
        sub = _merge_and_split(numbered_parts, min_len=1, max_len=max_len)
        # Re-merge any still-too-long parts at sentence boundaries
        final: list[str] = []
        for part in sub:
            if len(part) > max_len:
                final.extend(_split_long_text_fallback(part, max_len))
            else:
                final.append(part)
        return final

    return _split_long_text_fallback(text, max_len)


def _split_long_text_fallback(text: str, max_len: int) -> list[str]:
    """Fallback: split at Chinese sentence-ending punctuation."""
    segments: list[str] = []
    sentences = re.split(r"(?<=[。！？；\n])", text)

    current = ""
    for sent in sentences:
        if not sent.strip():
            continue
        if len(current) + len(sent) <= max_len:
            current += sent
        else:
            if current:
                segments.append(current.strip())
            current = sent

    if current:
        segments.append(current.strip())

    return segments

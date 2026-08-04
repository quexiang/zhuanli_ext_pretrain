"""Regulation text cleaning, segmentation, and JSONL output.

Unlike patent documents (scanned-PDF OCR), regulation documents are
typically HTML-based government notices.  They have a different structure:

- Metadata headers: 【颁布日期】【实施日期】【发文字号】etc.
- Document title (no section header prefix)
- Body organised by Chinese numbered sections: 一、二、三、...
- Subsections: （一）（二）（三）…  or  1. 2. 3. …
- Tables (application forms, lists) which we try to linearise
- Attachments and signature blocks

Processing pipeline
--------------------
1. Clean: remove HTML noise, normalise whitespace, strip metadata markers.
2. Segment into chunks respecting [120, 2000] character bounds.
3. Write JSONL (reuses ``text_processor.write_jsonl``).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Patterns to clean ─────────────────────────────────────────────────

# ── Garbled text (mojibake) detection ────────────────────────────
# GBK bytes interpreted as Latin-1 produce characters like:
#   ú ¿ ╩ ╘ ╨ ⌐ ┬ ╔ ╩ ª ░ ∞ └ φ ╣ ╙ ╨ ╣ ╜ ╥ ╡ ╬ ╒ ╨ ═ ╢ ╤ ╓ ╥
# These are strong indicators of encoding corruption.
# Lines with ≥ 30% of these CP1252 extended chars (0x80-0xFF
# excluding CJK) are flagged as garbled.

# CP1252 / Latin-1 extended chars not in CJK ranges
_GARBLED_CHAR_RE = re.compile(
    r"[\x80-\xbf\xc0-\xff]"  # all Latin-1 high bytes
)
# CJK + common Chinese punctuation
_CJK_CHAR_RE = re.compile(
    r"[一-鿿　-〿＀-￯㐀-䶿"
    r"豈-﫿⾀0-⾡f"
    r"a-zA-Z0-9\s\.\,\;\:\!\?\-\+\=\(\)\[\]\{\}\"\'\/\\@#$%^&*_|~`<>"
    r"·•–—‘’“”… "
    r"、。，；：！？‘’"
    r"“”（）【】《》〈〉]",
)


def _is_mojibake(text: str) -> bool:
    """Check if *text* is likely a mojibake corruption.

    Returns ``True`` if the ratio of non-CJK extended Latin characters
    exceeds 30%, which is a strong signal of GBK→Latin-1 encoding failure.
    """
    if not text or len(text) < 6:
        return False
    # Count characters that look like GBK bytes decoded as Latin-1
    garbled = len(_GARBLED_CHAR_RE.findall(text))
    total = len(text)
    if total == 0:
        return False
    # If > 30% of chars are in the Latin-1 high range, it's likely garbled
    if garbled / total > 0.30:
        return True
    # Also check: many CP1252-specific box-drawing / accent chars
    box_chars = len(re.findall(r"[\x80-\x9f╠-╿═-╬─-╋]", text))
    if box_chars >= 3:
        return True
    return False


# Metadata bracket markers (remove the whole line)
_METADATA_BRACKET_LINE_RE = re.compile(
    r"^\s*【[^】]+】\s*$",
    re.MULTILINE,
)

# "附件：" or "附件1：" lines (keep, but don't let them dominate)
_ATTACHMENT_HEADER_RE = re.compile(
    r"^\s*(?:附件[：:\s]|附表[：:\s]|附图[：:\s]).*$",
    re.MULTILINE,
)

# Pure whitespace / table-artifact lines
_PURE_WHITESPACE_RE = re.compile(r"^\s*$")

# Lines that are only punctuation / table borders
_TABLE_BORDER_RE = re.compile(
    r"^[\s_\-=－═─━═\|\/\\\+]+$",
)

# Single CJK character on its own line (table spill-over)
_SINGLE_CJK_LINE_RE = re.compile(
    r"^\s*[一-鿿]\s*$",
)

# Boilerplate signature / date lines (keep them — they are part of the doc)
# but we want to normalise very short standalone lines

# ── Section detection patterns ────────────────────────────────────────

# Article / clause reference patterns (keep regardless of length)
_ARTICLE_REF_RE = re.compile(
    r"第\s*[一二三四五六七八九十百千\d]+\s*(?:条|款|项|节|章|部分|篇)",
)

# Chinese numbered sections: 一、 二、 … 十、 十一、
_CN_SECTION_RE = re.compile(
    r"^(?:[一二三四五六七八九十]{1,3})[、．.]",
)

# Parenthesised numbering: （一）（二）… or (1) (2)…
_PAREN_NUMBERING_RE = re.compile(
    r"^[（(]\s*(?:[一二三四五六七八九十]+|\d+)\s*[）)]",
)

# Arabic numbering: 1. 2.  or  1、 2、
_ARABIC_NUMBERING_RE = re.compile(
    r"^\d+[、．.]",
)

# Document title keywords — lines likely to be titles
_TITLE_KEYWORDS = [
    "通知", "办法", "规定", "细则", "条例", "决定",
    "意见", "公告", "通告", "方案", "指引", "指南",
    "批复", "函", "报告",
]


def _is_section_start(line: str) -> bool:
    """Check if a line starts a new section."""
    stripped = line.strip()
    if not stripped:
        return False
    if _CN_SECTION_RE.match(stripped):
        return True
    if _PAREN_NUMBERING_RE.match(stripped):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════
def process_regulation_document(
    raw_text: str,
    min_len: int = 120,
    max_len: int = 2000,
    category: str = "法规文本",
    source: str = "local_html",
    source_url: str = "",
) -> list[dict[str, str]]:
    """Full pipeline for regulation documents.

    Args:
        source: Origin type (``"local_html"``, ``"local_pdf"``, ``"zip_pdf"``).
        source_url: Original filename or URL.
    """
    cleaned = clean_regulation_text(raw_text)
    segments = segment_regulation_text(cleaned, min_len, max_len)
    records = [
        {"text": seg, "category": category}
        for seg in segments
    ]
    records = _filter_quality(records, min_len)
    return records


# ═══════════════════════════════════════════════════════════════════
# Cleaning
# ═══════════════════════════════════════════════════════════════════
def clean_regulation_text(text: str) -> str:
    """Aggressive cleaning for regulation documents.

    Regulation HTML files contain a mix of structured text, tables,
    form fields, and metadata.  We:

    1. Remove encoding artifacts (�, BOM, zero-width chars, control chars).
    2. Strip inline 【…】 bracket metadata markers.
    3. Remove table border / box-drawing lines.
    4. Remove form-field artifact lines (lone □, single-CJK, pure digits).
    5. Remove short orphaned lines (≤ 3 chars) that aren't numbered.
    6. Collapse excessive whitespace and blank lines.
    """
    if not text.strip():
        return ""

    # ── 1. Encoding artifacts ──────────────────────────────
    text = text.replace("﻿", "")       # BOM
    text = text.replace("​", "")       # zero-width space
    text = text.replace("‎", "")       # LTR mark
    text = text.replace("‏", "")       # RTL mark
    text = text.replace("\xa0", " ")        # non-breaking space → space
    text = text.replace("�", "")       # replacement char �
    text = text.replace("\r", "\n")         # CR → LF

    # Control chars except \n, \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # ── 1b. Remove mojibake lines (GBK decoded as Latin-1) ────
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped and _is_mojibake(stripped):
            logger.debug("Dropping mojibake line: %s…", stripped[:40])
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)

    # ── 2. Remove inline 【…】 bracket markers ───────────────
    # Lines that are only 【...】 → remove entirely
    # Inline 【...】 markers (e.g. "【全文】通知内容...") → strip the marker
    text = _METADATA_BRACKET_LINE_RE.sub("", text)
    text = re.sub(r"【[^】]*】", "", text)  # remove all bracket markers

    # ── 3. Table artifacts ─────────────────────────────────
    text = _TABLE_BORDER_RE.sub("", text)

    # Box-drawing characters
    text = re.sub(r"[─-╿]", "", text)

    # Lines that are mostly spaces/punctuation fillers
    text = re.sub(r"^\s*[．。□▲△▼▽◆◇○●◎◇◆\s]{3,}\s*$", "", text, flags=re.MULTILINE)

    # ── 4. Form-field artifacts ────────────────────────────
    text = _SINGLE_CJK_LINE_RE.sub("", text)

    # Lone □ checkbox artifact
    text = re.sub(r"^\s*□+\s*$", "", text, flags=re.MULTILINE)

    # Pure digit line (form field spill-over)
    text = re.sub(r"^\s*\d{1,2}\s*$", "", text, flags=re.MULTILINE)

    # ── 5. Table data artifacts ────────────────────────────
    # Pure phone / mobile number
    text = re.sub(r"^\s*\d{7,15}\s*$", "", text, flags=re.MULTILINE)
    # Pure qualification / grade labels
    text = re.sub(r"^\s*(?:甲级|乙级|丙级|一级|二级|三级|合格|优秀|良好)\s*$", "", text, flags=re.MULTILINE)
    # Table schema header lines (just a column name)
    text = re.sub(r"^\s*(?:序号|单位名称|等级|联系人|联系电话|手机|通信地址|邮政编码|备注|姓名|性别|出生年月|学历|职称|职务|工作单位|身份证号)\s*$", "", text, flags=re.MULTILINE)
    # Pure digit/letter sequence (likely a serial number artifact)
    text = re.sub(r"^\s*\d{4}-\d{6,}[A-Z0-9]*\s*$", "", text, flags=re.MULTILINE)

    # ── 6. Short orphaned lines ────────────────────────────
    # Lines with ≤ 2 CJK chars that aren't section headers or article
    # references are likely noise (table spill-over, form artifacts).
    # We use a stricter threshold here (≤ 2 vs the original ≤ 3) and
    # whitelist article/clause markers to avoid dropping meaningful
    # content like "第一条" or "第三款".
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        # Keep section headers regardless of length
        if _CN_SECTION_RE.match(stripped) or _PAREN_NUMBERING_RE.match(stripped):
            cleaned_lines.append(line)
            continue
        # Keep article / clause references
        if _ARTICLE_REF_RE.search(stripped):
            cleaned_lines.append(line)
            continue
        # Discard only if extremely short and not meaningful
        cjk_count = len(re.findall(r"[一-鿿]", stripped))
        if cjk_count <= 2 and len(stripped) <= 5:
            continue  # discard short noise lines
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # ── 7. Whitespace normalisation ────────────────────────
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)      # collapse horizontal whitespace

    # Final strip of each line
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Segmentation
# ═══════════════════════════════════════════════════════════════════
def segment_regulation_text(
    text: str,
    min_len: int = 120,
    max_len: int = 2000,
) -> list[str]:
    """Split cleaned regulation text into chunks respecting length bounds.

    Strategy:
    1. Split by Chinese numbered sections (一、二、三、…)
    2. If no sections found, split by double newlines (paragraphs)
    3. Merge & split to fit [min_len, max_len]
    """
    if not text.strip():
        return []

    # Try splitting by Chinese numbered sections first
    groups = _split_by_sections(text)

    # If only one group, try paragraph splitting
    if len(groups) <= 1:
        groups = _split_by_paragraphs(text)

    return _merge_and_split(groups, min_len, max_len)


def _split_by_sections(text: str) -> list[str]:
    """Split at section-numbered lines (一、…  （一）…)."""
    lines = text.split("\n")
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        if _CN_SECTION_RE.match(line.strip()) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    # Filter empty
    return [s for s in sections if s]


def _split_by_paragraphs(text: str) -> list[str]:
    """Split at double-newline paragraph boundaries."""
    parts = re.split(r"\n\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _merge_and_split(
    groups: list[str],
    min_len: int,
    max_len: int,
) -> list[str]:
    """Merge short segments and split long ones to fit [min_len, max_len]."""
    result: list[str] = []
    buffer = ""

    for group in groups:
        g_len = len(group)

        if g_len > max_len:
            # Flush buffer first
            if buffer:
                result.append(buffer)
                buffer = ""
            result.extend(_split_long(group, max_len))
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


def _split_long(text: str, max_len: int) -> list[str]:
    """Split a long segment at sentence boundaries."""
    # Try splitting at double newline first
    parts = re.split(r"\n\n+", text)
    if len(parts) >= 2:
        sub = _merge_and_split([p.strip() for p in parts if p.strip()], 1, max_len)
        final: list[str] = []
        for part in sub:
            if len(part) > max_len:
                final.extend(_split_at_sentences(part, max_len))
            else:
                final.append(part)
        return final

    return _split_at_sentences(text, max_len)


def _split_at_sentences(text: str, max_len: int) -> list[str]:
    """Fallback: split at sentence-ending punctuation."""
    segments: list[str] = []
    sentences = re.split(r"(?<=[。！？；])", text)

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


# ═══════════════════════════════════════════════════════════════════
# Quality filtering
# ═══════════════════════════════════════════════════════════════════
def _filter_quality(
    records: list[dict[str, str]],
    min_len: int = 120,
) -> list[dict[str, str]]:
    """Remove low-quality records.

    A record is removed if:
    - It is empty or whitespace-only
    - It is shorter than ``min_len`` chars
    - It contains mojibake (GBK→Latin-1 encoding corruption)
    """
    result: list[dict[str, str]] = []
    for rec in records:
        text = rec.get("text", "").strip()
        if not text:
            continue
        if len(text) < min_len:
            continue
        # Discard records with garbled text
        if _is_mojibake(text):
            logger.warning(
                "Dropping mojibake record (len=%d): %s…",
                len(text), text[:80],
            )
            continue
        result.append(rec)
    return result

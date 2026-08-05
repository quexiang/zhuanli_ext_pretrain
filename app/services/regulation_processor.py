"""Regulation / legal document text extraction (PDF + HTML).

Multi-stage cleaning pipeline for LLM pre-training data quality:
  1. Format-specific raw text extraction (PyMuPDF / BeautifulSoup)
  2. Line-level noise removal (covers, headers, footers, page numbers)
  3. Multi-line footer/header block removal
  4. Orphaned article/clause numbering attachment
  5. Regulation metadata & boilerplate removal
  6. Paragraph-level deduplication
  7. Line-break joining (reconstruct mid-sentence breaks from PDF)
  8. Regulation structure parsing (chapters / articles / paragraphs)
  9. Article-based chunking with length control [min, max]
  10. Quality filtering (min length, noise ratio, boilerplate, global dedup)

Supports both PDF (.pdf) and HTML (.html / .htm) input formats.
Auto-detection is done by the caller (router) based on file extension.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Chinese numeral utilities
# ═══════════════════════════════════════════════════════════════════════

_CN_NUM_MAP: dict[str, int] = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "百": 100, "千": 1000, "万": 10000, "亿": 100000000,
}


def _cn_num_to_int(cn: str) -> int:
    """Convert Chinese numeral string to integer.

    Examples:
        "一百二十三" → 123
        "十二" → 12
        "三百零五" → 305
        "十" → 10
        "二十三" → 23
    """
    if not cn:
        return 0
    # Strip leading "第" if present
    cn = cn.lstrip("第")
    if cn.isdigit():
        return int(cn)

    total = 0
    section = 0
    for ch in cn:
        val = _CN_NUM_MAP.get(ch)
        if val is None:
            continue
        if val >= 10000:
            total += (section or 1) * val
            section = 0
        elif val >= 10:
            section = (section or 1) * val
            total += section
            section = 0
        else:
            section = val
    total += section
    return total


def _extract_cn_number(match_str: str) -> int:
    """Extract the numeric value from a Chinese number string like '第一百二十三'."""
    # Remove structural prefix
    for prefix in ["第", "共", "约"]:
        if match_str.startswith(prefix):
            match_str = match_str[1:]
    # Remove structural suffix
    for suffix in ["章", "节", "条", "款", "项", "编"]:
        if match_str.endswith(suffix):
            match_str = match_str[:-1]
    return _cn_num_to_int(match_str) if match_str else 0


# ═══════════════════════════════════════════════════════════════════════
# Core regex patterns
# ═══════════════════════════════════════════════════════════════════════

# ── Structure patterns ─────────────────────────────────────────────────
# 第X章 / 第X节 / 第X条 / 第X款 — supports both Chinese and Arabic numerals
_CN_OR_DIGIT = r"[一二三四五六七八九十百零\d]+"

CHAPTER_RE = re.compile(
    rf"^第({_CN_OR_DIGIT})章\s*(.+)?$"
)
SECTION_RE = re.compile(
    rf"^第({_CN_OR_DIGIT})节\s*(.+)?$"
)
ARTICLE_RE = re.compile(
    rf"^第({_CN_OR_DIGIT})条\s+"
)
PARAGRAPH_RE = re.compile(
    rf"^第({_CN_OR_DIGIT})款\s+"
)
SUB_ITEM_RE = re.compile(
    rf"^[（(]({_CN_OR_DIGIT})[）)]\s*"
)
# 编 (part/volume)
PART_RE = re.compile(
    rf"^第({_CN_OR_DIGIT})编\s*(.+)?$"
)

# ── Document metadata patterns ─────────────────────────────────────────

# 发文字号: 国发〔2023〕5号, X政发〔2023〕5号, X政办发〔2023〕10号, etc.
DOC_NUMBER_RE = re.compile(
    r"^[一-鿿\w]+"
    r"(?:发|办发|办函|函|通|通字|令|公告|通知)"
    r"[〔\[]\d{4}[〕\]］]?\s*\d+号?\s*$"
)

# 令号: 第X号 (主席令, 国务院令, etc.)
ORDER_NUMBER_RE = re.compile(
    rf"^.*?令?\s*第({_CN_OR_DIGIT})号\s*$"
)

# 公布/发布日期行
_PUBLISH_DATE_RE = re.compile(
    r"^\s*"
    r"(?:公布日期|发布日期|发布日期|通过日期|批准日期|"
    r"签发日期|发文日期|制定日期|颁布日期)"
    r"\s*[：:]\s*\d{4}年\d{1,2}月\d{1,2}日"
)

# 施行/实施日期行
_EFFECTIVE_DATE_RE = re.compile(
    r"^\s*"
    r"(?:施行日期|实施日期|生效日期|执行日期)"
    r"\s*[：:]\s*\d{4}年\d{1,2}月\d{1,2}日"
)

# Date-only lines: XXXX年XX月XX日
_DATE_ONLY_RE = re.compile(
    r"^\s*\d{4}年\d{1,2}月\d{1,2}日\s*$"
)

# 通过信息: （XXXX年XX月XX日第X届全国人民代表大会常务委员会第X次会议通过）
_ADOPTION_RE = re.compile(
    r"[（(]\s*\d{4}年\d{1,2}月\d{1,2}日"
    r".*?"
    r"(?:通过|批准|公布|发布|审议)"
    r"[）)]"
)

# 修订/修正声明
_REVISION_RE = re.compile(
    r"^.*?"
    r"根据\s*(?:\d{4}年\d{1,2}月\d{1,2}日)?"
    r".*?"
    r"(?:修正|修订|修改|决定修正|决定修订|决定修改)"
    r".*?$"
)

# 施行条款: 第X条 本法/本条例/本办法自...起施行
_ENACTMENT_RE = re.compile(
    rf"第({_CN_OR_DIGIT})条\s+"
    r"本(?:法|条例|办法|规定|决定|细则|规则|规程|规范|准则|通知|意见)"
    r".*?起施行"
)

# 废止声明: 自本法/本条例施行之日起，...同时废止
_ABOLISH_RE = re.compile(
    r"(?:自本(?:法|条例|办法|规定)施行之日起|"
    r"本(?:法|条例|办法|规定)施行后|"
    r"原(?:法|条例|办法|规定|决定)|"
    r"同时废止)"
)

# 发布机关行: XX部 / XX局 / XX委员会 / XX人民政府 发布/印发/制定
_ISSUER_RE = re.compile(
    r"^\s*"
    r"[一-鿿\w]+"
    r"(?:部|局|委员会|人民政府|办公厅|办公室|行|署|院|社|会|中心|"
    r"总局|总署|总会|联合会|协会)"
    r"\s*(?:发布|公布|印发|制定|颁布|签发|令|通知|公告)?\s*$"
)

# ── Cover/surface page patterns ────────────────────────────────────────

_COVER_PATTERNS: list[re.Pattern] = [
    re.compile(r"^中华人民共和国$"),
    re.compile(r"^全国人民代表大会$"),
    re.compile(r"^全国人民代表大会常务委员会$"),
    re.compile(r"^国务院$"),
    re.compile(r"^(?:法律|行政法规|部门规章|地方性法规|地方政府规章|规范性文件)$"),
    re.compile(r"^(?:主席令|国务院令|部令|局令)$"),
    re.compile(r"^$"),
]

# ── Page number patterns ───────────────────────────────────────────────

_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
_ROMAN_PAGE_RE = re.compile(r"^\s*[IVXLCivxlc]+\.?\s*$")
_PAGE_DASH_RE = re.compile(r"^\s*[-—–\-]{1,4}\s*\d{1,4}\s*[-—–\-]{1,4}\s*$")

# ── Noise / watermark patterns ────────────────────────────────────────

_WATERMARK_RE = re.compile(
    r"^(?:征求意见稿|送审稿|报批稿|草案|试行|暂行|"
    r"修订稿|修正稿|对照表|修改稿|"
    r"征求意见稿\.pdf|报批稿\.pdf)$"
)

_GARBLED_LINE_RE = re.compile(r"^[\s\*\-=_+／\\|》\.\，\。]{4,}$")
_SINGLE_PUNCT_RE = re.compile(r"^[,\、\。\，\？\！\；\：\-\—\～\·]{1,2}$")
_SINGLE_CHAR_RE = re.compile(r"^[一-鿿]$")
_TABLE_CELL_RE = re.compile(r"^\s*\|\s*$")
_ORPHAN_SINGLE_LETTER_RE = re.compile(r"^[A-Za-z]\s*$")
_SERIAL_RE = re.compile(r"^\d{4}-[A-Za-z0-9]+$")
_PARTIAL_PAT_NUM_RE = re.compile(r"^N\d{6,}$")

# ── TOC (目录) patterns ────────────────────────────────────────────────

_TOC_RE = re.compile(
    r"(?:"
    r"\.{10,}\s*(?:\d+|$)"           # dots leading to page number
    r"|"
    r"\s{4,}\d+\s*$"                 # wide space before page number
    r"|"
    r"^[IVXLC]+\s*\.{5,}"            # roman numeral TOC entry
    r")"
)

# ── Company / institution / person name patterns ────────────────────────

_COMPANY_SUFFIXES = (
    "有限公司", "集团公司", "有限责任公司", "股份公司",
    "分公司", "公司",
    "大学", "学院", "研究所", "研究院", "设计院", "设计所",
    "分局", "总队", "支队", "办公室", "委员会", "管理处",
    "集团", "厂", "局", "院", "所", "校", "社", "部", "处", "室", "中心",
    "企业", "协会", "学会", "商会", "研究会",
)

_PERSON_NAME_LINE_RE = re.compile(
    r"^\s*[一-鿿]{2,4}"
    r"(?:\s+[一-鿿]{2,4}){1,6}"
    r"\s*$"
)

# ── Text normalisation patterns ─────────────────────────────────────────

# Zero-width and invisible characters to strip entirely
_ZERO_WIDTH_RE = re.compile(
    r"[​‌‍‎‏⁠﻿­⁡⁢"
    r"⁣⁤￹￺￻"
    r"]"
)

# Whitespace normalisation: convert various spaces / line separators to ASCII
_SPACE_NORMALISE = {
    " ": " ",   # NBSP → space
    "　": " ",   # fullwidth space → space
    " ": "\n",  # line separator → newline
    " ": "\n",  # paragraph separator → newline
    " ": " ",   # narrow NBSP → space
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "": "\n",  # form feed → newline
}

# Control chars to strip (keep \n \t)
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)

# ── Content-bearing patterns (for quality checks) ──────────────────────

_SUBSTANTIVE_KEYWORDS_RE = re.compile(
    r"(?:规定|要求|应|不应|宜|不宜|不得|可|必须|允许|禁止|"
    r"应当|可以|有权|权利|义务|责任|负责|主管|"
    r"依照|按照|参照|适用|不适用|"
    r"第[一二三四五六七八九十百零\d]+条|"
    r"[一-鿿]{10,})"
)


# ═══════════════════════════════════════════════════════════════════════
# Text normalisation (applied after extraction + before output)
# ═══════════════════════════════════════════════════════════════════════

def _is_attachment_only_page(text: str) -> bool:
    """Detect pages not suitable for LLM pre-training: attachment notices,
    metadata shells, abolition announcements, forwarding pages, etc.

    Returns True to BLOCK (skip this file), False to KEEP (process it).
    """
    if not text:
        return True

    cjk = len(re.findall(r"[一-鿿]", text))

    # ── 1. Has regulation article/chapter structure → keep ─────────
    if re.search(r"第[一二三四五六七八九十百零\d]+条", text):
        return False
    if re.search(r"第[一二三四五六七八九十百零\d]+章", text):
        return False

    # ── 2. Has numbered item structure + enough CJK → keep ─────────
    #     Arabic: 1.  2.  （1） （2）  (1)  (2)
    #     Chinese: 一、 二、 （一） （二）
    _numbered = re.findall(
        r"(?:^\d+[\.、．]|^[（(]\d+[）)]|"
        r"^[一二三四五六七八九十]+[、．]|^[（(][一二三四五六七八九十]+[）)])",
        text,
        re.MULTILINE,
    )
    if len(_numbered) >= 3 and cjk >= 200:
        return False

    # ── 3. Substantial body text → keep (long-form regulation without
    #       explicit markers, or narrative-style legal text) ──
    if cjk >= 500:
        return False

    # ── 4. Attachment page: 附件 + file links ─────────────────────
    has_attach = "附件" in text
    has_link = bool(re.search(r"https?://|\.pdf|\.docx?|\.xlsx?|\.rar|\.zip", text))
    if has_attach and has_link:
        return True

    # ── 5. Metadata shell: just dates + title, no body ────────────
    has_metadata = bool(re.search(
        r"(?:颁布日期|实施日期|发布日期|施行日期|发文单位|发布机关|"
        r"发文字号|文号|【全文】|【颁布】|【实施】)",
        text,
    ))
    if has_metadata and cjk < 200:
        return True

    # ── 6. Abolition/announcement page ─────────────────────────
    #     特此公告/通知/通告, 废止声明, 投诉举报渠道公布等
    is_announcement = bool(re.search(
        r"(?:特此公告|特此通知|特此通告|特此公示|"
        r"决定废止|现予废止|予以废止|"
        r"经清理，本机关决定|决定修改|现予修改|予以修改|"
        r"投诉举报电话|投诉举报邮箱|举报电话|举报邮箱|"
        r"监督电话|联系电话：|联系邮箱：)",
        text,
    ))
    if is_announcement and cjk < 400:
        return True

    # ── 7. Notice with signatory ending but no article structure ─
    #      Check only the LAST 200 chars for signatory + date pattern
    _tail = text[-200:] if len(text) > 200 else text
    _has_signatory = bool(re.search(
        r"(?:[厅局部委署院会处室办社所中心分局总队]"
        r"[一-鿿（）()\w]{0,30}(?:〇[一二三四五六七八九]|[一二三四五六七八九]〇|"
        r"[二三四五六七八九]一|[一二三四五六七八九]二|[一二三四五六七八九]三|"
        r"[一二三四五六七八九]四|[一二三四五六七八九]五|[一二三四五六七八九]六|"
        r"[一二三四五六七八九]七|[一二三四五六七八九]八|[一二三四五六七八九]九|"
        r"[一二三四五六七八九]十)\s*年)",
        _tail,
    ))
    if _has_signatory and cjk < 400:
        return True

    return False


def _normalise_text(text: str) -> str:
    """Aggressively normalise text for LLM pre-training quality.

    Removes / replaces:
    - BOM (U+FEFF)
    - Zero-width characters (ZWSP, ZWNJ, ZWJ, WJ, soft hyphen, etc.)
    - All Unicode whitespace variants → ASCII space
    - Control characters (except \\n, \\t)
    - HTML entities (un-escaped)
    - Direction markers (LRM, RLM, etc.)
    - Surrogate characters
    """
    if not text:
        return ""

    # 1) Strip BOM explicitly
    text = text.lstrip("﻿")

    # 2) Remove zero-width / invisible characters
    text = _ZERO_WIDTH_RE.sub("", text)

    # 3) Replace various Unicode spaces / line separators with ASCII equivalents
    for src, dst in _SPACE_NORMALISE.items():
        text = text.replace(src, dst)

    # 4) Remove remaining control characters (keep \\n and \\t)
    text = _CONTROL_CHAR_RE.sub("", text)

    # 5) Decode HTML entities (belt-and-suspenders — BS4 should have handled most)
    import html as _html
    text = _html.unescape(text)

    # 6) Collapse consecutive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 7) Strip leading/trailing whitespace per line, remove blank lines
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    text = "\n".join(lines)

    return text


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Format-specific text extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_regulation_pdf(pdf_bytes: bytes) -> str:
    """Extract raw text from regulation PDF using PyMuPDF.

    Most Chinese regulations are born-digital PDFs with embedded text,
    not scanned images, so direct text extraction works well.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts: list[str] = []
    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():
            texts.append(page_text.strip())
    doc.close()
    raw = "\n\n".join(texts)
    return _normalise_text(raw)


def _extract_regulation_html(html_bytes: bytes) -> str:
    """Extract clean body text from regulation HTML using BeautifulSoup.

    Removes navigation, header, footer, script, style elements.
    Handles multiple encodings (UTF-8, GB2312, GBK) gracefully.
    """
    from bs4 import BeautifulSoup

    # Try UTF-8 with BOM strip first, fall back to common Chinese encodings
    html_text = None
    for encoding in ("utf-8-sig", "utf-8", "gb2312", "gbk", "gb18030", "latin-1"):
        try:
            html_text = html_bytes.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if html_text is None:
        html_text = html_bytes.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html_text, "lxml")

    # Remove non-content elements
    for tag_name in ("script", "style", "nav", "header", "footer",
                     "noscript", "iframe", "form", "link", "meta"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove hidden elements
    for tag in soup.find_all(style=True):
        style = tag.get("style", "")
        if "display:none" in style.replace(" ", "") or \
           "display: none" in style:
            tag.decompose()

    # Try to find the main content container
    body = None
    # Common Chinese government content selectors
    for selector in (
        "#content", ".content", ".article-content", ".Article_content",
        "#mainText", ".mainText", ".TRS_Editor", "#UCAP-CONTENT",
        "article", ".article", ".main-content", "#main-content",
        ".text-content", ".doc-content", "#docContent",
        ".xxgk_content", ".info_con", ".con_txt", ".details-cont",
    ):
        found = soup.select_one(selector)
        if found and len(found.get_text(strip=True)) > 200:
            body = found
            break

    if body is None:
        body = soup.find("body") or soup

    # Get text with newline separators
    text = body.get_text(separator="\n", strip=True)

    # Decode HTML entities that BeautifulSoup may have missed
    import html as _html
    text = _html.unescape(text)

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return _normalise_text(text)


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Line-level noise cleaning
# ═══════════════════════════════════════════════════════════════════════

def _is_cover_line(line: str) -> bool:
    """Detect regulation cover/surface page lines."""
    stripped = line.strip()
    if not stripped:
        return False
    for pat in _COVER_PATTERNS:
        if pat.match(stripped):
            return True
    return False


def _is_metadata_line(line: str) -> bool:
    """Detect regulation metadata lines: document numbers, dates, issuers."""
    stripped = line.strip()
    if not stripped:
        return False

    if DOC_NUMBER_RE.match(stripped):
        return True
    if ORDER_NUMBER_RE.match(stripped):
        return True
    if _PUBLISH_DATE_RE.match(stripped):
        return True
    if _EFFECTIVE_DATE_RE.match(stripped):
        return True
    if _DATE_ONLY_RE.match(stripped):
        return True
    if _ISSUER_RE.match(stripped):
        return True
    return False


def _is_page_number(line: str) -> bool:
    """Detect standalone page number lines."""
    stripped = line.strip()
    if not stripped:
        return False
    if _PAGE_NUM_RE.match(stripped) and len(stripped) <= 4:
        return True
    if _ROMAN_PAGE_RE.match(stripped):
        return True
    if _PAGE_DASH_RE.match(stripped):
        return True
    return False


def _is_watermark(line: str) -> bool:
    """Detect watermark text lines."""
    return bool(_WATERMARK_RE.match(line.strip()))


def _is_noise_line(line: str) -> bool:
    """Detect meaningless noise lines."""
    stripped = line.strip()
    if not stripped:
        return False
    if _GARBLED_LINE_RE.match(stripped):
        return True
    if _SINGLE_PUNCT_RE.match(stripped):
        return True
    if _SINGLE_CHAR_RE.match(stripped):
        return True
    if _TABLE_CELL_RE.match(stripped):
        return True
    if _ORPHAN_SINGLE_LETTER_RE.match(stripped):
        return True
    if _SERIAL_RE.match(stripped):
        return True
    if _PARTIAL_PAT_NUM_RE.match(stripped):
        return True
    return False


def _is_company_line(line: str) -> bool:
    """Detect standalone company/institution name lines."""
    stripped = line.strip()
    if not stripped or len(stripped) < 6:
        return False
    for suffix in _COMPANY_SUFFIXES:
        if stripped.endswith(suffix):
            prefix = stripped[:-len(suffix)]
            if re.match(r"^[一-鿿（）()A-Za-z\d\.]+$", prefix):
                return True
    return False


def _is_person_name_line(line: str) -> bool:
    """Detect standalone person name lines."""
    stripped = line.strip()
    if not stripped or len(stripped) < 4 or len(stripped) > 50:
        return False
    parts = stripped.split()
    if len(parts) < 2:
        return False
    for part in parts:
        if not re.match(r"^[一-鿿]{2,4}$", part):
            return False
    return True


def _is_toc_line(line: str) -> bool:
    """Detect table-of-contents lines (dot leaders, page numbers at end)."""
    stripped = line.strip()
    if not stripped:
        return False
    if _TOC_RE.search(stripped):
        return True
    # Dots count > 10 usually means TOC
    if stripped.count(".") > 10:
        return True
    return False


def _clean_lines(text: str) -> str:
    """Stage 2: Multi-rule line-level noise removal."""
    lines = text.split("\n")
    cleaned: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines (will re-add single newlines at end)
        if not stripped:
            cleaned.append("")
            continue

        if _is_cover_line(stripped):
            continue
        if _is_metadata_line(stripped):
            continue
        if _is_watermark(stripped):
            continue
        if _is_page_number(stripped):
            continue
        if _is_noise_line(stripped):
            continue
        if _is_company_line(stripped):
            continue
        if _is_person_name_line(stripped):
            continue
        if _is_toc_line(stripped):
            continue

        # Remove adoption/passage info embedded in text
        stripped_no_adoption = _ADOPTION_RE.sub("", stripped).strip()
        if stripped_no_adoption:
            cleaned.append(stripped_no_adoption)
        else:
            continue

    # Collapse consecutive empty lines
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Stage 3: Multi-line footer/header block removal
# ═══════════════════════════════════════════════════════════════════════

_FOOTER_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*\d+\s*/\s*\d+\s*[页頁]\s*$"),
    re.compile(r"^\s*[-—–\-]{2,}\s*\d+\s*[-—–\-]{2,}\s*$"),
    re.compile(r"^\s*—\s*\d+\s*—\s*$"),
    re.compile(r"^\s*本(?:法|条例|办法|规定).*?页\s*$"),
    # Document IDs in footers
    re.compile(r"^[A-Z]+[/\d]+[-—]\d{4}$"),
]


def _is_footer_line(line: str) -> bool:
    """Check if a line matches known footer patterns."""
    stripped = line.strip()
    if not stripped:
        return False
    for pat in _FOOTER_PATTERNS:
        if pat.match(stripped):
            return True
    return False


def _remove_footer_blocks(text: str) -> str:
    """Remove consecutive footer-line runs (2+ lines)."""
    lines = text.split("\n")
    if len(lines) < 2:
        return text

    is_footer = [_is_footer_line(ln) for ln in lines]
    keep = [True] * len(lines)
    i = 0
    while i < len(lines):
        if is_footer[i]:
            j = i + 1
            while j < len(lines) and is_footer[j]:
                j += 1
            if j - i >= 2:
                for k in range(i, j):
                    keep[k] = False
            i = j
            continue
        i += 1

    kept = [ln for ln, k in zip(lines, keep) if k]
    return "\n".join(kept)


# ═══════════════════════════════════════════════════════════════════════
# Stage 4: Orphaned article/clause numbering attachment
# ═══════════════════════════════════════════════════════════════════════

_ORPHANED_ARTICLE_RE = re.compile(rf"^(第({_CN_OR_DIGIT})条)\s*$")
_ORPHANED_CHAPTER_RE = re.compile(rf"^(第({_CN_OR_DIGIT})章)\s*$")
_ORPHANED_SECTION_RE = re.compile(rf"^(第({_CN_OR_DIGIT})节)\s*$")
_ORPHANED_PARAGRAPH_RE = re.compile(rf"^(第({_CN_OR_DIGIT})款)\s*$")


def _attach_orphaned_numbering(text: str) -> str:
    """Merge orphaned article/chapter numbers with the next line.

    e.g. "第十五条\n内容" → "第十五条 内容"
    e.g. "第一章\n总则" → "第一章 总则"
    """
    lines = text.split("\n")
    result: list[str] = []

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        is_orphan = (
            _ORPHANED_ARTICLE_RE.match(stripped) or
            _ORPHANED_CHAPTER_RE.match(stripped) or
            _ORPHANED_SECTION_RE.match(stripped) or
            _ORPHANED_PARAGRAPH_RE.match(stripped)
        )

        if is_orphan and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and not _is_toc_line(next_line):
                result.append(stripped + " " + next_line)
                i += 2
                continue

        result.append(lines[i])
        i += 1

    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════════
# Stage 5: Regulation metadata & boilerplate removal
# ═══════════════════════════════════════════════════════════════════════

def _remove_boilerplate(text: str) -> str:
    """Remove regulation-specific boilerplate from the entire text.

    Handles:
    - Adoption/passage information in parentheses
    - Revision/amendment declarations
    - Enactment boilerplate (第X条 本法自...起施行)
    - Abolishment declarations
    - Signatory lines at end
    """
    # Remove adoption info blocks (inline in text)
    text = _ADOPTION_RE.sub("", text)

    # Remove revision declaration lines
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue

        # Skip standalone revision declarations
        if _REVISION_RE.match(stripped):
            continue

        # Skip enactment boilerplate articles
        if _ENACTMENT_RE.search(stripped):
            # Only skip if it's primarily boilerplate (short article)
            if len(stripped) < 80:
                continue

        # Skip abolishment declarations (short lines only)
        if _ABOLISH_RE.search(stripped) and len(stripped) < 60:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def _remove_signatory_lines(text: str) -> str:
    """Remove signatory/date lines at the end of regulations.

    Patterns like:
        总理  李克强
        XXXX年XX月XX日
        （签署日期）
    """
    lines = text.split("\n")
    # Process from the end to find the signatory block
    # A signatory block is: optional title line + date line at the very end

    # Find trailing signatory lines
    signatory_start = len(lines)
    for i in range(len(lines) - 1, max(len(lines) - 10, 0), -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        # Date line at end
        if _DATE_ONLY_RE.match(stripped):
            signatory_start = i
            continue
        # Signatory title line (e.g. "主席 习近平", "总理 李克强")
        if re.match(r"^[一-鿿]{2,10}\s+[一-鿿]{2,6}$", stripped):
            signatory_start = min(signatory_start, i)
            continue
        # Not a signatory line → stop
        if signatory_start < len(lines) and i == signatory_start - 1:
            signatory_start = i + 1
        break

    if signatory_start < len(lines) - 1:
        lines = lines[:signatory_start]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Stage 6: Paragraph-level deduplication
# ═══════════════════════════════════════════════════════════════════════

def _remove_paragraph_duplicates(text: str, threshold: float = 0.85) -> str:
    """Remove near-duplicate consecutive paragraphs."""
    lines = text.split("\n")
    if len(lines) <= 1:
        return text

    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if kept and kept[-1].strip() and _jaccard(stripped, kept[-1].strip()) >= threshold:
            continue
        kept.append(line)

    return "\n".join(kept)


# ═══════════════════════════════════════════════════════════════════════
# Stage 7: Line-break joining
# ═══════════════════════════════════════════════════════════════════════

def _is_structure_boundary(line: str) -> bool:
    """Check if a line represents a structural boundary."""
    stripped = line.strip()
    if not stripped:
        return False
    if CHAPTER_RE.match(stripped) or SECTION_RE.match(stripped):
        return True
    if ARTICLE_RE.match(stripped) or PARAGRAPH_RE.match(stripped):
        return True
    if PART_RE.match(stripped):
        return True
    if SUB_ITEM_RE.match(stripped):
        return True
    return False


def _join_line_breaks(text: str) -> str:
    """Join lines broken mid-sentence by PDF text extraction."""
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

        if i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            if next_stripped:
                # Never join across structural boundaries
                if _is_structure_boundary(next_stripped):
                    result.append(current)
                    i += 1
                    continue
                if _is_structure_boundary(stripped):
                    result.append(current)
                    i += 1
                    continue
                # Sentence-ending punctuation → boundary
                if re.search(r"[。！？；]\s*$", stripped):
                    result.append(current)
                    i += 1
                    continue
                # Closing bracket → boundary
                if re.search(r"[）\]】]\s*$", stripped):
                    result.append(current)
                    i += 1
                    continue
                # Both lines have Chinese text → likely mid-sentence break
                if (
                    re.search(r"[一-鿿]", stripped)
                    and re.search(r"[一-鿿]", next_stripped)
                    and not re.search(r"[。！？；]\s*$", stripped)
                    and not re.match(r"^[A-Z0-9\[（(第\d]", next_stripped)
                ):
                    result.append(current + next_stripped)
                    i += 2
                    continue

        result.append(current)
        i += 1

    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════════
# Stage 8: Regulation structure parsing
# ═══════════════════════════════════════════════════════════════════════

def _parse_regulation_structure(text: str) -> list[dict]:
    """Parse regulation document structure into chapters and articles.

    Returns a list of dicts, each representing either a chapter heading
    or an article. Structure:
        {"type": "chapter", "title": "第一章 总则", "number": "1"}
        {"type": "article", "number": "1", "content": "第一条 ..."}
    """
    lines = text.split("\n")
    units: list[dict] = []

    # Find body start: skip preamble before first chapter/article
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if CHAPTER_RE.match(stripped) or ARTICLE_RE.match(stripped):
            body_start = i
            break

    i = body_start
    current_chapter: str | None = None
    current_article_buffer: list[str] = []

    def _flush_article():
        nonlocal current_article_buffer
        if not current_article_buffer:
            return
        content = " ".join(current_article_buffer).strip()
        # Check first line for article number
        art_match = ARTICLE_RE.match(current_article_buffer[0].strip())
        art_num = ""
        if art_match:
            art_num = art_match.group(1)
        if content and len(content) >= 10:
            units.append({
                "type": "article",
                "number": art_num,
                "content": content,
                "chapter": current_chapter,
            })
        current_article_buffer = []

    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            _flush_article()
            i += 1
            continue

        # Chapter heading
        ch_match = CHAPTER_RE.match(stripped)
        if ch_match:
            _flush_article()
            ch_num = ch_match.group(1)
            ch_title = ch_match.group(2) or ""
            current_chapter = f"第{ch_num}章 {ch_title}".strip()
            units.append({
                "type": "chapter",
                "title": current_chapter,
                "number": ch_num,
            })
            i += 1
            continue

        # Section heading (within a chapter)
        sec_match = SECTION_RE.match(stripped)
        if sec_match:
            _flush_article()
            units.append({
                "type": "section",
                "title": stripped,
                "number": sec_match.group(1),
                "chapter": current_chapter,
            })
            i += 1
            continue

        # Article boundary
        if ARTICLE_RE.match(stripped):
            _flush_article()
            current_article_buffer = [stripped]
            i += 1
            continue

        # Content line belonging to current article
        if current_article_buffer:
            current_article_buffer.append(stripped)
        else:
            # Text before first article/chapter: preamble, may still contain useful content
            current_article_buffer = [stripped]

        i += 1

    _flush_article()

    return units


# ═══════════════════════════════════════════════════════════════════════
# Stage 9: Article-based chunking with length control
# ═══════════════════════════════════════════════════════════════════════

def _chunk_text(
    text: str,
    min_size: int = 50,
    max_size: int = 2000,
    target_size: int = 150,
) -> list[str]:
    """Split text into chunks with length bounds.

    For regulations, the primary boundary is article (条) boundaries.
    Falls back to sentence boundaries when no article structure is found.
    """
    # Remove residual noise before chunking
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Try article-based splitting first
    parts = re.split(rf"(\n?第{_CN_OR_DIGIT}条\s+)", text)

    if len(parts) >= 3:  # At least one article boundary found
        chunks: list[str] = []
        i = 0
        # Handle preamble (before first article)
        if parts[0].strip():
            preamble = parts[0].strip()
            if len(preamble) >= min_size:
                chunks.extend(_split_long_text(preamble, max_size))
            i = 1

        while i < len(parts):
            if re.match(rf"^\n?第{_CN_OR_DIGIT}条\s+", parts[i]):
                chunk = parts[i]
                i += 1
                while i < len(parts):
                    if re.match(rf"^\n?第{_CN_OR_DIGIT}条\s+", parts[i]):
                        break
                    chunk += parts[i]
                    i += 1
                chunk = chunk.strip()
                # Handle too-long articles
                if len(chunk) > max_size:
                    chunks.extend(_split_long_text(chunk, max_size))
                else:
                    chunks.append(chunk)
            else:
                if parts[i].strip():
                    chunks.append(parts[i].strip())
                i += 1

        # Merge short consecutive chunks
        merged: list[str] = []
        buffer = ""
        for chunk in chunks:
            if len(chunk) < min_size:
                if buffer:
                    candidate = buffer + " " + chunk
                    if len(candidate) <= max_size:
                        buffer = candidate
                    else:
                        merged.append(buffer.strip())
                        buffer = chunk
                else:
                    buffer = chunk
            else:
                if buffer:
                    merged.append((buffer + " " + chunk).strip())
                    buffer = ""
                else:
                    merged.append(chunk)

        if buffer:
            if merged:
                merged[-1] = merged[-1] + " " + buffer
            else:
                merged.append(buffer.strip())

        # Filter: ensure each chunk meets min_size
        return [c for c in merged if len(c.strip()) >= min_size]

    # Fallback: no article structure found, split by paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(para) > max_size:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(_split_long_text(para, max_size))
        elif buffer and len(buffer) + len(para) + 2 <= target_size:
            buffer += "\n" + para
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = para

    if buffer:
        chunks.append(buffer.strip())

    # Merge undersized chunks
    merged: list[str] = []
    buf2 = ""
    for chunk in chunks:
        if len(chunk) < min_size:
            if buf2:
                buf2 += " " + chunk
            else:
                buf2 = chunk
        else:
            if buf2:
                merged.append((buf2 + " " + chunk).strip())
                buf2 = ""
            else:
                merged.append(chunk)
    if buf2 and merged:
        merged[-1] = merged[-1] + " " + buf2
    elif buf2:
        merged.append(buf2)

    return [c for c in merged if len(c.strip()) >= min_size]


def _split_long_text(text: str, max_len: int) -> list[str]:
    """Split long text at sentence/paragraph boundaries."""
    # Try splitting at sentence punctuation
    sentences = re.split(r"(?<=[。！？；\n])", text)
    segments: list[str] = []
    current = ""
    for sent in sentences:
        if not sent.strip():
            continue
        if len(current) + len(sent) <= max_len:
            current += sent
        else:
            if current:
                segments.append(current.strip())
            # If single sentence exceeds max, split by clause punctuation
            if len(sent) > max_len:
                sub_parts = re.split(r"(?<=[，、：）\)])", sent)
                sub_current = ""
                for sp in sub_parts:
                    if len(sub_current) + len(sp) <= max_len:
                        sub_current += sp
                    else:
                        if sub_current:
                            segments.append(sub_current.strip())
                        sub_current = sp
                if sub_current:
                    segments.append(sub_current.strip())
            else:
                current = sent

    if current:
        segments.append(current.strip())

    return segments


# ═══════════════════════════════════════════════════════════════════════
# Stage 10: Quality filtering
# ═══════════════════════════════════════════════════════════════════════

def _noise_ratio(text: str) -> float:
    """Calculate ratio of non-CJK, non-alphanumeric characters."""
    if not text:
        return 1.0
    meaningful = len(re.findall(
        r"[一-鿿a-zA-Z0-9，。！？；：、,.!?;:\-—–（）()\[\]【】{}《》<>]",
        text,
    ))
    return 1.0 - (meaningful / len(text)) if text else 1.0


def _has_substantive_content(text: str) -> bool:
    """Check if text has substantive legal/regulatory content."""
    stripped = text.strip()
    if len(stripped) < 10:
        return False
    if re.search(r"[一-鿿]", stripped):
        return True
    return bool(_SUBSTANTIVE_KEYWORDS_RE.search(stripped))


def _filter_quality(
    records: list[dict[str, str]],
    min_len: int = 50,
    max_len: int = 2000,
) -> list[dict[str, str]]:
    """Filter records: keep only those useful for LLM pre-training.

    A record is KEPT if it looks like a regulation article/clause/paragraph.
    A record is DISCARDED if it's metadata, contact info, an announcement shell,
    an attachment reference, or otherwise lacks substantive legal content.
    """
    result: list[dict[str, str]] = []
    for rec in records:
        text = rec.get("text", "").strip()
        if not text:
            continue
        if len(text) < min_len:
            continue
        if len(text) > max_len:
            continue
        if _noise_ratio(text) >= 0.3:
            continue

        cjk = len(re.findall(r"[一-鿿]", text))

        # ── Discard: generic signals this is NOT regulation body text ──
        #
        #  General principle: if a record contains contact info
        #  (any phone/email/address) or is just an attachment stub
        #  or metadata shell, discard it.
        #  No specific-keyword lists — use pattern classes.

        # Any phone number (landline or mobile)
        _has_phone = bool(re.search(r"\d{3,4}-\d{7,8}|\b1[3-9]\d{9}\b", text))
        # Any email address
        _has_email = bool(re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text))
        # Any street address fragment (路/街/道/巷 + nearby 号, or 大厦/广场/园区)
        _has_addr = bool(re.search(
            r"(?:[一-鿿]{2,10}(?:路|街|道|巷).{0,10}\d+号|"
            r"[一-鿿]{2,10}(?:大厦|广场|园区|小区|单元|楼)\d*)",
            text,
        ))
        # Contact info → not a regulation body (regulations don't list phone/email/address)
        # Only exception: if it has article structure, it might mention them in examples
        if (_has_phone or _has_email or _has_addr):
            if not re.search(r"第[一二三四五六七八九十百零\d]+条", text):
                continue

        # Attachment-only stub: "附件" + link/file + almost no body
        if "附件" in text and re.search(r"https?://|\.pdf|\.docx?|\.xlsx?", text) and cjk < 80:
            continue

        # Metadata wrapper + no article structure → likely a notice shell, not regulation
        _has_meta = bool(re.search(
            r"(?:颁布日期|实施日期|发布日期|施行日期|【全文】|【颁布】|【实施】|发文单位|发文字号)",
            text,
        ))
        _has_articles = bool(re.search(r"第[一二三四五六七八九十百零\d]+条", text))
        if _has_meta and not _has_articles and cjk < 500:
            continue

        # Short boilerplate/enactment/abolition lines
        if _ENACTMENT_RE.search(text) and len(text) < 80:
            continue
        if _ABOLISH_RE.search(text) and len(text) < 80:
            continue

        # ── Keep rules ──────────────────────────────────────────

        # Has article structure → keep
        if re.search(r"第[一二三四五六七八九十百零\d]+条", text):
            result.append(rec)
            continue
        if re.search(r"第[一二三四五六七八九十百零\d]+章", text):
            result.append(rec)
            continue

        # Has numbered items + enough CJK → keep
        _num = len(re.findall(
            r"(?:(?:^|[。！？；\n])\s*\d+[\.、．]|"
            r"(?:^|[。！？；\n])\s*[（(]\d+[）)]|"
            r"(?:^|[。！？；\n])\s*[一二三四五六七八九十]+[、．]|"
            r"(?:^|[。！？；\n])\s*[（(][一二三四五六七八九十]+[）)])",
            text,
        ))
        if _num >= 2 and cjk >= 50:
            result.append(rec)
            continue

        # Substantial CJK with legal keywords → keep
        if cjk >= 120 and _has_substantive_content(text):
            result.append(rec)
            continue

    return result


def _deduplicate(
    records: list[dict[str, str]],
    threshold: float = 0.9,
) -> list[dict[str, str]]:
    """Remove near-duplicate records using character bigram Jaccard."""
    def _bigrams(s: str) -> set[str]:
        return set(s[i:i+2] for i in range(len(s) - 1))

    seen: list[str] = []
    kept: list[dict[str, str]] = []

    for rec in records:
        text = rec.get("text", "").strip()
        if not text:
            continue

        bg = _bigrams(text)
        if not bg:
            continue

        is_dup = False
        for existing_text in seen:
            eb = _bigrams(existing_text)
            if not eb:
                continue
            intersection = bg & eb
            union = bg | eb
            sim = len(intersection) / len(union) if union else 0.0
            if sim >= threshold:
                is_dup = True
                break

        if not is_dup:
            seen.append(text)
            kept.append(rec)

    return kept


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def process_regulation_bytes(
    file_bytes: bytes,
    file_format: str,
    chunk_size: int = 150,
    min_chunk_len: int = 50,
    max_chunk_len: int = 2000,
    max_workers: int = 1,
) -> list[dict[str, str]]:
    """Full pipeline: extract → clean → segment → chunk → quality filter.

    Args:
        file_bytes: Raw file bytes (PDF or HTML).
        file_format: ``"pdf"`` or ``"html"``.
        chunk_size: Target chunk size in characters (for merging short articles).
        min_chunk_len: Minimum allowed chunk length.
        max_chunk_len: Maximum allowed chunk length.
        max_workers: Number of worker threads for parallel chapter processing.

    Returns:
        List of ``{"text": "...", "category": "法律法规"}`` records.
    """
    # ── Stage 1: Raw text extraction ──────────────────────────
    if file_format == "pdf":
        raw_text = _extract_regulation_pdf(file_bytes)
    elif file_format in ("html", "htm"):
        raw_text = _extract_regulation_html(file_bytes)
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    if not raw_text or len(raw_text) < 50:
        logger.warning("Extracted text too short or empty.")
        return []

    # ── Stage 2: Line-level cleaning ──────────────────────────
    cleaned_text = _clean_lines(raw_text)
    if not cleaned_text or len(cleaned_text) < 50:
        return []

    # ── Stage 3: Multi-line footer block removal ──────────────
    cleaned_text = _remove_footer_blocks(cleaned_text)

    # ── Stage 4: Orphaned numbering attachment ────────────────
    cleaned_text = _attach_orphaned_numbering(cleaned_text)

    # ── Stage 5: Boilerplate removal ──────────────────────────
    cleaned_text = _remove_boilerplate(cleaned_text)
    cleaned_text = _remove_signatory_lines(cleaned_text)

    # ── Stage 6: Paragraph deduplication ─────────────────────
    cleaned_text = _remove_paragraph_duplicates(cleaned_text)

    # ── Stage 7: Line-break joining ───────────────────────────
    cleaned_text = _join_line_breaks(cleaned_text)

    # ── Stage 8-9: Structure parsing + chunking ───────────────
    # Use the all-in-one chunking that handles article boundaries
    chunks = _chunk_text(
        cleaned_text,
        min_size=min_chunk_len,
        max_size=max_chunk_len,
        target_size=chunk_size,
    )

    if not chunks:
        return []

    # ── Stage 10: Quality filtering + wrap ────────────────────
    records = [{"text": _normalise_text(chunk), "category": "法律法规"} for chunk in chunks]
    records = _filter_quality(records, min_len=min_chunk_len, max_len=max_chunk_len)
    records = _deduplicate(records)

    logger.info(
        "Regulation processing complete: %d records (format=%s, len=%d chars)",
        len(records),
        file_format,
        len(cleaned_text),
    )

    return records

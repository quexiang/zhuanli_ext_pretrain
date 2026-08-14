"""Standard document PDF processing (national/industry/local standards).

Multi-stage cleaning pipeline for pre-training data quality:
  1. Raw text extraction (PyMuPDF direct)
  2. Line-level noise removal (headers, footers, page numbers, covers)
  3. Multi-line footer block removal (consecutive header lines)
  4. Orphaned numbering attachment (e.g. "6.1.1\ncontent" → "6.1.1 content")
  5. Company/person name line removal
  6. Paragraph-level deduplication & watermark removal
  7. Line-break joining (reconstruct mid-sentence breaks)
  8. Chapter-aware segmentation with category inference
  9. Clause-based chunking with length control [min, max]
  10. Quality filtering (min length, noise ratio, boilerplate, dedup)
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ── Functional category definitions ──────────────────────────────────
# Categories based on content function, not document structure.
# Only 4 functional categories for model training quality.

FUNC_CATEGORY_NAMES = ["指标规格", "操作步骤", "判定规则", "术语解释"]
# No "通用" — every chunk forced into one of the 4 categories via best-match fallback.

# ── 指标规格: specifications with parameters, values, limits ──────────
_SPEC_KEYWORDS = [
    "pH 值", "pH值", "pH ",
    "浓度", "含量", "比例", "尺寸", "长度", "宽度", "高度",
    "厚度", "直径", "半径", "面积", "体积", "重量", "质量",
    "密度", "强度", "硬度", "粘度", "湿度", "温度", "压力",
    "电压", "电流", "功率", "频率", "速度", "坡度", "梯度",
    "模量", "弹性", "韧性", "耐久性", "耐腐蚀", "耐磨",
    "抗拉", "抗压", "抗弯", "抗剪", "抗扭",
    "不大于", "不超过", "不宜超过", "不应小于", "不得低于",
    "不应高于", "不宜高于", "不应大于", "不宜大于",
    "应不小于", "应不低于", "应不高于", "应不大于",
    "最大值", "最小值", "上限", "下限", "限值",
    "MPa", "kPa", "mm", "cm", "km", "g", "kg", "W", "kW", "Hz", "℃", "%",
    "正负", "允许偏差", "公差", "误差范围",
    "技术参数", "技术指标", "性能指标", "规格参数", "规格型号",
    "不得小于", "不得大于", "不宜超过",
    "不小于", "不大于", "不低于", "不高于",
    "应满足", "应符合", "应达到",
    "应比", "不得比", "不应比",
]

# ── 操作步骤: procedural/sequential instructions ──────────────────────
_STEP_KEYWORDS = [
    "首先", "然后", "接着", "随后", "最后", "最终",
    "依次", "逐步", "按顺序", "先行",
    "放入", "加入", "注入",
    "启动", "开启", "停止", "打开", "关闭",
    "取出", "卸下", "安装", "拆卸",
    "连接", "断开", "旋转", "调整", "设置", "校准",
    "清零", "复位", "切换",
    "浸泡", "搅拌", "混合", "涂布", "涂抹",
    "铺设", "压实", "振捣", "干燥", "固化",
    "加热", "冷却", "预热", "保温",
    "烘干", "风干", "晾干", "静置", "停放",
    "行驶", "运行", "维护", "保养", "检修",
    "试验", "检测", "检验", "测试", "测量", "测定",
    "监控", "监测", "保存", "储存", "存放", "保管",
    "维修", "改造", "更新", "报废", "处置", "处理",
    "清理", "清洁", "清扫", "清洗", "冲刷", "冲洗",
    "擦拭", "润滑", "紧固", "校验", "检定", "鉴定",
    "浇水", "施肥", "修剪", "浇灌", "灌溉",
    "检查", "查看", "观测",
    "养护", "管养", "打理",
    "养护",
]

# ── 判定规则: pass/fail criteria, sampling, evaluation ────────────────
_RULE_KEYWORDS = [
    "合格", "不合格",
    "判定", "判断", "评定", "评价", "评估",
    "验收", "拒收", "接收", "放行", "封锁",
    "停用", "返工", "返修", "让步", "降级",
    "抽样", "样本", "批量", "批次",
    "复验", "复检", "复查", "复核",
    "质量评定", "质量评价", "质量评估", "质量验收",
    "竣工验收", "过程验收", "隐蔽验收", "分段验收",
    "最终验收", "初步验收", "中间验收",
    "符合", "不符合", "满足", "不满足",
    "通过", "不通过",
    "评分", "打分", "评级", "分级", "分等",
]

# ── 术语解释: term definitions ───────────────────────────────────────
_TERM_KEYWORDS = [
    "是指", "指的是", "称为", "定义为",
    "术语和定义", "术语术语", "名词解释",
    "符号和缩略语", "符号说明",
    "以下简称", "以下简称之", "上述", "前述",
    "下文", "后文", "本规范中", "本标准中",
    "本文件中", "本要求中", "本规程中",
]

CHAPTER_CATEGORY_MAP: dict[int, str] = {
    1: "范围",
    3: "术语和定义",
    4: "分类",
}

SUBCHAPTER_RE = re.compile(r"^(\d+\.\d+\.\d+)\s+(.+)$")
CHAPTER_RE = re.compile(r"^(\d+)\s+(.+)$")

# Chapters to skip entirely (not useful for model training)
_SKIP_CHAPTER_TITLES: set[str] = {
    "范围", "前言", "引言", "目录", "目次", "规范性引用文件",
}

# ── Boilerplate / standard declaration patterns ─────────────────────

_BOILERPLATE_PATS = [
    r"本文件按照GB/T 1.1—\d{4}给出的规则起草",
    r"本文件代替GB/T \d+\.1—\d{4}",
    r"代替了GB \d+—\d{4}",
    r"代替了GB/T \d+\.1—\d{4}",
    r"代替了GB/T \d+\.2—\d{4}",
    r"代替了GB/T \d+\.3—\d{4}",
    r"代替了GB/T \d+\.4—\d{4}",
    r"本文件不适用于",
    r"实施之日起",
    r"提出并归口",
    r"起草单位",
    r"主要起草人",
    r"本文件版权归",
    r"所有 rights reserved",
    r"未经许可不得复制",
    r"请勿在经济上使用",
    r"仅供参考，以官方最新发布为准",
    r"仅供参考",
]
_BOILERPLATE_RE = re.compile(
    r"(?:" + "|".join(_BOILERPLATE_PATS) + r")"
)

# Standard declaration lines (entire-line removal)
_STANDARD_DECL_RE = re.compile(
    r"^\s*(?:请注意本文件的某些内容可能涉及专利|"
    r"本文件的某些内容可能涉及专利|"
    r"发布并实施|"
    r"代替原标准|"
    r"本标准代替|"
    r"本标准与GB/T 1.1—\d{4}一致|"
    r"本标准的某些内容可能涉及专利|"
    r"本标准的附录均为规范性附录|"
    r"本标准的附录均为资料性附录|"
    r")"
)


def _is_skip_chapter(title: str) -> bool:
    """Check if a chapter title should be skipped entirely."""
    clean = title.strip()
    if clean in _SKIP_CHAPTER_TITLES:
        return True
    return False


def _infer_category(title: str, chapter_number: str) -> str | None:
    """Infer structural category from chapter title and number.
    Returns None to filter out useless structural sections (范围, 前言, etc.).
    Does NOT classify content functionally — that happens per-chunk later.
    """
    title_clean = title.strip()

    if _is_skip_chapter(title_clean):
        return None

    if title_clean == "参考文献":
        return "参考文献"

    if "术语和定义" in title_clean or "术语" in title_clean:
        return "术语和定义"

    if "附录" in title_clean:
        return "附录"

    try:
        n = int(chapter_number)
    except (ValueError, TypeError):
        return None

    if n in CHAPTER_CATEGORY_MAP:
        return CHAPTER_CATEGORY_MAP[n]

    return None


def _infer_functional_category(text: str) -> str | None:
    """Classify text by its functional content into one of 4 categories.

    Returns None if text doesn't meet any category's threshold —
    such text is dropped rather than forced into a category.

    Priority: 术语解释 > 判定规则 > 指标规格 > 操作步骤
    """
    stripped = text.strip()

    def _keyword_count(keywords: list[str], text: str) -> int:
        return sum(1 for kw in keywords if kw in text)

    # ── 术语解释 ──────────────────────────────────────────────────────
    # Structural: X.X term (NOT X.X.X clause)
    term_struct = bool(re.match(r"^\s*\d+\.\d+\s+[一-鿿]{2,10}", stripped)
                       and not re.match(r"^\s*\d+\.\d+\.\d+", stripped))
    # Semantic keywords
    term_sem = _keyword_count(_TERM_KEYWORDS, text)
    if term_struct or term_sem >= 1:
        return "术语解释"

    # ── 判定规则: needs >= 2 hits ────────────────────────────────────
    rule_hits = _keyword_count(_RULE_KEYWORDS, text)
    if rule_hits >= 2:
        return "判定规则"

    # ── 指标规格: needs >= 2 hits (checked before 操作步骤) ──────────
    spec_hits = _keyword_count(_SPEC_KEYWORDS, text)
    if spec_hits >= 2:
        return "指标规格"

    # ── 操作步骤: needs >= 2 hits, OR "应+养护动作" pattern ──────────
    step_hits = _keyword_count(_STEP_KEYWORDS, text)
    action_pattern = bool(re.search(r"应\s*(?:浇水|施肥|修剪|浇灌|灌溉|清理|涂抹|干燥|固化|放置|设置)", text))
    if step_hits >= 2 or action_pattern:
        return "操作步骤"

    # ── No category matched → drop ───────────────────────────────────
    return None


# ═══════════════════════════════════════════════════════════════════
# Stage 1-2: Line-level cleaning
# ═══════════════════════════════════════════════════════════════════

_STD_HEADER_PATS = [
    re.compile(r"^\s*DB\d+/?\w*\s*[-—–\-→]\s*\d{4}\s*$"),
    re.compile(r"^\s*GB\s*/?\s*T?\s*\d+\s*[-—–]\s*\d{4}\s*$"),
    re.compile(r"^\s*GB\s+\d+\s*[-—–]\s*\d{4}\s*$"),
    re.compile(r"^\s*(?:JJF|HG|HJ|JC|JGJ|CJ|CJJ|CJ/T|SL|DL|TB|TY|NY|QB|JB|SH|SY|ZB)\s*/?\s*\w*\s*\d+\s*[-—–\-→]?\s*\d{4}\s*$"),
    re.compile(r"^\s*ICS\s+\d{1,2}\.\d{2}(?:\.\d{2})?\s*;\s*\d{1,2}\.\d{2}(?:\.\d{2})?\s*\*?\s*$"),
    re.compile(r"^\s*ICS\s+\d{1,2}\.\d{2}(?:\.\d{2})?\s*\*?\s*$"),
    re.compile(r"^\s*\d{1,2}\.\d{2}(?:\.\d{2})?\s*$"),
]

_cover_re_patterns = [
    r"^中华人民共和国$",
    r"^国家标准$",
    r"^行业标准$",
    r"^地方标准$",
    r"^企业标准$",
    r"^标准批准发布$",
    r"^标准实施$",
    r"^发布日期$",
    r"^实施日期$",
    r"^代替\s*\S+",
    r"^标准号$",
    r"^ICS\s+\d+",
    r"^UDC\s+\d+",
    r"^文档编号$",
    r"^文件编号$",
    r"^编号：",
    r"^编号：",
    r"^发布$",
    r"^批准$",
]
_cover_re = [re.compile(p) for p in _cover_re_patterns]

_PAGE_NUM_RE = re.compile(r"^\s*\d+\s*$")
_ROMAN_PAGE_RE = re.compile(r"^\s*[IVXLC]+\.?\s*$")
_ISOLATED_NUM_RE = re.compile(r"^\s*\d{1,6}\s*$")

_WATERMARK_RE = re.compile(
    r"^(?:征求意见稿|报批稿|试行|试验性实施|参考稿|草案|公开征求意见稿|"
    r"征求意见稿\.pdf|报批稿\.pdf)$",
)

_GARBLED_LINE_RE = re.compile(r"^[\s\*\-=_+／\\|》\.\，\。]{4,}$")
_SINGLE_PUNCT_RE = re.compile(r"^[\,\、\。\，\？\！\；\：\-\—\～\·]{1,2}$")
_SINGLE_CHAR_RE = re.compile(r"^[一-鿿]$")
_TABLE_CELL_RE = re.compile(r"^\s*\|\s*$")
_ORPHAN_SINGLE_LETTER_RE = re.compile(r"^[A-Za-z]$\s*$")
_SERIAL_RE = re.compile(r"^\d{4}-[A-Za-z0-9]+$")
_PARTIAL_PAT_NUM_RE = re.compile(r"^N\d{6,}$")

_COMPANY_SUFFIXES = (
    "有限公司", "集团公司", "有限责任公司", "股份公司",
    "分公司", "公司",
    "大学", "学院", "研究所", "研究院", "设计院", "设计所",
    "分局", "总队", "支队", "办公室", "委员会", "管理处",
    "集团", "厂", "局", "院", "所", "校", "社", "部", "处", "室", "中心",
    "集团", "企业", "协会", "学会", "商会", "研究会",
)

_PERSON_NAME_LINE_RE = re.compile(
    r"^\s*[一-鿿]{2,4}"
    r"(?:\s+[一-鿿]{2,4}){1,6}"
    r"\s*$"
)


def _is_cover_line(line: str) -> bool:
    stripped = line.strip()
    for pat in _cover_re:
        if pat.match(stripped):
            return True
    return False


def _is_standard_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    for pat in _STD_HEADER_PATS:
        if pat.match(stripped):
            return True
    return False


def _is_page_number(line: str) -> bool:
    stripped = line.strip()
    if _PAGE_NUM_RE.match(stripped):
        return True
    if _ROMAN_PAGE_RE.match(stripped):
        return True
    if _ISOLATED_NUM_RE.match(stripped) and len(stripped) <= 6:
        return True
    if re.match(r"^[\s\-\—\-－]{1,3}\d{1,4}[\-\—\-－]{1,3}\s*$", stripped):
        return True
    return False


def _is_watermark(line: str) -> bool:
    return bool(_WATERMARK_RE.match(line.strip()))


def _is_noise_line(line: str) -> bool:
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
    if re.match(r"^[\s\.\，\。\，\、\；\：\？\！\-\—\～\·]{2,}\s*$", stripped):
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
    if not stripped:
        return False
    for suffix in _COMPANY_SUFFIXES:
        if stripped.endswith(suffix) and len(stripped) > len(suffix) + 2:
            prefix = stripped[:-len(suffix)]
            if re.match(r"^[一-鿿（）()A-Za-z\d\.]+$", prefix):
                return True
    return False


def _is_person_name_line(line: str) -> bool:
    """Detect standalone person name lines (e.g. "张三 李四 王五")."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) < 4 or len(stripped) > 50:
        return False
    # Check if it looks like multiple names separated by spaces
    parts = stripped.split()
    if len(parts) < 2:
        return False
    for part in parts:
        if not re.match(r"^[一-鿿]{2,4}$", part):
            return False
    return True


def _is_toc_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.count(".") > 10:
        return True
    if re.search(r"\.{10,}\s+[IVX]+\.?\s*$", stripped):
        return True
    if re.search(r"\.{10,}\s+\d+\s*$", stripped):
        return True
    if re.search(r"\s{4,}\d+\s*$", stripped) and len(stripped) > 30:
        return True
    return False


def _clean_lines(text: str) -> str:
    """Stage 1-2: Line-level cleaning with multi-stage pipeline."""
    lines = text.split("\n")
    cleaned: list[str] = []

    for line in lines:
        stripped = line.strip()

        if _is_cover_line(stripped):
            continue
        if _is_watermark(stripped):
            continue
        if _is_standard_header(stripped):
            continue
        if _is_page_number(stripped):
            if len(stripped) <= 6 and not re.match(r"^\d+\s+[一-鿿]", stripped):
                continue
        if _is_noise_line(stripped):
            continue
        if _is_company_line(stripped):
            continue
        if _is_person_name_line(stripped):
            continue
        if _is_toc_line(stripped):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ═══════════════════════════════════════════════════════════════════
# Stage 3: Multi-line footer block removal
# ═══════════════════════════════════════════════════════════════════

_STD_FOOTER_PATTERNS = [
    re.compile(r"^\s*DB\d+/?\w*\s*[-—–\-→]\s*\d{4}\s*$"),
    re.compile(r"^\s*GB\s*/?\s*T?\s*\d+\s*[-—–]\s*\d{4}\s*$"),
    re.compile(r"^\s*GB\s+\d+\s*[-—–]\s*\d{4}\s*$"),
    re.compile(r"^\s*(?:JJF|HG|HJ|JC|JGJ|CJ|CJJ|CJ/T|SL|DL|TB|TY|NY|QB|JB|SH|SY)\s*/?\s*\w*\s*\d+\s*[-—–\-→]?\s*\d{4}\s*$"),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*页$"),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*頁$"),
]

_STANDARD_PAGE_COUNT_RE = re.compile(
    r"^\s*(?:\d+\s+页|标准\s*\d+\s+页|规范性引用文件\s*\d+\s+页)"
)


def _is_footer_line(line: str) -> bool:
    """Check if a single line matches standard footer patterns."""
    stripped = line.strip()
    if not stripped:
        return False
    for pat in _STD_FOOTER_PATTERNS:
        if pat.match(stripped):
            return True
    if _STANDARD_PAGE_COUNT_RE.match(stripped):
        return True
    return False


def _remove_footer_blocks(text: str) -> str:
    """Remove consecutive footer-line runs (blocks) of 2+ lines."""
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
            run_len = j - i
            if run_len >= 2:
                for k in range(i, j):
                    keep[k] = False
            i = j
            continue
        i += 1

    kept = [ln for ln, k in zip(lines, keep) if k]
    return "\n".join(kept)


# ═══════════════════════════════════════════════════════════════════
# Stage 4: Orphaned numbering attachment
# ═══════════════════════════════════════════════════════════════════

_ORPHANED_CLAUSE_RE = re.compile(r"^(\d+\.\d+\.\d+)\s*$")
_ORPHANED_SECTION_RE = re.compile(r"^(\d+)\s*$")


def _attach_orphaned_numbering(text: str) -> str:
    """Merge orphaned clause numbers onto the next line.

    e.g. "6.1.1\n内容" → "6.1.1 内容"
    e.g. "1\n范围" → "1 范围"
    """
    lines = text.split("\n")
    result: list[str] = []

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # Check if this is an orphaned clause number (e.g. "6.1.1")
        clause_m = _ORPHANED_CLAUSE_RE.match(stripped)
        section_m = _ORPHANED_SECTION_RE.match(stripped)

        if (clause_m or section_m) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and not _is_toc_line(next_line):
                # Attach to next line
                result.append(stripped + " " + next_line)
                i += 2
                continue

        result.append(lines[i])
        i += 1

    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════
# Stage 5-6: Paragraph deduplication & watermark removal
# ═══════════════════════════════════════════════════════════════════

def _remove_paragraph_duplicates(text: str, threshold: float = 0.85) -> str:
    """Remove near-duplicate consecutive paragraphs."""
    lines = text.split("\n")
    if len(lines) <= 1:
        return text

    def _similarity(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            return 0.0
        intersection = sa & sb
        union = sa | sb
        return len(intersection) / len(union) if union else 0.0

    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if kept and kept[-1].strip() and _similarity(stripped, kept[-1].strip()) >= threshold:
            continue
        kept.append(line)

    return "\n".join(kept)


# ═══════════════════════════════════════════════════════════════════
# Stage 7: Line-break joining
# ═══════════════════════════════════════════════════════════════════

def _is_section_boundary(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    # Clause numbering
    if re.match(r"^\d+\.\d+\.\d+\s+", stripped):
        return True
    # Section numbering
    if re.match(r"^\d+\s+[一-鿿]", stripped):
        return True
    # Chinese ordinal numbering
    if re.match(r"^[一二三四五六七八九十]+[、．.．]", stripped):
        return True
    # Structural chapter headers that should act as boundaries
    for header in _SKIP_CHAPTER_TITLES:
        if stripped == header:
            return True
    if stripped == "参考文献" or stripped == "附录" or stripped == "术语和定义":
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
                if _is_section_boundary(next_stripped):
                    result.append(current)
                    i += 1
                    continue
                if _is_section_boundary(stripped):
                    result.append(current)
                    i += 1
                    continue
                # If current line ends with sentence-ending punctuation, don't join
                if re.search(r"[。！？；]\s*$", stripped):
                    result.append(current)
                    i += 1
                    continue
                if re.search(r"[）\]）]\s*$", stripped):
                    result.append(current)
                    i += 1
                    continue
                # If current line ends with clause number, skip (already attached)
                if re.search(r"^\d+\.\d+\.\d+\s*$", stripped):
                    result.append(current)
                    i += 1
                    continue
                # Both lines are Chinese text and current doesn't end with sentence end
                if (
                    re.search(r"[一-鿿]", stripped)
                    and re.search(r"[一-鿿]", next_stripped)
                    and not re.search(r"[。！？；]\s*$", stripped)
                    and not re.match(r"^[A-Z0-9\[一二三四五六七八九十\d\(]", next_stripped)
                ):
                    result.append(current + next_stripped)
                    i += 2
                    continue

        result.append(current)
        i += 1

    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════
# Stage 8: Chapter-aware segmentation
# ═══════════════════════════════════════════════════════════════════

def _extract_chapters(text: str) -> list[dict]:
    """Parse standard document structure, extract chapters."""
    lines = text.split("\n")

    body_start = 0
    for i, line in enumerate(lines):
        if re.match(r"^1\s+范[围阐]?", line.strip()):
            body_start = i
            break

    body_lines = lines[body_start:]

    chapter_boundaries: list[dict] = []
    i = 0
    while i < len(body_lines):
        line = body_lines[i].strip()
        if not line:
            i += 1
            continue

        sub_m = SUBCHAPTER_RE.match(line)
        if sub_m:
            chapter_boundaries.append({
                "index": i, "level": 3,
                "number": sub_m.group(1), "title": sub_m.group(1),
            })
            i += 1
            continue

        ch_m = CHAPTER_RE.match(line)
        if ch_m and not any(b["index"] == i for b in chapter_boundaries):
            num = ch_m.group(1)
            title = ch_m.group(2).strip()
            if not _is_toc_line(line) and len(title) >= 2:
                chapter_boundaries.append({
                    "index": i, "level": 1,
                    "number": num, "title": title,
                })

        if not any(b["index"] == i for b in chapter_boundaries):
            if line.strip() == "参考文献":
                chapter_boundaries.append({
                    "index": i, "level": 1,
                    "number": "99", "title": "参考文献",
                })
            elif line.strip() in ("附", "附录") and i + 1 < len(body_lines):
                next_line = body_lines[i + 1].strip()
                if "录" in next_line or re.match(r"^[A-Z]", next_line):
                    num_m = re.match(r"^([A-Z])", line.strip())
                    num_val = num_m.group(1) if num_m else ""
                    title_text = f"附录{num_val}" if num_m else "附录"
                    chapter_boundaries.append({
                        "index": i, "level": 1,
                        "number": num_val if num_val else "99",
                        "title": title_text,
                    })
        i += 1

    if not chapter_boundaries:
        return [{"title": "全文", "content": text, "category": None,
                 "number": "0", "level": 1}]

    chapters: list[dict] = []
    for idx, b in enumerate(chapter_boundaries):
        start = b["index"]
        end = (chapter_boundaries[idx + 1]["index"]
               if idx + 1 < len(chapter_boundaries)
               else len(body_lines))

        content_lines = body_lines[start:end]
        content = "\n".join(content_lines).strip()
        content = re.sub(r"\n?\s*DB\d+[^\n]*\n?", "\n", content)
        content = re.sub(r"\n?\s*GB\s*/?\s*T?\s*\d+\s*[-—–]\s*\d{4}\s*\n?", "\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        chapters.append({
            "title": b["title"], "content": content,
            "number": b["number"], "level": b["level"],
        })

    return chapters


def _process_chapters_parallel(
    chapters: list[dict],
    min_chunk_len: int,
    max_chunk_len: int,
    chunk_size: int,
    max_workers: int,
) -> list[dict[str, str]]:
    """Process chapters in parallel using a thread pool.

    Thread-safe because all PyMuPDF I/O is done (text extracted into
    Python strings), and subsequent operations use only pure-Python
    string processing with no shared mutable state.
    """
    all_records: list[dict[str, str]] = []

    if len(chapters) <= 1 or max_workers <= 1:
        # Too few chapters or no parallelism — fall back to sequential
        return _process_chapters_sequential(chapters, min_chunk_len, max_chunk_len, chunk_size)

    def _process_single_chapter(chapter: dict) -> list[dict[str, str]]:
        """Process one chapter and return records."""
        content = chapter["content"]
        title = chapter["title"]
        number = chapter["number"]

        if not content or len(content) < 10:
            return []

        if not _has_substantive_content(content):
            return []

        category = _infer_category(title, number)
        if category is None:
            return []

        # Skip structural-only chapters (范围, 前言, 引言, 目录, 目次, 规范性引用文件)
        if _is_skip_chapter(title.strip()):
            return []

        # Special chapters
        if "术语和定义" in title or "术语" == title:
            defs = _extract_term_definitions(content)
            return [
                {"text": d, "category": "术语解释"}
                for d in defs
                if len(d) >= 20 and _has_substantive_content(d)
            ]

        if "参考文献" in title:
            refs = _extract_references(content)
            return [{"text": ref, "category": "参考文献"} for ref in refs if len(ref) >= 10]

        if "范围" == title.strip():
            return []

        if "附录" in title:
            chunks = _chunk_text(content, min_size=30, max_size=chunk_size * 2)
            return [
                {"text": chunk, "category": cat}
                for chunk in chunks
                if len(chunk) >= 30 and _has_substantive_content(chunk)
                and (cat := _infer_functional_category(chunk)) is not None
            ]

        # General chapters — classify by content function
        appendix_parts = _split_at_appendix(content)
        records: list[dict[str, str]] = []
        for part_title, part_content in appendix_parts:
            if "附录" in part_title:
                chunks = _chunk_text(part_content, min_size=30, max_size=chunk_size * 2)
                for chunk in chunks:
                    if len(chunk) >= 30 and _has_substantive_content(chunk):
                        cat = _infer_functional_category(chunk)
                        if cat is not None:
                            records.append({"text": chunk, "category": cat})
            elif "参考文献" in part_title:
                refs = _extract_references(part_content)
                for ref in refs:
                    if len(ref) >= 10:
                        records.append({"text": ref, "category": "参考文献"})
            else:
                chunks = _chunk_text(part_content, min_size=min_chunk_len, max_size=max_chunk_len)
                for chunk in chunks:
                    if len(chunk) >= min_chunk_len and _has_substantive_content(chunk):
                        if _noise_ratio(chunk) < 0.3:
                            func_cat = _infer_functional_category(chunk)
                            if func_cat is not None:
                                records.append({"text": chunk, "category": func_cat})
        return records

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_chapter, ch): ch for ch in chapters}
        for future in as_completed(futures):
            try:
                all_records.extend(future.result())
            except Exception as exc:
                logger.error("Chapter processing failed: %s", exc)

    return all_records


def _process_chapters_sequential(
    chapters: list[dict],
    min_chunk_len: int,
    max_chunk_len: int,
    chunk_size: int,
) -> list[dict[str, str]]:
    """Sequential chapter processing (fallback for small documents)."""
    all_records: list[dict[str, str]] = []
    for chapter in chapters:
        content = chapter["content"]
        title = chapter["title"]
        number = chapter["number"]

        if not content or len(content) < 10:
            continue

        if not _has_substantive_content(content):
            continue

        category = _infer_category(title, number)
        if category is None:
            continue

        # Skip structural-only chapters (范围, 前言, 引言, 目录, 目次, 规范性引用文件)
        if _is_skip_chapter(title.strip()):
            continue

        # Special chapters
        if "术语和定义" in title or "术语" == title:
            defs = _extract_term_definitions(content)
            for d in defs:
                if len(d) >= 20 and _has_substantive_content(d):
                    all_records.append({"text": d, "category": "术语解释"})
            continue

        if "参考文献" in title:
            refs = _extract_references(content)
            for ref in refs:
                if len(ref) >= 10:
                    all_records.append({"text": ref, "category": "参考文献"})
            continue

        if "范围" == title.strip():
            continue

        if "附录" in title:
            chunks = _chunk_text(content, min_size=30, max_size=chunk_size * 2)
            for chunk in chunks:
                if len(chunk) >= 30 and _has_substantive_content(chunk):
                    cat = _infer_functional_category(chunk)
                    if cat is not None:
                        all_records.append({"text": chunk, "category": cat})
            continue

        # General chapters — classify by content function
        appendix_parts = _split_at_appendix(content)

        for part_title, part_content in appendix_parts:
            if "附录" in part_title:
                chunks = _chunk_text(part_content, min_size=30, max_size=chunk_size * 2)
                for chunk in chunks:
                    if len(chunk) >= 30 and _has_substantive_content(chunk):
                        cat = _infer_functional_category(chunk)
                        if cat is not None:
                            all_records.append({"text": chunk, "category": cat})
            elif "参考文献" in part_title:
                refs = _extract_references(part_content)
                for ref in refs:
                    if len(ref) >= 10:
                        all_records.append({"text": ref, "category": "参考文献"})
            else:
                chunks = _chunk_text(part_content, min_size=min_chunk_len, max_size=max_chunk_len)
                for chunk in chunks:
                    if len(chunk) >= min_chunk_len and _has_substantive_content(chunk):
                        if _noise_ratio(chunk) < 0.3:
                            func_cat = _infer_functional_category(chunk)
                            if func_cat is not None:
                                all_records.append({"text": chunk, "category": func_cat})

    return all_records


# ═══════════════════════════════════════════════════════════════════
# Stage 9: Clause-based chunking with length control
# ═══════════════════════════════════════════════════════════════════

def _chunk_text(text: str, min_size: int = 50, max_size: int = 200) -> list[str]:
    """Split text into chunks based on clause numbering with length bounds."""
    # Remove residual standard number patterns
    text = re.sub(r"\n?\s*DB\d+/?\w*\s*[-—–\-→]\s*\d{4}[^\n]*\n?", "\n", text)
    for prefix in ["JJF", "HG", "HJ", "JC", "JGJ", "CJ", "CJJ", "CJ/T",
                   "SL", "DL", "TB", "TY", "NY", "DB", "GB", "QB", "SH",
                   "SY", "JB", "YS", "WR", "JT", "MT", "LD", "YZ", "WS",
                   "YY"]:
        text = re.sub(
            rf"\n?\s*{prefix}\s*/?\s*\w*\s*\d+\s*[-—–\-→]?\s*\d{{4}}\s*\n?",
            "\n", text,
        )
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    parts = re.split(r"(\d+\.\d+\.\d+\s+)", text)

    chunks: list[str] = []
    i = 0
    while i < len(parts):
        if not parts[i]:
            i += 1
            continue
        if re.match(r"^\d+\.\d+\.\d+\s+$", parts[i]):
            chunk = parts[i]
            i += 1
            while i < len(parts):
                if re.match(r"^\d+\.\d+\.\d+\s+$", parts[i]):
                    break
                chunk += parts[i]
                i += 1
            chunks.append(chunk.strip())
        else:
            chunks.append(parts[i].strip())
            i += 1

    # Merge short chunks
    merged: list[str] = []
    buffer = ""
    for chunk in chunks:
        if len(chunk) < min_size:
            buffer += " " + chunk
        else:
            if buffer:
                merged.append((buffer + " " + chunk).strip())
                buffer = ""
            else:
                merged.append(chunk)

    if buffer:
        merged.append(buffer.strip())

    # Remove trailing sub-headers from chunks
    sub_header_patterns = [
        "屋顶绿化", "架空层绿化", "墙面（体）绿化", "墙面绿化",
        "棚架绿化", "桥体绿化", "窗阳台绿化", "硬质边坡绿化",
        "围墙栅栏绿化", "驳岸绿化",
        "基础结构建造", "种植施工", "设施维护", "植物管养",
        "安全技术措施",
    ]
    cleaned: list[str] = []
    for chunk in merged:
        for pattern in sub_header_patterns:
            match = re.search(re.escape(pattern) + r"\s*$", chunk)
            if match:
                before = chunk[:match.start()].rstrip()
                if before:
                    chunk = before
                break
        if len(chunk.strip()) >= min_size:
            cleaned.append(chunk.strip())

    # Split chunks that exceed max_size
    final: list[str] = []
    for chunk in cleaned:
        if len(chunk) > max_size:
            final.extend(_split_long_text(chunk, max_size))
        else:
            final.append(chunk)

    return final


def _split_long_text(text: str, max_len: int) -> list[str]:
    """Split text at clause or sentence boundaries."""
    # Try clause number boundaries first
    parts = re.split(r"(\n?\s*\d+\.\d+\.\d+\s+)", text)
    if len(parts) >= 2:
        chunks: list[str] = []
        i = 0
        while i < len(parts):
            if not parts[i]:
                i += 1
                continue
            if re.match(r"\s*\d+\.\d+\.\d+\s+", parts[i]):
                chunk = parts[i]
                i += 1
                while i < len(parts) and not re.match(r"\s*\d+\.\d+\.\d+\s+", parts[i]):
                    chunk += parts[i]
                    i += 1
                if len(chunk.strip()) > max_len:
                    chunks.extend(_split_long_text_fallback(chunk.strip(), max_len))
                else:
                    chunks.append(chunk.strip())
            else:
                chunks.append(parts[i].strip())
                i += 1
        return chunks

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


def _extract_term_definitions(content: str) -> list[str]:
    """Extract term definitions from a terms chapter."""
    lines = content.split("\n")
    definitions: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        term_match = re.match(r"^(\d+\.\d+)\s+(.+)$", line)
        if term_match:
            def_lines = [line]
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if re.match(r"^\d+\.\d+\s+", next_line):
                    break
                if next_line:
                    def_lines.append(next_line)
                j += 1
            definitions.append(" ".join(def_lines).strip())
            i = j
        else:
            i += 1

    return definitions


def _extract_references(content: str) -> list[str]:
    """Extract standard references from a references chapter."""
    lines = content.split("\n")
    references: list[str] = []
    current_ref = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_ref:
                references.append(current_ref.strip())
                current_ref = ""
            continue
        if re.match(
            r"^(?:GB(?:/T)?|GB\s|JJF|HG|HJ|JC|JGJ|CJ|CJJ|CJ/T|"
            r"SL|DL|TB|TY|NY|QB|JB|SH|SY|YS|WR|JT|MT|LD|YZ|WS|YY)\b", stripped,
        ):
            if current_ref:
                references.append(current_ref.strip())
            current_ref = stripped
        elif current_ref:
            current_ref += " " + stripped

    if current_ref:
        references.append(current_ref.strip())

    return references


def _split_at_appendix(content: str) -> list[tuple[str, str]]:
    """Split content at appendix/references boundaries."""
    appendix_pattern = re.compile(
        r"[A-Z]\s*\n[A-Z]?\s*\n附\s*\n录\s*\n[A-Z]?\s*\n（资料性）"
    )
    ref_pattern = re.compile(r"\n+参考文献")

    splits: list[tuple[int, str]] = []
    for pattern in [appendix_pattern, ref_pattern]:
        for m in pattern.finditer(content):
            splits.append((m.start(), m.group()))

    if not splits:
        return [("全文", content)]

    splits.sort(key=lambda x: x[0])

    parts: list[tuple[str, str]] = []
    prev_end = 0
    for pos, marker in splits:
        if pos > prev_end:
            title = "全文" if not parts else parts[-1][0]
            parts.append((title, content[prev_end:pos].strip()))
        prev_end = pos

    parts.append(("剩余", content[prev_end:].strip()))
    parts = [(t, c) for t, c in parts if len(c) > 10]
    return parts if parts else [("全文", content)]


# ═══════════════════════════════════════════════════════════════════
# Stage 10: Quality filtering
# ═══════════════════════════════════════════════════════════════════

def _noise_ratio(text: str) -> float:
    """Calculate ratio of noise characters (non-CJK, non-alphanumeric)."""
    if not text:
        return 1.0
    meaningful = len(re.findall(r"[一-鿿a-zA-Z0-9，。！？；：、,.!?;:\-—–（）()\[\]【】{}《》<>]", text))
    return 1.0 - (meaningful / len(text))


def _has_substantive_content(text: str) -> bool:
    """Check if text has substantive (non-boilerplate) content."""
    stripped = text.strip()
    if len(stripped) < 10:
        return False

    if re.search(r"[一-鿿]", stripped):
        return True
    if re.search(r"\d+\.\d+\.\d+\s+", stripped):
        return True
    if re.search(r"(?:规定|要求|应|不应|宜|不宜|不得|可|必须|允许|禁止)", stripped):
        return True
    return False


def _has_boilerplate(text: str) -> bool:
    """Check if text contains standard boilerplate."""
    return bool(_BOILERPLATE_RE.search(text)) or bool(_STANDARD_DECL_RE.search(text))


def _filter_quality(
    records: list[dict[str, str]],
    min_len: int = 50,
    max_len: int = 2000,
) -> list[dict[str, str]]:
    """Final quality filtering: min length, noise ratio, boilerplate."""
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

        # Skip boilerplate-only records (but keep those with technical content)
        if _has_boilerplate(text):
            if not re.search(
                r"(?:规定|要求|应|不应|宜|不宜|不得|可|必须|允许|禁止|"
                r"[一-鿿]{10,}|\d+\.\d+\.\d+)", text
            ):
                continue

        result.append({"text": text, "category": rec.get("category", "")})

    return result


def _deduplicate(records: list[dict[str, str]], threshold: float = 0.9) -> list[dict[str, str]]:
    """Remove near-duplicate records using character bigram similarity."""
    seen: list[str] = []
    kept: list[dict[str, str]] = []

    for rec in records:
        text = rec.get("text", "").strip()
        if not text:
            continue

        is_dup = False
        for existing_text in seen:
            if _similarity(text, existing_text) >= threshold:
                is_dup = True
                break

        if not is_dup:
            seen.append(text)
            kept.append(rec)

    return kept


def _similarity(a: str, b: str) -> float:
    """Simple Jaccard similarity on character bigrams."""
    if not a or not b:
        return 0.0

    def _bigrams(s: str) -> set:
        return set(s[i:i+2] for i in range(len(s)-1))

    sa, sb = _bigrams(a), _bigrams(b)
    if not sa or not sb:
        return 0.0
    intersection = sa & sb
    union = sa | sb
    return len(intersection) / len(union) if union else 0.0


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def process_pdf_bytes(
    pdf_bytes: bytes,
    chunk_size: int = 150,
    min_chunk_len: int = 50,
    max_chunk_len: int = 2000,
    max_workers: int = 1,
) -> list[dict[str, str]]:
    """Full pipeline: extract → clean → segment → chunk → quality filter.

    Uses adaptive parallel processing for chapter-level chunking when
    the document has multiple chapters and ``max_workers`` > 1.

    Args:
        pdf_bytes: Raw PDF file bytes.
        chunk_size: Target chunk size in characters.
        min_chunk_len: Minimum allowed chunk length.
        max_chunk_len: Maximum allowed chunk length.
        max_workers: Number of worker threads for chapter processing.
                     Set to 1 for serial processing, > 1 for parallel.

    Returns:
        List of {"text": "...", "category": "..."} records.
    """
    # ── Stage 1: Raw text extraction ──────────────────────────
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    raw_texts = [page.get_text("text") for page in doc]
    doc.close()

    # ── Stage 2: Line-level cleaning ──────────────────────────
    cleaned_pages = [_clean_lines(t) for t in raw_texts if _clean_lines(t).strip()]
    full_text = "\n\n".join(cleaned_pages)

    if not full_text or len(full_text) < 100:
        return []

    # ── Stage 3: Multi-line footer block removal ──────────────
    full_text = _remove_footer_blocks(full_text)

    # ── Stage 4: Orphaned numbering attachment ────────────────
    full_text = _attach_orphaned_numbering(full_text)

    # ── Stage 5-6: Paragraph deduplication ───────────────────
    full_text = _remove_paragraph_duplicates(full_text)

    # ── Stage 7: Line-break joining ───────────────────────────
    full_text = _join_line_breaks(full_text)

    # ── Stage 8: Chapter segmentation ─────────────────────────
    chapters = _extract_chapters(full_text)

    # ── Stages 9-10: Chunking + quality filtering (parallel-aware) ──
    all_records = _process_chapters_parallel(
        chapters,
        min_chunk_len=min_chunk_len,
        max_chunk_len=max_chunk_len,
        chunk_size=chunk_size,
        max_workers=max_workers,
    )

    # ── Stage 10: Quality filtering ───────────────────────────
    all_records = _filter_quality(all_records, min_len=min_chunk_len, max_len=max_chunk_len)

    # ── Final deduplication ───────────────────────────────────
    all_records = _deduplicate(all_records)

    return all_records


def process_pdf(
    pdf_path: str | Path,
    chunk_size: int = 150,
    max_workers: int = 1,
) -> list[dict[str, str]]:
    """Process a single standard PDF from disk."""
    pdf_path = Path(pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return process_pdf_bytes(
        pdf_bytes, chunk_size=chunk_size, max_workers=max_workers
    )

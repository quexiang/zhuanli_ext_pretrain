"""Unit tests for the patent-text cleaning pipeline."""

import re
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.text_processor import (
    clean_text,
    process_document,
    _join_ocr_line_breaks,
    _attach_orphaned_numbering,
    _remove_footer_blocks,
    _remove_claims_footer_blocks,
    _remove_noise_fragments,
    _is_footer_line,
)


# ═══════════════════════════════════════════════════════════════════
# Footer removal
# ═══════════════════════════════════════════════════════════════════
class TestFooterRemoval:
    def test_cn_page_garbled_block(self):
        """CN + garbled 说明书 + page number footer block."""
        text = "收和管理底线价格信息、自动成交。\nCN 102955986 A\n机明节\n1/3页\n电子招投标管理系统"
        result = clean_text(text)
        assert "CN 102955986 A" not in result
        assert "机明节" not in result
        assert "1/3页" not in result
        assert "电子招投标管理系统" in result

    def test_cn_page_clean_block(self):
        """CN + 说明书 + page number (proper OCR)."""
        text = "正文内容。\nCN 102324080 A\n说明书\n4/6页\n继续正文。"
        result = clean_text(text)
        assert "CN 102324080 A" not in result
        assert "说明书" not in result
        assert "4/6页" not in result

    def test_garbled_cn_only(self):
        """Garbled CN number on its own."""
        text = "前面的话。\nNN 017 A\n后面的话。"
        result = clean_text(text)
        assert "NN 017" not in result

    def test_garbled_two_line(self):
        """NN pattern + properly formatted CN (two-line footer)."""
        text = "文本内容。\n全N 1999191 A\nCN 102799991 A"
        result = clean_text(text)
        assert "全N" not in result
        assert "CN 102799991 A" not in result

    def test_single_footer_line_preserved_if_alone(self):
        """A single CN line with no page footer is kept (it might be inline)."""
        text = "一些包含 CN 102324080 A 的文字。"
        result = clean_text(text)
        assert "CN" in result  # inline mention should survive

    def test_three_consecutive_footer_lines(self):
        """All three variants in a row are removed as a block."""
        text = "正文前。\nCN 102324080 A\n玩明书\n2/6页\n正文后。"
        result = clean_text(text)
        assert "玩明书" not in result
        assert "CN 102324080 A" not in result
        assert "2/6页" not in result
        assert "正文后" in result


# ═══════════════════════════════════════════════════════════════════
# 权利要求书 disambiguation
# ═══════════════════════════════════════════════════════════════════
class TestClaimsFooterDisambiguation:
    def test_claims_footer_removed(self):
        """权利要求书 as page footer (followed by page number only)."""
        text = "5. 根据权利要求1所述的系统。\n权利要求书\n2/2页\n6. 根据权利要求1所述的系统。"
        result = clean_text(text)
        assert "权利要求书" not in result
        assert "2/2页" not in result
        assert "6. 根据权利要求1" in result

    def test_claims_footer_with_cn_page(self):
        """权利要求书 + CN + page footer in middle of claims."""
        text = "8. 如权利要求1所述的系统。\n权利要求书\nCN 102324080 A\n2/2页\n9. 如权利要求2所述的系统。"
        result = clean_text(text)
        assert "权利要求书" not in result
        assert "9. 如权利要求2" in result

    def test_claims_header_preserved(self):
        """权利要求书 as section header (followed by claim numbering)."""
        text = "摘要内容。\n权利要求书\nCN 102955986 A\n1/1页\n1. 一种招投标系统。"
        result = clean_text(text)
        assert "1. 一种招投标系统" in result


# ═══════════════════════════════════════════════════════════════════
# Line joining (OCR mid-sentence break reconstruction)
# ═══════════════════════════════════════════════════════════════════
class TestLineJoining:
    def test_join_lowercase_continuation(self):
        """Line starting with lowercase joins to previous."""
        text = "本发明涉及一种电子招投标管理系统，其包\n括招投标基本信息管理模块。"
        result = clean_text(text)
        assert "包括" in result.split("。")[0]

    def test_join_chinese_mid_sentence(self):
        """Mid-sentence Chinese OCR line break joining."""
        text = "本申请公开了一种招投标一体化系统及招\n标方法。"
        result = clean_text(text)
        assert "招标方法" in result
        assert "招\n标方法" not in result

    def test_no_join_across_section_headers(self):
        """Section headers are NOT joined."""
        text = "摘要内容。\n技术领域\n本发明涉及..."
        result = clean_text(text)
        assert "技术领域" in result

    def test_no_join_across_numbering(self):
        """Numbered items are NOT joined."""
        text = "前面内容\n[0001] 一种系统\n[0002] 如权利要求1所述"
        result = clean_text(text)
        assert "[0001]" in result and "[0002]" in result

    def test_no_join_sentence_boundary(self):
        """No join across sentence-ending punctuation."""
        text = "完成招标。\n注册用户分类"
        result = clean_text(text)
        assert "。注册" not in result

    def test_no_join_footer_lines(self):
        """Footer lines are NOT joined with content."""
        text = "正文。\nCN 102324080 A\n说明书\n4/6页\n继续正文。"
        result = clean_text(text)
        # After footer removal, CN/说明书/4/6页 should be gone
        assert "说明书" not in result


# ═══════════════════════════════════════════════════════════════════
# Orphaned numbering attachment
# ═══════════════════════════════════════════════════════════════════
class TestOrphanedNumbering:
    def test_attach_numbering(self):
        """[0055] on its own line attaches to next line."""
        result = clean_text("[0055]\n候选人公示阶段：")
        assert "[0055] 候选人公示阶段：" in result

    def test_multiple_orphaned(self):
        """Multiple orphaned numbering lines are all attached."""
        text = "[0055]\n内容1。\n[0056]\n内容2。"
        result = clean_text(text)
        assert "[0055] 内容1" in result
        assert "[0056] 内容2" in result


# ═══════════════════════════════════════════════════════════════════
# Noise removal
# ═══════════════════════════════════════════════════════════════════
class TestNoiseRemoval:
    def test_remove_single_letter(self):
        """Single letter X on its own line removed."""
        text = "(19) 国家知识产权局\nX\n(12) 发明专利申请"
        result = clean_text(text)
        lines = result.split("\n")
        assert not any(l.strip() == "X" for l in lines)

    def test_remove_partial_patent_number(self):
        """Partial patent number N1110700 removed."""
        text = "生成评审报告。\nN1110700\n审报告。"
        result = clean_text(text)
        assert "N1110700" not in result

    def test_remove_serial_number(self):
        """Serial number 1101-194T20200528 removed."""
        text = "前面\n1101-194T20200528\n后面"
        result = clean_text(text)
        assert "1101-194T20200528" not in result

    def test_preserve_short_meaningful_content(self):
        """Short but meaningful lines (图1) are preserved."""
        text = "[0052] 图1 为本发明招投标系统的组成示意图。"
        result = clean_text(text)
        assert "图1" in result


# ═══════════════════════════════════════════════════════════════════
# Segmentation quality
# ═══════════════════════════════════════════════════════════════════
class TestSegmentation:
    def test_claims_content_intact(self):
        """权利要求书 section content remains after footer cleanup."""
        text = (
            "发明内容\n"
            "本发明实施例提供了一种招投标系统及方法，该系统包括交易平台、投标客户端和评标终端，"
            "通过结构化招标文件和自动化评审流程，实现了招投标全流程的电子化和智能化管理，"
            "有效提高了招投标效率并降低了人为干预风险。\n"
            "权利要求书\nCN 102955986 A\n1/1页\n"
            "1. 一种招投标系统，其特征在于，包括交易平台和投标客户端，所述交易平台包括招标单元、投标单元、开标单元和评标单元；"
            "所述招标单元用于提供多个标书模块模板，根据招标人的选择结果生成结构化招标文件；"
            "所述投标单元用于供潜在投标人下载招标文件并提交投标文件；"
            "所述开标单元用于对投标文件进行解密和开标操作。"
        )
        records = process_document(text)
        assert len(records) >= 1
        combined = " ".join(r["text"] for r in records)
        assert "1." in combined

    def test_segment_length_bounds(self):
        """All segments respect [120, 2000] bound."""
        body = "\n".join(
            f"[{i:04d}] 本发明涉及一种系统。该系统包括多个部件。"
            for i in range(1, 30)
        )
        records = process_document(body, min_len=120, max_len=2000)
        for rec in records:
            assert 120 <= len(rec["text"]) <= 2000, (
                f"Segment length {len(rec['text'])} out of bounds"
            )

    def test_short_nobid_filtered(self):
        """Records under 120 chars without bidding keywords are removed."""
        records = process_document("这是一个无关的简短句子。")
        # Should be filtered out (under 100 chars, no bidding keywords)
        assert len(records) == 0

    def test_short_even_with_bid_discarded(self):
        """Short records under 120 chars discarded even with bidding keywords."""
        text = "招标\n本发明涉及一种招投标系统。"
        records = process_document(text)
        assert len(records) == 0

    def test_preserve_section_integrity(self):
        """Section content not broken by footer removal."""
        text = (
            "具体实施方式\n"
            "[0001] 本发明涉及一种系统。\n"
            "[0002] 该系统包括：\n"
            "说明书\n"
            "CN 102324080 A\n"
            "4/6页\n"
            "[0003] 还包括："
        )
        result = clean_text(text)
        assert "4/6页" not in result
        assert "[0002]" in result
        assert "[0003]" in result


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════
class TestEdgeCases:
    def test_empty_input(self):
        """Empty string returns empty list."""
        assert process_document("") == []

    def test_pure_noise_input(self):
        """Only noise returns empty (or minimal)."""
        text = "X\nNN 017 A\nN1110700\n机明节\n1/3页"
        records = process_document(text)
        assert len(records) == 0

    def test_shuomingshu_garbled_variants(self):
        """All garbled 说明书 variants are caught."""
        for variant in ["机明节", "玩明书", "讥明节", "况明书"]:
            text = f"正文前。\nCN 102324080 A\n{variant}\n2/6页\n正文后。"
            result = clean_text(text)
            assert variant not in result, f"{variant} was not removed"

    def test_chinese_punctuation_boundary(self):
        """Chinese sentence boundaries are respected during line joining."""
        text = "步骤一：打开系统。\n步骤二：输入信息。"
        result = clean_text(text)
        # Since both lines end with 。, they should stay as separate lines
        assert "步骤一：打开系统。" in result
        assert "步骤二：输入信息。" in result

    def test_real_world_blend(self):
        """Real-world noise pattern from actual OCR output."""
        text = (
            "[0037] 行政监督平台包含一个子系统，即招投标监管子系统。\n"
            "况明书\n"
            "CN 107392570 A\n"
            "7/7页\n"
            "[0038] 投标、开标、评标的全过程。"
        )
        result = clean_text(text)
        assert "[0037]" in result
        assert "[0038]" in result
        assert "况明书" not in result

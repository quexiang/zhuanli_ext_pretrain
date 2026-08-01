"""Tests for standard_processor multi-stage cleaning pipeline."""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "..")

from app.services.standard_processor import (
    _clean_lines,
    _chunk_text,
    _extract_term_definitions,
    _extract_references,
    _has_substantive_content,
    _infer_category,
    _is_toc_line,
    _join_line_breaks,
    _noise_ratio,
    _remove_paragraph_duplicates,
    _split_at_appendix,
)


class TestCleanLines(unittest.TestCase):
    def test_removes_standard_header(self):
        text = "DB5101/T 214—2025\nSome content"
        result = _clean_lines(text)
        self.assertNotIn("DB5101", result)
        self.assertIn("Some content", result)

    def test_removes_page_number(self):
        text = "Some content\n123\nMore content"
        result = _clean_lines(text)
        self.assertNotIn("123", result)
        self.assertIn("Some content", result)

    def test_removes_toc_line(self):
        text = "1  范围 ............................................................................... 1"
        result = _clean_lines(text)
        self.assertEqual(result.strip(), "")

    def test_removes_watermark(self):
        text = "征求意见稿\nActual content"
        result = _clean_lines(text)
        self.assertNotIn("征求意见稿", result)
        self.assertIn("Actual content", result)

    def test_removes_cover_line(self):
        text = "中华人民共和国\n国家标准\nActual content"
        result = _clean_lines(text)
        self.assertNotIn("中华人民共和国", result)
        self.assertIn("Actual content", result)

    def test_keeps_chapter_number(self):
        text = "1 范围\nSome content"
        result = _clean_lines(text)
        self.assertIn("范围", result)


class TestLineBreakJoining(unittest.TestCase):
    def test_joins_broken_chinese(self):
        text = "本标准规定了城市绿地"\
               "\n分类与评价"
        result = _join_line_breaks(text)
        # Should have fewer lines
        self.assertLess(len(result.split("\n")), len(text.split("\n")))

    def test_does_not_join_across_sections(self):
        text = "1 范围\n2 术语和定义"
        result = _join_line_breaks(text)
        self.assertIn("1 范围", result)
        self.assertIn("2 术语和定义", result)


class TestParagraphDedup(unittest.TestCase):
    def test_removes_near_dups(self):
        text = "This is a test sentence.\n" + \
               "This is a test sentence.\n" + \
               "Different content."
        result = _remove_paragraph_duplicates(text)
        # Near-duplicate should be removed
        self.assertEqual(result.count("This is a test sentence"), 1)


class TestChunkText(unittest.TestCase):
    def test_chunks_by_clause_number(self):
        text = ("6.1.1 种植土应疏松透气，含水量不大于15%。\n"
                "6.1.2 种植土pH值应在6.5至7.5之间。")
        chunks = _chunk_text(text)
        self.assertGreater(len(chunks), 0)

    def test_merges_short_chunks(self):
        text = "短内容\n更短\n较长的内容"
        chunks = _chunk_text(text, min_size=5)
        self.assertGreater(len(chunks), 0)


class TestInferCategory(unittest.TestCase):
    def test_range(self):
        self.assertEqual(_infer_category("1 范围", "1"), "范围")

    def test_terms(self):
        self.assertEqual(_infer_category("术语和定义", "3"), "术语和定义")

    def test_filtered(self):
        self.assertIsNone(_infer_category("前言", "0"))
        self.assertIsNone(_infer_category("引言", "0"))
        self.assertIsNone(_infer_category("规范性引用文件", "2"))

    def test_default(self):
        self.assertEqual(_infer_category("技术要求", "6"), "技术要求")


class TestSubstantiveContent(unittest.TestCase):
    def test_has_content(self):
        self.assertTrue(_has_substantive_content("本标准规定了种植土的物理化学性能要求"))

    def test_no_content_short(self):
        self.assertFalse(_has_substantive_content("abc"))

    def test_no_content_noise(self):
        self.assertFalse(_has_substantive_content("***&&&!!!"))


class TestNoiseRatio(unittest.TestCase):
    def test_clean_text(self):
        ratio = _noise_ratio("本标准规定了城市绿地分类")
        self.assertLess(ratio, 0.1)

    def test_garbled_text(self):
        ratio = _noise_ratio("****&&&&&&!!!!")
        self.assertGreater(ratio, 0.5)


class TestTermDefinitions(unittest.TestCase):
    def test_extract_terms(self):
        content = ("3.1 城市绿地 urban green space\n"
                   "城市各类绿地范围的总称。\n"
                   "3.2 生产绿地 production green space\n"
                   "为城市绿化提供苗木、花草、种子的苗圃。")
        defs = _extract_term_definitions(content)
        self.assertGreaterEqual(len(defs), 2)
        self.assertIn("3.1", defs[0])
        self.assertIn("3.2", defs[1])


class TestReferences(unittest.TestCase):
    def test_extract_refs(self):
        content = ("GB 50001—2024 房屋建筑制图统一标准\n"
                   "GB/T 50083—2024 建筑工程结构设计规范\n"
                   "HJ 2.1—2016 建设项目环境影响评价技术导线")
        refs = _extract_references(content)
        self.assertGreaterEqual(len(refs), 3)
        self.assertIn("GB 50001", refs[0])


class TestAppendixSplit(unittest.TestCase):
    def test_splits_at_appendix(self):
        content = "正文内容...\nA\nA\n附\n录\nA\n（资料性）\n附录内容"
        parts = _split_at_appendix(content)
        self.assertGreaterEqual(len(parts), 1)


if __name__ == "__main__":
    unittest.main()

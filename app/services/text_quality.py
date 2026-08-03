"""文本质量检查工具：判断提取的文本是否为有效中文内容。"""

import re
from app.config import settings


def calculate_cjk_ratio(text: str) -> float:
    """计算文本中 CJK（中日韩）字符占比。

    CJK 范围：
    - 中文汉字: \u4e00-\u9fff (基本汉字)
    - 中文扩展: \u3400-\u4dbf (扩展A)
    - 兼容汉字: \uf900-\ufaff
    - 日文: \u3040-\u30ff
    - 韩文: \uac00-\ud7af
    """
    if not text:
        return 0.0

    cjk_pattern = re.compile(
        r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]'
    )
    cjk_count = len(cjk_pattern.findall(text))
    total = len(text)
    return cjk_count / total if total > 0 else 0.0


def is_valid_text_content(text: str) -> tuple[bool, str]:
    """检查文本是否为有效内容，返回 (是否有效, 原因说明)。

    检查条件：
    1. 非空
    2. 长度 >= min_valid_text_length
    3. CJK 占比 >= cjk_ratio_threshold
    """
    stripped = text.strip()

    if not stripped:
        return False, "文本为空"

    if len(stripped) < settings.min_valid_text_length:
        return False, f"文本过短（{len(stripped)} < {settings.min_valid_text_length}）"

    cjk_ratio = calculate_cjk_ratio(stripped)
    if cjk_ratio < settings.cjk_ratio_threshold:
        return False, (
            f"CJK 占比过低（{cjk_ratio*100:.1f}% < {settings.cjk_ratio_threshold*100:.1f}%），"
            "疑似字体编码损坏或扫描件"
        )

    return True, f"有效文本（CJK 占比 {cjk_ratio*100:.1f}%）"
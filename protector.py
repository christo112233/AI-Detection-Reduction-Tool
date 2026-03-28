import re

def get_protection_prompt_addition() -> str:
    """返回注入到系统提示词中的免检保护规则"""
    return "\n\n【核心指令：免检保护区】文中可能包含形如 [PROTECTED_0], [PROTECTED_1] 的特殊占位符。这些代表用户强制锁定的绝对安全内容。你必须在输出生成的句子时，将其原封不动地（包括方括号和大小写）保留在正确的上下文中，严禁改写、删减或翻译！"

def is_segment_fully_protected(segment: str) -> bool:
    """
    判断一个段落是否完全由保护占位符和空白/标点组成。
    如果是，则触发底层闪避（Bypass）逻辑，直接返还原句，0 Token消耗。
    """
    # 剥离所有 [PROTECTED_X] 占位符
    cleaned = re.sub(r'\[PROTECTED_\d+\]', '', segment)
    # 如果剥离占位符后，只剩下空白或者中英文基础标点，说明该段落完全被锁定，不需要 AI 介入
    if not cleaned.strip() or re.fullmatch(r'^[。！？；!?;,\.\s]+$', cleaned):
        return True
    return False
from fastapi import APIRouter
from pydantic import BaseModel
import re
from typing import List, Dict, Any

# 创建独立的路由，保持主程序高内聚低耦合
router = APIRouter()

class AnalyzeRequest(BaseModel):
    text: str

@router.post("/api/inspector/analyze")
def auto_detect_protection_zones(req: AnalyzeRequest) -> List[Dict[str, Any]]:
    """
    智能探针：接收长文本，按行拆分，并智能预测哪些行应该是“免检区”。
    主要针对：参考文献列表、章节标题、极短的公式或人名段落。
    """
    segments = req.text.split('\n')
    results = []
    
    for seg in segments:
        stripped = seg.strip()
        if not stripped:
            results.append({"text": seg, "suggest_lock": False})
            continue
            
        is_locked = False
        
        # 规则 1: 参考文献标识 (References / Bibliography)
        if re.match(r'^(参考文献|References|Bibliography|参考书目)[\s:：]*$', stripped, re.IGNORECASE):
            is_locked = True
        elif re.match(r'^\[\d+\]\s*.*', stripped): 
            # ✨修复: 匹配 "[1] Author" 格式，将 \s+ 改为 \s*，允许括号后没有空格
            is_locked = True
            
        # 规则 2: 章节标题标识
        elif re.match(r'^(第[\d一二三四五六七八九十百千万]+章|Chapter\s*\d+|[0-9]+(\.[0-9]+)*\s*[^\n。！？.!?]{0,40}|摘要|Abstract|致谢|Acknowledgement|引言|Introduction)[\s:：]*$', stripped, re.IGNORECASE):
            # ✨修复: 增加了 \d 支持"第1章"，并将 \s+ 改为 \s* 支持 "1.1标题" 无空格格式
            is_locked = True
            
        # 规则 3: 无标点符号的极短独立行（通常是小标题或孤立的数据）
        elif len(stripped) < 35 and not re.search(r'[。！？.!?]$', stripped):
            # ✨致命 Bug 修复: 加上结束符 $，只排除以句号结尾的完整长句。
            
            # 长度放宽到了 35 字符，兼容稍微长一点的标题
            is_locked = True
            
        results.append({
            "text": seg,
            "suggest_lock": is_locked
        })
        
    return results
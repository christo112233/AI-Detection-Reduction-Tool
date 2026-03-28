import os
import json
import uuid
from datetime import datetime

# 本地历史记录文件存放路径
HISTORY_FILE = "task_history.json"
# 最多保存的历史记录条数
MAX_HISTORY = 30

def load_history():
    """读取本地历史记录"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history_list(history_list):
    """保存数据到本地 JSON 文件"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_list, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[错误] 历史记录保存失败: {e}")

def add_record(source_text: str, result_text: str, think_text: str, prompt_type: str, total_tokens: int, cached_tokens: int, model_name: str = "未知模型"):
    """添加一条新的重塑记录"""
    history = load_history()
    
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "source_text": source_text,
        "result_text": result_text,
        "think_text": think_text,
        "prompt_type": prompt_type,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens
    }
    
    # 插入到最前面
    history.insert(0, record)
    # 截断，只保留最新的 30 条
    history = history[:MAX_HISTORY]
    
    save_history_list(history)
    return record

def delete_record(record_id: str):
    """删除指定的历史记录"""
    history = load_history()
    history = [r for r in history if r.get("id") != record_id]
    save_history_list(history)
    return True
import requests
import json
from typing import Tuple

# DeepVeri 本地微服务节点的默认监听地址
DEEPVERI_CHECK_URL = "http://127.0.0.1:5005/api/check"

def check_health() -> bool:
    """
    ✨ 新增：健康探针。
    用于极速检测本地 DeepVeri 5005 端口微服务是否已经启动。
    """
    try:
        # 发送极短文本进行存活试探，设定 3 秒极短超时防止阻塞主线程
        response = requests.post(
            DEEPVERI_CHECK_URL, 
            json={"text": "ping"}, 
            headers={"Content-Type": "application/json"}, 
            timeout=3
        )
        return response.status_code == 200
    except Exception:
        return False

def check_aigc_rate(text: str) -> Tuple[float, bool]:
    """
    调用本地 DeepVeri 微服务检测单段文本的 AI 生成率。
    返回元组: (ai_ratio, is_ignored)
    若文本过短或出错，is_ignored 返回 True，后续不计入加权平均。
    """
    if not text or not text.strip():
        return 0.0, True

    payload = {"text": text}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            DEEPVERI_CHECK_URL, 
            data=json.dumps(payload), 
            headers=headers, 
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        status = data.get("status")
        if status == "success":
            return float(data.get("ai_ratio", 0.0)), False
        elif status == "too_short":
            # 文本过短，打上忽略标记
            return 0.0, True
        else:
            return 0.0, True

    except requests.exceptions.ConnectionError:
        print("[DeepVeri] 连接失败: 请确保本地 5005 端口微服务已启动")
        return 0.0, True
    except requests.exceptions.Timeout:
        print("[DeepVeri] 检测请求超时")
        return 0.0, True
    except Exception as e:
        print(f"[DeepVeri] 未知异常: {e}")
        return 0.0, True
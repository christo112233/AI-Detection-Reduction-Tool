"""
公告模块 — 从 GitHub / Gitee 拉取 announcement.json，支持双源降级与本地缓存。
"""
import json
import os
import ssl
import asyncio
import urllib.request
from fastapi import APIRouter

# PyInstaller 打包后需要用 certifi 的证书 bundle 来验证 HTTPS
try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CONTEXT = ssl.create_default_context()

router = APIRouter(prefix="/api/announcement", tags=["announcement"])

CACHE_FILE = "announcement_cache.json"
READ_STATE_FILE = "announcement_read_state.json"

# 默认公告源（可在 config.json 中覆盖）
DEFAULT_GITHUB_URL = "https://raw.githubusercontent.com/christo112233/TraceLess-Announcement/main/announcement.json"
DEFAULT_GITEE_URL = "https://gitee.com/christo112233/TraceLess-Announcement/raw/main/announcement.json"
TIMEOUT = 5  # 每个源超时秒数


def _load_json(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_json(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[公告] 写入文件失败 {path}: {e}")


def _get_urls():
    """从 config.json 读取公告源 URL，未配置则使用默认值"""
    config = _load_json("config.json")
    return (
        config.get("announcement_url_gitee", DEFAULT_GITEE_URL),
        config.get("announcement_url_github", DEFAULT_GITHUB_URL),
    )


def _fetch_url_sync(url: str) -> dict | None:
    """同步从单个 URL 拉取 JSON，超时/失败返回 None"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TraceLess/4.4"})
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=_SSL_CONTEXT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[公告] 拉取失败: {e}")
        return None


async def _fetch_url(url: str) -> dict | None:
    """异步包装，在线程池中执行同步请求避免阻塞事件循环"""
    return await asyncio.to_thread(_fetch_url_sync, url)


def _get_cached() -> dict | None:
    data = _load_json(CACHE_FILE)
    return data if data else None


def _save_cache(data: dict):
    _save_json(CACHE_FILE, data)


def _get_read_ids() -> set:
    state = _load_json(READ_STATE_FILE)
    ids = state.get("read_ids", [])
    return set(ids)


def _mark_read(announcement_id: str):
    ids = _get_read_ids()
    ids.add(announcement_id)
    _save_json(READ_STATE_FILE, {"read_ids": list(ids)})


@router.get("/fetch")
async def fetch_announcements():
    """
   
    响应附带每个公告是否已读的标记。
    """
    gitee_url, github_url = _get_urls()

    data = await _fetch_url(gitee_url)
    source = "gitee"
    if data is None:
        data = await _fetch_url(github_url)
        source = "github"

    read_ids = _get_read_ids()

    if data and isinstance(data.get("announcements"), list):
        _save_cache(data)
        announcements = data["announcements"]
        for a in announcements:
            a["_read"] = a.get("id", "") in read_ids
        return {
            "status": "success",
            "source": source,
            "announcements": announcements,
        }

    # 都失败，回退缓存
    cached = _get_cached()
    if cached and isinstance(cached.get("announcements"), list):
        announcements = cached["announcements"]
        for a in announcements:
            a["_read"] = True
        return {
            "status": "success",
            "source": "cache",
            "announcements": announcements,
        }

    return {"status": "success", "source": "empty", "announcements": []}


@router.post("/mark_read")
def mark_read(payload: dict):
    """标记某条公告为已读"""
    aid = payload.get("id", "")
    if aid:
        print(f"[公告] 标记已读: {aid}")
        _mark_read(aid)
    return {"status": "ok"}

"""
Redis 缓存工具模块
用于热门问答缓存、会话管理、兜底回复
"""

import json
import hashlib
from typing import Optional, Any
from datetime import timedelta

import redis.asyncio as aioredis

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RedisCache:
    """Redis 异步缓存客户端"""

    def __init__(self):
        self._pool: Optional[aioredis.ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None

    async def connect(self):
        """建立 Redis 连接池"""
        try:
            self._pool = aioredis.ConnectionPool(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                max_connections=20,
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
            await self._client.ping()
            logger.info(f"Redis 连接成功: {settings.redis_host}:{settings.redis_port}")
        except Exception as e:
            logger.warning(f"Redis 连接失败 (缓存将不可用): {e}")
            self._client = None

    async def disconnect(self):
        """关闭 Redis 连接"""
        if self._pool:
            await self._pool.disconnect()
            logger.info("Redis 连接已关闭")

    @property
    def client(self) -> Optional[aioredis.Redis]:
        return self._client

    async def get(self, key: str) -> Optional[str]:
        """获取缓存值"""
        if not self._client:
            return None
        try:
            return await self._client.get(key)
        except Exception as e:
            logger.warning(f"Redis GET 失败: {e}")
            return None

    async def set(
        self, key: str, value: str, ttl: int = 3600
    ) -> bool:
        """设置缓存 (默认1小时过期)"""
        if not self._client:
            return False
        try:
            await self._client.set(key, value, ex=ttl)
            return True
        except Exception as e:
            logger.warning(f"Redis SET 失败: {e}")
            return False

    async def get_json(self, key: str) -> Optional[Any]:
        """获取 JSON 缓存"""
        raw = await self.get(key)
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        return None

    async def set_json(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """设置 JSON 缓存"""
        return await self.set(key, json.dumps(value, ensure_ascii=False), ttl)

    @staticmethod
    def make_query_key(query: str, subject: str = "") -> str:
        """生成查询缓存键"""
        raw = f"{subject}:{query}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"qa:cache:{digest}"

    async def get_cached_answer(
        self, query: str, subject: str = ""
    ) -> Optional[str]:
        """获取缓存的问答结果 (兜底用)"""
        key = self.make_query_key(query, subject)
        return await self.get(key)

    async def cache_answer(
        self, query: str, answer: str, subject: str = "", ttl: int = 3600
    ) -> bool:
        """缓存问答结果"""
        key = self.make_query_key(query, subject)
        return await self.set(key, answer, ttl)


# 全局缓存实例
cache = RedisCache()

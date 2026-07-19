"""
Milvus 客户端封装
管理 Collection 的创建、索引、数据写入与连接
"""

import time
from typing import List, Dict, Optional, Any

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException,
)

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 向量维度 (DashScope text-embedding-v3: 1024)
EMBEDDING_DIM = 1024


class MilvusClient:
    """Milvus 客户端 — 管理向量数据库连接与 Collection"""

    def __init__(self):
        self._collection: Optional[Collection] = None
        self._connected = False

    # ==================== 连接管理 ====================

    def connect(self) -> bool:
        """建立 Milvus 连接"""
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
            )
            self._connected = True
            logger.info(f"Milvus 连接成功: {settings.milvus_host}:{settings.milvus_port}")
            return True
        except MilvusException as e:
            logger.error(f"Milvus 连接失败: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """断开 Milvus 连接"""
        try:
            connections.disconnect("default")
            self._connected = False
            logger.info("Milvus 连接已断开")
        except Exception as e:
            logger.warning(f"Milvus 断开连接异常: {e}")

    @property
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connected

    # ==================== Collection 管理 ====================

    def has_collection(self, name: str = None) -> bool:
        """检查 Collection 是否存在"""
        name = name or settings.milvus_collection_name
        return utility.has_collection(name)

    def create_collection(self, name: str = None) -> Collection:
        """
        创建 Collection (Schema: id + text + embedding + metadata)

        Args:
            name: Collection 名称，默认使用配置值

        Returns:
            创建的 Collection 对象
        """
        name = name or settings.milvus_collection_name

        if self.has_collection(name):
            logger.info(f"Collection '{name}' 已存在，复用已有")
            self._collection = Collection(name)
            return self._collection

        logger.info(f"创建 Collection: {name}")

        # 定义 Schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=128,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=EMBEDDING_DIM,
            ),
            # 元数据字段 (支持过滤)
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="char_count", dtype=DataType.INT64),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="上市公司年报向量库 — 混合检索",
            enable_dynamic_field=True,  # 允许动态字段
        )

        self._collection = Collection(name=name, schema=schema)
        logger.info(f"Collection '{name}' 创建成功")

        return self._collection

    def get_collection(self, name: str = None) -> Optional[Collection]:
        """获取 Collection 引用"""
        name = name or settings.milvus_collection_name

        if self._collection and self._collection.name == name:
            return self._collection

        if self.has_collection(name):
            self._collection = Collection(name)
            return self._collection

        logger.warning(f"Collection '{name}' 不存在")
        return None

    # ==================== 索引管理 ====================

    def create_index(self, collection: Collection = None):
        """
        创建向量索引 (IVF_FLAT)

        Args:
            collection: Collection 对象，默认使用当前 collection
        """
        col = collection or self._collection
        if col is None:
            raise ValueError("Collection 未初始化")

        index_params = {
            "metric_type": "IP",  # Inner Product (内积，适合归一化向量)
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }

        col.create_index(
            field_name="embedding",
            index_params=index_params,
        )
        logger.info(f"向量索引创建成功 (IVF_FLAT, nlist=128)")

        # 创建标量字段索引 (加速元数据过滤)
        for field in ["file_type", "subject", "chapter"]:
            try:
                col.create_index(
                    field_name=field,
                    index_params={"index_type": "TRIE"},
                )
            except Exception:
                pass  # 某些版本不支持或已存在

    def load_collection(self, collection: Collection = None):
        """加载 Collection 到内存"""
        col = collection or self._collection
        if col:
            col.load()
            logger.info(f"Collection '{col.name}' 已加载到内存")

    # ==================== 数据操作 ====================

    def insert(
        self,
        data: List[Dict[str, Any]],
        collection: Collection = None,
    ) -> int:
        """
        批量插入数据

        Args:
            data: 数据列表，每条包含 id, text, embedding, file_name 等
            collection: Collection 对象

        Returns:
            插入的记录数
        """
        col = collection or self._collection
        if col is None:
            raise ValueError("Collection 未初始化")

        if not data:
            return 0

        # 确保字段顺序与 Schema 一致
        field_names = [
            "id", "text", "embedding",
            "file_name", "file_type", "subject", "chapter",
            "chunk_index", "char_count", "source",
        ]

        records = []
        for item in data:
            record = []
            for field in field_names:
                record.append(item.get(field, "" if field != "chunk_index" and field != "char_count" else 0))
            records.append(record)

        try:
            mr = col.insert(records)
            col.flush()
            logger.info(f"插入 {len(data)} 条记录, 耗时刷新完成")
            return len(data)
        except MilvusException as e:
            logger.error(f"Milvus 插入失败: {e}")
            raise

    def delete_by_ids(self, ids: List[str], collection: Collection = None):
        """按 ID 删除记录"""
        col = collection or self._collection
        if col is None:
            raise ValueError("Collection 未初始化")

        expr = f"id in {ids}"
        col.delete(expr)
        logger.info(f"删除 {len(ids)} 条记录")

    def get_count(self, collection: Collection = None) -> int:
        """获取 Collection 记录数"""
        col = collection or self._collection
        if col is None:
            return 0
        return col.num_entities

    # ==================== 便捷方法 ====================

    def setup(self) -> Collection:
        """
        一键初始化: 连接 → 创建 Collection → 创建索引 → 加载
        """
        if not self._connected:
            self.connect()

        collection = self.create_collection()
        self.create_index(collection)
        self.load_collection(collection)

        return collection

    def ensure_loaded(self, collection: Collection = None):
        """确保 Collection 已加载"""
        col = collection or self._collection
        if col is None:
            return

        try:
            # 检查加载状态
            load_state = utility.load_state(col.name)
            if load_state.name != "Loaded":
                col.load()
        except Exception:
            col.load()


# 全局单例
milvus_client = MilvusClient()

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模型基类
定义所有ORM模型的基础配置
"""


# 创建基类
Base = declarative_base()


class TimestampMixin:
    """时间戳混入类，为模型添加创建和更新时间字段"""

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False, comment="更新时间")


class BaseModel(Base, TimestampMixin):
    """基础模型类"""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True, comment="主键ID")

    def to_dict(self):
        """将模型转换为字典"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def __repr__(self):
        """方法描述"""


class DataSyncStatus(Base, TimestampMixin):
    """数据同步状态模型"""

    __tablename__ = "data_sync_status"

    sync_type = Column(String(100), primary_key=True, comment="同步类型")
    last_sync_time = Column(DateTime, comment="上次同步时间")
    status = Column(String(50), comment="同步状态")  # success, failed, running
    message = Column(String(500), comment="同步消息")

    def __repr__(self):
        """方法描述"""


# 数据库会话工厂
SessionLocal = None


def init_db_session(engine=None):
    """初始化数据库会话"""
    global SessionLocal

    if engine is None:
        # 使用默认的数据库连接
        import os
        from pathlib import Path

        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        db_path = project_root / "data" / "stock_data.db"

        # 确保数据目录存在
        db_path.parent.mkdir(exist_ok=True)

        # 创建SQLite引擎
        database_url = f"sqlite:///{db_path}"
        engine = create_engine(database_url, echo=False)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal


def get_db_session():
    """获取数据库会话"""
    global SessionLocal

    if SessionLocal is None:
        # 自动初始化数据库会话
        init_db_session()

    return SessionLocal()

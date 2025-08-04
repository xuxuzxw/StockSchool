#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算引擎工厂
"""

from typing import Dict, Any, Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from loguru import logger

from .factor_models import FactorType
from .technical_factor_engine import TechnicalFactorEngine
from .fundamental_factor_engine import FundamentalFactorEngine
from .sentiment_factor_engine import SentimentFactorEngine


class EngineConnectionManager:
    """数据库连接管理器"""
    
    def __init__(self, database_url: str, pool_size: int = 5):
        self.database_url = database_url
        self.pool_size = pool_size
        self._engine: Optional[Engine] = None
    
    def get_engine(self) -> Engine:
        """获取数据库引擎（单例模式）"""
        if self._engine is None:
            self._engine = create_engine(
                self.database_url,
                pool_size=self.pool_size,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600
            )
        return self._engine
    
    def dispose(self):
        """释放连接"""
        if self._engine:
            self._engine.dispose()
            self._engine = None


class FactorEngineFactory:
    """因子计算引擎工厂"""
    
    # 引擎类型映射
    ENGINE_MAPPING = {
        'technical': TechnicalFactorEngine,
        'fundamental': FundamentalFactorEngine,
        'sentiment': SentimentFactorEngine,
        FactorType.TECHNICAL: TechnicalFactorEngine,
        FactorType.FUNDAMENTAL: FundamentalFactorEngine,
        FactorType.SENTIMENT: SentimentFactorEngine,
    }
    
    def __init__(self, connection_manager: EngineConnectionManager):
        self.connection_manager = connection_manager
        self._engine_cache: Dict[str, Any] = {}
    
    def create_engine(self, factor_type: str):
        """创建因子计算引擎"""
        # 检查缓存
        if factor_type in self._engine_cache:
            return self._engine_cache[factor_type]
        
        # 获取引擎类
        engine_class = self.ENGINE_MAPPING.get(factor_type)
        if not engine_class:
            raise ValueError(f"不支持的因子类型: {factor_type}")
        
        # 创建引擎实例
        db_engine = self.connection_manager.get_engine()
        factor_engine = engine_class(db_engine)
        
        # 缓存引擎
        self._engine_cache[factor_type] = factor_engine
        
        logger.debug(f"创建 {factor_type} 类型因子引擎")
        return factor_engine
    
    def clear_cache(self):
        """清空引擎缓存"""
        self._engine_cache.clear()
    
    @classmethod
    def create_for_worker(cls, database_url: str, factor_type: str):
        """为工作进程创建引擎（无缓存）"""
        connection_manager = EngineConnectionManager(database_url)
        factory = cls(connection_manager)
        
        try:
            return factory.create_engine(factor_type)
        except Exception as e:
            logger.error(f"工作进程创建引擎失败: {e}")
            connection_manager.dispose()
            raise
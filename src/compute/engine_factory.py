import logging
from typing import Optional, Dict, Any
from sqlalchemy import Engine

from src.database.connection import get_db_engine
from src.compute.technical_factor_engine import TechnicalFactorEngine
from src.compute.fundamental_factor_engine import FundamentalFactorEngine
from src.compute.sentiment_factor_engine import SentimentFactorEngine

logger = logging.getLogger(__name__)

class EngineConnectionManager:
    """引擎连接管理器"""
    
    def __init__(self):
        """初始化连接管理器"""
        self._engine = get_db_engine()
    
    @property
    def engine(self) -> Engine:
        """获取数据库引擎"""
        return self._engine

class FactorEngineFactory:
    """因子计算引擎工厂"""
    
    ENGINE_MAPPING = {
        'technical': TechnicalFactorEngine,
        'fundamental': FundamentalFactorEngine,
        'sentiment': SentimentFactorEngine
    }
    
    def __init__(self):
        """初始化工厂"""
        self.connection_manager = EngineConnectionManager()
    
    def create_engine(self, engine_type: str) -> Optional[Any]:
        """
        创建因子计算引擎
        
        Args:
            engine_type: 引擎类型 ('technical', 'fundamental', 'sentiment')
            
        Returns:
            对应的因子计算引擎实例
        """
        if engine_type not in self.ENGINE_MAPPING:
            logger.error(f"不支持的引擎类型: {engine_type}")
            return None
        
        engine_class = self.ENGINE_MAPPING[engine_type]
        return engine_class(self.connection_manager.engine)
    
    def get_available_engines(self) -> list:
        """获取可用的引擎类型"""
        return list(self.ENGINE_MAPPING.keys())

class EngineFactory:
    """向后兼容的引擎工厂"""
    
    @staticmethod
    def create_technical_engine() -> TechnicalFactorEngine:
        """创建技术因子引擎"""
        return TechnicalFactorEngine(get_db_engine())
    
    @staticmethod
    def create_fundamental_engine() -> FundamentalFactorEngine:
        """创建基本面因子引擎"""
        return FundamentalFactorEngine(get_db_engine())
    
    @staticmethod
    def create_sentiment_engine() -> SentimentFactorEngine:
        """创建情绪因子引擎"""
        return SentimentFactorEngine(get_db_engine())

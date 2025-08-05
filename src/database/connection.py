import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """数据库连接管理类"""
    
    _instance = None
    _engine = None
    _session_factory = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._engine is None:
            self.initialize_connection()
    
    def initialize_connection(self):
        """初始化数据库连接"""
        try:
            from src.config.unified_config import config
            
            db_config = config.get('database', {})
            db_type = db_config.get('type', 'sqlite')
            
            if db_type == 'postgresql':
                # PostgreSQL连接
                host = db_config.get('host', 'localhost')
                port = db_config.get('port', 5432)
                database = db_config.get('database', 'stockschool')
                username = db_config.get('username', 'postgres')
                password = db_config.get('password', '')
                
                connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            else:
                # SQLite连接
                db_path = db_config.get('path', 'data/stockschool.db')
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                connection_string = f"sqlite:///{db_path}"
            
            self._engine = create_engine(connection_string, echo=False)
            self._session_factory = sessionmaker(bind=self._engine)
            
            # 测试连接
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"数据库连接成功: {db_type}")
            
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def get_engine(self):
        """获取数据库引擎"""
        return self._engine
    
    def get_session(self):
        """获取数据库会话"""
        return self._session_factory()

# 全局实例
db_connection = DatabaseConnection()

def get_db_engine():
    """获取数据库引擎"""
    return db_connection.get_engine()

def get_db_session():
    """获取数据库会话"""
    return db_connection.get_session()

def get_db_connection():
    """获取数据库连接（兼容旧代码）"""
    return get_db_engine()

def test_database_connection() -> Dict[str, Any]:
    """测试数据库连接"""
    try:
        engine = get_db_engine()
        
        # 测试基本连接
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            conn.commit()
            
        # 获取数据库信息
        db_info = {
            'connected': True,
            'engine': str(engine.url),
            'database_type': 'postgresql' if 'postgresql' in str(engine.url) else 'sqlite'
        }
        
        logger.info("数据库连接测试成功")
        return db_info
        
    except SQLAlchemyError as e:
        logger.error(f"数据库连接测试失败: {e}")
        return {
            'connected': False,
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"数据库连接测试失败: {e}")
        return {
            'connected': False,
            'error': str(e)
        }

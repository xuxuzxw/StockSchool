#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的因子注册表实现
分离元数据管理和配置管理的职责
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from threading import Lock
import logging

from .factor_models import FactorMetadata, FactorConfig, FactorType, FactorCategory

logger = logging.getLogger(__name__)


class FactorMetadataManager:
    """因子元数据管理器"""
    
    def __init__(self):
        self._metadata: Dict[str, FactorMetadata] = {}
        self._lock = Lock()
    
    def register_metadata(self, metadata: FactorMetadata) -> None:
        """注册因子元数据"""
        with self._lock:
            if metadata.name in self._metadata:
                logger.warning(f"因子 {metadata.name} 已存在，将被覆盖")
            self._metadata[metadata.name] = metadata
            logger.info(f"注册因子元数据: {metadata.name}")
    
    def get_metadata(self, name: str) -> Optional[FactorMetadata]:
        """获取因子元数据"""
        return self._metadata.get(name)
    
    def list_factors(self, factor_type: Optional[FactorType] = None) -> List[str]:
        """列出因子名称"""
        if factor_type is None:
            return list(self._metadata.keys())
        
        return [name for name, metadata in self._metadata.items() 
                if metadata.factor_type == factor_type]
    
    def get_factors_by_category(self, category: FactorCategory) -> List[str]:
        """按分类获取因子"""
        return [name for name, metadata in self._metadata.items() 
                if metadata.category == category]


class FactorConfigManager:
    """因子配置管理器"""
    
    def __init__(self):
        self._configs: Dict[str, FactorConfig] = {}
        self._lock = Lock()
    
    def register_config(self, config: FactorConfig) -> None:
        """注册因子配置"""
        with self._lock:
            self._configs[config.name] = config
            logger.info(f"注册因子配置: {config.name}")
    
    def get_config(self, name: str) -> Optional[FactorConfig]:
        """获取因子配置"""
        return self._configs.get(name)
    
    def update_config(self, name: str, **kwargs) -> bool:
        """更新因子配置"""
        with self._lock:
            if name not in self._configs:
                return False
            
            config = self._configs[name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    config.set_parameter(key, value)
            
            logger.info(f"更新因子配置: {name}")
            return True


class FactorRegistry:
    """改进的因子注册表"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.metadata_manager = FactorMetadataManager()
            self.config_manager = FactorConfigManager()
            self._initialized = True
    
    def register_factor(self, metadata: FactorMetadata, config: Optional[FactorConfig] = None):
        """注册因子"""
        self.metadata_manager.register_metadata(metadata)
        
        if config is None:
            config = FactorConfig(name=metadata.name)
        
        self.config_manager.register_config(config)
    
    def get_factor_metadata(self, name: str) -> Optional[FactorMetadata]:
        """获取因子元数据"""
        return self.metadata_manager.get_metadata(name)
    
    def get_factor_config(self, name: str) -> Optional[FactorConfig]:
        """获取因子配置"""
        return self.config_manager.get_config(name)


# 单例实例
factor_registry = FactorRegistry()
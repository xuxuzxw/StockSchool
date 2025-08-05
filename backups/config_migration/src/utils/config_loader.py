#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载器 - 向后兼容接口
现在使用统一配置管理系统
"""

# 导入新的统一配置管理器
from ..config.unified_config import config as unified_config, get_config, get_factor_params, get_training_params, get_monitoring_params, get_database_params, get_api_params

class Config:
    """配置管理类 - 向后兼容接口"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def get(self, key_path: str, default=None):
        """获取配置值"""
        return unified_config.get(key_path, default)
    
    def get_section(self, section: str):
        """获取配置段"""
        return unified_config.get_section(section)
    
    def reload(self):
        """重新加载配置"""
        return unified_config.reload()
    
    def set(self, key_path: str, value):
        """设置配置值"""
        return unified_config.set(key_path, value)
    
    def has(self, key_path: str) -> bool:
        """检查配置项是否存在"""
        return unified_config.has(key_path)
    
    @property
    def all_config(self):
        """获取所有配置"""
        return unified_config.all_config

# 创建一个全局可访问的配置实例
config = Config()
"""
配置管理器
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigManager:
    """统一的配置管理器"""
    
    # 默认配置常量
    DEFAULT_CONFIG = {
        'db_host': 'localhost',
        'db_port': 5432,
        'db_name': 'stockschool',
        'db_user': 'stockschool',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'test_timeout': 300,
        'performance_test_enabled': True,
        'ai_analysis_test_enabled': True,
        'max_concurrent_tests': 5,
        'retry_attempts': 3,
        'retry_delay': 1.0
    }
    
    def __init__(self, config_file: str = '.env.acceptance'):
        """初始化配置管理器"""
        self.config_file = config_file
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        config = self.DEFAULT_CONFIG.copy()
        
        # 加载环境变量配置文件
        if os.path.exists(self.config_file):
            config.update(self._load_env_file())
        
        # 加载YAML配置文件（如果存在）
        yaml_config_file = self.config_file.replace('.env', '.yml')
        if os.path.exists(yaml_config_file):
            config.update(self._load_yaml_file(yaml_config_file))
        
        # 环境变量覆盖
        config.update(self._load_env_variables())
        
        return config
    
    def _load_env_file(self) -> Dict[str, Any]:
        """加载.env格式的配置文件"""
        config = {}
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = self._parse_value(value.strip())
        except Exception as e:
            print(f"警告: 加载配置文件失败: {e}")
        
        return config
    
    def _load_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """加载YAML格式的配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"警告: 加载YAML配置文件失败: {e}")
            return {}
    
    def _load_env_variables(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        env_mapping = {
            'POSTGRES_PASSWORD': 'db_password',
            'REDIS_PASSWORD': 'redis_password',
            'TUSHARE_TOKEN': 'tushare_token',
            'AI_API_KEY': 'ai_api_key',
            'AI_API_BASE_URL': 'ai_api_base_url'
        }
        
        config = {}
        for env_key, config_key in env_mapping.items():
            value = os.getenv(env_key)
            if value:
                config[config_key] = value
        
        return config
    
    def _parse_value(self, value: str) -> Any:
        """解析配置值的类型"""
        # 布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # 数字
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # 字符串
        return value.strip('"\'')
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_database_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        return {
            'host': self.get('db_host'),
            'port': self.get('db_port'),
            'database': self.get('db_name'),
            'user': self.get('db_user'),
            'password': self.get('db_password')
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """获取Redis配置"""
        return {
            'host': self.get('redis_host'),
            'port': self.get('redis_port'),
            'password': self.get('redis_password'),
            'decode_responses': True
        }
    
    def validate_required_config(self) -> List[str]:
        """验证必需的配置项"""
        required_keys = [
            'db_password',
            'tushare_token'
        ]
        
        missing_keys = []
        for key in required_keys:
            if not self.get(key):
                missing_keys.append(key)
        
        return missing_keys
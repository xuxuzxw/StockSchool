# src/utils/config_loader.py
import yaml
from pathlib import Path
from typing import Any, Optional

class Config:
    """全局配置管理类，使用单例模式确保配置的一致性"""
    _instance = None
    _config = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
            cls._instance.load_config()
        return cls._instance
    
    def load_config(self):
        """加载配置文件"""
        try:
            config_path = Path(__file__).parent.parent.parent / 'config.yml'
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"配置文件未找到: {config_path}")
            self._config = {}
        except yaml.YAMLError as e:
            print(f"配置文件格式错误: {e}")
            self._config = {}
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key_path: 配置键路径，使用点号分隔，如 'factor_params.rsi.window'
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        if not self._config:
            return default
        
        # 如果key_path为空，返回整个配置
        if not key_path:
            return self._config
            
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> dict:
        """获取配置段
        
        Args:
            section: 配置段名称
            
        Returns:
            配置段字典
        """
        return self._config.get(section, {})
    
    def reload(self):
        """重新加载配置文件"""
        self.load_config()
    
    def set(self, key_path: str, value: Any):
        """设置配置值（运行时修改，不会保存到文件）
        
        Args:
            key_path: 配置键路径
            value: 配置值
        """
        if not self._config:
            self._config = {}
            
        keys = key_path.split('.')
        current = self._config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def has(self, key_path: str) -> bool:
        """检查配置项是否存在
        
        Args:
            key_path: 配置键路径
            
        Returns:
            是否存在
        """
        if not self._config:
            return False
            
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return False
        
        return True
    
    @property
    def all_config(self) -> dict:
        """获取所有配置"""
        return self._config or {}

# 创建一个全局可访问的配置实例
config = Config()

# 便捷函数
def get_config(key_path: str, default: Any = None) -> Any:
    """获取配置值的便捷函数"""
    return config.get(key_path, default)

def get_factor_params() -> dict:
    """获取因子参数配置"""
    return config.get_section('factor_params')

def get_training_params() -> dict:
    """获取训练参数配置"""
    return config.get_section('training_params')

def get_monitoring_params() -> dict:
    """获取监控参数配置"""
    return config.get_section('monitoring_params')

def get_database_params() -> dict:
    """获取数据库参数配置"""
    return config.get_section('database_params')

def get_api_params() -> dict:
    """获取API参数配置"""
    return config.get_section('api_params')
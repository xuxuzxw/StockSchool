"""
配置加载器 - 专门负责配置文件的加载和合并
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
    
    def load_all_configs(self, environment: str) -> Dict[str, Any]:
        """加载所有配置文件"""
        config = {}
        
        # 加载基础配置
        base_config = self._load_config_file(self.config_dir / "base.yml")
        if base_config:
            config.update(base_config)
        
        # 加载环境特定配置
        env_config = self._load_config_file(self.config_dir / f"{environment}.yml")
        if env_config:
            self._merge_config(config, env_config)
        
        # 加载主配置文件（向后兼容）
        main_config = self._load_config_file(Path("config.yml"))
        if main_config:
            self._merge_config(config, main_config)
        
        logger.info(f"配置加载完成，环境: {environment}")
        return config
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """加载单个配置文件"""
        if not file_path.exists():
            return {}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f) or {}
                
                logger.debug(f"加载配置文件: {file_path}")
                return config
        except Exception as e:
            logger.error(f"加载配置文件失败 {file_path}: {e}")
            return {}
    
    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """深度合并配置"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
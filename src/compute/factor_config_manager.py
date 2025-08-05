import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import yaml

from src.config.config_loader import ConfigLoader as BaseConfigLoader

from ..compute.factor_registry_improved import FactorConfigManager as BaseFactorConfigManager

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子配置管理器
支持动态配置加载和热更新
"""


logger = logging.getLogger(__name__)


@dataclass
class FactorConfigSchema:
    """因子配置模式"""

    name: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    calculation_order: int = 0
    cache_enabled: bool = True
    parallel_enabled: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        for param_name, rules in self.validation_rules.items():
            if param_name in parameters:
                value = parameters[param_name]

                # 类型检查
                if "type" in rules:
                    expected_type = rules["type"]
                    if not isinstance(value, expected_type):
                        raise ValueError(f"参数 {param_name} 类型错误，期望 {expected_type}，实际 {type(value)}")

                # 范围检查
                if "min" in rules and value < rules["min"]:
                    raise ValueError(f"参数 {param_name} 值 {value} 小于最小值 {rules['min']}")

                if "max" in rules and value > rules["max"]:
                    raise ValueError(f"参数 {param_name} 值 {value} 大于最大值 {rules['max']}")

                # 选项检查
                if "choices" in rules and value not in rules["choices"]:
                    raise ValueError(f"参数 {param_name} 值 {value} 不在允许的选项中: {rules['choices']}")

        return True


# 使用统一的配置加载器


class ConfigLoader:
    """因子配置加载器 - 使用统一配置加载器"""

    @staticmethod
    def load_from_yaml(file_path: Path) -> Dict[str, Any]:
        """从YAML文件加载配置"""
        try:
            return BaseConfigLoader(file_path.parent)._load_config_file(file_path)
        except Exception as e:
            logger.error(f"加载YAML配置文件失败 {file_path}: {e}")
            raise

    @staticmethod
    def load_from_json(file_path: Path) -> Dict[str, Any]:
        """从JSON文件加载配置"""
        try:
            return BaseConfigLoader(file_path.parent)._load_config_file(file_path)
        except Exception as e:
            logger.error(f"加载JSON配置文件失败 {file_path}: {e}")
            raise

    @staticmethod
    def save_to_yaml(data: Dict[str, Any], file_path: Path) -> None:
        """保存配置到YAML文件"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"保存YAML配置文件失败 {file_path}: {e}")
            raise


# 使用统一的因子配置管理器
from .factor_models import FactorConfig as BaseFactorConfig


class FileBasedFactorConfigManager:
    """基于文件的因子配置管理器 - 使用标准实现"""

    def __init__(self, config_dir: Path = None):
        """方法描述"""
        self.configs: Dict[str, FactorConfigSchema] = {}
        self._lock = Lock()
        self._last_modified: Dict[str, datetime] = {}

        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_all_configs(self) -> None:
        """加载所有配置文件"""
        with self._lock:
            self.configs.clear()
            self._last_modified.clear()

            # 加载YAML配置文件
            for yaml_file in self.config_dir.glob("*.yml"):
                self._load_config_file(yaml_file)

            for yaml_file in self.config_dir.glob("*.yaml"):
                self._load_config_file(yaml_file)

            # 加载JSON配置文件
            for json_file in self.config_dir.glob("*.json"):
                self._load_config_file(json_file)

            logger.info(f"加载了 {len(self.configs)} 个因子配置")

    def _load_config_file(self, file_path: Path) -> None:
        """加载单个配置文件"""
        try:
            if file_path.suffix.lower() in [".yml", ".yaml"]:
                data = ConfigLoader.load_from_yaml(file_path)
            elif file_path.suffix.lower() == ".json":
                data = ConfigLoader.load_from_json(file_path)
            else:
                logger.warning(f"不支持的配置文件格式: {file_path}")
                return

            # 解析配置
            for factor_name, config_data in data.items():
                config = FactorConfigSchema(name=factor_name, **config_data)
                self.configs[factor_name] = config
                self._last_modified[factor_name] = datetime.fromtimestamp(file_path.stat().st_mtime)

            logger.debug(f"加载配置文件: {file_path}")

        except Exception as e:
            logger.error(f"加载配置文件失败 {file_path}: {e}")

    def get_config(self, factor_name: str) -> Optional[FactorConfigSchema]:
        """获取因子配置"""
        return self.configs.get(factor_name)

    def set_config(self, factor_name: str, config: FactorConfigSchema) -> None:
        """设置因子配置"""
        with self._lock:
            self.configs[factor_name] = config
            logger.info(f"设置因子配置: {factor_name}")

    def update_config(self, factor_name: str, **kwargs) -> bool:
        """更新因子配置"""
        with self._lock:
            if factor_name not in self.configs:
                logger.warning(f"因子配置不存在: {factor_name}")
                return False

            config = self.configs[factor_name]

            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    config.parameters[key] = value

            logger.info(f"更新因子配置: {factor_name}")
            return True

    def save_config(self, factor_name: str, file_path: Path = None) -> None:
        """保存因子配置到文件"""
        if factor_name not in self.configs:
            raise ValueError(f"因子配置不存在: {factor_name}")

        if file_path is None:
            file_path = self.config_dir / f"{factor_name}.yml"

        config = self.configs[factor_name]
        config_data = {
            factor_name: {
                "enabled": config.enabled,
                "parameters": config.parameters,
                "dependencies": config.dependencies,
                "calculation_order": config.calculation_order,
                "cache_enabled": config.cache_enabled,
                "parallel_enabled": config.parallel_enabled,
                "validation_rules": config.validation_rules,
            }
        }

        ConfigLoader.save_to_yaml(config_data, file_path)
        logger.info(f"保存因子配置: {factor_name} -> {file_path}")

    def list_factors(self, enabled_only: bool = False) -> List[str]:
        """列出因子名称"""
        if enabled_only:
            return [name for name, config in self.configs.items() if config.enabled]
        return list(self.configs.keys())

    def get_calculation_order(self) -> List[str]:
        """获取按计算顺序排序的因子列表"""
        sorted_factors = sorted(self.configs.items(), key=lambda x: x[1].calculation_order)
        return [name for name, config in sorted_factors if config.enabled]

    def validate_dependencies(self) -> Dict[str, List[str]]:
        """验证因子依赖关系"""
        missing_dependencies = {}

        for factor_name, config in self.configs.items():
            missing = []
            for dep in config.dependencies:
                if dep not in self.configs:
                    missing.append(dep)

            if missing:
                missing_dependencies[factor_name] = missing

        return missing_dependencies

    def check_for_updates(self) -> List[str]:
        """检查配置文件更新"""
        updated_factors = []

        for config_file in self.config_dir.glob("*.yml"):
            file_mtime = datetime.fromtimestamp(config_file.stat().st_mtime)

            # 检查文件是否有更新
            for factor_name in self.configs:
                if factor_name in self._last_modified and file_mtime > self._last_modified[factor_name]:
                    updated_factors.append(factor_name)

        return updated_factors

    def reload_updated_configs(self) -> List[str]:
        """重新加载已更新的配置"""
        updated_factors = self.check_for_updates()

        if updated_factors:
            logger.info(f"检测到配置更新，重新加载: {updated_factors}")
            self.load_all_configs()

        return updated_factors


# 默认配置模板
DEFAULT_FACTOR_CONFIGS = {
    "rsi_14": {
        "enabled": True,
        "parameters": {"window": 14, "threshold_oversold": 30, "threshold_overbought": 70},
        "dependencies": [],
        "calculation_order": 1,
        "cache_enabled": True,
        "parallel_enabled": True,
        "validation_rules": {
            "window": {"type": int, "min": 1, "max": 252},
            "threshold_oversold": {"type": (int, float), "min": 0, "max": 50},
            "threshold_overbought": {"type": (int, float), "min": 50, "max": 100},
        },
    },
    "macd": {
        "enabled": True,
        "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "dependencies": [],
        "calculation_order": 2,
        "cache_enabled": True,
        "parallel_enabled": True,
        "validation_rules": {
            "fast_period": {"type": int, "min": 1, "max": 100},
            "slow_period": {"type": int, "min": 1, "max": 100},
            "signal_period": {"type": int, "min": 1, "max": 50},
        },
    },
}


def create_default_configs(config_dir: Path) -> None:
    """创建默认配置文件"""
    config_dir.mkdir(parents=True, exist_ok=True)

    for factor_name, config_data in DEFAULT_FACTOR_CONFIGS.items():
        config_file = config_dir / f"{factor_name}.yml"
        if not config_file.exists():
            ConfigLoader.save_to_yaml({factor_name: config_data}, config_file)
            logger.info(f"创建默认配置文件: {config_file}")


if __name__ == "__main__":
    # 测试代码
    config_manager = FactorConfigManager()

    # 创建默认配置
    create_default_configs(config_manager.config_dir)

    # 加载配置
    config_manager.load_all_configs()

    # 测试配置获取
    rsi_config = config_manager.get_config("rsi_14")
    if rsi_config:
        print(f"RSI配置: {rsi_config}")

        # 测试参数验证
        try:
            rsi_config.validate_parameters({"window": 14, "threshold_oversold": 30})
            print("参数验证通过")
        except ValueError as e:
            print(f"参数验证失败: {e}")

    # 测试依赖验证
    missing_deps = config_manager.validate_dependencies()
    if missing_deps:
        print(f"缺少依赖: {missing_deps}")
    else:
        print("所有依赖都满足")

    print("配置管理器测试完成")

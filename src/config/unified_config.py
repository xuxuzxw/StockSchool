import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理器
整合现有的配置系统，提供向后兼容的接口
"""


class UnifiedConfig:
    """
    统一配置管理器
    提供与旧版config_loader兼容的接口，同时使用新的配置管理系统
    """

    _instance: Optional["UnifiedConfig"] = None
    _config_manager: Optional["ConfigManager"] = None

    def __new__(cls, *args, **kwargs):
        """方法描述"""
        if cls._instance is None:
            cls._instance = super(UnifiedConfig, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化配置管理器"""
        try:
            # 降级到简单配置加载，避免循环导入
            self._fallback_to_simple_config()
            logger.info("统一配置管理器初始化完成")
        except Exception as e:
            logger.error(f"配置管理器初始化失败: {e}")
            self._simple_config = {}

    def _load_legacy_config(self):
        """加载旧版config.yml文件以保持向后兼容"""
        legacy_config_path = Path(__file__).parent.parent.parent / "config.yml"

        if legacy_config_path.exists():
            try:
                with open(legacy_config_path, "r", encoding="utf-8") as f:
                    legacy_config = yaml.safe_load(f)

                # 将旧配置合并到新配置管理器中
                if legacy_config and self._config_manager:
                    self._config_manager._merge_config(legacy_config)
                    logger.info("旧版配置文件已合并到新配置系统")

            except Exception as e:
                logger.warning(f"加载旧版配置文件失败: {e}")

    def _fallback_to_simple_config(self):
        """降级到简单配置加载模式"""
        try:
            # 由ConfigManager调用时，使用其配置目录
            # 否则使用默认配置目录
            self._simple_config = {}

            # 检查是否有配置管理器提供的目录
            if hasattr(self, "_config_dir") and self._config_dir:
                config_dir = Path(self._config_dir)
            else:
                config_dir = Path(__file__).parent.parent.parent / "config"

            # 加载基础配置
            base_config = config_dir / "base.yml"
            if base_config.exists():
                with open(base_config, "r", encoding="utf-8") as f:
                    base_data = yaml.safe_load(f)
                    if base_data:
                        self._simple_config.update(base_data)

            # 加载环境特定配置
            env = os.getenv("ENVIRONMENT", "development")
            env_config = config_dir / f"{env}.yml"
            if env_config.exists():
                with open(env_config, "r", encoding="utf-8") as f:
                    env_data = yaml.safe_load(f)
                    if env_data:
                        self._simple_config.update(env_data)

            # 加载主配置文件作为后备
            main_config = Path(__file__).parent.parent.parent / "config.yml"
            if main_config.exists():
                with open(main_config, "r", encoding="utf-8") as f:
                    main_data = yaml.safe_load(f)
                    if main_data:
                        self._simple_config.update(main_data)

            if not self._simple_config:
                logger.warning("未找到任何配置文件，使用空配置")

        except Exception as e:
            logger.error(f"降级配置加载失败: {e}")
            self._simple_config = {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（兼容旧版接口）

        Args:
            key_path: 配置键路径，如 'factor_params.rsi.window'
            default: 默认值

        Returns:
            配置值
        """
        try:
            return self._get_from_simple_config(key_path, default)
        except Exception as e:
            logger.warning(f"获取配置失败 {key_path}: {e}")
            return default

    def _get_from_simple_config(self, key_path: str, default: Any = None) -> Any:
        """从简单配置中获取值"""
        if not hasattr(self, "_simple_config"):
            return default

        if not key_path:
            return self._simple_config

        keys = key_path.split(".")
        value = self._simple_config

        try:
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置段（兼容旧版接口）

        Args:
            section: 配置段名称

        Returns:
            配置段字典
        """
        return self.get(section, {})

    def set(self, key_path: str, value: Any) -> None:
        """
        设置配置值

        Args:
            key_path: 配置键路径
            value: 配置值
        """
        try:
            self._set_in_simple_config(key_path, value)
        except Exception as e:
            logger.error(f"设置配置失败 {key_path}: {e}")

    def _set_in_simple_config(self, key_path: str, value: Any) -> None:
        """在简单配置中设置值"""
        if not hasattr(self, "_simple_config"):
            self._simple_config = {}

        keys = key_path.split(".")
        current = self._simple_config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def reload(self) -> None:
        """
        重新加载配置（兼容旧版接口）
        """
        try:
            self._fallback_to_simple_config()
            logger.info("配置重新加载完成")
        except Exception as e:
            logger.error(f"配置重新加载失败: {e}")

    @property
    def all_config(self) -> Dict[str, Any]:
        """
        获取所有配置（兼容旧版接口）

        Returns:
            所有配置的字典
        """
        try:
            return getattr(self, "_simple_config", {})
        except Exception as e:
            logger.error(f"获取所有配置失败: {e}")
            return {}

    def get_environment(self) -> str:
        """
        获取当前环境

        Returns:
            环境名称
        """
        return os.getenv("ENVIRONMENT", "development")

    def has(self, key_path: str) -> bool:
        """
        检查配置项是否存在

        Args:
            key_path: 配置键路径

        Returns:
            配置项是否存在
        """
        try:
            if self._config_manager:
                return self._config_manager.has(key_path)
            else:
                # 降级模式
                return self._has_in_simple_config(key_path)
        except Exception as e:
            logger.warning(f"检查配置项失败 {key_path}: {e}")
            return False

    def _has_in_simple_config(self, key_path: str) -> bool:
        """在简单配置中检查配置项是否存在"""
        if not hasattr(self, "_simple_config"):
            return False

        if not key_path:
            return True

        keys = key_path.split(".")
        value = self._simple_config

        try:
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return False
            return True
        except (KeyError, TypeError):
            return False

    def validate(self) -> bool:
        """
        验证配置

        Returns:
            验证是否通过
        """
        try:
            # 简单模式下的基本验证
            return hasattr(self, "_simple_config") and isinstance(self._simple_config, dict)
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False


# 创建全局实例（兼容旧版接口）
config = UnifiedConfig()


# 兼容函数（保持与旧版config_loader.py的兼容性）
def get_config(key_path: str, default: Any = None) -> Any:
    """获取配置值"""
    return config.get(key_path, default)


def get_factor_params() -> Dict[str, Any]:
    """获取因子参数"""
    return config.get_section("factor_params")


def get_training_params() -> Dict[str, Any]:
    """获取训练参数"""
    return config.get_section("training_params")


def get_monitoring_params() -> Dict[str, Any]:
    """获取监控参数"""
    return config.get_section("monitoring_params")


def get_database_params() -> Dict[str, Any]:
    """获取数据库参数"""
    return config.get_section("database_params")


def get_api_params() -> Dict[str, Any]:
    """获取API参数"""
    return config.get_section("api_params")

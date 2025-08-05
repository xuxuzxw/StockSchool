import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子配置管理
提供配置驱动的因子计算参数管理
"""


class FactorConfig:
    """因子配置管理类"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径，支持json和yaml格式
        """
        self.config = self._load_default_config()

        if config_path:
            self.load_config(config_path)

    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            "technical": {
                "enabled": True,
                "max_periods": 500,
                "momentum": {"enabled": True, "periods": [5, 10, 20], "rsi_period": 14, "williams_period": 14},
                "trend": {
                    "enabled": True,
                    "sma_periods": [5, 10, 20, 60],
                    "ema_periods": [12, 26],
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                },
                "volatility": {"enabled": True, "periods": [5, 20, 60], "atr_period": 14, "bb_period": 20, "bb_std": 2},
                "volume": {"enabled": True, "sma_periods": [5, 20], "mfi_period": 14},
            },
            "fundamental": {
                "enabled": True,
                "profitability": {"enabled": True, "metrics": ["roe", "roa", "roic", "gross_margin", "net_margin"]},
                "valuation": {"enabled": True, "metrics": ["pe_ttm", "pb", "ps_ttm", "pcf_ttm", "ev_ebitda", "peg"]},
                "growth": {"enabled": True, "metrics": ["revenue_yoy", "net_profit_yoy"], "periods": [1, 3, 5]},  # 年数
                "quality": {"enabled": True, "metrics": ["debt_to_equity", "current_ratio", "quick_ratio"]},
            },
            "sentiment": {
                "enabled": False,
                "news": {"enabled": False, "sources": ["sina", "eastmoney"], "sentiment_model": "bert"},
                "social": {"enabled": False, "platforms": ["weibo", "xueqiu"]},
            },
            "performance": {
                "parallel_processing": True,
                "max_workers": 4,
                "batch_size": 100,
                "cache_enabled": True,
                "cache_ttl": 3600,  # 缓存时间（秒）
            },
            "database": {"batch_insert": True, "batch_size": 1000, "connection_pool_size": 10, "max_overflow": 20},
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_enabled": True,
                "file_path": "logs/factor_engine.log",
                "max_file_size": "10MB",
                "backup_count": 5,
            },
        }

    def load_config(self, config_path: str):
        """从文件加载配置"""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                if config_file.suffix.lower() == ".json":
                    file_config = json.load(f)
                elif config_file.suffix.lower() in [".yml", ".yaml"]:
                    file_config = yaml.safe_load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_file.suffix}")

            # 深度合并配置
            self.config = self._deep_merge(self.config, file_config)

        except Exception as e:
            raise ValueError(f"加载配置文件失败: {e}")

    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """深度合并字典"""
        result = base_dict.copy()

        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config

    def get_technical_config(self) -> Dict[str, Any]:
        """获取技术因子配置"""
        return self.config.get("technical", {})

    def get_fundamental_config(self) -> Dict[str, Any]:
        """获取基本面因子配置"""
        return self.config.get("fundamental", {})

    def get_sentiment_config(self) -> Dict[str, Any]:
        """获取情感因子配置"""
        return self.config.get("sentiment", {})

    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能配置"""
        return self.config.get("performance", {})

    def get_database_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        return self.config.get("database", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.config.get("logging", {})

    def is_factor_enabled(self, factor_type: str, factor_category: str = None) -> bool:
        """
        检查因子是否启用

        Args:
            factor_type: 因子类型 (technical, fundamental, sentiment)
            factor_category: 因子分类 (momentum, trend, etc.)

        Returns:
            是否启用
        """
        factor_config = self.config.get(factor_type, {})

        if not factor_config.get("enabled", False):
            return False

        if factor_category:
            category_config = factor_config.get(factor_category, {})
            return category_config.get("enabled", True)

        return True

    def get_factor_params(self, factor_type: str, factor_category: str) -> Dict[str, Any]:
        """
        获取特定因子的参数

        Args:
            factor_type: 因子类型
            factor_category: 因子分类

        Returns:
            因子参数字典
        """
        factor_config = self.config.get(factor_type, {})
        return factor_config.get(factor_category, {})

    def update_config(self, config_updates: Dict[str, Any]):
        """更新配置"""
        self.config = self._deep_merge(self.config, config_updates)

    def save_config(self, config_path: str):
        """保存配置到文件"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                if config_file.suffix.lower() == ".json":
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                elif config_file.suffix.lower() in [".yml", ".yaml"]:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_file.suffix}")
        except Exception as e:
            raise ValueError(f"保存配置文件失败: {e}")

    def validate_config(self) -> Dict[str, Any]:
        """验证配置有效性"""
        errors = []
        warnings = []

        # 验证技术因子配置
        technical_config = self.get_technical_config()
        if technical_config.get("enabled"):
            if technical_config.get("max_periods", 0) <= 0:
                errors.append("technical.max_periods 必须大于0")

            # 验证动量因子参数
            momentum_config = technical_config.get("momentum", {})
            if momentum_config.get("enabled"):
                periods = momentum_config.get("periods", [])
                if not periods or not all(isinstance(p, int) and p > 0 for p in periods):
                    errors.append("momentum.periods 必须是正整数列表")

        # 验证基本面因子配置
        fundamental_config = self.get_fundamental_config()
        if fundamental_config.get("enabled"):
            for category in ["profitability", "valuation", "growth", "quality"]:
                category_config = fundamental_config.get(category, {})
                if category_config.get("enabled"):
                    metrics = category_config.get("metrics", [])
                    if not metrics:
                        warnings.append(f"fundamental.{category}.metrics 为空")

        # 验证性能配置
        performance_config = self.get_performance_config()
        max_workers = performance_config.get("max_workers", 1)
        if max_workers <= 0:
            errors.append("performance.max_workers 必须大于0")

        batch_size = performance_config.get("batch_size", 1)
        if batch_size <= 0:
            errors.append("performance.batch_size 必须大于0")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        summary = {
            "technical_enabled": self.is_factor_enabled("technical"),
            "fundamental_enabled": self.is_factor_enabled("fundamental"),
            "sentiment_enabled": self.is_factor_enabled("sentiment"),
            "parallel_processing": self.get_performance_config().get("parallel_processing", False),
            "max_workers": self.get_performance_config().get("max_workers", 1),
            "batch_size": self.get_performance_config().get("batch_size", 100),
        }

        # 统计启用的因子数量
        technical_config = self.get_technical_config()
        if technical_config.get("enabled"):
            enabled_technical = sum(
                1
                for category in ["momentum", "trend", "volatility", "volume"]
                if self.is_factor_enabled("technical", category)
            )
            summary["enabled_technical_categories"] = enabled_technical

        fundamental_config = self.get_fundamental_config()
        if fundamental_config.get("enabled"):
            enabled_fundamental = sum(
                1
                for category in ["profitability", "valuation", "growth", "quality"]
                if self.is_factor_enabled("fundamental", category)
            )
            summary["enabled_fundamental_categories"] = enabled_fundamental

        return summary


# 创建默认配置实例
default_config = FactorConfig()

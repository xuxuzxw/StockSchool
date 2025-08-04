#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算配置管理
提供可配置的因子计算参数
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml
from pathlib import Path


@dataclass
class MomentumFactorConfig:
    """动量类因子配置"""
    rsi_windows: List[int] = field(default_factory=lambda: [6, 14])
    williams_r_windows: List[int] = field(default_factory=lambda: [14])
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    roc_windows: List[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class TrendFactorConfig:
    """趋势类因子配置"""
    sma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    ema_windows: List[int] = field(default_factory=lambda: [12, 26])
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9


@dataclass
class VolatilityFactorConfig:
    """波动率类因子配置"""
    volatility_windows: List[int] = field(default_factory=lambda: [5, 20, 60])
    atr_windows: List[int] = field(default_factory=lambda: [14])
    bollinger_window: int = 20
    bollinger_std: float = 2.0


@dataclass
class VolumeFactorConfig:
    """成交量类因子配置"""
    volume_sma_windows: List[int] = field(default_factory=lambda: [5, 20])
    volume_ratio_windows: List[int] = field(default_factory=lambda: [5, 20])
    mfi_window: int = 14


@dataclass
class TechnicalFactorCalculationConfig:
    """技术面因子计算配置"""
    momentum: MomentumFactorConfig = field(default_factory=MomentumFactorConfig)
    trend: TrendFactorConfig = field(default_factory=TrendFactorConfig)
    volatility: VolatilityFactorConfig = field(default_factory=VolatilityFactorConfig)
    volume: VolumeFactorConfig = field(default_factory=VolumeFactorConfig)
    
    # 通用配置
    min_data_points: int = 252  # 最少需要一年的数据
    enable_data_quality_check: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'TechnicalFactorCalculationConfig':
        """
        从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置实例
        """
        if not config_path.exists():
            return cls()  # 返回默认配置
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def to_yaml(self, config_path: Path) -> None:
        """
        保存配置到YAML文件
        
        Args:
            config_path: 配置文件路径
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, allow_unicode=True)
    
    def get_factor_config(self, factor_category: str) -> Dict[str, Any]:
        """
        获取特定类别的因子配置
        
        Args:
            factor_category: 因子类别 (momentum, trend, volatility, volume)
            
        Returns:
            因子配置字典
        """
        category_configs = {
            'momentum': self.momentum,
            'trend': self.trend,
            'volatility': self.volatility,
            'volume': self.volume
        }
        
        return category_configs.get(factor_category, {})


# 默认配置实例
DEFAULT_CONFIG = TechnicalFactorCalculationConfig()


def get_default_config() -> TechnicalFactorCalculationConfig:
    """获取默认配置"""
    return DEFAULT_CONFIG


def load_config_from_file(config_path: str = "config/factor_calculation.yml") -> TechnicalFactorCalculationConfig:
    """
    从文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置实例
    """
    path = Path(config_path)
    return TechnicalFactorCalculationConfig.from_yaml(path)
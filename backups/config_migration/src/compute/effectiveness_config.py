#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子有效性分析配置管理
集中管理所有配置参数和常量
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum


class AnalysisType(Enum):
    """分析类型枚举"""
    IC_ANALYSIS = "ic_analysis"
    IR_ANALYSIS = "ir_analysis"
    LAYERED_BACKTEST = "layered_backtest"
    DECAY_ANALYSIS = "decay_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"


class StatisticalConstants:
    """统计分析常量"""
    
    # 样本数量要求
    MIN_SAMPLES_IC = 10
    MIN_SAMPLES_LAYERED = 20
    MIN_SAMPLES_DECAY = 30
    
    # 置信区间
    CONFIDENCE_LEVEL_95 = 0.95
    CONFIDENCE_LEVEL_99 = 0.99
    ALPHA_05 = 0.05
    ALPHA_01 = 0.01
    
    # 显著性检验
    T_CRITICAL_95 = 1.96
    T_CRITICAL_99 = 2.58
    
    # 默认窗口期
    DEFAULT_IC_WINDOW = 20
    DEFAULT_IR_WINDOW = 20
    DEFAULT_DECAY_PERIODS = 20
    
    # 分层参数
    DEFAULT_LAYERS = 10
    MIN_STOCKS_PER_LAYER = 2
    
    # 相关性阈值
    HIGH_CORRELATION_THRESHOLD = 0.8
    MODERATE_CORRELATION_THRESHOLD = 0.5


@dataclass
class ICAnalysisConfig:
    """IC分析配置"""
    min_samples: int = StatisticalConstants.MIN_SAMPLES_IC
    confidence_level: float = StatisticalConstants.CONFIDENCE_LEVEL_95
    include_spearman: bool = True
    include_pearson: bool = True
    window_size: int = StatisticalConstants.DEFAULT_IC_WINDOW


@dataclass
class IRAnalysisConfig:
    """IR分析配置"""
    window_size: int = StatisticalConstants.DEFAULT_IR_WINDOW
    confidence_level: float = StatisticalConstants.CONFIDENCE_LEVEL_95
    include_cumulative: bool = True
    include_t_stat: bool = True


@dataclass
class LayeredBacktestConfig:
    """分层回测配置"""
    n_layers: int = StatisticalConstants.DEFAULT_LAYERS
    min_stocks_per_layer: int = StatisticalConstants.MIN_STOCKS_PER_LAYER
    include_monotonicity_test: bool = True
    include_significance_test: bool = True
    quantile_method: str = 'qcut'  # 'qcut' or 'cut'


@dataclass
class DecayAnalysisConfig:
    """衰减分析配置"""
    max_periods: int = StatisticalConstants.DEFAULT_DECAY_PERIODS
    min_samples: int = StatisticalConstants.MIN_SAMPLES_DECAY
    include_half_life: bool = True
    include_pattern_analysis: bool = True


@dataclass
class DatabaseConfig:
    """数据库配置"""
    stock_daily_table: str = 'stock_daily'
    factor_table: str = 'stock_factors'
    batch_size: int = 1000
    query_timeout: int = 300


@dataclass
class EffectivenessAnalysisConfig:
    """因子有效性分析总配置"""
    ic_config: ICAnalysisConfig = field(default_factory=ICAnalysisConfig)
    ir_config: IRAnalysisConfig = field(default_factory=IRAnalysisConfig)
    layered_config: LayeredBacktestConfig = field(default_factory=LayeredBacktestConfig)
    decay_config: DecayAnalysisConfig = field(default_factory=DecayAnalysisConfig)
    database_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # 通用配置
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # 日志配置
    log_level: str = 'INFO'
    enable_performance_logging: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EffectivenessAnalysisConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'ic_config': self.ic_config.__dict__,
            'ir_config': self.ir_config.__dict__,
            'layered_config': self.layered_config.__dict__,
            'decay_config': self.decay_config.__dict__,
            'database_config': self.database_config.__dict__,
            'enable_caching': self.enable_caching,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'enable_parallel_processing': self.enable_parallel_processing,
            'max_workers': self.max_workers,
            'log_level': self.log_level,
            'enable_performance_logging': self.enable_performance_logging
        }


# 默认配置实例
DEFAULT_EFFECTIVENESS_CONFIG = EffectivenessAnalysisConfig()


def get_analysis_config(analysis_type: AnalysisType, 
                       custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    获取特定分析类型的配置
    
    Args:
        analysis_type: 分析类型
        custom_config: 自定义配置覆盖
        
    Returns:
        配置字典
    """
    base_config = DEFAULT_EFFECTIVENESS_CONFIG
    
    if analysis_type == AnalysisType.IC_ANALYSIS:
        config = base_config.ic_config.__dict__
    elif analysis_type == AnalysisType.IR_ANALYSIS:
        config = base_config.ir_config.__dict__
    elif analysis_type == AnalysisType.LAYERED_BACKTEST:
        config = base_config.layered_config.__dict__
    elif analysis_type == AnalysisType.DECAY_ANALYSIS:
        config = base_config.decay_config.__dict__
    else:
        config = {}
    
    # 应用自定义配置
    if custom_config:
        config.update(custom_config)
    
    return config
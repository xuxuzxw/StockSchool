#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API配置管理
统一管理API相关的配置项
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from src.utils.config_loader import config


@dataclass
class APIConfig:
    """API配置类"""
    
    # 请求限制
    max_stocks_per_request: int = 1000
    max_factors_per_request: int = 100
    max_date_range_days: int = 365
    
    # 分页配置
    default_page_size: int = 20
    max_page_size: int = 1000
    
    # 缓存配置
    cache_enabled: bool = True
    default_cache_ttl: int = 300  # 5分钟
    
    # 任务配置
    task_timeout: int = 1800  # 30分钟
    max_concurrent_tasks: int = 10
    
    # 认证配置
    token_expire_hours: int = 24
    require_authentication: bool = True
    
    # 限流配置
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 1000
    
    @classmethod
    def from_config(cls) -> 'APIConfig':
        """从配置文件创建API配置"""
        api_config = config.get('api', {})
        
        return cls(
            max_stocks_per_request=api_config.get('max_stocks_per_request', 1000),
            max_factors_per_request=api_config.get('max_factors_per_request', 100),
            max_date_range_days=api_config.get('max_date_range_days', 365),
            default_page_size=api_config.get('default_page_size', 20),
            max_page_size=api_config.get('max_page_size', 1000),
            cache_enabled=api_config.get('cache_enabled', True),
            default_cache_ttl=api_config.get('default_cache_ttl', 300),
            task_timeout=api_config.get('task_timeout', 1800),
            max_concurrent_tasks=api_config.get('max_concurrent_tasks', 10),
            token_expire_hours=api_config.get('token_expire_hours', 24),
            require_authentication=api_config.get('require_authentication', True),
            rate_limit_per_minute=api_config.get('rate_limit_per_minute', 100),
            rate_limit_per_hour=api_config.get('rate_limit_per_hour', 1000)
        )


# 全局API配置实例
api_config = APIConfig.from_config()


class FactorMetadata:
    """因子元数据管理"""
    
    # 支持的因子类型
    SUPPORTED_FACTOR_TYPES = ['technical', 'fundamental', 'sentiment']
    
    # 技术面因子
    TECHNICAL_FACTORS = [
        'rsi_6', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'sma_5', 'sma_10', 'sma_20', 'sma_60', 'ema_12', 'ema_26',
        'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
        'atr_14', 'volatility_5', 'volatility_20', 'volatility_60',
        'volume_sma_5', 'volume_sma_20', 'volume_ratio', 'vpt', 'mfi'
    ]
    
    # 基本面因子
    FUNDAMENTAL_FACTORS = [
        'pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio',
        'ev_ebitda', 'peg_ratio', 'roe', 'roa', 'roic',
        'gross_margin', 'net_margin', 'operating_margin'
    ]
    
    # 情绪面因子
    SENTIMENT_FACTORS = [
        'money_flow', 'net_inflow_rate', 'large_order_ratio',
        'medium_order_ratio', 'small_order_ratio', 'turnover_rate',
        'attention_score', 'attention_change_rate', 'price_momentum_sentiment',
        'volatility_sentiment', 'bull_bear_ratio', 'sentiment_volatility',
        'abnormal_volume', 'abnormal_return', 'limit_up_signal',
        'limit_down_signal', 'gap_signal', 'volume_anomaly'
    ]
    
    @classmethod
    def get_all_factors(cls) -> List[str]:
        """获取所有支持的因子"""
        return cls.TECHNICAL_FACTORS + cls.FUNDAMENTAL_FACTORS + cls.SENTIMENT_FACTORS
    
    @classmethod
    def get_factors_by_type(cls, factor_type: str) -> List[str]:
        """根据类型获取因子列表"""
        factor_map = {
            'technical': cls.TECHNICAL_FACTORS,
            'fundamental': cls.FUNDAMENTAL_FACTORS,
            'sentiment': cls.SENTIMENT_FACTORS
        }
        return factor_map.get(factor_type, [])
    
    @classmethod
    def validate_factors(cls, factor_names: List[str]) -> List[str]:
        """验证因子名称，返回无效的因子列表"""
        all_factors = set(cls.get_all_factors())
        return [f for f in factor_names if f not in all_factors]


# 错误消息常量
class ErrorMessages:
    """错误消息常量"""
    
    INVALID_STOCK_CODE = "无效的股票代码格式"
    INVALID_FACTOR_NAME = "不支持的因子名称"
    INVALID_DATE_RANGE = "无效的日期范围"
    INSUFFICIENT_PERMISSION = "权限不足"
    TASK_NOT_FOUND = "任务不存在"
    DATABASE_ERROR = "数据库连接错误"
    CALCULATION_ERROR = "因子计算失败"
    STANDARDIZATION_ERROR = "因子标准化失败"
    ANALYSIS_ERROR = "有效性分析失败"
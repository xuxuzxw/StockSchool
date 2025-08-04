#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的因子计算数据模型
添加数据验证、常量定义和性能优化
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Protocol
from datetime import datetime, date
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from .factor_exceptions import InvalidFactorParameterError, InsufficientDataError


# 常量定义
class FactorConstants:
    """因子计算相关常量"""
    DEFAULT_PERCENTILES = [5, 10, 25, 50, 75, 90, 95]
    MIN_WINDOW_SIZE = 1
    MAX_WINDOW_SIZE = 252  # 一年交易日
    DEFAULT_RSI_WINDOW = 14
    DEFAULT_MACD_FAST = 12
    DEFAULT_MACD_SLOW = 26
    DEFAULT_MACD_SIGNAL = 9


class FactorType(Enum):
    """因子类型枚举"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"


class FactorCategory(Enum):
    """因子分类枚举"""
    # 技术面分类
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    
    # 基本面分类
    VALUATION = "valuation"
    PROFITABILITY = "profitability"
    GROWTH = "growth"
    QUALITY = "quality"
    LEVERAGE = "leverage"
    
    # 情绪面分类
    ATTENTION = "attention"
    FLOW = "flow"
    SENTIMENT_STRENGTH = "sentiment_strength"
    EVENT = "event"


class CalculationStatus(Enum):
    """计算状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# 协议定义
class Serializable(Protocol):
    """可序列化协议"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        ...


class Validatable(Protocol):
    """可验证协议"""
    
    def validate(self) -> bool:
        """验证数据有效性"""
        ...


# 基础类
class BaseFactorModel(ABC):
    """因子模型基类"""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """验证数据有效性"""
        pass


@dataclass
class FactorMetadata(BaseFactorModel):
    """改进的因子元数据"""
    __slots__ = ('name', 'description', 'factor_type', 'category', 'formula', 
                 'parameters', 'data_requirements', 'min_periods', 'created_at', 'updated_at')
    
    name: str
    description: str
    factor_type: FactorType
    category: FactorCategory
    formula: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    data_requirements: List[str] = field(default_factory=list)
    min_periods: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """初始化后验证"""
        self.validate()
    
    def validate(self) -> bool:
        """验证数据有效性"""
        if not self.name or not isinstance(self.name, str):
            raise InvalidFactorParameterError("name", self.name, "因子名称不能为空且必须是字符串")
        
        if not self.description or not isinstance(self.description, str):
            raise InvalidFactorParameterError("description", self.description, "因子描述不能为空且必须是字符串")
        
        if self.min_periods < FactorConstants.MIN_WINDOW_SIZE:
            raise InvalidFactorParameterError("min_periods", self.min_periods, 
                                            f"最小周期不能小于 {FactorConstants.MIN_WINDOW_SIZE}")
        
        if self.min_periods > FactorConstants.MAX_WINDOW_SIZE:
            raise InvalidFactorParameterError("min_periods", self.min_periods,
                                            f"最小周期不能大于 {FactorConstants.MAX_WINDOW_SIZE}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'factor_type': self.factor_type.value,
            'category': self.category.value,
            'formula': self.formula,
            'parameters': self.parameters,
            'data_requirements': self.data_requirements,
            'min_periods': self.min_periods,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class FactorValue(BaseFactorModel):
    """改进的因子值"""
    __slots__ = ('ts_code', 'trade_date', 'factor_name', 'raw_value', 
                 'standardized_value', 'percentile_rank', 'industry_rank', 'is_valid', 'created_at')
    
    ts_code: str
    trade_date: date
    factor_name: str
    raw_value: Optional[float]
    standardized_value: Optional[float] = None
    percentile_rank: Optional[float] = None
    industry_rank: Optional[float] = None
    is_valid: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """初始化后验证"""
        self.validate()
    
    def validate(self) -> bool:
        """验证数据有效性"""
        if not self.ts_code or not isinstance(self.ts_code, str):
            raise InvalidFactorParameterError("ts_code", self.ts_code, "股票代码不能为空")
        
        if not self.factor_name or not isinstance(self.factor_name, str):
            raise InvalidFactorParameterError("factor_name", self.factor_name, "因子名称不能为空")
        
        # 验证分位数排名范围
        if self.percentile_rank is not None and not (0 <= self.percentile_rank <= 1):
            raise InvalidFactorParameterError("percentile_rank", self.percentile_rank, 
                                            "分位数排名必须在0-1之间")
        
        if self.industry_rank is not None and not (0 <= self.industry_rank <= 1):
            raise InvalidFactorParameterError("industry_rank", self.industry_rank,
                                            "行业排名必须在0-1之间")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'ts_code': self.ts_code,
            'trade_date': self.trade_date.isoformat(),
            'factor_name': self.factor_name,
            'raw_value': self.raw_value,
            'standardized_value': self.standardized_value,
            'percentile_rank': self.percentile_rank,
            'industry_rank': self.industry_rank,
            'is_valid': self.is_valid,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class FactorStatistics(BaseFactorModel):
    """改进的因子统计信息"""
    __slots__ = ('factor_name', 'calculation_date', 'total_count', 'valid_count',
                 'mean_value', 'standard_deviation', 'min_value', 'max_value', 
                 'percentiles', 'coverage_rate')
    
    factor_name: str
    calculation_date: date
    total_count: int
    valid_count: int
    mean_value: Optional[float] = None
    standard_deviation: Optional[float] = None  # 改名提高可读性
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    percentiles: Dict[str, float] = field(default_factory=dict)
    coverage_rate: float = 0.0
    
    def __post_init__(self):
        """初始化后验证"""
        self.validate()
    
    def validate(self) -> bool:
        """验证数据有效性"""
        if self.total_count < 0:
            raise InvalidFactorParameterError("total_count", self.total_count, "总数量不能为负数")
        
        if self.valid_count < 0 or self.valid_count > self.total_count:
            raise InvalidFactorParameterError("valid_count", self.valid_count, 
                                            "有效数量必须在0到总数量之间")
        
        return True
    
    def calculate_statistics(self, values: List[float], 
                           percentiles: List[int] = None) -> None:
        """计算统计信息"""
        if percentiles is None:
            percentiles = FactorConstants.DEFAULT_PERCENTILES
        
        if not values:
            return
        
        valid_values = self._filter_valid_values(values)
        self.valid_count = len(valid_values)
        self.coverage_rate = self._calculate_coverage_rate()
        
        if valid_values:
            self._calculate_basic_statistics(valid_values)
            self._calculate_percentiles(valid_values, percentiles)
    
    def _filter_valid_values(self, values: List[float]) -> List[float]:
        """过滤有效值"""
        return [v for v in values if not pd.isna(v) and np.isfinite(v)]
    
    def _calculate_coverage_rate(self) -> float:
        """计算覆盖率"""
        return self.valid_count / self.total_count if self.total_count > 0 else 0.0
    
    def _calculate_basic_statistics(self, valid_values: List[float]) -> None:
        """计算基础统计量"""
        self.mean_value = float(np.mean(valid_values))
        self.standard_deviation = float(np.std(valid_values))
        self.min_value = float(np.min(valid_values))
        self.max_value = float(np.max(valid_values))
    
    def _calculate_percentiles(self, valid_values: List[float], 
                             percentiles: List[int]) -> None:
        """计算分位数"""
        for p in percentiles:
            self.percentiles[f'p{p}'] = float(np.percentile(valid_values, p))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'calculation_date': self.calculation_date.isoformat(),
            'total_count': self.total_count,
            'valid_count': self.valid_count,
            'mean_value': self.mean_value,
            'standard_deviation': self.standard_deviation,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'percentiles': self.percentiles,
            'coverage_rate': self.coverage_rate
        }


# 工厂类
class FactorMetadataFactory:
    """因子元数据工厂"""
    
    @staticmethod
    def create_technical_factor(name: str, description: str, category: FactorCategory,
                              **kwargs) -> FactorMetadata:
        """创建技术面因子元数据"""
        return FactorMetadata(
            name=name,
            description=description,
            factor_type=FactorType.TECHNICAL,
            category=category,
            **kwargs
        )
    
    @staticmethod
    def create_fundamental_factor(name: str, description: str, category: FactorCategory,
                                **kwargs) -> FactorMetadata:
        """创建基本面因子元数据"""
        return FactorMetadata(
            name=name,
            description=description,
            factor_type=FactorType.FUNDAMENTAL,
            category=category,
            **kwargs
        )
    
    @staticmethod
    def create_sentiment_factor(name: str, description: str, category: FactorCategory,
                              **kwargs) -> FactorMetadata:
        """创建情绪面因子元数据"""
        return FactorMetadata(
            name=name,
            description=description,
            factor_type=FactorType.SENTIMENT,
            category=category,
            **kwargs
        )
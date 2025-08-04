#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算引擎抽象基类
定义因子计算引擎的统一接口和抽象方法
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from datetime import date, datetime
import pandas as pd
from loguru import logger

from .factor_models import (
    FactorType, FactorResult, CalculationTask, FactorMetadata,
    FactorConfig, CalculationStatus
)


class BaseFactorEngine(ABC):
    """因子计算引擎抽象基类"""
    
    def __init__(self, engine, factor_type: FactorType):
        """
        初始化因子引擎
        
        Args:
            engine: 数据库引擎
            factor_type: 因子类型
        """
        self.engine = engine
        self.factor_type = factor_type
        self._factor_configs: Dict[str, FactorConfig] = {}
        self._factor_metadata: Dict[str, FactorMetadata] = {}
        
        # 初始化因子配置
        self._initialize_factors()
        
        logger.info(f"{self.factor_type.value}因子引擎初始化完成")
    
    @abstractmethod
    def _initialize_factors(self):
        """初始化因子配置和元数据（子类实现）"""
        pass
    
    @abstractmethod
    def calculate_factors(self, 
                         ts_code: str,
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None,
                         factor_names: Optional[List[str]] = None) -> FactorResult:
        """
        计算因子（子类实现）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            factor_names: 指定计算的因子名称列表
            
        Returns:
            因子计算结果
        """
        pass
    
    @abstractmethod
    def get_required_data(self, 
                         ts_code: str,
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None) -> pd.DataFrame:
        """
        获取计算所需的数据（子类实现）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            所需数据的DataFrame
        """
        pass    

    def register_factor(self, metadata: FactorMetadata, config: Optional[FactorConfig] = None):
        """注册因子"""
        self._factor_metadata[metadata.name] = metadata
        if config:
            self._factor_configs[metadata.name] = config
        else:
            self._factor_configs[metadata.name] = FactorConfig(name=metadata.name)
    
    def get_factor_metadata(self, name: str) -> Optional[FactorMetadata]:
        """获取因子元数据"""
        return self._factor_metadata.get(name)
    
    def get_factor_config(self, name: str) -> Optional[FactorConfig]:
        """获取因子配置"""
        return self._factor_configs.get(name)
    
    def list_factors(self) -> List[str]:
        """列出所有因子名称"""
        return list(self._factor_metadata.keys())
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        验证数据完整性
        
        Args:
            data: 数据DataFrame
            required_columns: 必需的列名列表
            
        Returns:
            是否通过验证
        """
        if data.empty:
            logger.warning("数据为空")
            return False
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"缺少必需的列: {missing_columns}")
            return False
        
        return True
    
    def handle_calculation_error(self, ts_code: str, error: Exception) -> FactorResult:
        """
        处理计算错误
        
        Args:
            ts_code: 股票代码
            error: 异常对象
            
        Returns:
            错误结果
        """
        logger.error(f"计算股票 {ts_code} 的{self.factor_type.value}因子时出错: {str(error)}")
        
        return FactorResult(
            ts_code=ts_code,
            calculation_date=datetime.now(),
            factor_type=self.factor_type,
            status=CalculationStatus.FAILED,
            error_message=str(error)
        )


class BaseFactorCalculator(ABC):
    """因子计算器抽象基类"""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        计算因子值
        
        Args:
            data: 输入数据
            **kwargs: 计算参数
            
        Returns:
            计算结果
        """
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """
        验证计算参数
        
        Args:
            **kwargs: 参数字典
            
        Returns:
            是否通过验证
        """
        return True


class BaseFactorStandardizer(ABC):
    """因子标准化器抽象基类"""
    
    @abstractmethod
    def standardize(self, 
                   factor_values: pd.Series,
                   method: str = "zscore") -> pd.Series:
        """
        标准化因子值
        
        Args:
            factor_values: 因子值序列
            method: 标准化方法
            
        Returns:
            标准化后的因子值
        """
        pass
    
    @abstractmethod
    def calculate_percentile_rank(self, factor_values: pd.Series) -> pd.Series:
        """
        计算分位数排名
        
        Args:
            factor_values: 因子值序列
            
        Returns:
            分位数排名序列
        """
        pass


if __name__ == "__main__":
    print("因子计算引擎抽象基类定义完成")
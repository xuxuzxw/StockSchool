#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证混入类
提供通用的数据验证功能
"""

import pandas as pd
from typing import List
from loguru import logger


class DataValidationError(Exception):
    """数据验证异常"""
    pass


class InsufficientDataError(DataValidationError):
    """数据不足异常"""
    pass


class DataValidationMixin:
    """数据验证混入类"""
    
    def validate_required_columns(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证必需的数据列
        
        Args:
            data: 输入数据
            required_columns: 必需的列名列表
            
        Raises:
            DataValidationError: 当缺少必需列时
        """
        if data.empty:
            raise DataValidationError("输入数据不能为空")
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataValidationError(f"缺少必要列: {missing_columns}")
    
    def validate_data_length(self, data: pd.DataFrame, min_length: int, 
                           factor_name: str = "") -> None:
        """
        验证数据长度
        
        Args:
            data: 输入数据
            min_length: 最小长度要求
            factor_name: 因子名称（用于错误信息）
            
        Raises:
            InsufficientDataError: 当数据长度不足时
        """
        if len(data) < min_length:
            error_msg = f"数据长度不足，{factor_name}需要至少 {min_length} 条记录，实际 {len(data)} 条"
            raise InsufficientDataError(error_msg)
    
    def validate_numeric_column(self, data: pd.DataFrame, column: str) -> None:
        """
        验证数值列的有效性
        
        Args:
            data: 输入数据
            column: 列名
            
        Raises:
            DataValidationError: 当列数据无效时
        """
        if column not in data.columns:
            raise DataValidationError(f"缺少列: {column}")
        
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise DataValidationError(f"列 {column} 必须是数值类型")
        
        # 检查是否全为空值
        if data[column].isna().all():
            raise DataValidationError(f"列 {column} 全部为空值")
    
    def log_data_quality_info(self, data: pd.DataFrame, factor_name: str) -> None:
        """
        记录数据质量信息
        
        Args:
            data: 输入数据
            factor_name: 因子名称
        """
        total_rows = len(data)
        null_counts = data.isnull().sum()
        
        logger.debug(f"{factor_name} 数据质量信息:")
        logger.debug(f"  总行数: {total_rows}")
        
        for col, null_count in null_counts.items():
            if null_count > 0:
                null_ratio = null_count / total_rows * 100
                logger.debug(f"  {col} 空值: {null_count} ({null_ratio:.1f}%)")
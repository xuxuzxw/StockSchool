#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子有效性分析模板方法模式实现
标准化分析流程，提高代码复用性
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
from loguru import logger

from .factor_exceptions import DataValidationError, InsufficientDataError
from .performance_decorators import timing_decorator, memory_monitor_decorator


class EffectivenessAnalysisTemplate(ABC):
    """因子有效性分析模板基类"""
    
    def __init__(self, engine, config: Optional[Dict[str, Any]] = None):
        self.engine = engine
        self.config = config or self._get_default_config()
        self._validation_errors = []
    
    @timing_decorator
    @memory_monitor_decorator
    def analyze(self, factor_data: pd.DataFrame, return_data: pd.DataFrame, 
                **kwargs) -> Dict[str, Any]:
        """
        模板方法：执行完整的有效性分析流程
        
        Args:
            factor_data: 因子数据
            return_data: 收益率数据
            **kwargs: 额外参数
            
        Returns:
            分析结果字典
        """
        start_time = datetime.now()
        
        try:
            # 1. 数据预处理和验证
            self._preprocess_data(factor_data, return_data)
            self._validate_input_data(factor_data, return_data)
            
            # 2. 数据合并
            merged_data = self._merge_data(factor_data, return_data)
            
            # 3. 执行具体分析（子类实现）
            analysis_results = self._execute_analysis(merged_data, **kwargs)
            
            # 4. 后处理和验证结果
            final_results = self._postprocess_results(analysis_results)
            
            # 5. 生成分析报告
            report = self._generate_report(final_results, start_time)
            
            return report
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__} 分析失败: {e}")
            return self._handle_analysis_error(e, start_time)
    
    def _preprocess_data(self, factor_data: pd.DataFrame, return_data: pd.DataFrame):
        """数据预处理"""
        # 标准化日期格式
        for df in [factor_data, return_data]:
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 数据类型转换
        if 'factor_value' in factor_data.columns:
            factor_data['factor_value'] = pd.to_numeric(
                factor_data['factor_value'], errors='coerce'
            )
        
        if 'return_rate' in return_data.columns:
            return_data['return_rate'] = pd.to_numeric(
                return_data['return_rate'], errors='coerce'
            )
    
    def _validate_input_data(self, factor_data: pd.DataFrame, return_data: pd.DataFrame):
        """验证输入数据"""
        self._validation_errors.clear()
        
        # 检查数据是否为空
        if factor_data.empty:
            self._validation_errors.append("因子数据为空")
        
        if return_data.empty:
            self._validation_errors.append("收益率数据为空")
        
        # 检查必要列
        required_factor_cols = self._get_required_factor_columns()
        missing_factor_cols = [col for col in required_factor_cols 
                              if col not in factor_data.columns]
        if missing_factor_cols:
            self._validation_errors.append(f"因子数据缺少列: {missing_factor_cols}")
        
        required_return_cols = self._get_required_return_columns()
        missing_return_cols = [col for col in required_return_cols 
                              if col not in return_data.columns]
        if missing_return_cols:
            self._validation_errors.append(f"收益率数据缺少列: {missing_return_cols}")
        
        # 检查数据量
        min_samples = self.config.get('min_samples', 30)
        if len(factor_data) < min_samples:
            self._validation_errors.append(
                f"因子数据量不足，需要至少{min_samples}条，实际{len(factor_data)}条"
            )
        
        if self._validation_errors:
            raise DataValidationError("; ".join(self._validation_errors))
    
    def _merge_data(self, factor_data: pd.DataFrame, return_data: pd.DataFrame) -> pd.DataFrame:
        """合并因子数据和收益率数据"""
        merge_keys = ['ts_code', 'trade_date']
        
        merged_data = pd.merge(
            factor_data, return_data,
            on=merge_keys,
            how='inner',
            suffixes=('_factor', '_return')
        )
        
        if merged_data.empty:
            raise DataValidationError("因子数据和收益率数据无法匹配")
        
        logger.info(f"数据合并完成，共{len(merged_data)}条记录")
        return merged_data
    
    @abstractmethod
    def _execute_analysis(self, merged_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """执行具体的分析逻辑（子类实现）"""
        pass
    
    def _postprocess_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """后处理分析结果"""
        # 添加元数据
        results['metadata'] = {
            'analysis_type': self.__class__.__name__,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _generate_report(self, results: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """生成分析报告"""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'success': True,
            'results': results,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_analysis_error(self, error: Exception, start_time: datetime) -> Dict[str, Any]:
        """处理分析错误"""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'success': False,
            'error': str(error),
            'error_type': type(error).__name__,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        }
    
    @abstractmethod
    def _get_required_factor_columns(self) -> List[str]:
        """获取必需的因子数据列（子类实现）"""
        pass
    
    @abstractmethod
    def _get_required_return_columns(self) -> List[str]:
        """获取必需的收益率数据列（子类实现）"""
        pass
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置（子类实现）"""
        pass
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子服务层
分离业务逻辑，提高代码可维护性和可测试性
"""

from typing import List, Dict, Any, Optional
from datetime import date
import pandas as pd
import logging

from src.utils.db import get_db_engine
from src.compute.factor_engine import FactorEngine
from src.compute.factor_standardizer import FactorStandardizer
from src.compute.manual_calculation_trigger import ManualCalculationTrigger, CalculationRequest, CalculationMode

logger = logging.getLogger(__name__)


class FactorService:
    """因子业务服务类"""
    
    def __init__(self):
        self.engine = get_db_engine()
        self.factor_engine = FactorEngine(self.engine)
        self.standardizer = FactorStandardizer(self.engine)
        self.manual_trigger = ManualCalculationTrigger(self.engine)
    
    def query_factors(
        self,
        ts_codes: List[str],
        factor_names: Optional[List[str]] = None,
        factor_types: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        standardized: bool = False
    ) -> Dict[str, Any]:
        """
        查询因子数据
        
        Args:
            ts_codes: 股票代码列表
            factor_names: 因子名称列表
            factor_types: 因子类型列表
            start_date: 开始日期
            end_date: 结束日期
            standardized: 是否返回标准化值
            
        Returns:
            包含因子数据的字典
        """
        try:
            # 构建查询参数
            query_params = self._build_query_params(
                ts_codes, factor_names, factor_types, start_date, end_date
            )
            
            # 执行查询
            data = self._execute_factor_query(query_params, standardized)
            
            # 格式化响应数据
            return self._format_factor_response(data, query_params)
            
        except Exception as e:
            logger.error(f"查询因子数据失败: {e}")
            raise
    
    def _build_query_params(
        self,
        ts_codes: List[str],
        factor_names: Optional[List[str]],
        factor_types: Optional[List[str]],
        start_date: Optional[date],
        end_date: Optional[date]
    ) -> Dict[str, Any]:
        """构建查询参数"""
        query_params = {
            'ts_codes': ts_codes,
            'start_date': start_date,
            'end_date': end_date or date.today()
        }
        
        if factor_names:
            query_params['factor_names'] = factor_names
        
        if factor_types and 'all' not in factor_types:
            query_params['factor_types'] = factor_types
            
        return query_params
    
    def _execute_factor_query(
        self, 
        query_params: Dict[str, Any], 
        standardized: bool
    ) -> pd.DataFrame:
        """执行因子查询"""
        if standardized:
            return self.factor_engine.get_standardized_factors(**query_params)
        else:
            return self.factor_engine.get_factors(**query_params)
    
    def _format_factor_response(
        self, 
        data: pd.DataFrame, 
        query_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """格式化因子响应数据"""
        result_data = []
        
        if not data.empty:
            for _, row in data.iterrows():
                factor_values = {
                    col: row[col] for col in data.columns 
                    if col not in ['ts_code', 'factor_date']
                }
                result_data.append({
                    'ts_code': row['ts_code'],
                    'factor_date': row['factor_date'],
                    'factor_values': factor_values
                })
        
        return {
            "factors": result_data,
            "total_count": len(result_data),
            "query_params": query_params
        }


class CalculationModeStrategy:
    """计算模式策略基类"""
    
    def determine_mode(self, request_data: Dict[str, Any]) -> CalculationMode:
        """确定计算模式"""
        raise NotImplementedError


class DefaultCalculationModeStrategy(CalculationModeStrategy):
    """默认计算模式策略"""
    
    def determine_mode(self, request_data: Dict[str, Any]) -> CalculationMode:
        """根据请求数据确定计算模式"""
        ts_codes = request_data.get('ts_codes', [])
        factor_names = request_data.get('factor_names', [])
        start_date = request_data.get('start_date')
        end_date = request_data.get('end_date')
        
        if ts_codes and len(ts_codes) == 1:
            return CalculationMode.SINGLE_STOCK
        elif ts_codes and len(ts_codes) > 1:
            return CalculationMode.MULTIPLE_STOCKS
        elif factor_names and len(factor_names) == 1:
            return CalculationMode.SINGLE_FACTOR
        elif factor_names and len(factor_names) > 1:
            return CalculationMode.MULTIPLE_FACTORS
        elif start_date and end_date:
            return CalculationMode.DATE_RANGE
        else:
            return CalculationMode.FULL_MARKET


class FactorCalculationService:
    """因子计算服务类"""
    
    def __init__(self, mode_strategy: CalculationModeStrategy = None):
        self.engine = get_db_engine()
        self.manual_trigger = ManualCalculationTrigger(self.engine)
        self.mode_strategy = mode_strategy or DefaultCalculationModeStrategy()
    
    def submit_calculation_request(self, request_data: Dict[str, Any]) -> str:
        """
        提交因子计算请求
        
        Args:
            request_data: 计算请求数据
            
        Returns:
            任务ID
        """
        try:
            # 确定计算模式
            mode = self.mode_strategy.determine_mode(request_data)
            
            # 创建计算请求
            calc_request = self._create_calculation_request(request_data, mode)
            
            # 提交任务
            task_id = self.manual_trigger.submit_calculation_request(calc_request)
            
            return task_id
            
        except Exception as e:
            logger.error(f"提交计算请求失败: {e}")
            raise
    
    def _create_calculation_request(
        self, 
        request_data: Dict[str, Any], 
        mode: CalculationMode
    ) -> CalculationRequest:
        """创建计算请求对象"""
        from datetime import datetime
        
        return CalculationRequest(
            request_id=f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode=mode,
            ts_codes=request_data.get('ts_codes', []),
            factor_names=request_data.get('factor_names', []),
            factor_types=request_data.get('factor_types', []),
            calculation_date=request_data.get('calculation_date'),
            start_date=request_data.get('start_date'),
            end_date=request_data.get('end_date'),
            force_recalculate=request_data.get('force_recalculate', False),
            priority=request_data.get('priority', 'normal')
        )
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        return self.manual_trigger.get_task_status(task_id)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动计算触发器
实现特定因子的手动重新计算、参数调整和结果验证
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from loguru import logger
from sqlalchemy import text
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

from .factor_models import FactorType, FactorResult, CalculationStatus
from .technical_factor_engine import TechnicalFactorEngine
from .fundamental_factor_engine import FundamentalFactorEngine
from .sentiment_factor_engine import SentimentFactorEngine
from .task_scheduler import TaskConfig, TaskType, TaskPriority


class CalculationMode(Enum):
    """计算模式枚举"""
    SINGLE_STOCK = "single_stock"
    MULTIPLE_STOCKS = "multiple_stocks"
    SINGLE_FACTOR = "single_factor"
    MULTIPLE_FACTORS = "multiple_factors"
    DATE_RANGE = "date_range"


class ValidationResult(Enum):
    """验证结果枚举"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class CalculationRequest:
    """计算请求"""
    request_id: str
    mode: CalculationMode
    ts_codes: List[str]
    factor_names: List[str]
    factor_types: List[str]
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    parameters: Optional[Dict[str, Any]] = None
    force_recalculate: bool = False
    validate_results: bool = True
    created_time: datetime = None
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()


@dataclass
class CalculationComparison:
    """计算结果比较"""
    factor_name: str
    ts_code: str
    date_value: date
    old_value: Optional[float]
    new_value: Optional[float]
    difference: Optional[float]
    percentage_change: Optional[float]
    is_significant: bool


class ManualCalculationTrigger:
    """手动计算触发器"""
    
    def __init__(self, engine):
        """初始化手动计算触发器"""
        self.engine = engine
        
        # 初始化因子计算引擎
        self.technical_engine = TechnicalFactorEngine(engine)
        self.fundamental_engine = FundamentalFactorEngine(engine)
        self.sentiment_engine = SentimentFactorEngine(engine)
        
        # 计算历史记录
        self.calculation_history = {}  # request_id -> results
        
        # 验证规则
        self.validation_rules = self._setup_validation_rules()
    
    def _setup_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """设置验证规则"""
        return {
            'value_range': {
                'rsi_14': {'min': 0, 'max': 100},
                'pe_ttm': {'min': -100, 'max': 1000},
                'pb': {'min': 0, 'max': 50},
                'roe': {'min': -100, 'max': 100}
            },
            'change_threshold': {
                'default': 0.1,  # 10%变化阈值
                'rsi_14': 0.05,  # RSI变化5%就算显著
                'pe_ttm': 0.2    # PE变化20%算显著
            },
            'null_tolerance': {
                'default': 0.05,  # 默认5%的空值容忍度
                'technical': 0.02,  # 技术指标2%
                'fundamental': 0.1  # 基本面指标10%
            }
        }
    
    def submit_calculation_request(self, request: CalculationRequest) -> str:
        """提交计算请求"""
        try:
            # 验证请求参数
            self._validate_request(request)
            
            # 保存请求记录
            self._save_request_to_db(request)
            
            logger.info(f"提交手动计算请求: {request.request_id}")
            
            return request.request_id
            
        except Exception as e:
            logger.error(f"提交计算请求失败: {e}")
            raise
    
    def execute_calculation_request(self, request_id: str) -> Dict[str, Any]:
        """执行计算请求"""
        try:
            # 加载请求
            request = self._load_request_from_db(request_id)
            if not request:
                raise ValueError(f"未找到计算请求: {request_id}")
            
            logger.info(f"开始执行计算请求: {request_id}")
            
            # 备份现有数据（如果需要比较）
            backup_data = None
            if request.validate_results:
                backup_data = self._backup_existing_data(request)
            
            # 执行计算
            calculation_results = self._execute_calculation(request)
            
            # 验证结果
            validation_results = []
            if request.validate_results:
                validation_results = self._validate_calculation_results(
                    request, calculation_results, backup_data
                )
            
            # 保存结果
            execution_result = {
                'request_id': request_id,
                'status': 'completed',
                'calculation_results': calculation_results,
                'validation_results': validation_results,
                'execution_time': datetime.now(),
                'summary': self._generate_execution_summary(calculation_results, validation_results)
            }
            
            # 保存到历史记录
            self.calculation_history[request_id] = execution_result
            
            # 保存到数据库
            self._save_execution_result(execution_result)
            
            logger.info(f"计算请求执行完成: {request_id}")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"执行计算请求失败: {e}")
            
            # 保存失败结果
            error_result = {
                'request_id': request_id,
                'status': 'failed',
                'error_message': str(e),
                'execution_time': datetime.now()
            }
            
            self.calculation_history[request_id] = error_result
            self._save_execution_result(error_result)
            
            raise
    
    def _validate_request(self, request: CalculationRequest):
        """验证计算请求"""
        # 检查股票代码
        if not request.ts_codes:
            raise ValueError("股票代码列表不能为空")
        
        # 检查因子名称
        if not request.factor_names and not request.factor_types:
            raise ValueError("必须指定因子名称或因子类型")
        
        # 检查日期范围
        if request.start_date and request.end_date:
            if request.start_date > request.end_date:
                raise ValueError("开始日期不能晚于结束日期")
        
        # 检查股票代码格式
        for ts_code in request.ts_codes:
            if not self._is_valid_ts_code(ts_code):
                raise ValueError(f"无效的股票代码: {ts_code}")
    
    def _is_valid_ts_code(self, ts_code: str) -> bool:
        """检查股票代码格式是否有效"""
        import re
        pattern = r'^\d{6}\.(SZ|SH)$'
        return bool(re.match(pattern, ts_code))
    
    def _save_request_to_db(self, request: CalculationRequest):
        """保存请求到数据库"""
        try:
            request_data = {
                'request_id': request.request_id,
                'mode': request.mode.value,
                'ts_codes': json.dumps(request.ts_codes),
                'factor_names': json.dumps(request.factor_names),
                'factor_types': json.dumps(request.factor_types),
                'start_date': request.start_date,
                'end_date': request.end_date,
                'parameters': json.dumps(request.parameters) if request.parameters else None,
                'force_recalculate': request.force_recalculate,
                'validate_results': request.validate_results,
                'created_time': request.created_time,
                'status': 'submitted'
            }
            
            df = pd.DataFrame([request_data])
            df.to_sql(
                'manual_calculation_requests',
                self.engine,
                if_exists='append',
                index=False
            )
            
        except Exception as e:
            logger.error(f"保存计算请求失败: {e}")
            raise
    
    def _load_request_from_db(self, request_id: str) -> Optional[CalculationRequest]:
        """从数据库加载请求"""
        try:
            query = text("""
                SELECT * FROM manual_calculation_requests 
                WHERE request_id = :request_id
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'request_id': request_id})
                row = result.fetchone()
                
                if not row:
                    return None
                
                return CalculationRequest(
                    request_id=row.request_id,
                    mode=CalculationMode(row.mode),
                    ts_codes=json.loads(row.ts_codes),
                    factor_names=json.loads(row.factor_names),
                    factor_types=json.loads(row.factor_types),
                    start_date=row.start_date,
                    end_date=row.end_date,
                    parameters=json.loads(row.parameters) if row.parameters else None,
                    force_recalculate=row.force_recalculate,
                    validate_results=row.validate_results,
                    created_time=row.created_time
                )
                
        except Exception as e:
            logger.error(f"加载计算请求失败: {e}")
            return None    
 
   def _backup_existing_data(self, request: CalculationRequest) -> Dict[str, pd.DataFrame]:
        """备份现有数据"""
        backup_data = {}
        
        try:
            for ts_code in request.ts_codes:
                for factor_name in request.factor_names:
                    # 构建查询条件
                    where_conditions = ["ts_code = :ts_code", "factor_name = :factor_name"]
                    params = {'ts_code': ts_code, 'factor_name': factor_name}
                    
                    if request.start_date:
                        where_conditions.append("factor_date >= :start_date")
                        params['start_date'] = request.start_date
                    
                    if request.end_date:
                        where_conditions.append("factor_date <= :end_date")
                        params['end_date'] = request.end_date
                    
                    where_clause = " AND ".join(where_conditions)
                    
                    query = text(f"""
                        SELECT factor_date, factor_value 
                        FROM factor_values 
                        WHERE {where_clause}
                        ORDER BY factor_date
                    """)
                    
                    with self.engine.connect() as conn:
                        result = conn.execute(query, params)
                        data = pd.DataFrame(result.fetchall(), columns=result.keys())
                        
                        if not data.empty:
                            backup_key = f"{ts_code}_{factor_name}"
                            backup_data[backup_key] = data
            
            logger.info(f"备份了 {len(backup_data)} 个因子的现有数据")
            
        except Exception as e:
            logger.error(f"备份现有数据失败: {e}")
        
        return backup_data
    
    def _execute_calculation(self, request: CalculationRequest) -> Dict[str, Any]:
        """执行计算"""
        calculation_results = {
            'successful_calculations': [],
            'failed_calculations': [],
            'total_factor_values': 0,
            'execution_summary': {}
        }
        
        try:
            for factor_type in request.factor_types:
                # 选择对应的计算引擎
                if factor_type == 'technical':
                    engine = self.technical_engine
                elif factor_type == 'fundamental':
                    engine = self.fundamental_engine
                elif factor_type == 'sentiment':
                    engine = self.sentiment_engine
                else:
                    logger.warning(f"未知的因子类型: {factor_type}")
                    continue
                
                # 为每个股票执行计算
                for ts_code in request.ts_codes:
                    try:
                        # 执行因子计算
                        result = engine.calculate_factors(
                            ts_code=ts_code,
                            start_date=request.start_date,
                            end_date=request.end_date,
                            factor_names=request.factor_names
                        )
                        
                        if result.status == CalculationStatus.SUCCESS:
                            calculation_results['successful_calculations'].append({
                                'ts_code': ts_code,
                                'factor_type': factor_type,
                                'data_points': result.data_points,
                                'execution_time': result.execution_time.total_seconds(),
                                'factor_count': len(result.factors)
                            })
                            
                            # 统计因子值数量
                            for factor_values in result.factors.values():
                                calculation_results['total_factor_values'] += len(factor_values)
                        else:
                            calculation_results['failed_calculations'].append({
                                'ts_code': ts_code,
                                'factor_type': factor_type,
                                'error_message': result.error_message,
                                'status': result.status.value
                            })
                    
                    except Exception as e:
                        logger.error(f"计算股票 {ts_code} 的 {factor_type} 因子失败: {e}")
                        calculation_results['failed_calculations'].append({
                            'ts_code': ts_code,
                            'factor_type': factor_type,
                            'error_message': str(e),
                            'status': 'exception'
                        })
            
            # 生成执行摘要
            calculation_results['execution_summary'] = {
                'total_stocks': len(request.ts_codes),
                'total_factor_types': len(request.factor_types),
                'successful_count': len(calculation_results['successful_calculations']),
                'failed_count': len(calculation_results['failed_calculations']),
                'success_rate': (len(calculation_results['successful_calculations']) / 
                               max(1, len(request.ts_codes) * len(request.factor_types))) * 100
            }
            
        except Exception as e:
            logger.error(f"执行计算失败: {e}")
            raise
        
        return calculation_results
    
    def _validate_calculation_results(self, request: CalculationRequest, 
                                    calculation_results: Dict[str, Any],
                                    backup_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """验证计算结果"""
        validation_results = []
        
        try:
            # 获取新计算的数据
            new_data = self._get_calculated_data(request)
            
            for ts_code in request.ts_codes:
                for factor_name in request.factor_names:
                    backup_key = f"{ts_code}_{factor_name}"
                    
                    # 比较新旧数据
                    if backup_key in backup_data and backup_key in new_data:
                        comparisons = self._compare_factor_data(
                            factor_name, ts_code,
                            backup_data[backup_key], 
                            new_data[backup_key]
                        )
                        
                        # 验证数据质量
                        quality_check = self._check_data_quality(
                            factor_name, new_data[backup_key]
                        )
                        
                        validation_results.append({
                            'ts_code': ts_code,
                            'factor_name': factor_name,
                            'comparisons': comparisons,
                            'quality_check': quality_check,
                            'validation_status': self._determine_validation_status(
                                comparisons, quality_check
                            )
                        })
            
        except Exception as e:
            logger.error(f"验证计算结果失败: {e}")
        
        return validation_results
    
    def _get_calculated_data(self, request: CalculationRequest) -> Dict[str, pd.DataFrame]:
        """获取新计算的数据"""
        new_data = {}
        
        try:
            for ts_code in request.ts_codes:
                for factor_name in request.factor_names:
                    # 查询最新计算的数据
                    query = text("""
                        SELECT factor_date, factor_value 
                        FROM factor_values 
                        WHERE ts_code = :ts_code 
                        AND factor_name = :factor_name
                        AND created_time >= :created_time
                        ORDER BY factor_date
                    """)
                    
                    with self.engine.connect() as conn:
                        result = conn.execute(query, {
                            'ts_code': ts_code,
                            'factor_name': factor_name,
                            'created_time': request.created_time
                        })
                        
                        data = pd.DataFrame(result.fetchall(), columns=result.keys())
                        
                        if not data.empty:
                            new_key = f"{ts_code}_{factor_name}"
                            new_data[new_key] = data
            
        except Exception as e:
            logger.error(f"获取新计算数据失败: {e}")
        
        return new_data
    
    def _compare_factor_data(self, factor_name: str, ts_code: str,
                           old_data: pd.DataFrame, new_data: pd.DataFrame) -> List[CalculationComparison]:
        """比较因子数据"""
        comparisons = []
        
        try:
            # 合并数据进行比较
            merged = pd.merge(
                old_data, new_data,
                on='factor_date',
                how='outer',
                suffixes=('_old', '_new')
            )
            
            # 获取变化阈值
            change_threshold = self.validation_rules['change_threshold'].get(
                factor_name, 
                self.validation_rules['change_threshold']['default']
            )
            
            for _, row in merged.iterrows():
                old_value = row.get('factor_value_old')
                new_value = row.get('factor_value_new')
                
                # 计算差异
                difference = None
                percentage_change = None
                is_significant = False
                
                if pd.notna(old_value) and pd.notna(new_value):
                    difference = new_value - old_value
                    if old_value != 0:
                        percentage_change = abs(difference / old_value)
                        is_significant = percentage_change > change_threshold
                
                comparisons.append(CalculationComparison(
                    factor_name=factor_name,
                    ts_code=ts_code,
                    date_value=row['factor_date'],
                    old_value=old_value,
                    new_value=new_value,
                    difference=difference,
                    percentage_change=percentage_change,
                    is_significant=is_significant
                ))
            
        except Exception as e:
            logger.error(f"比较因子数据失败: {e}")
        
        return comparisons
    
    def _check_data_quality(self, factor_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """检查数据质量"""
        quality_check = {
            'total_records': len(data),
            'null_count': data['factor_value'].isnull().sum(),
            'null_percentage': 0,
            'value_range_check': True,
            'outlier_count': 0,
            'quality_score': 100
        }
        
        try:
            if len(data) > 0:
                # 计算空值比例
                quality_check['null_percentage'] = (quality_check['null_count'] / len(data)) * 100
                
                # 检查值范围
                if factor_name in self.validation_rules['value_range']:
                    value_range = self.validation_rules['value_range'][factor_name]
                    valid_data = data['factor_value'].dropna()
                    
                    if not valid_data.empty:
                        out_of_range = ((valid_data < value_range['min']) | 
                                      (valid_data > value_range['max'])).sum()
                        
                        if out_of_range > 0:
                            quality_check['value_range_check'] = False
                            quality_check['outlier_count'] = out_of_range
                
                # 计算质量评分
                quality_score = 100
                
                # 空值扣分
                null_tolerance = self.validation_rules['null_tolerance'].get(
                    factor_name, 
                    self.validation_rules['null_tolerance']['default']
                )
                
                if quality_check['null_percentage'] > null_tolerance * 100:
                    quality_score -= min(50, quality_check['null_percentage'])
                
                # 异常值扣分
                if quality_check['outlier_count'] > 0:
                    outlier_ratio = quality_check['outlier_count'] / len(data)
                    quality_score -= min(30, outlier_ratio * 100)
                
                quality_check['quality_score'] = max(0, quality_score)
            
        except Exception as e:
            logger.error(f"检查数据质量失败: {e}")
            quality_check['quality_score'] = 0
        
        return quality_check
    
    def _determine_validation_status(self, comparisons: List[CalculationComparison],
                                   quality_check: Dict[str, Any]) -> ValidationResult:
        """确定验证状态"""
        # 检查数据质量
        if quality_check['quality_score'] < 60:
            return ValidationResult.FAILED
        
        # 检查显著变化
        significant_changes = sum(1 for comp in comparisons if comp.is_significant)
        total_comparisons = len(comparisons)
        
        if total_comparisons > 0:
            significant_ratio = significant_changes / total_comparisons
            
            if significant_ratio > 0.5:  # 超过50%的数据有显著变化
                return ValidationResult.WARNING
            elif significant_ratio > 0.1:  # 超过10%的数据有显著变化
                return ValidationResult.WARNING
        
        return ValidationResult.PASSED
    
    def _generate_execution_summary(self, calculation_results: Dict[str, Any],
                                  validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成执行摘要"""
        summary = {
            'calculation_summary': calculation_results.get('execution_summary', {}),
            'validation_summary': {
                'total_validations': len(validation_results),
                'passed_count': 0,
                'warning_count': 0,
                'failed_count': 0
            },
            'recommendations': []
        }
        
        # 统计验证结果
        for validation in validation_results:
            status = validation.get('validation_status', ValidationResult.FAILED)
            if status == ValidationResult.PASSED:
                summary['validation_summary']['passed_count'] += 1
            elif status == ValidationResult.WARNING:
                summary['validation_summary']['warning_count'] += 1
            else:
                summary['validation_summary']['failed_count'] += 1
        
        # 生成建议
        if summary['validation_summary']['failed_count'] > 0:
            summary['recommendations'].append("存在数据质量问题，建议检查计算逻辑")
        
        if summary['validation_summary']['warning_count'] > 0:
            summary['recommendations'].append("存在显著数据变化，建议人工审核")
        
        if calculation_results.get('execution_summary', {}).get('success_rate', 0) < 90:
            summary['recommendations'].append("计算成功率较低，建议检查系统状态")
        
        return summary
    
    def _save_execution_result(self, execution_result: Dict[str, Any]):
        """保存执行结果"""
        try:
            result_data = {
                'request_id': execution_result['request_id'],
                'status': execution_result['status'],
                'execution_time': execution_result['execution_time'],
                'calculation_results': json.dumps(execution_result.get('calculation_results')),
                'validation_results': json.dumps(execution_result.get('validation_results')),
                'summary': json.dumps(execution_result.get('summary')),
                'error_message': execution_result.get('error_message')
            }
            
            df = pd.DataFrame([result_data])
            df.to_sql(
                'manual_calculation_results',
                self.engine,
                if_exists='append',
                index=False
            )
            
        except Exception as e:
            logger.error(f"保存执行结果失败: {e}")
    
    def get_calculation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取计算历史"""
        try:
            query = text("""
                SELECT r.request_id, r.mode, r.ts_codes, r.factor_names, 
                       r.created_time, res.status, res.execution_time,
                       res.summary
                FROM manual_calculation_requests r
                LEFT JOIN manual_calculation_results res ON r.request_id = res.request_id
                ORDER BY r.created_time DESC
                LIMIT :limit
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'limit': limit})
                
                history = []
                for row in result.fetchall():
                    history.append({
                        'request_id': row.request_id,
                        'mode': row.mode,
                        'ts_codes': json.loads(row.ts_codes) if row.ts_codes else [],
                        'factor_names': json.loads(row.factor_names) if row.factor_names else [],
                        'created_time': row.created_time,
                        'status': row.status,
                        'execution_time': row.execution_time,
                        'summary': json.loads(row.summary) if row.summary else {}
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"获取计算历史失败: {e}")
            return []
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """获取请求状态"""
        try:
            query = text("""
                SELECT r.*, res.status as execution_status, res.execution_time,
                       res.summary, res.error_message
                FROM manual_calculation_requests r
                LEFT JOIN manual_calculation_results res ON r.request_id = res.request_id
                WHERE r.request_id = :request_id
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'request_id': request_id})
                row = result.fetchone()
                
                if not row:
                    return {'error': '请求不存在'}
                
                return {
                    'request_id': row.request_id,
                    'mode': row.mode,
                    'ts_codes': json.loads(row.ts_codes),
                    'factor_names': json.loads(row.factor_names),
                    'factor_types': json.loads(row.factor_types),
                    'created_time': row.created_time,
                    'status': row.execution_status or 'submitted',
                    'execution_time': row.execution_time,
                    'summary': json.loads(row.summary) if row.summary else {},
                    'error_message': row.error_message
                }
                
        except Exception as e:
            logger.error(f"获取请求状态失败: {e}")
            return {'error': str(e)}
    
    def create_quick_calculation_request(self, ts_codes: List[str], 
                                       factor_names: List[str],
                                       **kwargs) -> str:
        """创建快速计算请求"""
        request_id = str(uuid.uuid4())
        
        # 根据因子名称推断因子类型
        factor_types = self._infer_factor_types(factor_names)
        
        request = CalculationRequest(
            request_id=request_id,
            mode=CalculationMode.MULTIPLE_STOCKS if len(ts_codes) > 1 else CalculationMode.SINGLE_STOCK,
            ts_codes=ts_codes,
            factor_names=factor_names,
            factor_types=factor_types,
            **kwargs
        )
        
        return self.submit_calculation_request(request)
    
    def _infer_factor_types(self, factor_names: List[str]) -> List[str]:
        """根据因子名称推断因子类型"""
        factor_types = set()
        
        # 技术指标
        technical_factors = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'kdj']
        # 基本面指标
        fundamental_factors = ['pe', 'pb', 'ps', 'roe', 'roa', 'revenue', 'profit']
        # 情绪面指标
        sentiment_factors = ['money_flow', 'attention', 'sentiment', 'volume_spike']
        
        for factor_name in factor_names:
            factor_lower = factor_name.lower()
            
            if any(tech in factor_lower for tech in technical_factors):
                factor_types.add('technical')
            elif any(fund in factor_lower for fund in fundamental_factors):
                factor_types.add('fundamental')
            elif any(sent in factor_lower for sent in sentiment_factors):
                factor_types.add('sentiment')
            else:
                # 默认添加所有类型
                factor_types.update(['technical', 'fundamental', 'sentiment'])
        
        return list(factor_types)
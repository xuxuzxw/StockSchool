import json
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np
import pandas as pd
import structlog
from loguru import logger
from sqlalchemy import text

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进后的手动计算触发器
使用设计模式和最佳实践重构
"""


from .factor_models import CalculationStatus, FactorResult, FactorType
from .fundamental_factor_engine import FundamentalFactorEngine
from .sentiment_factor_engine import SentimentFactorEngine
from .task_scheduler import TaskConfig, TaskPriority, TaskType
from .technical_factor_engine import TechnicalFactorEngine


# ==================== 配置管理 ====================
@dataclass
class ValidationRules:
    """验证规则配置"""
    value_ranges: Dict[str, Dict[str, float]]
    change_thresholds: Dict[str, float]
    null_tolerances: Dict[str, float]

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'ValidationRules':
        """从配置字典创建验证规则"""
        return cls(
            value_ranges=config_dict.get('value_range', {}),
            change_thresholds=config_dict.get('change_threshold', {}),
            null_tolerances=config_dict.get('null_tolerance', {})
        )


@dataclass
class TableConfig:
    """数据库表配置"""
    requests_table: str = 'manual_calculation_requests'
    results_table: str = 'manual_calculation_results'
    factors_table: str = 'factor_values'


# ==================== 异常体系 ====================
class CalculationRequestError(Exception):
    """计算请求基础异常"""
    pass


class ValidationError(CalculationRequestError):
    """验证错误"""
    pass


class DatabaseError(CalculationRequestError):
    """数据库操作错误"""
    pass


class CalculationExecutionError(CalculationRequestError):
    """计算执行错误"""
    pass


# ==================== 枚举定义 ====================
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


class RequestStatus(Enum):
    """请求状态枚举"""
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ==================== 数据模型 ====================
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
        """方法描述"""
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


# ==================== 验证策略 ====================
class ValidationStrategy(ABC):
    """验证策略接口"""

    @abstractmethod
    def validate(self, request: CalculationRequest) -> List[str]:
        """验证请求，返回错误消息列表"""
        pass


class BasicRequestValidator(ValidationStrategy):
    """基础请求验证器"""

    def validate(self, request: CalculationRequest) -> List[str]:
        """验证基础请求参数"""
        errors = []

        if not request.ts_codes:
            errors.append("股票代码列表不能为空")

        if not request.factor_names and not request.factor_types:
            errors.append("必须指定因子名称或因子类型")

        if request.start_date and request.end_date:
            if request.start_date > request.end_date:
                errors.append("开始日期不能晚于结束日期")

        return errors


class StockCodeValidator(ValidationStrategy):
    """股票代码验证器"""

    def validate(self, request: CalculationRequest) -> List[str]:
        """验证股票代码格式"""
        import re
        errors = []
        pattern = r'^\d{6}\.(SZ|SH)$'

        for ts_code in request.ts_codes:
            if not re.match(pattern, ts_code):
                errors.append(f"无效的股票代码: {ts_code}")

        return errors


class CompositeValidator(ValidationStrategy):
    """组合验证器"""

    def __init__(self, validators: List[ValidationStrategy]):
        """方法描述"""

    def validate(self, request: CalculationRequest) -> List[str]:
        """执行所有验证器"""
        all_errors = []
        for validator in self.validators:
            errors = validator.validate(request)
            all_errors.extend(errors)
        return all_errors


# ==================== 工厂模式 ====================
class FactorEngineFactory:
    """因子引擎工厂"""

    @staticmethod
    def create_engine(factor_type: str, engine) -> Any:
        """创建因子计算引擎"""
        engines = {
            'technical': TechnicalFactorEngine,
            'fundamental': FundamentalFactorEngine,
            'sentiment': SentimentFactorEngine
        }

        engine_class = engines.get(factor_type)
        if not engine_class:
            raise ValueError(f"不支持的因子类型: {factor_type}")

        return engine_class(engine)


# ==================== 数据访问层 ====================
class CalculationRequestRepository:
    """计算请求数据访问层"""

    def __init__(self, engine, table_config: TableConfig):
        """方法描述"""
        self.table_config = table_config
        self.logger = structlog.get_logger()

    @contextmanager
    def transaction(self):
        """事务管理上下文"""
        conn = self.engine.connect()
        trans = conn.begin()
        try:
            yield conn
            trans.commit()
        except Exception:
            trans.rollback()
            raise
        finally:
            conn.close()

    def save_request(self, request: CalculationRequest) -> None:
        """保存计算请求"""
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
                'status': RequestStatus.SUBMITTED.value
            }

            with self.transaction() as conn:
                df = pd.DataFrame([request_data])
                df.to_sql(
                    self.table_config.requests_table,
                    conn,
                    if_exists='append',
                    index=False
                )

            self.logger.info("计算请求已保存", request_id=request.request_id)

        except Exception as e:
            self.logger.error("保存计算请求失败", request_id=request.request_id, error=str(e))
            raise DatabaseError(f"保存请求失败: {e}") from e

    def load_request(self, request_id: str) -> Optional[CalculationRequest]:
        """从数据库加载请求"""
        try:
            query = text(f"""
                SELECT * FROM {self.table_config.requests_table}
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
            self.logger.error("加载计算请求失败", request_id=request_id, error=str(e))
            raise DatabaseError(f"加载请求失败: {e}") from e

    def update_request_status(self, request_id: str, status: RequestStatus) -> None:
        """更新请求状态"""
        try:
            query = text(f"""
                UPDATE {self.table_config.requests_table}
                SET status = :status, updated_time = :updated_time
                WHERE request_id = :request_id
            """)

            with self.transaction() as conn:
                conn.execute(query, {
                    'request_id': request_id,
                    'status': status.value,
                    'updated_time': datetime.now()
                })

        except Exception as e:
            self.logger.error("更新请求状态失败", request_id=request_id, error=str(e))
            raise DatabaseError(f"更新状态失败: {e}") from e


# ==================== 业务逻辑层 ====================
class CalculationExecutor:
    """计算执行器"""

    def __init__(self, engine, factory: FactorEngineFactory):
        """方法描述"""
        self.factory = factory
        self.logger = structlog.get_logger()

    def execute_calculation(self, request: CalculationRequest) -> Dict[str, Any]:
        """执行计算"""
        calculation_results = {
            'successful_calculations': [],
            'failed_calculations': [],
            'total_factor_values': 0,
            'execution_summary': {}
        }

        try:
            for factor_type in request.factor_types:
                engine = self.factory.create_engine(factor_type, self.engine)

                for ts_code in request.ts_codes:
                    try:
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
                        self.logger.error("计算失败", ts_code=ts_code, factor_type=factor_type, error=str(e))
                        calculation_results['failed_calculations'].append({
                            'ts_code': ts_code,
                            'factor_type': factor_type,
                            'error_message': str(e),
                            'status': 'exception'
                        })

            # 生成执行摘要
            calculation_results['execution_summary'] = self._generate_summary(request, calculation_results)

        except Exception as e:
            self.logger.error("执行计算失败", error=str(e))
            raise CalculationExecutionError(f"计算执行失败: {e}") from e

        return calculation_results

    def _generate_summary(self, request: CalculationRequest, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成执行摘要"""
        total_tasks = len(request.ts_codes) * len(request.factor_types)
        successful_count = len(results['successful_calculations'])
        failed_count = len(results['failed_calculations'])

        return {
            'total_stocks': len(request.ts_codes),
            'total_factor_types': len(request.factor_types),
            'successful_count': successful_count,
            'failed_count': failed_count,
            'success_rate': (successful_count / max(1, total_tasks)) * 100
        }


# ==================== 主控制器 ====================
class ManualCalculationTrigger:
    """手动计算触发器 - 重构后的版本"""

    def __init__(self,
                 engine,
                 validation_rules: ValidationRules,
                 table_config: TableConfig):
        self.engine = engine
        self.validation_rules = validation_rules
        self.table_config = table_config

        # 初始化组件
        self.validator = CompositeValidator([
            BasicRequestValidator(),
            StockCodeValidator()
        ])
        self.repository = CalculationRequestRepository(engine, table_config)
        self.executor = CalculationExecutor(engine, FactorEngineFactory())

        # 计算历史记录
        self.calculation_history = {}

        self.logger = structlog.get_logger()

    def submit_calculation_request(self, request: CalculationRequest) -> str:
        """提交计算请求"""
        try:
            # 验证请求参数
            validation_errors = self.validator.validate(request)
            if validation_errors:
                raise ValidationError(f"请求验证失败: {'; '.join(validation_errors)}")

            # 保存请求记录
            self.repository.save_request(request)

            self.logger.info("提交手动计算请求", request_id=request.request_id)

            return request.request_id

        except Exception as e:
            self.logger.error("提交计算请求失败", request_id=request.request_id, error=str(e))
            raise

    def execute_calculation_request(self, request_id: str) -> Dict[str, Any]:
        """执行计算请求"""
        try:
            # 加载并验证请求
            request = self._load_and_validate_request(request_id)

            # 更新状态为运行中
            self.repository.update_request_status(request_id, RequestStatus.RUNNING)

            # 执行计算
            with self._measure_execution_time("calculation_execution", request_id=request_id):
                calculation_results = self.executor.execute_calculation(request)

            # 构建执行结果
            execution_result = {
                'request_id': request_id,
                'status': RequestStatus.COMPLETED.value,
                'calculation_results': calculation_results,
                'execution_time': datetime.now(),
                'summary': calculation_results.get('execution_summary', {})
            }

            # 更新状态为完成
            self.repository.update_request_status(request_id, RequestStatus.COMPLETED)

            # 保存到历史记录
            self.calculation_history[request_id] = execution_result

            self.logger.info("计算请求执行完成", request_id=request_id)

            return execution_result

        except Exception as e:
            self.logger.error("执行计算请求失败", request_id=request_id, error=str(e))

            # 更新状态为失败
            self.repository.update_request_status(request_id, RequestStatus.FAILED)

            # 保存失败结果
            error_result = {
                'request_id': request_id,
                'status': RequestStatus.FAILED.value,
                'error_message': str(e),
                'execution_time': datetime.now()
            }

            self.calculation_history[request_id] = error_result

            raise

    def _load_and_validate_request(self, request_id: str) -> CalculationRequest:
        """加载并验证请求"""
        request = self.repository.load_request(request_id)
        if not request:
            raise ValueError(f"未找到计算请求: {request_id}")
        return request

    @contextmanager
    def _measure_execution_time(self, operation: str, **context):
        """测量执行时间"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.logger.info(
                "operation_completed",
                operation=operation,
                duration_seconds=duration,
                **context
            )

    def create_quick_calculation_request(self,
                                       ts_codes: List[str],
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

    @lru_cache(maxsize=128)
    def _infer_factor_types(self, factor_names: tuple) -> List[str]:
        """根据因子名称推断因子类型"""
        # 将tuple转换为list以便处理
        factor_names_list = list(factor_names)

        technical_factors = {'rsi', 'macd', 'sma', 'ema', 'bollinger', 'atr'}
        fundamental_factors = {'pe', 'pb', 'roe', 'roa', 'debt_ratio'}
        sentiment_factors = {'money_flow', 'attention', 'sentiment'}

        factor_types = set()

        for factor_name in factor_names_list:
            factor_base = factor_name.split('_')[0].lower()

            if factor_base in technical_factors:
                factor_types.add('technical')
            elif factor_base in fundamental_factors:
                factor_types.add('fundamental')
            elif factor_base in sentiment_factors:
                factor_types.add('sentiment')
            else:
                # 默认为技术面
                factor_types.add('technical')

        return list(factor_types)

    def get_calculation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取计算历史"""
        try:
            query = text(f"""
                SELECT request_id, mode, ts_codes, factor_names,
                       created_time, status
                FROM {self.table_config.requests_table}
                ORDER BY created_time DESC
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
                        'status': row.status
                    })

                return history

        except Exception as e:
            self.logger.error("获取计算历史失败", error=str(e))
            return []
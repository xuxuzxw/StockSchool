"""
数据验证器 - 使用策略模式处理不同类型的数据验证
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass
import pandas as pd
import re
from .data_service_constants import DataServiceConstants


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    score: float
    issues: List[str]
    metrics: Dict[str, Any]
    message: str = ""


class DataValidator(ABC):
    """数据验证器基类"""
    
    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """验证数据"""
        pass


class StockBasicValidator(DataValidator):
    """股票基础信息验证器"""
    
    def validate(self, data: pd.DataFrame, **kwargs) -> ValidationResult:
        """验证股票基础信息数据"""
        issues = []
        metrics = {}
        
        if data.empty:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=["数据为空"],
                metrics={},
                message="股票基础信息数据为空"
            )
        
        # 检查必需列
        required_columns = DataServiceConstants.STOCK_BASIC_REQUIRED_COLUMNS
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"缺少必要字段: {missing_columns}")
        
        # 计算数据质量指标
        total_records = len(data)
        null_records = data.isnull().sum().sum()
        total_cells = total_records * len(data.columns)
        
        completeness = (1 - null_records / total_cells) * 100 if total_cells > 0 else 0
        
        # 验证股票代码格式
        if 'ts_code' in data.columns:
            invalid_codes = data[~data['ts_code'].str.match(DataServiceConstants.STOCK_CODE_PATTERN)]
            invalid_codes_count = len(invalid_codes)
            accuracy = (1 - invalid_codes_count / total_records) * 100 if total_records > 0 else 0
            
            if invalid_codes_count > total_records * DataServiceConstants.INVALID_CODE_THRESHOLD:
                issues.append(f"股票代码格式错误过多: {invalid_codes_count}/{total_records}")
        else:
            accuracy = 0
            issues.append("缺少股票代码列")
        
        # 计算综合评分
        overall_score = (completeness * 0.6 + accuracy * 0.4)
        
        metrics = {
            "total_records": total_records,
            "null_records": null_records,
            "completeness": completeness,
            "accuracy": accuracy,
            "invalid_codes_count": invalid_codes_count if 'ts_code' in data.columns else 0,
            "markets": data['market'].value_counts().to_dict() if 'market' in data.columns else {}
        }
        
        return ValidationResult(
            is_valid=len(issues) == 0 and overall_score >= 80,
            score=overall_score,
            issues=issues,
            metrics=metrics,
            message=f"股票基础信息验证完成，评分: {overall_score:.2f}"
        )


class DailyDataValidator(DataValidator):
    """日线数据验证器"""
    
    def validate(self, data: pd.DataFrame, **kwargs) -> ValidationResult:
        """验证日线数据"""
        issues = []
        metrics = {}
        
        if data.empty:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=["数据为空"],
                metrics={},
                message="日线数据为空"
            )
        
        # 检查必需列
        required_columns = DataServiceConstants.DAILY_DATA_REQUIRED_COLUMNS
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"缺少必要字段: {missing_columns}")
        
        # 验证价格数据逻辑
        logical_errors = data[
            (data['high'] < data['low']) |
            (data['open'] <= 0) |
            (data['close'] <= 0) |
            (data['vol'] < 0)
        ]
        
        if len(logical_errors) > 0:
            issues.append(f"价格数据存在逻辑错误: {len(logical_errors)} 条记录")
        
        # 计算数据质量指标
        total_records = len(data)
        null_records = data.isnull().sum().sum()
        total_cells = total_records * len(data.columns)
        
        completeness = (1 - null_records / total_cells) * 100 if total_cells > 0 else 0
        accuracy = (1 - len(logical_errors) / total_records) * 100 if total_records > 0 else 0
        
        overall_score = (completeness * 0.5 + accuracy * 0.5)
        
        metrics = {
            "total_records": total_records,
            "null_records": null_records,
            "logical_errors": len(logical_errors),
            "completeness": completeness,
            "accuracy": accuracy,
            "price_range": {
                "min_close": float(data['close'].min()) if 'close' in data.columns else 0,
                "max_close": float(data['close'].max()) if 'close' in data.columns else 0,
                "avg_volume": float(data['vol'].mean()) if 'vol' in data.columns else 0
            }
        }
        
        return ValidationResult(
            is_valid=len(issues) == 0 and overall_score >= 80,
            score=overall_score,
            issues=issues,
            metrics=metrics,
            message=f"日线数据验证完成，评分: {overall_score:.2f}"
        )


class TradeCalendarValidator(DataValidator):
    """交易日历验证器"""
    
    def validate(self, data: pd.DataFrame, **kwargs) -> ValidationResult:
        """验证交易日历数据"""
        issues = []
        metrics = {}
        
        if data.empty:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=["数据为空"],
                metrics={},
                message="交易日历数据为空"
            )
        
        # 检查必需列
        required_columns = DataServiceConstants.TRADE_CAL_REQUIRED_COLUMNS
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"缺少必要字段: {missing_columns}")
        
        # 统计交易日和非交易日
        trading_days = data[data['is_open'] == 1] if 'is_open' in data.columns else pd.DataFrame()
        non_trading_days = data[data['is_open'] == 0] if 'is_open' in data.columns else pd.DataFrame()
        
        # 验证交易日比例
        trading_ratio = len(trading_days) / len(data) if len(data) > 0 else 0
        if (trading_ratio < DataServiceConstants.TRADING_RATIO_MIN or 
            trading_ratio > DataServiceConstants.TRADING_RATIO_MAX):
            issues.append(f"交易日比例异常: {trading_ratio:.2%}")
        
        # 计算数据质量
        null_count = data.isnull().sum().sum()
        completeness = 100.0 if null_count == 0 else 90.0
        
        overall_score = completeness
        
        metrics = {
            "total_days": len(data),
            "trading_days": len(trading_days),
            "non_trading_days": len(non_trading_days),
            "trading_ratio": trading_ratio,
            "completeness": completeness
        }
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=overall_score,
            issues=issues,
            metrics=metrics,
            message=f"交易日历验证完成，评分: {overall_score:.2f}"
        )


class ValidatorFactory:
    """验证器工厂"""
    
    _validators = {
        'stock_basic': StockBasicValidator,
        'daily_data': DailyDataValidator,
        'trade_calendar': TradeCalendarValidator,
    }
    
    @classmethod
    def create_validator(cls, validator_type: str) -> DataValidator:
        """创建验证器"""
        if validator_type not in cls._validators:
            raise ValueError(f"不支持的验证器类型: {validator_type}")
        
        return cls._validators[validator_type]()
    
    @classmethod
    def register_validator(cls, validator_type: str, validator_class: type):
        """注册新的验证器"""
        cls._validators[validator_type] = validator_class
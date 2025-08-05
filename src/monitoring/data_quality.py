import asyncio
import json
import logging
import pickle
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from src.config.unified_config import config
from src.monitoring.alerts import Alert, AlertRule, AlertSeverity, AlertStatus, AlertType
from src.monitoring.collectors import HealthMetric

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量监控系统

实现数据完整性、准确性、时效性、一致性检测
提供数据质量综合评分和告警功能

作者: StockSchool Team
创建时间: 2025-01-02
"""



# 配置日志
logger = logging.getLogger(__name__)

class DataQualityDimension(Enum):
    """数据质量维度"""
    COMPLETENESS = "completeness"  # 完整性
    ACCURACY = "accuracy"          # 准确性
    TIMELINESS = "timeliness"      # 时效性
    CONSISTENCY = "consistency"    # 一致性
    VALIDITY = "validity"          # 有效性
    UNIQUENESS = "uniqueness"      # 唯一性

class QualityCheckType(Enum):
    """质量检查类型"""
    NULL_CHECK = "null_check"              # 空值检查
    RANGE_CHECK = "range_check"            # 范围检查
    FORMAT_CHECK = "format_check"          # 格式检查
    OUTLIER_CHECK = "outlier_check"        # 异常值检查
    FRESHNESS_CHECK = "freshness_check"    # 新鲜度检查
    DUPLICATE_CHECK = "duplicate_check"    # 重复值检查
    CROSS_SOURCE_CHECK = "cross_source_check"  # 跨源一致性检查
    TREND_CHECK = "trend_check"            # 趋势检查

@dataclass
class QualityRule:
    """数据质量规则"""
    id: str
    name: str
    description: str
    dimension: DataQualityDimension
    check_type: QualityCheckType
    table_name: str
    column_name: Optional[str] = None
    threshold: Optional[float] = None
    parameters: Dict[str, Any] = None
    enabled: bool = True
    severity: AlertSeverity = AlertSeverity.WARNING
    tags: List[str] = None

    def __post_init__(self):
        """方法描述"""
        if self.parameters is None:
            self.parameters = {}
        if self.tags is None:
            self.tags = []

@dataclass
class QualityResult:
    """数据质量检查结果"""
    rule_id: str
    table_name: str
    column_name: Optional[str]
    dimension: DataQualityDimension
    check_type: QualityCheckType
    passed: bool
    score: float  # 0-100分
    value: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        """方法描述"""
        if self.details is None:
            self.details = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class QualityReport:
    """数据质量报告"""
    table_name: str
    overall_score: float
    dimension_scores: Dict[DataQualityDimension, float]
    results: List[QualityResult]
    issues_count: Dict[AlertSeverity, int]
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """方法描述"""
        if self.metadata is None:
            self.metadata = {}

class DataQualityMonitor:
    """数据质量监控器"""

    def __init__(self, db_path: str = None):
        """方法描述"""
        self.engine = None
        self.rules: Dict[str, QualityRule] = {}
        self.anomaly_models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # 初始化数据库
        self._init_database()

        # 加载默认规则
        self._load_default_rules()

        # 初始化异常检测模型
        self._init_anomaly_detection()

    def _init_database(self):
        """初始化数据库"""
        try:
            # 创建数据库连接
            database_url = config.get('database.url', 'sqlite:///stockschool.db')
            self.engine = create_engine(database_url)

            # 创建质量监控相关表
            with self.engine.connect() as conn:
                # 质量规则表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS data_quality_rules (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        dimension TEXT NOT NULL,
                        check_type TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        column_name TEXT,
                        threshold REAL,
                        parameters TEXT,
                        enabled BOOLEAN DEFAULT TRUE,
                        severity TEXT DEFAULT 'WARNING',
                        tags TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

                # 质量检查结果表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS data_quality_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        rule_id TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        column_name TEXT,
                        dimension TEXT NOT NULL,
                        check_type TEXT NOT NULL,
                        passed BOOLEAN NOT NULL,
                        score REAL NOT NULL,
                        value REAL,
                        threshold REAL,
                        message TEXT,
                        details TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (rule_id) REFERENCES data_quality_rules (id)
                    )
                """))

                # 质量报告表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS data_quality_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        table_name TEXT NOT NULL,
                        overall_score REAL NOT NULL,
                        dimension_scores TEXT NOT NULL,
                        issues_count TEXT NOT NULL,
                        metadata TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

                conn.commit()
        except Exception as e:
            self.logger.error(f"初始化数据库失败: {e}")
            raise

    def _init_anomaly_detection(self):
        """初始化异常检测模型"""
        try:
            # 为每个表初始化异常检测模型
            tables = ['stock_basic', 'daily_data', 'trade_cal']
            for table in tables:
                self.anomaly_models[table] = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
                self.scalers[table] = StandardScaler()

        except Exception as e:
            self.logger.error(f"初始化异常检测模型失败: {e}")

    def _load_default_rules(self):
        """加载默认质量规则""
        default_rules = [
            # 股票数据完整性检查
            QualityRule(
                id="stock_data_completeness",
                name="股票数据完整性检查",
                description="检查股票数据表的空值比例",
                dimension=DataQualityDimension.COMPLETENESS,
                check_type=QualityCheckType.NULL_CHECK,
                table_name="stock_data",
                threshold=0.05,  # 空值比例不超过5%
                severity=AlertSeverity.WARNING,
                tags=["stock", "completeness"]
            ),

            # 价格数据范围检查
            QualityRule(
                id="price_range_check",
                name="股价范围检查",
                description="检查股价是否在合理范围内",
                dimension=DataQualityDimension.ACCURACY,
                check_type=QualityCheckType.RANGE_CHECK,
                table_name="stock_data",
                column_name="close_price",
                parameters={"min_value": 0.01, "max_value": 10000},
                severity=AlertSeverity.ERROR,
                tags=["stock", "price", "accuracy"]
            ),

            # 交易量异常检查
            QualityRule(
                id="volume_outlier_check",
                name="交易量异常检查",
                description="检测交易量异常值",
                dimension=DataQualityDimension.ACCURACY,
                check_type=QualityCheckType.OUTLIER_CHECK,
                table_name="stock_data",
                column_name="volume",
                parameters={"method": "zscore", "threshold": 3},
                severity=AlertSeverity.WARNING,
                tags=["stock", "volume", "outlier"]
            ),

            # 数据新鲜度检查
            QualityRule(
                id="data_freshness_check",
                name="数据新鲜度检查",
                description="检查数据更新时效性",
                dimension=DataQualityDimension.TIMELINESS,
                check_type=QualityCheckType.FRESHNESS_CHECK,
                table_name="stock_data",
                column_name="update_time",
                threshold=24,  # 24小时内必须有更新
                parameters={"unit": "hours"},
                severity=AlertSeverity.CRITICAL,
                tags=["stock", "freshness"]
            ),

            # 重复数据检查
            QualityRule(
                id="duplicate_check",
                name="重复数据检查",
                description="检查重复的股票数据记录",
                dimension=DataQualityDimension.UNIQUENESS,
                check_type=QualityCheckType.DUPLICATE_CHECK,
                table_name="stock_data",
                parameters={"key_columns": ["stock_code", "trade_date"]},
                threshold=0.01,  # 重复率不超过1%
                severity=AlertSeverity.WARNING,
                tags=["stock", "duplicate"]
            )
        ]

        for rule in default_rules:
            self.add_rule(rule)

    def add_rule(self, rule: QualityRule):
        """添加质量规则"""
        self.rules[rule.id] = rule

        # 保存到数据库
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT OR REPLACE INTO data_quality_rules
                    (id, name, description, dimension, check_type, table_name,
                     column_name, threshold, parameters, enabled, severity, tags)
                    VALUES (:id, :name, :description, :dimension, :check_type, :table_name,
                            :column_name, :threshold, :parameters, :enabled, :severity, :tags)
                """), {
                    'id': rule.id,
                    'name': rule.name,
                    'description': rule.description,
                    'dimension': rule.dimension.value,
                    'check_type': rule.check_type.value,
                    'table_name': rule.table_name,
                    'column_name': rule.column_name,
                    'threshold': rule.threshold,
                    'parameters': json.dumps(rule.parameters),
                    'enabled': rule.enabled,
                    'severity': rule.severity.value,
                    'tags': json.dumps(rule.tags)
                })
                conn.commit()
        except Exception as e:
            self.logger.error(f"保存质量规则失败: {e}")

    def remove_rule(self, rule_id: str):
        """删除质量规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]

        try:
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM data_quality_rules WHERE id = :id"), {'id': rule_id})
                conn.commit()
        except Exception as e:
            self.logger.error(f"删除质量规则失败: {e}")

    async def check_data_quality(self, table_name: str = None) -> List[QualityResult]:
        """执行数据质量检查"""
        results = []

        # 筛选要检查的规则
        rules_to_check = [
            rule for rule in self.rules.values()
            if rule.enabled and (table_name is None or rule.table_name == table_name)
        ]

        for rule in rules_to_check:
            try:
                result = await self._execute_quality_check(rule)
                if result:
                    results.append(result)

                    # 保存结果到数据库
                    await self._save_result(result)
            except Exception as e:
                self.logger.error(f"执行质量检查失败 {rule.id}: {e}")

                # 创建错误结果
                error_result = QualityResult(
                    rule_id=rule.id,
                    table_name=rule.table_name,
                    column_name=rule.column_name,
                    dimension=rule.dimension,
                    check_type=rule.check_type,
                    passed=False,
                    score=0.0,
                    message=f"检查执行失败: {str(e)}"
                )
                results.append(error_result)

        return results

    async def _execute_quality_check(self, rule: QualityRule) -> Optional[QualityResult]:
        """执行具体的质量检查"""
        if rule.check_type == QualityCheckType.NULL_CHECK:
            return await self._check_null_values(rule)
        elif rule.check_type == QualityCheckType.RANGE_CHECK:
            return await self._check_value_range(rule)
        elif rule.check_type == QualityCheckType.OUTLIER_CHECK:
            return await self._check_outliers(rule)
        elif rule.check_type == QualityCheckType.FRESHNESS_CHECK:
            return await self._check_data_freshness(rule)
        elif rule.check_type == QualityCheckType.DUPLICATE_CHECK:
            return await self._check_duplicates(rule)
        elif rule.check_type == QualityCheckType.FORMAT_CHECK:
            return await self._check_format(rule)
        else:
            self.logger.warning(f"不支持的检查类型: {rule.check_type}")
            return None

    async def _check_null_values(self, rule: QualityRule) -> QualityResult:
        """检查空值比例"""
        try:
            with self.engine.connect() as conn:
                # 获取总记录数
                total_query = f"SELECT COUNT(*) as total FROM {rule.table_name}"
                total_result = conn.execute(text(total_query)).fetchone()
                total_count = total_result[0] if total_result else 0

                if total_count == 0:
                    return QualityResult(
                        rule_id=rule.id,
                        table_name=rule.table_name,
                        column_name=rule.column_name,
                        dimension=rule.dimension,
                        check_type=rule.check_type,
                        passed=False,
                        score=0.0,
                        message="表中没有数据"
                    )

                if rule.column_name:
                    # 检查特定列的空值
                    null_query = f"SELECT COUNT(*) as null_count FROM {rule.table_name} WHERE {rule.column_name} IS NULL"
                else:
                    # 检查所有列的空值（简化版本）
                    null_query = f"SELECT COUNT(*) as null_count FROM {rule.table_name} WHERE 1=0"  # 需要动态构建

                null_result = conn.execute(text(null_query)).fetchone()
                null_count = null_result[0] if null_result else 0

                null_ratio = null_count / total_count
                passed = null_ratio <= (rule.threshold or 0.05)
                score = max(0, 100 * (1 - null_ratio / (rule.threshold or 0.05)))

                return QualityResult(
                    rule_id=rule.id,
                    table_name=rule.table_name,
                    column_name=rule.column_name,
                    dimension=rule.dimension,
                    check_type=rule.check_type,
                    passed=passed,
                    score=min(100, score),
                    value=null_ratio,
                    threshold=rule.threshold,
                    message=f"空值比例: {null_ratio:.2%} (阈值: {rule.threshold:.2%})",
                    details={
                        "total_count": total_count,
                        "null_count": null_count,
                        "null_ratio": null_ratio
                    }
                )
        except Exception as e:
            self.logger.error(f"空值检查失败: {e}")
            raise

    async def _check_value_range(self, rule: QualityRule) -> QualityResult:
        """检查数值范围"""
        try:
            params = rule.parameters or {}
            min_value = params.get('min_value')
            max_value = params.get('max_value')

            if not rule.column_name:
                raise ValueError("范围检查需要指定列名")

            with self.engine.connect() as conn:
                # 获取总记录数
                total_query = f"SELECT COUNT(*) as total FROM {rule.table_name} WHERE {rule.column_name} IS NOT NULL"
                total_result = conn.execute(text(total_query)).fetchone()
                total_count = total_result[0] if total_result else 0

                if total_count == 0:
                    return QualityResult(
                        rule_id=rule.id,
                        table_name=rule.table_name,
                        column_name=rule.column_name,
                        dimension=rule.dimension,
                        check_type=rule.check_type,
                        passed=False,
                        score=0.0,
                        message="没有有效数据进行范围检查"
                    )

                # 构建范围查询条件
                conditions = []
                if min_value is not None:
                    conditions.append(f"{rule.column_name} >= {min_value}")
                if max_value is not None:
                    conditions.append(f"{rule.column_name} <= {max_value}")

                if not conditions:
                    raise ValueError("范围检查需要指定min_value或max_value")

                range_condition = " AND ".join(conditions)
                valid_query = f"SELECT COUNT(*) as valid_count FROM {rule.table_name} WHERE {rule.column_name} IS NOT NULL AND {range_condition}"

                valid_result = conn.execute(text(valid_query)).fetchone()
                valid_count = valid_result[0] if valid_result else 0

                valid_ratio = valid_count / total_count
                passed = valid_ratio >= 0.95  # 95%的数据应该在合理范围内
                score = 100 * valid_ratio

            try:
                return QualityResult(
                    rule_id=rule.id,
                    table_name=rule.table_name,
                    column_name=rule.column_name,
                    dimension=rule.dimension,
                    check_type=rule.check_type,
                    passed=passed,
                    score=score,
                    value=valid_ratio,
                    message=f"范围内数据比例: {valid_ratio:.2%}",
                    details={
                        "total_count": total_count,
                        "valid_count": valid_count,
                        "valid_ratio": valid_ratio,
                        "min_value": min_value,
                        "max_value": max_value
                    }
                )
        try:

        except Exception as e:
            self.logger.error(f"范围检查失败: {e}")
            raise

    async def _check_outliers(self, rule: QualityRule) -> QualityResult:
        """检查异常值"""
        try:
            if not rule.column_name:
                raise ValueError("异常值检查需要指定列名")

            params = rule.parameters or {}
            method = params.get('method', 'zscore')
            threshold = params.get('threshold', 3)

            with self.engine.connect() as conn:
                # 获取数据
                query = f"SELECT {rule.column_name} FROM {rule.table_name} WHERE {rule.column_name} IS NOT NULL"
                result = conn.execute(text(query))
                values = [row[0] for row in result.fetchall()]

                if len(values) < 10:  # 数据太少无法进行异常检测
                    return QualityResult(
                        rule_id=rule.id,
                        table_name=rule.table_name,
                        column_name=rule.column_name,
                        dimension=rule.dimension,
                        check_type=rule.check_type,
                        passed=True,
                        score=100.0,
                        message="数据量不足，跳过异常检测"
                    )

                # 执行异常检测
                if method == 'zscore':
                    z_scores = np.abs(stats.zscore(values))
                    outliers = np.sum(z_scores > threshold)
                elif method == 'iqr':
                    q1, q3 = np.percentile(values, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = np.sum((values < lower_bound) | (values > upper_bound))
                else:
                    raise ValueError(f"不支持的异常检测方法: {method}")

                total_count = len(values)
                outlier_ratio = outliers / total_count
                passed = outlier_ratio <= 0.05  # 异常值比例不超过5%
                score = max(0, 100 * (1 - outlier_ratio / 0.05))

            try:
                return QualityResult(
                    rule_id=rule.id,
                    table_name=rule.table_name,
                    column_name=rule.column_name,
                    dimension=rule.dimension,
                    check_type=rule.check_type,
                    passed=passed,
                    score=min(100, score),
                    value=outlier_ratio,
                    message=f"异常值比例: {outlier_ratio:.2%} (方法: {method})",
                    details={
                        "total_count": total_count,
                        "outlier_count": outliers,
                        "outlier_ratio": outlier_ratio,
                        "method": method,
                        "threshold": threshold
                    }
                )
        try:

        except Exception as e:
            self.logger.error(f"异常值检查失败: {e}")
            raise

    async def _check_data_freshness(self, rule: QualityRule) -> QualityResult:
        """检查数据新鲜度"""
        try:
            if not rule.column_name:
                raise ValueError("新鲜度检查需要指定时间列名")

            params = rule.parameters or {}
            unit = params.get('unit', 'hours')
            threshold_hours = rule.threshold or 24

            # 转换时间单位
            if unit == 'minutes':
                threshold_hours = threshold_hours / 60
            elif unit == 'days':
                threshold_hours = threshold_hours * 24

            with self.engine.connect() as conn:
                # 获取最新数据时间
                query = f"SELECT MAX({rule.column_name}) as latest_time FROM {rule.table_name}"
                result = conn.execute(text(query)).fetchone()

                if not result or not result[0]:
                    return QualityResult(
                        rule_id=rule.id,
                        table_name=rule.table_name,
                        column_name=rule.column_name,
                        dimension=rule.dimension,
                        check_type=rule.check_type,
                        passed=False,
                        score=0.0,
                        message="没有找到时间数据"
                    )

                latest_time = result[0]
                if isinstance(latest_time, str):
                    latest_time = datetime.fromisoformat(latest_time.replace('Z', '+00:00'))

                now = datetime.now()
                time_diff = now - latest_time
                hours_diff = time_diff.total_seconds() / 3600

                passed = hours_diff <= threshold_hours
                score = max(0, 100 * (1 - hours_diff / (threshold_hours * 2)))  # 超过2倍阈值时分数为0

            try:
                return QualityResult(
                    rule_id=rule.id,
                    table_name=rule.table_name,
                    column_name=rule.column_name,
                    dimension=rule.dimension,
                    check_type=rule.check_type,
                    passed=passed,
                    score=min(100, score),
                    value=hours_diff,
                    threshold=threshold_hours,
                    message=f"数据延迟: {hours_diff:.1f}小时 (阈值: {threshold_hours}小时)",
                    details={
                        "latest_time": latest_time.isoformat(),
                        "current_time": now.isoformat(),
                        "hours_diff": hours_diff,
                        "threshold_hours": threshold_hours
                    }
                )
        try:

        except Exception as e:
            self.logger.error(f"新鲜度检查失败: {e}")
            raise

    async def _check_duplicates(self, rule: QualityRule) -> QualityResult:
        """检查重复数据"""
        try:
            params = rule.parameters or {}
            key_columns = params.get('key_columns', [])

            if not key_columns:
                raise ValueError("重复检查需要指定key_columns")

            key_columns_str = ", ".join(key_columns)

            with self.engine.connect() as conn:
                # 获取总记录数
                total_query = f"SELECT COUNT(*) as total FROM {rule.table_name}"
                total_result = conn.execute(text(total_query)).fetchone()
                total_count = total_result[0] if total_result else 0

                if total_count == 0:
                    return QualityResult(
                        rule_id=rule.id,
                        table_name=rule.table_name,
                        column_name=None,
                        dimension=rule.dimension,
                        check_type=rule.check_type,
                        passed=True,
                        score=100.0,
                        message="表中没有数据"
                    )

                # 获取唯一记录数
                unique_query = f"SELECT COUNT(DISTINCT {key_columns_str}) as unique_count FROM {rule.table_name}"
                unique_result = conn.execute(text(unique_query)).fetchone()
                unique_count = unique_result[0] if unique_result else 0

                duplicate_count = total_count - unique_count
                duplicate_ratio = duplicate_count / total_count if total_count > 0 else 0

                threshold = rule.threshold or 0.01
                passed = duplicate_ratio <= threshold
                score = max(0, 100 * (1 - duplicate_ratio / threshold))

            try:
                return QualityResult(
                    rule_id=rule.id,
                    table_name=rule.table_name,
                    column_name=None,
                    dimension=rule.dimension,
                    check_type=rule.check_type,
                    passed=passed,
                    score=min(100, score),
                    value=duplicate_ratio,
                    threshold=threshold,
                    message=f"重复数据比例: {duplicate_ratio:.2%} (阈值: {threshold:.2%})",
                    details={
                        "total_count": total_count,
                        "unique_count": unique_count,
                        "duplicate_count": duplicate_count,
                        "duplicate_ratio": duplicate_ratio,
                        "key_columns": key_columns
                    }
                )
        try:

        except Exception as e:
            self.logger.error(f"重复检查失败: {e}")
            raise

    async def _check_format(self, rule: QualityRule) -> QualityResult:
        """检查数据格式"""
        # 这里可以实现格式检查逻辑，比如日期格式、邮箱格式等
        # 暂时返回一个占位符结果
        return QualityResult(
            rule_id=rule.id,
            table_name=rule.table_name,
            column_name=rule.column_name,
            dimension=rule.dimension,
            check_type=rule.check_type,
            passed=True,
            score=100.0,
            message="格式检查功能待实现"
        )

    async def _save_result(self, result: QualityResult):
        """保存检查结果到数据库"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO data_quality_results
                    (rule_id, table_name, column_name, dimension, check_type,
                     passed, score, value, threshold, message, details, timestamp)
                    VALUES (:rule_id, :table_name, :column_name, :dimension, :check_type,
                            :passed, :score, :value, :threshold, :message, :details, :timestamp)
                """), {
                    'rule_id': result.rule_id,
                    'table_name': result.table_name,
                    'column_name': result.column_name,
                    'dimension': result.dimension.value,
                    'check_type': result.check_type.value,
                    'passed': result.passed,
                    'score': result.score,
                    'value': result.value,
                    'threshold': result.threshold,
                    'message': result.message,
                    'details': json.dumps(result.details),
                    'timestamp': result.timestamp
                })
        try:
                conn.commit()
        except Exception as e:
            self.logger.error(f"保存检查结果失败: {e}")

    async def generate_quality_report(self, table_name: str = None) -> QualityReport:
        """生成数据质量报告"""
        # 执行质量检查
        results = await self.check_data_quality(table_name)

        if not results:
            return QualityReport(
                table_name=table_name or "all",
                overall_score=0.0,
                dimension_scores={},
                results=[],
                issues_count={},
                timestamp=datetime.now()
            )

        # 按维度分组计算分数
        dimension_scores = {}
        dimension_results = defaultdict(list)

        for result in results:
            dimension_results[result.dimension].append(result)

        for dimension, dim_results in dimension_results.items():
            avg_score = sum(r.score for r in dim_results) / len(dim_results)
            dimension_scores[dimension] = avg_score

        # 计算总体分数
        overall_score = sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0

        # 统计问题数量
        issues_count = defaultdict(int)
        for result in results:
            if not result.passed:
                # 根据分数确定严重程度
                if result.score < 50:
                    issues_count[AlertSeverity.CRITICAL] += 1
                elif result.score < 80:
                    issues_count[AlertSeverity.ERROR] += 1
                else:
                    issues_count[AlertSeverity.WARNING] += 1

        report = QualityReport(
            table_name=table_name or "all",
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            results=results,
            issues_count=dict(issues_count),
            timestamp=datetime.now(),
            metadata={
                "total_checks": len(results),
                "passed_checks": sum(1 for r in results if r.passed),
                "failed_checks": sum(1 for r in results if not r.passed)
            }
        )

        # 保存报告到数据库
        await self._save_report(report)

        return report

    async def _save_report(self, report: QualityReport):
        """保存质量报告到数据库"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO data_quality_reports
                    (table_name, overall_score, dimension_scores, issues_count, metadata, timestamp)
                    VALUES (:table_name, :overall_score, :dimension_scores, :issues_count, :metadata, :timestamp)
                """), {
                    'table_name': report.table_name,
                    'overall_score': report.overall_score,
                    'dimension_scores': json.dumps({k.value: v for k, v in report.dimension_scores.items()}),
                    'issues_count': json.dumps({k.value: v for k, v in report.issues_count.items()}),
                    'metadata': json.dumps(report.metadata),
                    'timestamp': report.timestamp
                })
        try:
                conn.commit()
        except Exception as e:
            self.logger.error(f"保存质量报告失败: {e}")

    def get_quality_history(self, table_name: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """获取质量历史数据"""
        try:
            with self.engine.connect() as conn:
                where_clause = ""
                params = {'days': days}

                if table_name:
                    where_clause = "AND table_name = :table_name"
                    params['table_name'] = table_name

                query = f"""
                    SELECT * FROM data_quality_reports
                    WHERE timestamp >= datetime('now', '-{days} days') {where_clause}
                    ORDER BY timestamp DESC
                """

                result = conn.execute(text(query), params)
            try:
                return [dict(row._mapping) for row in result.fetchall()]
        try:

        except Exception as e:
            self.logger.error(f"获取质量历史失败: {e}")
            return []

    async def create_quality_alerts(self, results: List[QualityResult]) -> List[Alert]:
        """根据质量检查结果创建告警"""
        alerts = []

        for result in results:
            if not result.passed:
                # 确定告警严重程度
                if result.score < 50:
                    severity = AlertSeverity.CRITICAL
                elif result.score < 80:
                    severity = AlertSeverity.ERROR
                else:
                    severity = AlertSeverity.WARNING

                alert = Alert(
                    id=f"dq_{result.rule_id}_{int(result.timestamp.timestamp())}",
                    rule_id=result.rule_id,
                    title=f"数据质量问题: {result.table_name}",
                    message=result.message,
                    severity=severity,
                    alert_type=AlertType.DATA_QUALITY,
                    status=AlertStatus.ACTIVE,
                    created_at=result.timestamp,
                    updated_at=result.timestamp,
                    source="data_quality_monitor",
                    tags=["data_quality", result.dimension.value, result.check_type.value],
                    metadata={
                        "table_name": result.table_name,
                        "column_name": result.column_name,
                        "dimension": result.dimension.value,
                        "check_type": result.check_type.value,
                        "score": result.score,
                        "details": result.details
                    }
                )
                alerts.append(alert)

        return alerts

    async def validate_data_stream(self, table_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """实时数据流验证

        Args:
            table_name: 表名
            data: 待验证的数据

        Returns:
            Dict[str, Any]: 验证结果
        """
        try:
            results = []

            # 实时完整性检查
            null_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            results.append({
                'check': 'completeness',
                'score': max(0, 100 - null_ratio * 100),
                'details': {'null_ratio': null_ratio}
            })

            # 实时异常检测
            if table_name in self.anomaly_models and not data.empty:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # 选择数值列进行异常检测
                    X = data[numeric_cols].fillna(0)

                    # 标准化数据
                    scaler = self.scalers[table_name]
                    X_scaled = scaler.fit_transform(X)

                    # 异常检测
                    model = self.anomaly_models[table_name]
                    predictions = model.fit_predict(X_scaled)
                    anomaly_ratio = sum(predictions == -1) / len(predictions)

                    results.append({
                        'check': 'anomaly_detection',
                        'score': max(0, 100 - anomaly_ratio * 100),
                        'details': {'anomaly_ratio': anomaly_ratio, 'anomaly_count': sum(predictions == -1)}
                    })

            # 实时一致性检查
            if 'trade_date' in data.columns:
                date_consistency = data['trade_date'].nunique() == len(data)
                results.append({
                    'check': 'consistency',
                    'score': 100 if date_consistency else 50,
                    'details': {'date_consistency': date_consistency}
                })

            # 计算综合评分
            overall_score = sum(r['score'] for r in results) / len(results) if results else 100

            return {
                'table_name': table_name,
                'overall_score': overall_score,
                'checks': results,
                'timestamp': datetime.now().isoformat()
            }
        try:

        except Exception as e:
            self.logger.error(f"实时数据流验证失败: {e}")
            return {
                'table_name': table_name,
                'overall_score': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def train_anomaly_model(self, table_name: str, historical_data: pd.DataFrame):
        """训练异常检测模型

        Args:
            table_name: 表名
            historical_data: 历史数据用于训练
        """
        try:
            if historical_data.empty:
                self.logger.warning(f"表 {table_name} 没有足够的历史数据进行训练")
                return

            # 选择数值列
            numeric_cols = historical_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                self.logger.warning(f"表 {table_name} 没有数值列用于异常检测")
                return

            # 数据预处理
            X = historical_data[numeric_cols].fillna(0)

            # 标准化
            scaler = self.scalers[table_name]
            X_scaled = scaler.fit_transform(X)

            # 训练异常检测模型
            model = self.anomaly_models[table_name]
            model.fit(X_scaled)

            self.logger.info(f"表 {table_name} 的异常检测模型训练完成")
        try:

        except Exception as e:
            self.logger.error(f"训练异常检测模型失败: {e}")

    def get_data_lineage(self, table_name: str) -> Dict[str, Any]:
        """获取数据血缘信息

        Args:
            table_name: 表名

        Returns:
            Dict[str, Any]: 数据血缘信息
        """
        return {
            'table_name': table_name,
            'upstream_sources': [],  # 上游数据源
            'downstream_consumers': [],  # 下游消费者
            'transformation_rules': [],  # 转换规则
            'last_updated': datetime.now().isoformat(),
            'data_freshness': 'unknown'
        }

# 全局实例
data_quality_monitor = DataQualityMonitor()

# 导出主要类和函数
__all__ = [
    'DataQualityMonitor',
    'QualityRule',
    'QualityResult',
    'QualityReport',
    'DataQualityDimension',
    'QualityCheckType',
    'data_quality_monitor'
]
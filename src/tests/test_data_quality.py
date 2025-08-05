from datetime import datetime, timedelta

from sqlalchemy import create_engine, text

import asyncio
import numpy as np
import os
import pandas as pd
import pytest
import tempfile
from unittest.mock import AsyncMock, Mock, patch

from src.monitoring.alerts import AlertSeverity, AlertType
from src.monitoring.data_quality import (
    DataQualityDimension,
    DataQualityMonitor,
    QualityCheckType,
    QualityReport,
    QualityResult,
    QualityRule,
)


class TestDataQualityMonitor:
    """数据质量监控器测试类"""

    @pytest.fixture
    def temp_db(self):
        """创建临时数据库"""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        os.unlink(path)

    @pytest.fixture
    def monitor(self, temp_db):
        """创建测试用的数据质量监控器"""
        with patch("src.monitoring.data_quality.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "monitoring.data_quality.db_path": temp_db,
                "database.url": f"sqlite:///{temp_db}",
            }.get(key, default)

            monitor = DataQualityMonitor()
            return monitor

    @pytest.fixture
    def sample_data(self, monitor):
        """创建测试数据"""
        # 创建测试表
        with monitor.engine.connect() as conn:
            # 创建股票数据表
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY,
                    stock_code TEXT,
                    trade_date DATE,
                    close_price REAL,
                    volume INTEGER,
                    update_time TIMESTAMP
                )
            """
                )
            )

            # 插入测试数据
            test_data = [
                ("000001", "2024-01-01", 10.5, 1000000, "2024-01-01 15:00:00"),
                ("000001", "2024-01-02", 10.8, 1200000, "2024-01-02 15:00:00"),
                ("000002", "2024-01-01", 25.3, 800000, "2024-01-01 15:00:00"),
                ("000002", "2024-01-02", None, 900000, "2024-01-02 15:00:00"),  # 空值
                ("000003", "2024-01-01", -5.0, 500000, "2024-01-01 15:00:00"),  # 异常值
                ("000003", "2024-01-02", 15000.0, 600000, "2024-01-02 15:00:00"),  # 异常值
                ("000001", "2024-01-01", 10.5, 1000000, "2024-01-01 15:00:00"),  # 重复数据
            ]

            for data in test_data:
                conn.execute(
                    text(
                        """
                    INSERT INTO stock_data (stock_code, trade_date, close_price, volume, update_time)
                    VALUES (?, ?, ?, ?, ?)
                """
                    ),
                    data,
                )

            conn.commit()

    def test_init_database(self, monitor):
        """测试数据库初始化"""
        # 检查表是否创建成功
        with monitor.engine.connect() as conn:
            tables = conn.execute(
                text(
                    """
                SELECT name FROM sqlite_master WHERE type='table'
            """
                )
            ).fetchall()

            table_names = [table[0] for table in tables]
            assert "data_quality_rules" in table_names
            assert "data_quality_results" in table_names
            assert "data_quality_reports" in table_names

    def test_add_remove_rule(self, monitor):
        """测试添加和删除质量规则"""
        rule = QualityRule(
            id="test_rule",
            name="测试规则",
            description="测试用的质量规则",
            dimension=DataQualityDimension.COMPLETENESS,
            check_type=QualityCheckType.NULL_CHECK,
            table_name="test_table",
            threshold=0.05,
        )

        # 添加规则
        monitor.add_rule(rule)
        assert "test_rule" in monitor.rules

        # 检查数据库中是否保存
        with monitor.engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM data_quality_rules WHERE id = 'test_rule'")).fetchone()
            assert result is not None
            assert result[1] == "测试规则"  # name字段

        # 删除规则
        monitor.remove_rule("test_rule")
        assert "test_rule" not in monitor.rules

        # 检查数据库中是否删除
        with monitor.engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM data_quality_rules WHERE id = 'test_rule'")).fetchone()
            assert result is None

    @pytest.mark.asyncio
    async def test_check_null_values(self, monitor, sample_data):
        """测试空值检查"""
        rule = QualityRule(
            id="null_check_test",
            name="空值检查测试",
            description="测试空值检查功能",
            dimension=DataQualityDimension.COMPLETENESS,
            check_type=QualityCheckType.NULL_CHECK,
            table_name="stock_data",
            column_name="close_price",
            threshold=0.1,  # 10%阈值
        )

        result = await monitor._check_null_values(rule)

        assert result is not None
        assert result.rule_id == "null_check_test"
        assert result.table_name == "stock_data"
        assert result.column_name == "close_price"
        assert result.dimension == DataQualityDimension.COMPLETENESS
        assert result.check_type == QualityCheckType.NULL_CHECK
        assert result.value is not None  # 空值比例
        assert result.score >= 0
        assert "空值比例" in result.message

    @pytest.mark.asyncio
    async def test_check_value_range(self, monitor, sample_data):
        """测试数值范围检查"""
        rule = QualityRule(
            id="range_check_test",
            name="范围检查测试",
            description="测试范围检查功能",
            dimension=DataQualityDimension.ACCURACY,
            check_type=QualityCheckType.RANGE_CHECK,
            table_name="stock_data",
            column_name="close_price",
            parameters={"min_value": 0.01, "max_value": 1000},
        )

        result = await monitor._check_value_range(rule)

        assert result is not None
        assert result.rule_id == "range_check_test"
        assert result.dimension == DataQualityDimension.ACCURACY
        assert result.check_type == QualityCheckType.RANGE_CHECK
        assert result.value is not None  # 有效数据比例
        assert "范围内数据比例" in result.message

    @pytest.mark.asyncio
    async def test_check_outliers(self, monitor, sample_data):
        """测试异常值检查"""
        rule = QualityRule(
            id="outlier_check_test",
            name="异常值检查测试",
            description="测试异常值检查功能",
            dimension=DataQualityDimension.ACCURACY,
            check_type=QualityCheckType.OUTLIER_CHECK,
            table_name="stock_data",
            column_name="close_price",
            parameters={"method": "zscore", "threshold": 2},
        )

        result = await monitor._check_outliers(rule)

        assert result is not None
        assert result.rule_id == "outlier_check_test"
        assert result.dimension == DataQualityDimension.ACCURACY
        assert result.check_type == QualityCheckType.OUTLIER_CHECK
        assert "异常值比例" in result.message

    @pytest.mark.asyncio
    async def test_check_data_freshness(self, monitor, sample_data):
        """测试数据新鲜度检查"""
        rule = QualityRule(
            id="freshness_check_test",
            name="新鲜度检查测试",
            description="测试新鲜度检查功能",
            dimension=DataQualityDimension.TIMELINESS,
            check_type=QualityCheckType.FRESHNESS_CHECK,
            table_name="stock_data",
            column_name="update_time",
            threshold=24,  # 24小时
            parameters={"unit": "hours"},
        )

        result = await monitor._check_data_freshness(rule)

        assert result is not None
        assert result.rule_id == "freshness_check_test"
        assert result.dimension == DataQualityDimension.TIMELINESS
        assert result.check_type == QualityCheckType.FRESHNESS_CHECK
        assert "数据延迟" in result.message

    @pytest.mark.asyncio
    async def test_check_duplicates(self, monitor, sample_data):
        """测试重复数据检查"""
        rule = QualityRule(
            id="duplicate_check_test",
            name="重复检查测试",
            description="测试重复检查功能",
            dimension=DataQualityDimension.UNIQUENESS,
            check_type=QualityCheckType.DUPLICATE_CHECK,
            table_name="stock_data",
            parameters={"key_columns": ["stock_code", "trade_date"]},
            threshold=0.1,  # 10%阈值
        )

        result = await monitor._check_duplicates(rule)

        assert result is not None
        assert result.rule_id == "duplicate_check_test"
        assert result.dimension == DataQualityDimension.UNIQUENESS
        assert result.check_type == QualityCheckType.DUPLICATE_CHECK
        assert "重复数据比例" in result.message
        assert result.value > 0  # 应该检测到重复数据

    @pytest.mark.asyncio
    async def test_check_data_quality(self, monitor, sample_data):
        """测试完整的数据质量检查"""
        # 添加一些测试规则
        rules = [
            QualityRule(
                id="completeness_test",
                name="完整性测试",
                description="测试数据完整性",
                dimension=DataQualityDimension.COMPLETENESS,
                check_type=QualityCheckType.NULL_CHECK,
                table_name="stock_data",
                column_name="close_price",
                threshold=0.2,
            ),
            QualityRule(
                id="accuracy_test",
                name="准确性测试",
                description="测试数据准确性",
                dimension=DataQualityDimension.ACCURACY,
                check_type=QualityCheckType.RANGE_CHECK,
                table_name="stock_data",
                column_name="close_price",
                parameters={"min_value": 0.01, "max_value": 1000},
            ),
        ]

        for rule in rules:
            monitor.add_rule(rule)

        # 执行质量检查
        results = await monitor.check_data_quality("stock_data")

        assert len(results) >= 2  # 至少有我们添加的两个规则的结果

        # 检查结果类型
        for result in results:
            assert isinstance(result, QualityResult)
            assert result.table_name == "stock_data"
            assert result.score >= 0 and result.score <= 100

    @pytest.mark.asyncio
    async def test_generate_quality_report(self, monitor, sample_data):
        """测试生成质量报告"""
        # 添加测试规则
        rule = QualityRule(
            id="report_test",
            name="报告测试",
            description="测试报告生成",
            dimension=DataQualityDimension.COMPLETENESS,
            check_type=QualityCheckType.NULL_CHECK,
            table_name="stock_data",
            column_name="close_price",
            threshold=0.1,
        )
        monitor.add_rule(rule)

        # 生成报告
        report = await monitor.generate_quality_report("stock_data")

        assert isinstance(report, QualityReport)
        assert report.table_name == "stock_data"
        assert report.overall_score >= 0 and report.overall_score <= 100
        assert len(report.results) > 0
        assert len(report.dimension_scores) > 0
        assert isinstance(report.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_create_quality_alerts(self, monitor):
        """测试创建质量告警"""
        # 创建一些测试结果
        results = [
            QualityResult(
                rule_id="test_rule_1",
                table_name="test_table",
                column_name="test_column",
                dimension=DataQualityDimension.COMPLETENESS,
                check_type=QualityCheckType.NULL_CHECK,
                passed=False,
                score=30.0,  # 低分数，应该产生CRITICAL告警
                message="严重的数据质量问题",
            ),
            QualityResult(
                rule_id="test_rule_2",
                table_name="test_table",
                column_name="test_column",
                dimension=DataQualityDimension.ACCURACY,
                check_type=QualityCheckType.RANGE_CHECK,
                passed=False,
                score=70.0,  # 中等分数，应该产生WARNING告警
                message="轻微的数据质量问题",
            ),
            QualityResult(
                rule_id="test_rule_3",
                table_name="test_table",
                column_name="test_column",
                dimension=DataQualityDimension.TIMELINESS,
                check_type=QualityCheckType.FRESHNESS_CHECK,
                passed=True,
                score=95.0,  # 高分数，不应该产生告警
                message="数据质量良好",
            ),
        ]

        alerts = await monitor.create_quality_alerts(results)

        # 应该只有2个告警（passed=False的结果）
        assert len(alerts) == 2

        # 检查告警严重程度
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        warning_alerts = [a for a in alerts if a.severity == AlertSeverity.WARNING]

        assert len(critical_alerts) == 1
        assert len(warning_alerts) == 1

        # 检查告警类型
        for alert in alerts:
            assert alert.alert_type == AlertType.DATA_QUALITY
            assert "data_quality" in alert.tags

    def test_get_quality_history(self, monitor):
        """测试获取质量历史数据"""
        # 先插入一些历史数据
        with monitor.engine.connect() as conn:
            test_reports = [
                {
                    "table_name": "test_table",
                    "overall_score": 85.0,
                    "dimension_scores": '{"completeness": 90.0, "accuracy": 80.0}',
                    "issues_count": '{"WARNING": 2}',
                    "metadata": "{}",
                    "timestamp": datetime.now() - timedelta(days=1),
                },
                {
                    "table_name": "test_table",
                    "overall_score": 78.0,
                    "dimension_scores": '{"completeness": 85.0, "accuracy": 71.0}',
                    "issues_count": '{"ERROR": 1, "WARNING": 1}',
                    "metadata": "{}",
                    "timestamp": datetime.now() - timedelta(days=2),
                },
            ]

            for report in test_reports:
                conn.execute(
                    text(
                        """
                    INSERT INTO data_quality_reports
                    (table_name, overall_score, dimension_scores, issues_count, metadata, timestamp)
                    VALUES (:table_name, :overall_score, :dimension_scores, :issues_count, :metadata, :timestamp)
                """
                    ),
                    report,
                )
            conn.commit()

        # 获取历史数据
        history = monitor.get_quality_history("test_table", days=7)

        assert len(history) == 2
        assert all(h["table_name"] == "test_table" for h in history)
        assert history[0]["overall_score"] == 85.0  # 最新的记录在前

    def test_default_rules_loaded(self, monitor):
        """测试默认规则是否正确加载"""
        # 检查是否加载了默认规则
        assert len(monitor.rules) > 0

        # 检查特定的默认规则
        expected_rules = [
            "stock_data_completeness",
            "price_range_check",
            "volume_outlier_check",
            "data_freshness_check",
            "duplicate_check",
        ]

        for rule_id in expected_rules:
            assert rule_id in monitor.rules
            rule = monitor.rules[rule_id]
            assert isinstance(rule, QualityRule)
            assert rule.enabled

    @pytest.mark.asyncio
    async def test_error_handling(self, monitor):
        """测试错误处理"""
        # 测试不存在的表
        rule = QualityRule(
            id="error_test",
            name="错误测试",
            description="测试错误处理",
            dimension=DataQualityDimension.COMPLETENESS,
            check_type=QualityCheckType.NULL_CHECK,
            table_name="non_existent_table",
            column_name="non_existent_column",
            threshold=0.1,
        )

        monitor.add_rule(rule)

        # 执行检查，应该处理错误并返回错误结果
        results = await monitor.check_data_quality("non_existent_table")

        assert len(results) > 0
        error_result = results[0]
        assert not error_result.passed
        assert error_result.score == 0.0
        assert "失败" in error_result.message


if __name__ == "__main__":
    pytest.main([__file__])

from datetime import datetime, timedelta

from sqlalchemy import text

from src.data.data_quality_monitor import (
    DataQualityMonitor,
    DataQualityReport,
    MagicMock,
    Mock,
    QualityLevel,
    QualityMetrics,
    """,
    from,
    import,
)
from src.data.data_quality_monitor import numpy as np
from src.data.data_quality_monitor import pandas as pd
from src.data.data_quality_monitor import patch, pytest, unittest.mock, 数据质量监控系统测试


class TestDataQualityMonitor:
    """数据质量监控器测试类"""

    @pytest.fixture
    def monitor(self):
        """创建数据质量监控器实例"""
        with patch('src.data.data_quality_monitor.get_db_engine'):
            return DataQualityMonitor()

    @pytest.fixture
    def sample_daily_data(self):
        """创建样本日线数据"""
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        data = []

        for i, date in enumerate(dates):
            base_price = 10 + i * 0.1
            data.append({
                'ts_code': '000001.SZ',
                'trade_date': date.strftime('%Y-%m-%d'),
                'open': base_price,
                'high': base_price * 1.05,
                'low': base_price * 0.95,
                'close': base_price * 1.02,
                'volume': 1000000 + i * 10000
            })

        return pd.DataFrame(data)

    @pytest.fixture
    def sample_financial_data(self):
        """创建样本财务数据"""
        return pd.DataFrame([
            {
                'ts_code': '000001.SZ',
                'end_date': '2024-03-31',
                'revenue': 1000000,
                'net_profit': 100000,
                'total_assets': 5000000,
                'total_liab': 3000000,
                'total_share': 1000000
            },
            {
                'ts_code': '000002.SZ',
                'end_date': '2024-03-31',
                'revenue': 800000,
                'net_profit': 80000,
                'total_assets': 4000000,
                'total_liab': 2500000,
                'total_share': 800000
            }
        ])

    @pytest.fixture
    def sample_sentiment_data(self):
        """创建样本情绪数据"""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        data = []

        for date in dates:
            data.append({
                'ts_code': '000001.SZ',
                'news_date': date.strftime('%Y-%m-%d'),
                'sentiment_score': np.random.uniform(-0.5, 0.5),
                'positive_count': np.random.randint(10, 100),
                'negative_count': np.random.randint(5, 50),
                'neutral_count': np.random.randint(20, 200),
                'news_volume': np.random.randint(50, 500)
            })

        return pd.DataFrame(data)

    def test_determine_quality_level(self, monitor):
        """测试质量等级判定"""
        assert monitor._determine_quality_level(95) == QualityLevel.EXCELLENT
        assert monitor._determine_quality_level(85) == QualityLevel.GOOD
        assert monitor._determine_quality_level(75) == QualityLevel.FAIR
        assert monitor._determine_quality_level(65) == QualityLevel.POOR
        assert monitor._determine_quality_level(55) == QualityLevel.CRITICAL

    def test_calculate_completeness_daily_data(self, monitor, sample_daily_data):
        """测试日线数据完整性计算"""
        with patch.object(monitor, '_get_trade_dates') as mock_trade_dates:
            mock_trade_dates.return_value = [
                '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'
            ]

            with patch.object(monitor.engine, 'connect') as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value.__enter__.return_value = mock_conn

                # 模拟股票列表查询
                mock_conn.execute.return_value = Mock()
                with patch('pandas.read_sql') as mock_read_sql:
                    mock_read_sql.return_value = pd.DataFrame({'ts_code': ['000001.SZ', '000002.SZ']})

                    # 测试完整数据
                    completeness = monitor._calculate_completeness(
                        sample_daily_data.head(10), 'daily', '2024-01-01', '2024-01-05'
                    )

                    assert completeness == 100.0  # 10条数据 / (5个交易日 * 2只股票) = 100%

    def test_calculate_accuracy_daily_data(self, monitor, sample_daily_data):
        """测试日线数据准确性计算"""
        # 测试正常数据
        accuracy = monitor._calculate_accuracy(sample_daily_data, 'daily')
        assert accuracy >= 90.0  # 正常数据应该有很高的准确性

        # 测试包含错误的数据
        error_data = sample_daily_data.copy()
        error_data.loc[0, 'high'] = error_data.loc[0, 'low'] - 1  # 高价低于低价
        error_data.loc[1, 'open'] = -1  # 负价格

        accuracy_with_errors = monitor._calculate_accuracy(error_data, 'daily')
        assert accuracy_with_errors < accuracy  # 有错误的数据准确性应该更低

    def test_calculate_accuracy_financial_data(self, monitor, sample_financial_data):
        """测试财务数据准确性计算"""
        # 测试正常数据
        accuracy = monitor._calculate_accuracy(sample_financial_data, 'financial')
        assert accuracy >= 90.0

        # 测试包含错误的数据
        error_data = sample_financial_data.copy()
        error_data.loc[0, 'total_assets'] = error_data.loc[0, 'total_liab'] - 1000  # 资产小于负债
        error_data.loc[1, 'revenue'] = -100000  # 负收入

        accuracy_with_errors = monitor._calculate_accuracy(error_data, 'financial')
        assert accuracy_with_errors < accuracy

    def test_calculate_accuracy_sentiment_data(self, monitor, sample_sentiment_data):
        """测试情绪数据准确性计算"""
        # 测试正常数据
        accuracy = monitor._calculate_accuracy(sample_sentiment_data, 'sentiment')
        assert accuracy >= 90.0

        # 测试包含错误的数据
        error_data = sample_sentiment_data.copy()
        error_data.loc[0, 'sentiment_score'] = 2.0  # 超出范围
        error_data.loc[1, 'positive_count'] = -10  # 负计数

        accuracy_with_errors = monitor._calculate_accuracy(error_data, 'sentiment')
        assert accuracy_with_errors < accuracy

    def test_calculate_timeliness(self, monitor, sample_daily_data):
        """测试数据时效性计算"""
        # 测试最新数据
        recent_data = sample_daily_data.copy()
        recent_data['trade_date'] = datetime.now().strftime('%Y-%m-%d')

        timeliness = monitor._calculate_timeliness(recent_data, 'daily')
        assert timeliness == 100.0  # 最新数据应该有100%时效性

        # 测试过期数据
        old_data = sample_daily_data.copy()
        old_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        old_data['trade_date'] = old_date

        timeliness_old = monitor._calculate_timeliness(old_data, 'daily')
        assert timeliness_old < 100.0  # 过期数据时效性应该更低

    def test_calculate_consistency_daily_data(self, monitor, sample_daily_data):
        """测试日线数据一致性计算"""
        # 测试正常数据
        consistency = monitor._calculate_consistency(sample_daily_data, 'daily')
        assert consistency >= 90.0

        # 测试包含重复数据
        duplicate_data = pd.concat([sample_daily_data, sample_daily_data.head(2)])
        consistency_with_duplicates = monitor._calculate_consistency(duplicate_data, 'daily')
        assert consistency_with_duplicates < consistency

        # 测试缺少必要列
        incomplete_data = sample_daily_data.drop(columns=['volume'])
        consistency_incomplete = monitor._calculate_consistency(incomplete_data, 'daily')
        assert consistency_incomplete < consistency

    def test_identify_issues(self, monitor):
        """测试问题识别"""
        data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

        # 测试低质量分数
        issues = monitor._identify_issues(data, 'daily', 70, 70, 70, 70)
        assert len(issues) == 4  # 四个维度都低于80分

        # 测试高质量分数
        issues_good = monitor._identify_issues(data, 'daily', 90, 90, 90, 90)
        assert len(issues_good) == 0  # 没有问题

        # 测试高缺失值比例
        data_with_nulls = pd.DataFrame({
            'col1': [1, None, None, None, None],
            'col2': [None, None, None, None, 5]
        })
        issues_nulls = monitor._identify_issues(data_with_nulls, 'daily', 90, 90, 90, 90)
        assert any('缺失值比例过高' in issue for issue in issues_nulls)

    def test_generate_recommendations(self, monitor):
        """测试建议生成"""
        # 测试严重质量问题
        issues = ["数据完整性不足: 50.0%", "数据准确性不足: 60.0%"]
        recommendations = monitor._generate_recommendations(issues, QualityLevel.CRITICAL)

        assert len(recommendations) > 0
        assert any('立即停止使用' in rec for rec in recommendations)
        assert any('增加数据同步频率' in rec for rec in recommendations)
        assert any('加强数据验证规则' in rec for rec in recommendations)

        # 测试良好质量
        recommendations_good = monitor._generate_recommendations([], QualityLevel.EXCELLENT)
        assert len(recommendations_good) == 0

    def test_create_quality_details(self, monitor, sample_daily_data):
        """测试质量详细信息创建"""
        details = monitor._create_quality_details(sample_daily_data, 'daily')

        assert details['record_count'] == len(sample_daily_data)
        assert details['column_count'] == len(sample_daily_data.columns)
        assert 'null_count' in details
        assert 'duplicate_count' in details
        assert 'date_range' in details
        assert 'numeric_stats' in details

        # 测试空数据
        empty_details = monitor._create_quality_details(pd.DataFrame(), 'daily')
        assert empty_details['record_count'] == 0
        assert empty_details['null_count'] == 0

    @patch('src.data.data_quality_monitor.get_db_engine')
    def test_assess_data_quality_integration(self, mock_engine, sample_daily_data):
        """测试数据质量评估集成功能"""
        # 设置模拟
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

        monitor = DataQualityMonitor()

        # 模拟数据获取
        with patch.object(monitor, '_fetch_data') as mock_fetch:
            mock_fetch.return_value = sample_daily_data

            with patch.object(monitor, '_get_trade_dates') as mock_trade_dates:
                mock_trade_dates.return_value = ['2024-01-01', '2024-01-02', '2024-01-03']

                with patch.object(monitor, '_save_quality_metrics'):
                    # 执行质量评估
                    report = monitor.assess_data_quality(
                        data_source='tushare',
                        data_type='daily',
                        start_date='2024-01-01',
                        end_date='2024-01-03'
                    )

                    # 验证报告结构
                    assert isinstance(report, DataQualityReport)
                    assert report.data_source == 'tushare'
                    assert report.data_type == 'daily'
                    assert isinstance(report.metrics, QualityMetrics)
                    assert isinstance(report.metrics.level, QualityLevel)
                    assert 0 <= report.metrics.overall_score <= 100

    def test_create_empty_report(self, monitor):
        """测试空数据报告创建"""
        report = monitor._create_empty_report('tushare', 'daily', '2024-01-01')

        assert report.data_source == 'tushare'
        assert report.data_type == 'daily'
        assert report.report_date == '2024-01-01'
        assert report.metrics.overall_score == 0.0
        assert report.metrics.level == QualityLevel.CRITICAL
        assert '未找到数据' in report.metrics.issues
        assert len(report.recommendations) > 0

    def test_get_quarters_in_range(self, monitor):
        """测试季度范围获取"""
        quarters = monitor._get_quarters_in_range('2024-01-01', '2024-12-31')

        assert len(quarters) == 4
        assert '2024Q1' in quarters
        assert '2024Q2' in quarters
        assert '2024Q3' in quarters
        assert '2024Q4' in quarters

        # 测试跨年季度
        quarters_cross_year = monitor._get_quarters_in_range('2023-10-01', '2024-03-31')
        assert len(quarters_cross_year) == 2
        assert '2023Q4' in quarters_cross_year
        assert '2024Q1' in quarters_cross_year

    @patch('src.data.data_quality_monitor.get_db_engine')
    def test_generate_daily_quality_report(self, mock_engine):
        """测试每日质量报告生成"""
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

        monitor = DataQualityMonitor()

        with patch.object(monitor, 'assess_data_quality') as mock_assess:
            # 模拟质量评估结果
            mock_report = DataQualityReport(
                report_date='2024-01-01',
                data_source='tushare',
                data_type='daily',
                metrics=QualityMetrics(
                    completeness=90.0,
                    accuracy=95.0,
                    timeliness=85.0,
                    consistency=92.0,
                    overall_score=90.5,
                    level=QualityLevel.EXCELLENT,
                    issues=[]
                ),
                details={'record_count': 1000},
                recommendations=[]
            )
            mock_assess.return_value = mock_report

            # 生成每日报告
            reports = monitor.generate_daily_quality_report('2024-01-01')

            # 验证报告
            assert len(reports) > 0
            assert 'tushare_daily' in reports
            assert isinstance(reports['tushare_daily'], DataQualityReport)

    @patch('src.data.data_quality_monitor.get_db_engine')
    def test_get_quality_trend(self, mock_engine):
        """测试质量趋势获取"""
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

        # 模拟趋势数据
        trend_data = pd.DataFrame({
            'report_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'overall_score': [85.0, 87.0, 90.0],
            'quality_level': ['good', 'good', 'excellent'],
            'completeness_score': [80.0, 85.0, 90.0],
            'accuracy_score': [90.0, 89.0, 90.0],
            'timeliness_score': [85.0, 87.0, 90.0],
            'consistency_score': [85.0, 87.0, 90.0]
        })

        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = trend_data

            monitor = DataQualityMonitor()
            trend = monitor.get_quality_trend('tushare', 'daily', 30)

            assert len(trend) == 3
            assert 'overall_score' in trend.columns
            assert 'quality_level' in trend.columns

    @patch('src.data.data_quality_monitor.get_db_engine')
    def test_check_quality_alerts(self, mock_engine):
        """测试质量告警检查"""
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

        # 模拟告警数据
        alert_data = pd.DataFrame({
            'data_source': ['tushare', 'akshare'],
            'data_type': ['daily', 'sentiment'],
            'overall_score': [65.0, 55.0],
            'quality_level': ['poor', 'critical'],
            'issues': ['数据完整性不足', '数据准确性不足']
        })

        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = alert_data

            monitor = DataQualityMonitor()
            alerts = monitor.check_quality_alerts(70.0)

            assert len(alerts) == 2
            assert alerts[0]['data_source'] == 'tushare'
            assert alerts[1]['data_source'] == 'akshare'
            assert all(alert['overall_score'] < 70.0 for alert in alerts)


class TestQualityMetrics:
    """质量指标测试类"""

    def test_quality_metrics_creation(self):
        """测试质量指标创建"""
        metrics = QualityMetrics(
            completeness=90.0,
            accuracy=95.0,
            timeliness=85.0,
            consistency=92.0,
            overall_score=90.5,
            level=QualityLevel.EXCELLENT,
            issues=["测试问题"]
        )

        assert metrics.completeness == 90.0
        assert metrics.accuracy == 95.0
        assert metrics.timeliness == 85.0
        assert metrics.consistency == 92.0
        assert metrics.overall_score == 90.5
        assert metrics.level == QualityLevel.EXCELLENT
        assert "测试问题" in metrics.issues


class TestDataQualityReport:
    """数据质量报告测试类"""

    def test_report_creation(self):
        """测试报告创建"""
        metrics = QualityMetrics(
            completeness=90.0,
            accuracy=95.0,
            timeliness=85.0,
            consistency=92.0,
            overall_score=90.5,
            level=QualityLevel.EXCELLENT,
            issues=[]
        )

        report = DataQualityReport(
            report_date='2024-01-01',
            data_source='tushare',
            data_type='daily',
            metrics=metrics,
            details={'record_count': 1000},
            recommendations=['建议1', '建议2']
        )

        assert report.report_date == '2024-01-01'
        assert report.data_source == 'tushare'
        assert report.data_type == 'daily'
        assert report.metrics == metrics
        assert report.details['record_count'] == 1000
        assert len(report.recommendations) == 2


if __name__ == '__main__':
    pytest.main([__file__])
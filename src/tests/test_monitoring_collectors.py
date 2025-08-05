from datetime import datetime

from src.monitoring.collectors import (
    APIHealthCollector,
    AsyncMock,
    BaseCollector,
    CeleryHealthCollector,
    CollectorConfig,
    DatabaseHealthCollector,
    Mock,
    RedisHealthCollector,
    StockSchool,
    SystemHealthCollector,
    Team,
    2025-01-02,
    """,
    asyncio,
    collect_system_health,
    from,
    import,
    logging,
    patch,
    pytest,
    unittest.mock,
    作者:,
    创建时间:,
    包含数据库、Redis、Celery、API服务等组件的监控测试,
    测试各种系统组件健康状态数据收集的准确性,
    监控数据收集器集成测试,
)


class TestCollectorConfig:
    """测试收集器配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = Collectorget_config()

        assert config.database_url == "postgresql://stockschool:stockschool123@localhost:15432/stockschool"
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.collection_interval == 30.0
        assert config.max_retries == 3

    def test_custom_config(self):
        """测试自定义配置"""
        config = CollectorConfig(
            database_url="postgresql://test:test@localhost:5432/test",
            redis_host="redis-server",
            redis_port=6380,
            collection_interval=60.0
        )

        assert config.database_url == "postgresql://test:test@localhost:5432/test"
        assert config.redis_host == "redis-server"
        assert config.redis_port == 6380
        assert config.collection_interval == 60.0


class MockCollector(BaseCollector):
    """用于测试的模拟收集器"""

    def __init__(self, config: CollectorConfig, should_fail: bool = False):
        """方法描述"""
        self.should_fail = should_fail
        self.collect_call_count = 0

    async def collect(self):
        """模拟收集数据"""
        self.collect_call_count += 1

        if self.should_fail:
            raise Exception("模拟收集失败")

        return {
            'status': 'healthy',
            'value': 100,
            'timestamp': datetime.now().isoformat()
        }


class TestBaseCollector:
    """测试基础收集器"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return CollectorConfig(max_retries=2, retry_delay=0.1)

    @pytest.mark.asyncio
    async def test_successful_collection(self, config):
        """测试成功收集数据"""
        collector = MockCollector(config, should_fail=False)

        result = await collector.collect_with_retry()

        assert result['status'] == 'healthy'
        assert result['value'] == 100
        assert 'collection_time_ms' in result
        assert 'collection_timestamp' in result
        assert 'collector_name' in result
        assert result['attempt_number'] == 1
        assert collector.collect_call_count == 1

    @pytest.mark.asyncio
    async def test_failed_collection_with_retry(self, config):
        """测试失败收集数据的重试机制"""
        collector = MockCollector(config, should_fail=True)

        result = await collector.collect_with_retry()

        assert result['status'] == 'error'
        assert 'error_message' in result
        assert result['total_attempts'] == 3  # 1 + 2 retries
        assert collector.collect_call_count == 3

    def test_collector_stats(self, config):
        """测试收集器统计信息"""
        collector = MockCollector(config)

        # 初始统计
        stats = collector.get_stats()
        assert stats['collection_count'] == 0
        assert stats['error_count'] == 0
        assert stats['success_rate'] == 0

    @pytest.mark.asyncio
    async def test_collector_stats_after_collection(self, config):
        """测试收集后的统计信息"""
        collector = MockCollector(config, should_fail=False)

        await collector.collect_with_retry()

        stats = collector.get_stats()
        assert stats['collection_count'] == 1
        assert stats['error_count'] == 0
        assert stats['success_rate'] == 100.0
        assert stats['last_collection_time'] is not None


class TestDatabaseHealthCollector:
    """测试数据库健康收集器"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return Collectorget_config()

    @pytest.mark.asyncio
    async def test_mock_database_collection(self, config):
        """测试模拟数据库收集"""
        collector = DatabaseHealthCollector(config)

        result = await collector.collect()

        # 验证返回的数据结构
        required_fields = [
            'connection_status', 'connection_time_ms', 'connection_count',
            'active_connections', 'query_avg_time_ms', 'slow_queries_count',
            'database_size_mb', 'table_count', 'last_error'
        ]

        for field in required_fields:
            assert field in result, f"缺少字段: {field}"

        # 验证数据类型和范围
        assert isinstance(result['connection_time_ms'], (int, float))
        assert result['connection_time_ms'] >= 0
        assert isinstance(result['connection_count'], int)
        assert result['connection_count'] >= 0
        assert isinstance(result['active_connections'], int)
        assert result['active_connections'] >= 0
        assert result['connection_status'] in ['healthy', 'warning', 'critical']

    @pytest.mark.asyncio
    async def test_database_collection_with_retry(self, config):
        """测试带重试的数据库收集"""
        collector = DatabaseHealthCollector(config)

        result = await collector.collect_with_retry()

        assert 'collection_time_ms' in result
        assert 'collection_timestamp' in result
        assert 'collector_name' in result
        assert result['collector_name'] == 'DatabaseHealthCollector'


class TestRedisHealthCollector:
    """测试Redis健康收集器"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return Collectorget_config()

    @pytest.mark.asyncio
    async def test_mock_redis_collection(self, config):
        """测试模拟Redis收集"""
        collector = RedisHealthCollector(config)

        result = await collector.collect()

        # 验证返回的数据结构
        required_fields = [
            'connection_status', 'connection_time_ms', 'memory_usage_mb',
            'memory_usage_percent', 'connected_clients', 'cache_hit_rate',
            'keys_count', 'commands_processed', 'uptime_seconds', 'last_error'
        ]

        for field in required_fields:
            assert field in result, f"缺少字段: {field}"

        # 验证数据类型和范围
        assert isinstance(result['memory_usage_percent'], (int, float))
        assert 0 <= result['memory_usage_percent'] <= 100
        assert isinstance(result['cache_hit_rate'], (int, float))
        assert 0 <= result['cache_hit_rate'] <= 100
        assert result['connection_status'] in ['healthy', 'warning', 'critical']

    @pytest.mark.asyncio
    async def test_redis_status_determination(self, config):
        """测试Redis状态判断逻辑"""
        collector = RedisHealthCollector(config)

        # 多次收集，验证状态逻辑
        for _ in range(5):
            result = await collector.collect()
            memory_percent = result['memory_usage_percent']
            status = result['connection_status']

            if memory_percent > 90:
                assert status == 'critical'
            elif memory_percent > 80:
                assert status == 'warning'
            else:
                assert status == 'healthy'


class TestCeleryHealthCollector:
    """测试Celery健康收集器"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return Collectorget_config()

    @pytest.mark.asyncio
    async def test_mock_celery_collection(self, config):
        """测试模拟Celery收集"""
        collector = CeleryHealthCollector(config)

        result = await collector.collect()

        # 验证返回的数据结构
        required_fields = [
            'connection_status', 'connection_time_ms', 'active_tasks',
            'pending_tasks', 'failed_tasks', 'success_rate',
            'avg_task_time_seconds', 'worker_count', 'task_types', 'last_error'
        ]

        for field in required_fields:
            assert field in result, f"缺少字段: {field}"

        # 验证数据类型和范围
        assert isinstance(result['active_tasks'], int)
        assert result['active_tasks'] >= 0
        assert isinstance(result['success_rate'], (int, float))
        assert 0 <= result['success_rate'] <= 100
        assert isinstance(result['worker_count'], int)
        assert result['worker_count'] >= 0
        assert result['connection_status'] in ['healthy', 'warning', 'critical']

    @pytest.mark.asyncio
    async def test_celery_status_determination(self, config):
        """测试Celery状态判断逻辑"""
        collector = CeleryHealthCollector(config)

        # 多次收集，验证状态逻辑
        for _ in range(5):
            result = await collector.collect()
            worker_count = result['worker_count']
            success_rate = result['success_rate']
            status = result['connection_status']

            if worker_count == 0:
                assert status == 'critical'
            elif success_rate < 90:
                assert status == 'warning'
            else:
                assert status == 'healthy'


class TestAPIHealthCollector:
    """测试API健康收集器"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return Collectorget_config()

    @pytest.mark.asyncio
    async def test_mock_api_collection(self, config):
        """测试模拟API收集"""
        collector = APIHealthCollector(config)

        result = await collector.collect()

        # 验证返回的数据结构
        required_fields = [
            'status', 'response_time_ms', 'http_status_code',
            'request_count_1h', 'error_count_1h', 'error_rate',
            'active_connections', 'uptime_seconds', 'version', 'last_error'
        ]

        for field in required_fields:
            assert field in result, f"缺少字段: {field}"

        # 验证数据类型和范围
        assert isinstance(result['response_time_ms'], (int, float))
        assert result['response_time_ms'] >= 0
        assert isinstance(result['error_rate'], (int, float))
        assert 0 <= result['error_rate'] <= 100
        assert result['status'] in ['healthy', 'warning', 'critical']

    @pytest.mark.asyncio
    async def test_api_status_determination(self, config):
        """测试API状态判断逻辑"""
        collector = APIHealthCollector(config)

        # 多次收集，验证状态逻辑
        for _ in range(5):
            result = await collector.collect()
            response_time = result['response_time_ms']
            error_rate = result['error_rate']
            status = result['status']

            if response_time > 2000 or error_rate > 5:
                assert status in ['warning', 'critical']
            else:
                assert status == 'healthy'

    @pytest.mark.asyncio
    async def test_api_collector_close(self, config):
        """测试API收集器关闭"""
        collector = APIHealthCollector(config)

        # 调用close方法不应该抛出异常
        await collector.close()


class TestSystemHealthCollector:
    """测试系统健康综合收集器"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return CollectorConfig(max_retries=1, retry_delay=0.1)

    @pytest.mark.asyncio
    async def test_system_health_collection(self, config):
        """测试系统健康综合收集"""
        collector = SystemHealthCollector(config)

        result = await collector.collect()

        # 验证返回的数据结构
        required_fields = [
            'database', 'redis', 'celery', 'api',
            'overall_status', 'timestamp', 'collection_summary'
        ]

        for field in required_fields:
            assert field in result, f"缺少字段: {field}"

        # 验证各组件数据
        assert isinstance(result['database'], dict)
        assert isinstance(result['redis'], dict)
        assert isinstance(result['celery'], dict)
        assert isinstance(result['api'], dict)

        # 验证整体状态
        assert result['overall_status'] in ['healthy', 'warning', 'critical']

        # 验证收集摘要
        summary = result['collection_summary']
        assert summary['total_components'] == 4
        assert isinstance(summary['healthy_components'], int)
        assert isinstance(summary['warning_components'], int)
        assert isinstance(summary['critical_components'], int)

        # 验证组件数量总和
        total = (summary['healthy_components'] +
                summary['warning_components'] +
                summary['critical_components'])
        assert total == 4

    @pytest.mark.asyncio
    async def test_system_health_overall_status_logic(self, config):
        """测试系统整体状态逻辑"""
        collector = SystemHealthCollector(config)

        # 多次收集，验证整体状态逻辑
        for _ in range(3):
            result = await collector.collect()

            # 获取各组件状态
            db_status = result['database'].get('connection_status', 'critical')
            redis_status = result['redis'].get('connection_status', 'critical')
            celery_status = result['celery'].get('connection_status', 'critical')
            api_status = result['api'].get('status', 'critical')

            statuses = [db_status, redis_status, celery_status, api_status]
            overall_status = result['overall_status']

            # 验证整体状态逻辑
            if 'critical' in statuses:
                assert overall_status == 'critical'
            elif 'warning' in statuses:
                assert overall_status == 'warning'
            else:
                assert overall_status == 'healthy'

    @pytest.mark.asyncio
    async def test_system_collector_stats(self, config):
        """测试系统收集器统计信息"""
        collector = SystemHealthCollector(config)

        # 执行收集
        await collector.collect_with_retry()

        # 获取统计信息
        stats = collector.get_all_stats()

        # 验证统计信息结构
        expected_collectors = [
            'database_collector', 'redis_collector',
            'celery_collector', 'api_collector', 'system_collector'
        ]

        for collector_name in expected_collectors:
            assert collector_name in stats
            collector_stats = stats[collector_name]
            assert 'collection_count' in collector_stats
            assert 'error_count' in collector_stats
            assert 'success_rate' in collector_stats

    @pytest.mark.asyncio
    async def test_system_collector_close(self, config):
        """测试系统收集器关闭"""
        collector = SystemHealthCollector(config)

        # 调用close方法不应该抛出异常
        await collector.close()


class TestConvenienceFunctions:
    """测试便捷函数"""

    @pytest.mark.asyncio
    async def test_collect_system_health_function(self):
        """测试系统健康收集便捷函数"""
        result = await collect_system_health()

        # 验证返回的数据结构
        assert isinstance(result, dict)
        assert 'database' in result
        assert 'redis' in result
        assert 'celery' in result
        assert 'api' in result
        assert 'overall_status' in result

    @pytest.mark.asyncio
    async def test_collect_system_health_with_custom_config(self):
        """测试使用自定义配置的系统健康收集"""
        config = CollectorConfig(max_retries=1)
        result = await collect_system_health(config)

        assert isinstance(result, dict)
        assert 'overall_status' in result


class TestErrorHandling:
    """测试错误处理"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return CollectorConfig(max_retries=1, retry_delay=0.1)

    @pytest.mark.asyncio
    async def test_collector_exception_handling(self, config):
        """测试收集器异常处理"""
        collector = MockCollector(config, should_fail=True)

        result = await collector.collect_with_retry()

        assert result['status'] == 'error'
        assert 'error_message' in result
        assert result['total_attempts'] == 2  # 1 + 1 retry

    @pytest.mark.asyncio
    async def test_system_collector_partial_failure(self, config):
        """测试系统收集器部分失败处理"""
        collector = SystemHealthCollector(config)

        # 模拟部分收集器失败
        with patch.object(collector.database_collector, 'collect_with_retry',
                         side_effect=Exception("Database error")):
            result = await collector.collect()

            # 系统应该仍然返回结果，但数据库部分会显示错误
            assert isinstance(result, dict)
            assert 'database' in result
            assert result['database'].get('connection_status') == 'critical'


class TestPerformance:
    """测试性能相关"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return Collectorget_config()

    @pytest.mark.asyncio
    async def test_collection_timing(self, config):
        """测试收集时间记录"""
        collector = MockCollector(config)

        result = await collector.collect_with_retry()

        assert 'collection_time_ms' in result
        assert isinstance(result['collection_time_ms'], (int, float))
        assert result['collection_time_ms'] >= 0

    @pytest.mark.asyncio
    async def test_concurrent_collection(self, config):
        """测试并发收集"""
        collector = SystemHealthCollector(config)

        # 并发执行多次收集
        tasks = [collector.collect() for _ in range(3)]
        results = await asyncio.gather(*tasks)

        # 验证所有收集都成功
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert 'overall_status' in result


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行测试
    pytest.main([__file__, "-v"])
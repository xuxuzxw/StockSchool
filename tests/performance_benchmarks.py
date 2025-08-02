#!/usr/bin/env python3
"""
监控系统性能基准测试

测试数据库查询性能、缓存效率、WebSocket传输性能等
提供性能基准数据和优化建议

作者: StockSchool Team
创建时间: 2025-01-02
"""

import asyncio
import time
import statistics
import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

# 导入测试依赖
try:
    import pytest
    import pytest_asyncio
    from unittest.mock import Mock, AsyncMock
except ImportError as e:
    logging.warning(f"测试依赖不可用: {e}")
    pytest = None

# 导入项目模块
try:
    from src.services.monitoring_service import (
        MonitoringService, CacheManager, MetricsStorage, CacheConfig, StorageConfig
    )
    from src.database.performance_optimization import DatabaseOptimizer, QueryOptimizer
    from src.websocket.monitoring_websocket import (
        MonitoringWebSocketServer, ConnectionManager, WebSocketMessage, MessageType
    )
    from src.schemas.monitoring_schemas import MonitoringMetricSchema, MetricType
except ImportError as e:
    logging.warning(f"导入项目模块失败: {e}")
    # 创建模拟类用于测试
    class MonitoringService:
        async def initialize(self): pass
        async def close(self): pass
    
    class CacheManager:
        def __init__(self, config): pass
        async def initialize(self): pass
        async def close(self): pass
    
    class DatabaseOptimizer:
        def __init__(self, url): pass
        async def initialize(self): pass
        async def close(self): pass

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    duration_seconds: float
    operations_count: int
    ops_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_count: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    min_time: float
    max_time: float
    avg_time: float
    median_time: float
    p95_time: float
    p99_time: float
    std_dev: float
    
    @classmethod
    def from_times(cls, times: List[float]) -> 'PerformanceMetrics':
        """从时间列表创建性能指标"""
        if not times:
            return cls(0, 0, 0, 0, 0, 0, 0)
        
        sorted_times = sorted(times)
        return cls(
            min_time=min(times),
            max_time=max(times),
            avg_time=statistics.mean(times),
            median_time=statistics.median(times),
            p95_time=sorted_times[int(len(sorted_times) * 0.95)],
            p99_time=sorted_times[int(len(sorted_times) * 0.99)],
            std_dev=statistics.stdev(times) if len(times) > 1 else 0
        )


class PerformanceBenchmark:
    """性能基准测试基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.results: List[BenchmarkResult] = []
        
    async def setup(self) -> None:
        """测试前设置"""
        pass
    
    async def teardown(self) -> None:
        """测试后清理"""
        pass
    
    async def run_benchmark(self, 
                          test_func: Callable,
                          iterations: int = 100,
                          warmup_iterations: int = 10,
                          **kwargs) -> BenchmarkResult:
        """运行基准测试"""
        # 预热
        self.logger.info(f"开始预热 {warmup_iterations} 次...")
        for _ in range(warmup_iterations):
            try:
                await test_func(**kwargs)
            except Exception as e:
                self.logger.warning(f"预热失败: {e}")
        
        # 垃圾回收
        gc.collect()
        
        # 记录初始状态
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行基准测试
        self.logger.info(f"开始基准测试 {iterations} 次...")
        times = []
        errors = 0
        start_time = time.time()
        
        for i in range(iterations):
            try:
                operation_start = time.perf_counter()
                await test_func(**kwargs)
                operation_end = time.perf_counter()
                times.append(operation_end - operation_start)
                
            except Exception as e:
                errors += 1
                self.logger.error(f"测试迭代 {i} 失败: {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 记录最终状态
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = process.cpu_percent()
        
        # 计算结果
        success_count = len(times)
        ops_per_second = success_count / total_duration if total_duration > 0 else 0
        success_rate = success_count / iterations if iterations > 0 else 0
        
        # 创建结果对象
        result = BenchmarkResult(
            test_name=self.name,
            duration_seconds=total_duration,
            operations_count=success_count,
            ops_per_second=ops_per_second,
            memory_usage_mb=final_memory - initial_memory,
            cpu_usage_percent=cpu_percent,
            success_rate=success_rate,
            error_count=errors,
            metadata={
                'performance_metrics': PerformanceMetrics.from_times(times).__dict__ if times else {},
                'iterations': iterations,
                'warmup_iterations': warmup_iterations,
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory
            }
        )
        
        self.results.append(result)
        return result


class DatabaseBenchmark(PerformanceBenchmark):
    """数据库性能基准测试"""
    
    def __init__(self, database_url: str):
        super().__init__("DatabaseBenchmark")
        self.database_url = database_url
        self.optimizer: Optional[DatabaseOptimizer] = None
        self.query_optimizer: Optional[QueryOptimizer] = None
    
    async def setup(self) -> None:
        """设置数据库连接"""
        try:
            self.optimizer = DatabaseOptimizer(self.database_url)
            await self.optimizer.initialize()
            self.query_optimizer = QueryOptimizer(self.optimizer)
            self.logger.info("数据库基准测试设置完成")
        except Exception as e:
            self.logger.error(f"数据库设置失败: {e}")
            # 使用模拟对象
            self.optimizer = Mock()
            self.query_optimizer = Mock()
    
    async def teardown(self) -> None:
        """清理数据库连接"""
        if self.optimizer and hasattr(self.optimizer, 'close'):
            await self.optimizer.close()
    
    async def test_query_performance(self, query_name: str, params: Dict[str, Any]) -> None:
        """测试查询性能"""
        if not self.query_optimizer:
            return
        
        try:
            results, metrics = await self.query_optimizer.execute_optimized_query(
                query_name, params
            )
            return len(results)
        except Exception as e:
            self.logger.error(f"查询执行失败: {e}")
            raise
    
    async def test_index_creation(self) -> None:
        """测试索引创建性能"""
        if not self.optimizer:
            return
        
        try:
            results = await self.optimizer.create_optimization_indexes()
            return len(results.get('created', []))
        except Exception as e:
            self.logger.error(f"索引创建失败: {e}")
            raise
    
    async def run_database_benchmarks(self) -> List[BenchmarkResult]:
        """运行所有数据库基准测试"""
        await self.setup()
        
        try:
            results = []
            
            # 测试查询性能
            query_params = {
                'start_time': datetime.now() - timedelta(hours=1),
                'end_time': datetime.now(),
                'metric_names': ['cpu_usage', 'memory_usage'],
                'limit': 1000
            }
            
            result = await self.run_benchmark(
                self.test_query_performance,
                iterations=50,
                query_name='get_recent_metrics',
                params=query_params
            )
            results.append(result)
            
            # 测试索引创建性能
            result = await self.run_benchmark(
                self.test_index_creation,
                iterations=1,  # 索引创建只需要测试一次
                warmup_iterations=0
            )
            results.append(result)
            
            return results
            
        finally:
            await self.teardown()


class CacheBenchmark(PerformanceBenchmark):
    """缓存性能基准测试"""
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        super().__init__("CacheBenchmark")
        self.cache_config = cache_config or CacheConfig()
        self.cache_manager: Optional[CacheManager] = None
    
    async def setup(self) -> None:
        """设置缓存管理器"""
        try:
            self.cache_manager = CacheManager(self.cache_config)
            await self.cache_manager.initialize()
            self.logger.info("缓存基准测试设置完成")
        except Exception as e:
            self.logger.error(f"缓存设置失败: {e}")
            self.cache_manager = Mock()
    
    async def teardown(self) -> None:
        """清理缓存连接"""
        if self.cache_manager and hasattr(self.cache_manager, 'close'):
            await self.cache_manager.close()
    
    async def test_cache_set(self, key: str, value: Any) -> None:
        """测试缓存设置性能"""
        if not self.cache_manager:
            return
        
        await self.cache_manager.set(key, value, 300)
    
    async def test_cache_get(self, key: str) -> Any:
        """测试缓存获取性能"""
        if not self.cache_manager:
            return None
        
        return await self.cache_manager.get(key)
    
    async def test_batch_operations(self, data: Dict[str, Any]) -> None:
        """测试批量操作性能"""
        if not self.cache_manager:
            return
        
        # 批量设置
        await self.cache_manager.batch_set(data, 300)
        
        # 批量获取
        keys = list(data.keys())
        results = await self.cache_manager.batch_get(keys)
        return len(results)
    
    async def run_cache_benchmarks(self) -> List[BenchmarkResult]:
        """运行所有缓存基准测试"""
        await self.setup()
        
        try:
            results = []
            
            # 准备测试数据
            test_data = {
                f"test_key_{i}": {
                    "timestamp": datetime.now().isoformat(),
                    "value": random.random(),
                    "metadata": {"index": i}
                }
                for i in range(100)
            }
            
            # 测试单个设置性能
            result = await self.run_benchmark(
                self.test_cache_set,
                iterations=1000,
                key="benchmark_key",
                value={"test": "data", "timestamp": datetime.now().isoformat()}
            )
            results.append(result)
            
            # 预设一些数据用于获取测试
            for key, value in list(test_data.items())[:10]:
                await self.test_cache_set(key, value)
            
            # 测试单个获取性能
            result = await self.run_benchmark(
                self.test_cache_get,
                iterations=1000,
                key="test_key_0"
            )
            results.append(result)
            
            # 测试批量操作性能
            result = await self.run_benchmark(
                self.test_batch_operations,
                iterations=100,
                data=test_data
            )
            results.append(result)
            
            return results
            
        finally:
            await self.teardown()


class WebSocketBenchmark(PerformanceBenchmark):
    """WebSocket性能基准测试"""
    
    def __init__(self):
        super().__init__("WebSocketBenchmark")
        self.websocket_server: Optional[MonitoringWebSocketServer] = None
        self.connection_manager: Optional[ConnectionManager] = None
    
    async def setup(self) -> None:
        """设置WebSocket服务器"""
        try:
            self.websocket_server = MonitoringWebSocketServer()
            await self.websocket_server.start()
            self.connection_manager = self.websocket_server.connection_manager
            self.logger.info("WebSocket基准测试设置完成")
        except Exception as e:
            self.logger.error(f"WebSocket设置失败: {e}")
            self.websocket_server = Mock()
            self.connection_manager = Mock()
    
    async def teardown(self) -> None:
        """清理WebSocket服务器"""
        if self.websocket_server and hasattr(self.websocket_server, 'stop'):
            await self.websocket_server.stop()
    
    async def test_message_compression(self, message_size: int) -> None:
        """测试消息压缩性能"""
        # 创建大消息
        large_data = {
            "type": "test_data",
            "payload": "x" * message_size,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"size": message_size}
        }
        
        message = WebSocketMessage(
            type=MessageType.DATA,
            data=large_data
        )
        
        # 测试压缩
        compressed_message = message.compress(1024)
        
        # 如果消息被压缩，测试解压缩
        if compressed_message.compressed:
            decompressed_message = WebSocketMessage.decompress(compressed_message)
            return len(json.dumps(decompressed_message.to_dict()))
        
        return len(json.dumps(message.to_dict()))
    
    async def test_batch_messaging(self, batch_size: int) -> None:
        """测试批量消息性能"""
        if not self.connection_manager:
            return
        
        # 创建模拟连接
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()
        
        from src.websocket.monitoring_websocket import ClientConnection, SubscriptionType
        
        connection = ClientConnection(
            client_id="test_client",
            websocket=mock_websocket,
            subscriptions={SubscriptionType.SYSTEM_HEALTH},
            connected_at=datetime.now(),
            last_ping=datetime.now(),
            metadata={},
            supports_batching=True,
            batch_size=batch_size
        )
        
        # 创建批量消息
        messages = []
        for i in range(batch_size):
            message = WebSocketMessage(
                type=MessageType.DATA,
                data={"index": i, "timestamp": datetime.now().isoformat()}
            )
            messages.append(message)
        
        # 测试批量发送
        await self.connection_manager._send_batch_messages(connection, messages)
        return batch_size
    
    async def run_websocket_benchmarks(self) -> List[BenchmarkResult]:
        """运行所有WebSocket基准测试"""
        await self.setup()
        
        try:
            results = []
            
            # 测试消息压缩性能
            result = await self.run_benchmark(
                self.test_message_compression,
                iterations=100,
                message_size=5000  # 5KB消息
            )
            results.append(result)
            
            # 测试批量消息性能
            result = await self.run_benchmark(
                self.test_batch_messaging,
                iterations=50,
                batch_size=20
            )
            results.append(result)
            
            return results
            
        finally:
            await self.teardown()


class IntegratedBenchmark:
    """集成性能基准测试"""
    
    def __init__(self, 
                 database_url: str = "postgresql://stockschool:stockschool123@localhost:15432/stockschool",
                 cache_config: Optional[CacheConfig] = None):
        self.database_url = database_url
        self.cache_config = cache_config or CacheConfig()
        self.logger = logging.getLogger(f"{__name__}.IntegratedBenchmark")
        self.results: Dict[str, List[BenchmarkResult]] = {}
    
    async def run_all_benchmarks(self) -> Dict[str, List[BenchmarkResult]]:
        """运行所有基准测试"""
        self.logger.info("🚀 开始运行性能基准测试...")
        
        # 数据库基准测试
        self.logger.info("📊 运行数据库基准测试...")
        db_benchmark = DatabaseBenchmark(self.database_url)
        self.results['database'] = await db_benchmark.run_database_benchmarks()
        
        # 缓存基准测试
        self.logger.info("🗄️ 运行缓存基准测试...")
        cache_benchmark = CacheBenchmark(self.cache_config)
        self.results['cache'] = await cache_benchmark.run_cache_benchmarks()
        
        # WebSocket基准测试
        self.logger.info("🔌 运行WebSocket基准测试...")
        ws_benchmark = WebSocketBenchmark()
        self.results['websocket'] = await ws_benchmark.run_websocket_benchmarks()
        
        self.logger.info("✅ 所有基准测试完成")
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "detailed_results": {},
            "recommendations": []
        }
        
        # 汇总结果
        total_tests = 0
        total_ops = 0
        avg_ops_per_second = 0
        
        for category, results in self.results.items():
            category_summary = {
                "test_count": len(results),
                "total_operations": sum(r.operations_count for r in results),
                "avg_ops_per_second": statistics.mean([r.ops_per_second for r in results]) if results else 0,
                "avg_success_rate": statistics.mean([r.success_rate for r in results]) if results else 0,
                "total_errors": sum(r.error_count for r in results)
            }
            
            report["summary"][category] = category_summary
            report["detailed_results"][category] = [r.to_dict() for r in results]
            
            total_tests += len(results)
            total_ops += category_summary["total_operations"]
            avg_ops_per_second += category_summary["avg_ops_per_second"]
        
        # 总体摘要
        report["summary"]["overall"] = {
            "total_test_categories": len(self.results),
            "total_tests": total_tests,
            "total_operations": total_ops,
            "avg_ops_per_second": avg_ops_per_second / len(self.results) if self.results else 0
        }
        
        # 生成建议
        report["recommendations"] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        # 分析数据库性能
        if 'database' in self.results:
            db_results = self.results['database']
            avg_db_ops = statistics.mean([r.ops_per_second for r in db_results]) if db_results else 0
            
            if avg_db_ops < 100:
                recommendations.append("数据库查询性能较低，建议检查索引配置和查询优化")
            
            high_error_tests = [r for r in db_results if r.error_count > 0]
            if high_error_tests:
                recommendations.append("数据库操作存在错误，建议检查连接配置和查询语法")
        
        # 分析缓存性能
        if 'cache' in self.results:
            cache_results = self.results['cache']
            avg_cache_ops = statistics.mean([r.ops_per_second for r in cache_results]) if cache_results else 0
            
            if avg_cache_ops < 1000:
                recommendations.append("缓存操作性能较低，建议检查Redis配置和网络延迟")
            
            high_memory_tests = [r for r in cache_results if r.memory_usage_mb > 100]
            if high_memory_tests:
                recommendations.append("缓存操作内存使用较高，建议优化数据结构和缓存策略")
        
        # 分析WebSocket性能
        if 'websocket' in self.results:
            ws_results = self.results['websocket']
            avg_ws_ops = statistics.mean([r.ops_per_second for r in ws_results]) if ws_results else 0
            
            if avg_ws_ops < 500:
                recommendations.append("WebSocket消息处理性能较低，建议启用压缩和批量传输")
        
        if not recommendations:
            recommendations.append("所有性能指标表现良好，系统运行正常")
        
        return recommendations
    
    def save_report(self, filename: str = None) -> str:
        """保存性能报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report = self.generate_report()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"性能报告已保存到: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"保存性能报告失败: {e}")
            raise


# 便捷函数
async def run_performance_benchmarks(
    database_url: str = "postgresql://stockschool:stockschool123@localhost:15432/stockschool",
    save_report: bool = True
) -> Dict[str, Any]:
    """运行性能基准测试"""
    benchmark = IntegratedBenchmark(database_url)
    
    try:
        await benchmark.run_all_benchmarks()
        report = benchmark.generate_report()
        
        if save_report:
            benchmark.save_report()
        
        return report
        
    except Exception as e:
        logging.error(f"性能基准测试失败: {e}")
        raise


# 主函数
async def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 开始监控系统性能基准测试...")
    
    try:
        report = await run_performance_benchmarks()
        
        print("\n📊 性能测试摘要:")
        print(f"总测试类别: {report['summary']['overall']['total_test_categories']}")
        print(f"总测试数量: {report['summary']['overall']['total_tests']}")
        print(f"总操作数量: {report['summary']['overall']['total_operations']}")
        print(f"平均操作/秒: {report['summary']['overall']['avg_ops_per_second']:.2f}")
        
        print("\n💡 优化建议:")
        for i, recommendation in enumerate(report['recommendations'], 1):
            print(f"{i}. {recommendation}")
        
        print("\n✅ 性能基准测试完成！")
        
    except Exception as e:
        print(f"❌ 性能基准测试失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
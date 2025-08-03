"""
数据同步调度器测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor

from src.data.data_sync_scheduler import (
    DataSyncScheduler,
    TaskDependencyManager,
    ResourceManager,
    FailureRecoveryManager,
    SyncTask,
    TaskStatus,
    TaskPriority,
    ResourcePool
)


class TestSyncTask:
    """同步任务测试类"""
    
    def test_sync_task_creation(self):
        """测试同步任务创建"""
        task = SyncTask(
            task_id="test_task_001",
            data_source="tushare",
            data_type="daily",
            target_date="2024-01-01",
            dependencies={"dep1", "dep2"},
            priority=TaskPriority.HIGH
        )
        
        assert task.task_id == "test_task_001"
        assert task.data_source == "tushare"
        assert task.data_type == "daily"
        assert task.target_date == "2024-01-01"
        assert task.dependencies == {"dep1", "dep2"}
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 0
    
    def test_sync_task_defaults(self):
        """测试同步任务默认值"""
        task = SyncTask(
            task_id="test_task_002",
            data_source="akshare",
            data_type="sentiment",
            target_date="2024-01-01"
        )
        
        assert task.dependencies == set()
        assert task.priority == TaskPriority.MEDIUM
        assert task.status == TaskStatus.PENDING
        assert task.max_retries == 3
        assert task.timeout == 300
        assert task.cpu_requirement == 1.0
        assert task.memory_requirement == 512


class TestTaskDependencyManager:
    """任务依赖管理器测试类"""
    
    @pytest.fixture
    def dependency_manager(self):
        """创建依赖管理器实例"""
        return TaskDependencyManager()
    
    @pytest.fixture
    def sample_tasks(self):
        """创建样本任务"""
        return [
            SyncTask("task_a", "tushare", "basic", "2024-01-01"),
            SyncTask("task_b", "tushare", "daily", "2024-01-01", dependencies={"task_a"}),
            SyncTask("task_c", "akshare", "sentiment", "2024-01-01", dependencies={"task_a"}),
            SyncTask("task_d", "tushare", "financial", "2024-01-01", dependencies={"task_b", "task_c"})
        ]
    
    def test_add_task(self, dependency_manager, sample_tasks):
        """测试添加任务"""
        task = sample_tasks[0]
        dependency_manager.add_task(task)
        
        assert task.task_id in dependency_manager.task_registry
        assert task.task_id in dependency_manager.dependency_graph
        assert dependency_manager.task_registry[task.task_id] == task
    
    def test_add_task_with_dependencies(self, dependency_manager, sample_tasks):
        """测试添加有依赖的任务"""
        # 先添加依赖任务
        dependency_manager.add_task(sample_tasks[0])  # task_a
        dependency_manager.add_task(sample_tasks[1])  # task_b (depends on task_a)
        
        # 检查依赖关系
        assert "task_a" in dependency_manager.dependency_graph["task_b"]
        assert "task_b" in dependency_manager.reverse_graph["task_a"]
    
    def test_remove_task(self, dependency_manager, sample_tasks):
        """测试移除任务"""
        # 添加任务
        for task in sample_tasks[:2]:
            dependency_manager.add_task(task)
        
        # 移除任务
        dependency_manager.remove_task("task_a")
        
        assert "task_a" not in dependency_manager.task_registry
        assert "task_a" not in dependency_manager.dependency_graph
        assert "task_a" not in dependency_manager.reverse_graph
        
        # 检查依赖关系是否正确清理
        assert "task_a" not in dependency_manager.dependency_graph["task_b"]
    
    def test_get_ready_tasks(self, dependency_manager, sample_tasks):
        """测试获取可执行任务"""
        # 添加所有任务
        for task in sample_tasks:
            dependency_manager.add_task(task)
        
        # 初始状态下只有task_a可以执行（无依赖）
        ready_tasks = dependency_manager.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "task_a"
        
        # 完成task_a后，task_b和task_c可以执行
        dependency_manager.mark_task_completed("task_a")
        ready_tasks = dependency_manager.get_ready_tasks()
        ready_task_ids = {task.task_id for task in ready_tasks}
        assert ready_task_ids == {"task_b", "task_c"}
        
        # 完成task_b和task_c后，task_d可以执行
        dependency_manager.mark_task_completed("task_b")
        dependency_manager.mark_task_completed("task_c")
        ready_tasks = dependency_manager.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "task_d"
    
    def test_get_ready_tasks_priority_order(self, dependency_manager):
        """测试按优先级排序的可执行任务"""
        tasks = [
            SyncTask("low_priority", "tushare", "basic", "2024-01-01", priority=TaskPriority.LOW),
            SyncTask("high_priority", "tushare", "basic", "2024-01-01", priority=TaskPriority.HIGH),
            SyncTask("medium_priority", "tushare", "basic", "2024-01-01", priority=TaskPriority.MEDIUM)
        ]
        
        for task in tasks:
            dependency_manager.add_task(task)
        
        ready_tasks = dependency_manager.get_ready_tasks()
        
        # 应该按优先级从高到低排序
        assert ready_tasks[0].task_id == "high_priority"
        assert ready_tasks[1].task_id == "medium_priority"
        assert ready_tasks[2].task_id == "low_priority"
    
    def test_mark_task_completed(self, dependency_manager, sample_tasks):
        """测试标记任务完成"""
        task = sample_tasks[0]
        dependency_manager.add_task(task)
        
        dependency_manager.mark_task_completed(task.task_id)
        
        assert task.status == TaskStatus.COMPLETED
        assert task.end_time is not None
    
    def test_mark_task_failed(self, dependency_manager, sample_tasks):
        """测试标记任务失败"""
        task = sample_tasks[0]
        dependency_manager.add_task(task)
        
        error_message = "测试错误"
        dependency_manager.mark_task_failed(task.task_id, error_message)
        
        assert task.status == TaskStatus.FAILED
        assert task.error_message == error_message
        assert task.end_time is not None
    
    def test_get_task_chain(self, dependency_manager, sample_tasks):
        """测试获取任务依赖链"""
        for task in sample_tasks:
            dependency_manager.add_task(task)
        
        # task_d的依赖链应该包含所有任务
        chain = dependency_manager.get_task_chain("task_d")
        
        # 应该按依赖顺序排列
        assert "task_a" in chain
        assert "task_b" in chain
        assert "task_c" in chain
        assert "task_d" in chain
        
        # task_a应该在task_b和task_c之前
        assert chain.index("task_a") < chain.index("task_b")
        assert chain.index("task_a") < chain.index("task_c")
        
        # task_d应该在最后
        assert chain.index("task_d") == len(chain) - 1
    
    def test_detect_circular_dependencies(self, dependency_manager):
        """测试检测循环依赖"""
        # 创建循环依赖：A -> B -> C -> A
        tasks = [
            SyncTask("task_a", "tushare", "basic", "2024-01-01", dependencies={"task_c"}),
            SyncTask("task_b", "tushare", "daily", "2024-01-01", dependencies={"task_a"}),
            SyncTask("task_c", "akshare", "sentiment", "2024-01-01", dependencies={"task_b"})
        ]
        
        for task in tasks:
            dependency_manager.add_task(task)
        
        cycles = dependency_manager.detect_circular_dependencies()
        
        assert len(cycles) > 0
        # 应该检测到包含这三个任务的循环
        cycle_tasks = set()
        for cycle in cycles:
            cycle_tasks.update(cycle)
        
        assert "task_a" in cycle_tasks
        assert "task_b" in cycle_tasks
        assert "task_c" in cycle_tasks
    
    def test_get_task_statistics(self, dependency_manager, sample_tasks):
        """测试获取任务统计信息"""
        for task in sample_tasks:
            dependency_manager.add_task(task)
        
        # 标记一些任务为不同状态
        dependency_manager.mark_task_completed("task_a")
        dependency_manager.mark_task_failed("task_b", "测试失败")
        
        stats = dependency_manager.get_task_statistics()
        
        assert stats['total_tasks'] == 4
        assert stats['status_distribution']['completed'] == 1
        assert stats['status_distribution']['failed'] == 1
        assert stats['status_distribution']['pending'] == 2


class TestResourceManager:
    """资源管理器测试类"""
    
    @pytest.fixture
    def resource_pool(self):
        """创建资源池"""
        return ResourcePool(
            max_cpu_cores=4.0,
            max_memory_mb=2048,
            max_api_calls_per_minute={'tushare': 200, 'akshare': 100}
        )
    
    @pytest.fixture
    def resource_manager(self, resource_pool):
        """创建资源管理器"""
        return ResourceManager(resource_pool)
    
    def test_can_allocate_resources_success(self, resource_manager):
        """测试资源分配检查 - 成功情况"""
        task = SyncTask(
            "test_task", "tushare", "daily", "2024-01-01",
            cpu_requirement=2.0,
            memory_requirement=1024,
            api_calls_required=50
        )
        
        assert resource_manager.can_allocate_resources(task) is True
    
    def test_can_allocate_resources_cpu_limit(self, resource_manager):
        """测试资源分配检查 - CPU限制"""
        # 先分配大部分CPU资源
        resource_manager.resource_pool.used_cpu_cores = 3.5
        
        task = SyncTask(
            "test_task", "tushare", "daily", "2024-01-01",
            cpu_requirement=1.0  # 超过剩余的0.5核
        )
        
        assert resource_manager.can_allocate_resources(task) is False
    
    def test_can_allocate_resources_memory_limit(self, resource_manager):
        """测试资源分配检查 - 内存限制"""
        # 先分配大部分内存资源
        resource_manager.resource_pool.used_memory_mb = 1800
        
        task = SyncTask(
            "test_task", "tushare", "daily", "2024-01-01",
            memory_requirement=512  # 超过剩余的248MB
        )
        
        assert resource_manager.can_allocate_resources(task) is False
    
    def test_allocate_and_release_resources(self, resource_manager):
        """测试资源分配和释放"""
        task = SyncTask(
            "test_task", "tushare", "daily", "2024-01-01",
            cpu_requirement=2.0,
            memory_requirement=1024,
            api_calls_required=50
        )
        
        # 分配资源
        assert resource_manager.allocate_resources(task) is True
        assert resource_manager.resource_pool.used_cpu_cores == 2.0
        assert resource_manager.resource_pool.used_memory_mb == 1024
        assert resource_manager.resource_pool.api_calls_used['tushare'] == 50
        
        # 释放资源
        resource_manager.release_resources(task)
        assert resource_manager.resource_pool.used_cpu_cores == 0.0
        assert resource_manager.resource_pool.used_memory_mb == 0
    
    def test_api_call_rate_limiting(self, resource_manager):
        """测试API调用频率限制"""
        # 创建大量API调用的任务
        task = SyncTask(
            "test_task", "tushare", "daily", "2024-01-01",
            api_calls_required=250  # 超过限制的200
        )
        
        assert resource_manager.can_allocate_resources(task) is False
    
    def test_get_resource_usage(self, resource_manager):
        """测试获取资源使用情况"""
        task = SyncTask(
            "test_task", "tushare", "daily", "2024-01-01",
            cpu_requirement=2.0,
            memory_requirement=1024,
            api_calls_required=50
        )
        
        resource_manager.allocate_resources(task)
        usage = resource_manager.get_resource_usage()
        
        assert usage['cpu']['used'] == 2.0
        assert usage['cpu']['max'] == 4.0
        assert usage['cpu']['usage_rate'] == 0.5
        
        assert usage['memory']['used'] == 1024
        assert usage['memory']['max'] == 2048
        assert usage['memory']['usage_rate'] == 0.5
        
        assert 'tushare' in usage['api_calls']
        assert usage['api_calls']['tushare']['current'] == 50


class TestFailureRecoveryManager:
    """失败恢复管理器测试类"""
    
    @pytest.fixture
    def dependency_manager(self):
        """创建依赖管理器"""
        return TaskDependencyManager()
    
    @pytest.fixture
    def recovery_manager(self, dependency_manager):
        """创建失败恢复管理器"""
        return FailureRecoveryManager(dependency_manager)
    
    def test_handle_task_failure_with_retry(self, recovery_manager, dependency_manager):
        """测试任务失败处理 - 可重试"""
        task = SyncTask("test_task", "tushare", "daily", "2024-01-01", max_retries=3)
        dependency_manager.add_task(task)
        
        error = Exception("网络错误")
        can_retry = recovery_manager.handle_task_failure(task, error)
        
        assert can_retry is True
        assert task.retry_count == 1
        assert task.error_message == "网络错误"
        assert task.status == TaskStatus.PENDING  # 重置为待执行状态
    
    def test_handle_task_failure_max_retries_reached(self, recovery_manager, dependency_manager):
        """测试任务失败处理 - 达到最大重试次数"""
        task = SyncTask("test_task", "tushare", "daily", "2024-01-01", max_retries=2)
        task.retry_count = 2  # 已经重试了2次
        dependency_manager.add_task(task)
        
        error = Exception("持续错误")
        can_retry = recovery_manager.handle_task_failure(task, error)
        
        assert can_retry is False
        assert task.status == TaskStatus.FAILED
    
    def test_exponential_backoff_retry(self, recovery_manager):
        """测试指数退避重试策略"""
        delay1 = recovery_manager._exponential_backoff_retry(1)
        delay2 = recovery_manager._exponential_backoff_retry(2)
        delay3 = recovery_manager._exponential_backoff_retry(3)
        
        assert delay1 == 2
        assert delay2 == 4
        assert delay3 == 8
        
        # 测试最大延迟限制
        delay_max = recovery_manager._exponential_backoff_retry(10)
        assert delay_max == 300  # 最大5分钟
    
    def test_fixed_interval_retry(self, recovery_manager):
        """测试固定间隔重试策略"""
        delay1 = recovery_manager._fixed_interval_retry(1)
        delay2 = recovery_manager._fixed_interval_retry(5)
        
        assert delay1 == 60
        assert delay2 == 60  # 固定1分钟
    
    def test_handle_dependent_tasks(self, recovery_manager, dependency_manager):
        """测试处理依赖任务"""
        # 创建依赖链：A -> B -> C
        task_a = SyncTask("task_a", "tushare", "basic", "2024-01-01")
        task_b = SyncTask("task_b", "tushare", "daily", "2024-01-01", dependencies={"task_a"})
        task_c = SyncTask("task_c", "akshare", "sentiment", "2024-01-01", dependencies={"task_b"})
        
        for task in [task_a, task_b, task_c]:
            dependency_manager.add_task(task)
        
        # 标记task_a失败
        dependency_manager.mark_task_failed("task_a", "测试失败")
        
        # 处理依赖任务
        recovery_manager._handle_dependent_tasks("task_a")
        
        # 检查依赖任务是否被正确处理
        # 这里的具体行为取决于实现策略


class TestDataSyncScheduler:
    """数据同步调度器测试类"""
    
    @pytest.fixture
    def scheduler(self):
        """创建调度器实例"""
        with patch('src.data.data_sync_scheduler.get_db_engine'):
            return DataSyncScheduler(max_workers=2)
    
    def test_scheduler_initialization(self, scheduler):
        """测试调度器初始化"""
        assert scheduler.max_workers == 2
        assert isinstance(scheduler.dependency_manager, TaskDependencyManager)
        assert isinstance(scheduler.resource_manager, ResourceManager)
        assert isinstance(scheduler.recovery_manager, FailureRecoveryManager)
        assert scheduler.is_running is False
    
    def test_add_sync_task(self, scheduler):
        """测试添加同步任务"""
        task = SyncTask("test_task", "tushare", "daily", "2024-01-01")
        scheduler.add_sync_task(task)
        
        assert task.task_id in scheduler.dependency_manager.task_registry
    
    def test_create_daily_sync_plan(self, scheduler):
        """测试创建每日同步计划"""
        target_date = "2024-01-01"
        tasks = scheduler.create_daily_sync_plan(target_date)
        
        assert len(tasks) > 0
        
        # 检查是否包含基础任务
        task_ids = [task.task_id for task in tasks]
        assert f"stock_basic_{target_date}" in task_ids
        assert f"trade_calendar_{target_date}" in task_ids
        assert f"daily_data_{target_date}" in task_ids
        
        # 检查依赖关系
        daily_task = next(task for task in tasks if task.task_id == f"daily_data_{target_date}")
        assert f"stock_basic_{target_date}" in daily_task.dependencies
        assert f"trade_calendar_{target_date}" in daily_task.dependencies
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, scheduler):
        """测试任务执行成功"""
        task = SyncTask("test_task", "tushare", "stock_basic", "2024-01-01")
        scheduler.add_sync_task(task)
        
        # 模拟同步器
        mock_synchronizer = Mock()
        mock_synchronizer.sync_stock_basic.return_value = {"success": True, "count": 100}
        scheduler.synchronizers['tushare'] = mock_synchronizer
        
        with patch.object(scheduler.resource_manager, 'allocate_resources', return_value=True):
            with patch.object(scheduler.resource_manager, 'release_resources'):
                result = await scheduler.execute_task(task)
                
                assert result is True
                assert task.status == TaskStatus.COMPLETED
                assert task.result is not None
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self, scheduler):
        """测试任务执行失败"""
        task = SyncTask("test_task", "tushare", "stock_basic", "2024-01-01")
        scheduler.add_sync_task(task)
        
        # 模拟同步器抛出异常
        mock_synchronizer = Mock()
        mock_synchronizer.sync_stock_basic.side_effect = Exception("同步失败")
        scheduler.synchronizers['tushare'] = mock_synchronizer
        
        with patch.object(scheduler.resource_manager, 'allocate_resources', return_value=True):
            with patch.object(scheduler.resource_manager, 'release_resources'):
                with patch.object(scheduler.recovery_manager, 'handle_task_failure', return_value=False):
                    result = await scheduler.execute_task(task)
                    
                    assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_task_resource_allocation_failure(self, scheduler):
        """测试资源分配失败"""
        task = SyncTask("test_task", "tushare", "daily", "2024-01-01")
        
        with patch.object(scheduler.resource_manager, 'allocate_resources', return_value=False):
            result = await scheduler.execute_task(task)
            
            assert result is False
    
    def test_get_scheduler_status(self, scheduler):
        """测试获取调度器状态"""
        # 添加一些任务
        tasks = [
            SyncTask("task1", "tushare", "basic", "2024-01-01"),
            SyncTask("task2", "tushare", "daily", "2024-01-01"),
        ]
        
        for task in tasks:
            scheduler.add_sync_task(task)
        
        status = scheduler.get_scheduler_status()
        
        assert 'is_running' in status
        assert 'running_tasks_count' in status
        assert 'task_statistics' in status
        assert 'resource_usage' in status
        assert 'max_workers' in status
        
        assert status['max_workers'] == 2
        assert status['task_statistics']['total_tasks'] == 2
    
    def test_stop_scheduler(self, scheduler):
        """测试停止调度器"""
        scheduler.is_running = True
        scheduler.stop_scheduler()
        
        assert scheduler.is_running is False
    
    @patch('src.data.data_sync_scheduler.get_db_engine')
    def test_save_task_status(self, mock_engine, scheduler):
        """测试保存任务状态"""
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        
        task = SyncTask("test_task", "tushare", "daily", "2024-01-01")
        task.status = TaskStatus.COMPLETED
        scheduler.add_sync_task(task)
        
        scheduler.save_task_status()
        
        # 验证数据库操作被调用
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()


class TestIntegration:
    """集成测试类"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self):
        """测试完整工作流程模拟"""
        with patch('src.data.data_sync_scheduler.get_db_engine'):
            scheduler = DataSyncScheduler(max_workers=2)
            
            # 创建简单的任务计划
            tasks = [
                SyncTask("task_a", "tushare", "basic", "2024-01-01", priority=TaskPriority.HIGH),
                SyncTask("task_b", "tushare", "daily", "2024-01-01", 
                        dependencies={"task_a"}, priority=TaskPriority.MEDIUM)
            ]
            
            # 添加任务
            for task in tasks:
                scheduler.add_sync_task(task)
            
            # 模拟同步器
            mock_synchronizer = Mock()
            mock_synchronizer.sync_stock_basic.return_value = {"success": True}
            mock_synchronizer.sync_daily_data.return_value = {"success": True}
            scheduler.synchronizers['tushare'] = mock_synchronizer
            
            # 检查初始状态
            ready_tasks = scheduler.dependency_manager.get_ready_tasks()
            assert len(ready_tasks) == 1
            assert ready_tasks[0].task_id == "task_a"
            
            # 执行第一个任务
            with patch.object(scheduler.resource_manager, 'allocate_resources', return_value=True):
                with patch.object(scheduler.resource_manager, 'release_resources'):
                    result = await scheduler.execute_task(ready_tasks[0])
                    assert result is True
            
            # 检查第二个任务是否可以执行
            ready_tasks = scheduler.dependency_manager.get_ready_tasks()
            assert len(ready_tasks) == 1
            assert ready_tasks[0].task_id == "task_b"
            
            # 执行第二个任务
            with patch.object(scheduler.resource_manager, 'allocate_resources', return_value=True):
                with patch.object(scheduler.resource_manager, 'release_resources'):
                    result = await scheduler.execute_task(ready_tasks[0])
                    assert result is True
            
            # 检查所有任务都已完成
            ready_tasks = scheduler.dependency_manager.get_ready_tasks()
            assert len(ready_tasks) == 0
            
            # 检查任务状态
            stats = scheduler.dependency_manager.get_task_statistics()
            assert stats['status_distribution']['completed'] == 2


if __name__ == '__main__':
    pytest.main([__file__])
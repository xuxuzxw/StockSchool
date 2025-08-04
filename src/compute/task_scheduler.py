#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务调度和监控系统
实现自动化任务调度、资源管理和进度监控
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta, time
from typing import List, Dict, Optional, Any, Callable, Union
from loguru import logger
from sqlalchemy import text
import threading
import time as time_module
import schedule
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
import json
import uuid
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil

from .factor_models import FactorType, FactorResult, CalculationStatus
from .parallel_factor_calculator import ParallelFactorCalculator
from .incremental_calculator import IncrementalFactorCalculator


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class TaskType(Enum):
    """任务类型枚举"""
    DAILY_CALCULATION = "daily_calculation"
    INCREMENTAL_UPDATE = "incremental_update"
    MANUAL_CALCULATION = "manual_calculation"
    DATA_MAINTENANCE = "data_maintenance"
    SYSTEM_MAINTENANCE = "system_maintenance"


@dataclass
class TaskConfig:
    """任务配置"""
    task_id: str
    task_name: str
    task_type: TaskType
    priority: TaskPriority
    dependencies: List[str]  # 依赖的任务ID列表
    max_retries: int = 3
    retry_delay: int = 60  # 重试延迟（秒）
    timeout: int = 3600  # 超时时间（秒）
    enabled: bool = True
    schedule_time: Optional[str] = None  # 调度时间，格式："HH:MM"
    schedule_days: List[str] = None  # 调度日期，["monday", "tuesday", ...]
    parameters: Dict[str, Any] = None


@dataclass
class TaskExecution:
    """任务执行记录"""
    execution_id: str
    task_id: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    resource_usage: Optional[Dict[str, float]] = None


class TaskDependencyManager:
    """任务依赖管理器"""
    
    def __init__(self):
        """初始化依赖管理器"""
        self.dependency_graph = defaultdict(set)  # task_id -> set of dependent tasks
        self.reverse_graph = defaultdict(set)     # task_id -> set of dependencies
    
    def add_dependency(self, task_id: str, dependency_id: str):
        """添加任务依赖"""
        self.dependency_graph[dependency_id].add(task_id)
        self.reverse_graph[task_id].add(dependency_id)
    
    def remove_dependency(self, task_id: str, dependency_id: str):
        """移除任务依赖"""
        self.dependency_graph[dependency_id].discard(task_id)
        self.reverse_graph[task_id].discard(dependency_id)
    
    def get_dependencies(self, task_id: str) -> set:
        """获取任务的依赖项"""
        return self.reverse_graph[task_id].copy()
    
    def get_dependents(self, task_id: str) -> set:
        """获取依赖于指定任务的任务列表"""
        return self.dependency_graph[task_id].copy()
    
    def is_ready_to_run(self, task_id: str, completed_tasks: set) -> bool:
        """检查任务是否可以运行（所有依赖都已完成）"""
        dependencies = self.get_dependencies(task_id)
        return dependencies.issubset(completed_tasks)
    
    def get_execution_order(self, task_ids: List[str]) -> List[str]:
        """获取任务执行顺序（拓扑排序）"""
        # 构建子图
        subgraph = defaultdict(set)
        in_degree = defaultdict(int)
        
        # 初始化
        for task_id in task_ids:
            in_degree[task_id] = 0
        
        # 构建子图的边和入度
        for task_id in task_ids:
            dependencies = self.get_dependencies(task_id)
            for dep_id in dependencies:
                if dep_id in task_ids:
                    subgraph[dep_id].add(task_id)
                    in_degree[task_id] += 1
        
        # 拓扑排序
        result = []
        queue = deque([task_id for task_id in task_ids if in_degree[task_id] == 0])
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for dependent in subgraph[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # 检查是否有循环依赖
        if len(result) != len(task_ids):
            logger.warning("检测到循环依赖，使用原始顺序")
            return task_ids
        
        return result


class ResourceManager:
    """计算资源管理器"""
    
    def __init__(self, max_cpu_percent: float = 80.0, max_memory_percent: float = 80.0):
        """初始化资源管理器"""
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.running_tasks = {}  # execution_id -> resource_info
        self.resource_lock = threading.Lock()
    
    def check_resource_availability(self) -> Dict[str, bool]:
        """检查资源可用性"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        return {
            'cpu_available': cpu_percent < self.max_cpu_percent,
            'memory_available': memory_percent < self.max_memory_percent,
            'overall_available': (cpu_percent < self.max_cpu_percent and 
                                memory_percent < self.max_memory_percent)
        }
    
    def estimate_task_resources(self, task_config: TaskConfig) -> Dict[str, float]:
        """估算任务资源需求"""
        # 基于任务类型和参数估算资源需求
        base_cpu = 20.0  # 基础CPU使用率
        base_memory = 500.0  # 基础内存使用量(MB)
        
        if task_config.task_type == TaskType.DAILY_CALCULATION:
            # 日常计算任务
            stock_count = task_config.parameters.get('stock_count', 1000) if task_config.parameters else 1000
            factor_count = task_config.parameters.get('factor_count', 50) if task_config.parameters else 50
            
            cpu_usage = base_cpu + (stock_count * factor_count * 0.001)
            memory_usage = base_memory + (stock_count * factor_count * 0.1)
            
        elif task_config.task_type == TaskType.INCREMENTAL_UPDATE:
            # 增量更新任务
            cpu_usage = base_cpu * 0.5
            memory_usage = base_memory * 0.3
            
        elif task_config.task_type == TaskType.DATA_MAINTENANCE:
            # 数据维护任务
            cpu_usage = base_cpu * 0.3
            memory_usage = base_memory * 2.0
            
        else:
            cpu_usage = base_cpu
            memory_usage = base_memory
        
        return {
            'estimated_cpu_percent': min(cpu_usage, 90.0),
            'estimated_memory_mb': memory_usage,
            'estimated_duration_minutes': task_config.timeout / 60
        }
    
    def can_run_task(self, task_config: TaskConfig) -> bool:
        """检查是否可以运行任务"""
        with self.resource_lock:
            # 检查当前资源状态
            availability = self.check_resource_availability()
            if not availability['overall_available']:
                return False
            
            # 估算任务资源需求
            estimated_resources = self.estimate_task_resources(task_config)
            
            # 检查是否有足够资源
            current_cpu = psutil.cpu_percent()
            current_memory = psutil.virtual_memory().percent
            
            projected_cpu = current_cpu + estimated_resources['estimated_cpu_percent']
            projected_memory_mb = estimated_resources['estimated_memory_mb']
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            
            return (projected_cpu < self.max_cpu_percent and 
                   projected_memory_mb < available_memory_mb)
    
    def register_running_task(self, execution_id: str, task_config: TaskConfig):
        """注册正在运行的任务"""
        with self.resource_lock:
            self.running_tasks[execution_id] = {
                'task_id': task_config.task_id,
                'start_time': datetime.now(),
                'estimated_resources': self.estimate_task_resources(task_config)
            }
    
    def unregister_task(self, execution_id: str):
        """注销任务"""
        with self.resource_lock:
            self.running_tasks.pop(execution_id, None)
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_info.percent,
            'memory_used_mb': memory_info.used / (1024 * 1024),
            'memory_available_mb': memory_info.available / (1024 * 1024),
            'disk_percent': disk_info.percent,
            'disk_used_gb': disk_info.used / (1024 * 1024 * 1024),
            'disk_free_gb': disk_info.free / (1024 * 1024 * 1024),
            'running_tasks_count': len(self.running_tasks),
            'running_tasks': list(self.running_tasks.keys())
        }


class TaskQueue:
    """任务队列管理器"""
    
    def __init__(self):
        """初始化任务队列"""
        self.pending_queue = []  # 待执行任务队列
        self.running_tasks = {}  # 正在执行的任务
        self.completed_tasks = {}  # 已完成的任务
        self.failed_tasks = {}  # 失败的任务
        self.queue_lock = threading.Lock()
    
    def add_task(self, task_config: TaskConfig, execution: TaskExecution):
        """添加任务到队列"""
        with self.queue_lock:
            # 按优先级插入队列
            priority_value = task_config.priority.value
            
            # 找到插入位置
            insert_index = 0
            for i, (config, exec_record) in enumerate(self.pending_queue):
                if config.priority.value < priority_value:
                    insert_index = i
                    break
                insert_index = i + 1
            
            self.pending_queue.insert(insert_index, (task_config, execution))
    
    def get_next_task(self) -> Optional[tuple]:
        """获取下一个待执行任务"""
        with self.queue_lock:
            if self.pending_queue:
                return self.pending_queue.pop(0)
            return None
    
    def move_to_running(self, execution_id: str, task_config: TaskConfig, execution: TaskExecution):
        """将任务移动到运行状态"""
        with self.queue_lock:
            self.running_tasks[execution_id] = (task_config, execution)
    
    def move_to_completed(self, execution_id: str, execution: TaskExecution):
        """将任务移动到完成状态"""
        with self.queue_lock:
            if execution_id in self.running_tasks:
                task_config, _ = self.running_tasks.pop(execution_id)
                self.completed_tasks[execution_id] = (task_config, execution)
    
    def move_to_failed(self, execution_id: str, execution: TaskExecution):
        """将任务移动到失败状态"""
        with self.queue_lock:
            if execution_id in self.running_tasks:
                task_config, _ = self.running_tasks.pop(execution_id)
                self.failed_tasks[execution_id] = (task_config, execution)
    
    def get_queue_status(self) -> Dict[str, int]:
        """获取队列状态"""
        with self.queue_lock:
            return {
                'pending': len(self.pending_queue),
                'running': len(self.running_tasks),
                'completed': len(self.completed_tasks),
                'failed': len(self.failed_tasks)
            }


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, engine, max_workers: int = 4):
        """初始化任务调度器"""
        self.engine = engine
        self.max_workers = max_workers
        
        # 初始化组件
        self.dependency_manager = TaskDependencyManager()
        self.resource_manager = ResourceManager()
        self.task_queue = TaskQueue()
        
        # 任务配置
        self.task_configs = {}  # task_id -> TaskConfig
        self.task_functions = {}  # task_id -> callable
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_futures = {}  # execution_id -> Future
        
        # 调度器状态
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # 统计信息
        self.execution_history = deque(maxlen=1000)  # 保留最近1000次执行记录
        
        # 初始化调度任务
        self._setup_default_schedules()
    
    def _setup_default_schedules(self):
        """设置默认调度任务"""
        # 每日收盘后因子计算
        schedule.every().day.at("16:30").do(self._trigger_daily_calculation)
        
        # 每小时增量更新
        schedule.every().hour.at(":05").do(self._trigger_incremental_update)
        
        # 每日数据维护
        schedule.every().day.at("02:00").do(self._trigger_data_maintenance)
    
    def register_task(self, task_config: TaskConfig, task_function: Callable):
        """注册任务"""
        self.task_configs[task_config.task_id] = task_config
        self.task_functions[task_config.task_id] = task_function
        
        # 注册依赖关系
        for dep_id in task_config.dependencies:
            self.dependency_manager.add_dependency(task_config.task_id, dep_id)
        
        logger.info(f"注册任务: {task_config.task_name} ({task_config.task_id})")
    
    def submit_task(self, task_id: str, parameters: Dict[str, Any] = None) -> str:
        """提交任务执行"""
        if task_id not in self.task_configs:
            raise ValueError(f"未知任务ID: {task_id}")
        
        task_config = self.task_configs[task_id]
        
        # 更新参数
        if parameters:
            if task_config.parameters:
                task_config.parameters.update(parameters)
            else:
                task_config.parameters = parameters
        
        # 创建执行记录
        execution_id = str(uuid.uuid4())
        execution = TaskExecution(
            execution_id=execution_id,
            task_id=task_id,
            status=TaskStatus.PENDING
        )
        
        # 添加到队列
        self.task_queue.add_task(task_config, execution)
        
        logger.info(f"提交任务: {task_config.task_name} (执行ID: {execution_id})")
        
        return execution_id
    
    def _execute_task(self, task_config: TaskConfig, execution: TaskExecution) -> TaskExecution:
        """执行单个任务"""
        execution_id = execution.execution_id
        
        try:
            # 更新状态
            execution.status = TaskStatus.RUNNING
            execution.start_time = datetime.now()
            
            # 注册资源使用
            self.resource_manager.register_running_task(execution_id, task_config)
            
            # 移动到运行队列
            self.task_queue.move_to_running(execution_id, task_config, execution)
            
            logger.info(f"开始执行任务: {task_config.task_name}")
            
            # 执行任务函数
            task_function = self.task_functions[task_config.task_id]
            result = task_function(task_config.parameters or {})
            
            # 更新执行结果
            execution.end_time = datetime.now()
            execution.duration = execution.end_time - execution.start_time
            execution.result = result
            execution.status = TaskStatus.SUCCESS
            execution.resource_usage = self.resource_manager.get_resource_usage()
            
            # 移动到完成队列
            self.task_queue.move_to_completed(execution_id, execution)
            
            logger.info(f"任务执行成功: {task_config.task_name}, 耗时: {execution.duration}")
            
        except Exception as e:
            # 处理执行失败
            execution.end_time = datetime.now()
            execution.duration = execution.end_time - execution.start_time if execution.start_time else timedelta(0)
            execution.error_message = str(e)
            execution.status = TaskStatus.FAILED
            
            # 检查是否需要重试
            if execution.retry_count < task_config.max_retries:
                execution.retry_count += 1
                execution.status = TaskStatus.RETRYING
                
                logger.warning(f"任务执行失败，准备重试: {task_config.task_name}, "
                             f"重试次数: {execution.retry_count}/{task_config.max_retries}, 错误: {e}")
                
                # 延迟后重新提交
                def retry_task():
                    time_module.sleep(task_config.retry_delay)
                    self.task_queue.add_task(task_config, execution)
                
                threading.Thread(target=retry_task, daemon=True).start()
            else:
                # 移动到失败队列
                self.task_queue.move_to_failed(execution_id, execution)
                logger.error(f"任务执行失败，已达最大重试次数: {task_config.task_name}, 错误: {e}")
        
        finally:
            # 注销资源
            self.resource_manager.unregister_task(execution_id)
            
            # 记录执行历史
            self.execution_history.append(execution)
            
            # 保存执行记录到数据库
            self._save_execution_record(execution)
        
        return execution
    
    def _save_execution_record(self, execution: TaskExecution):
        """保存执行记录到数据库"""
        try:
            record_data = {
                'execution_id': execution.execution_id,
                'task_id': execution.task_id,
                'status': execution.status.value,
                'start_time': execution.start_time,
                'end_time': execution.end_time,
                'duration_seconds': execution.duration.total_seconds() if execution.duration else None,
                'retry_count': execution.retry_count,
                'error_message': execution.error_message,
                'resource_usage': json.dumps(execution.resource_usage) if execution.resource_usage else None,
                'created_time': datetime.now()
            }
            
            df = pd.DataFrame([record_data])
            df.to_sql(
                'task_execution_log',
                self.engine,
                if_exists='append',
                index=False
            )
            
        except Exception as e:
            logger.error(f"保存执行记录失败: {e}")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        logger.info("任务调度器启动")
        
        while self.scheduler_running:
            try:
                # 检查定时任务
                schedule.run_pending()
                
                # 处理待执行任务
                self._process_pending_tasks()
                
                # 短暂休眠
                time_module.sleep(1)
                
            except Exception as e:
                logger.error(f"调度器循环异常: {e}")
                time_module.sleep(5)
        
        logger.info("任务调度器停止")
    
    def _process_pending_tasks(self):
        """处理待执行任务"""
        # 获取已完成的任务集合
        completed_task_ids = set()
        for execution_id, (task_config, execution) in self.task_queue.completed_tasks.items():
            if execution.status == TaskStatus.SUCCESS:
                completed_task_ids.add(task_config.task_id)
        
        # 检查待执行任务
        tasks_to_execute = []
        
        while True:
            next_task = self.task_queue.get_next_task()
            if not next_task:
                break
            
            task_config, execution = next_task
            
            # 检查依赖是否满足
            if not self.dependency_manager.is_ready_to_run(task_config.task_id, completed_task_ids):
                # 依赖未满足，重新加入队列
                self.task_queue.add_task(task_config, execution)
                break
            
            # 检查资源是否可用
            if not self.resource_manager.can_run_task(task_config):
                # 资源不足，重新加入队列
                self.task_queue.add_task(task_config, execution)
                break
            
            tasks_to_execute.append((task_config, execution))
            
            # 限制并发任务数
            if len(tasks_to_execute) >= self.max_workers:
                break
        
        # 提交任务执行
        for task_config, execution in tasks_to_execute:
            future = self.executor.submit(self._execute_task, task_config, execution)
            self.running_futures[execution.execution_id] = future
    
    def start_scheduler(self):
        """启动调度器"""
        if self.scheduler_running:
            logger.warning("调度器已在运行")
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("任务调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        if not self.scheduler_running:
            logger.warning("调度器未在运行")
            return
        
        self.scheduler_running = False
        
        # 等待调度器线程结束
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        
        # 关闭执行器
        self.executor.shutdown(wait=True)
        
        logger.info("任务调度器已停止")
    
    def _trigger_daily_calculation(self):
        """触发日常因子计算"""
        logger.info("触发日常因子计算")
        self.submit_task('daily_factor_calculation')
    
    def _trigger_incremental_update(self):
        """触发增量更新"""
        logger.info("触发增量更新")
        self.submit_task('incremental_factor_update')
    
    def _trigger_data_maintenance(self):
        """触发数据维护"""
        logger.info("触发数据维护")
        self.submit_task('data_maintenance')
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        queue_status = self.task_queue.get_queue_status()
        resource_usage = self.resource_manager.get_resource_usage()
        
        # 计算执行统计
        recent_executions = list(self.execution_history)[-100:]  # 最近100次执行
        success_count = sum(1 for exec in recent_executions if exec.status == TaskStatus.SUCCESS)
        failed_count = sum(1 for exec in recent_executions if exec.status == TaskStatus.FAILED)
        
        return {
            'scheduler_running': self.scheduler_running,
            'queue_status': queue_status,
            'resource_usage': resource_usage,
            'execution_stats': {
                'recent_executions': len(recent_executions),
                'success_count': success_count,
                'failed_count': failed_count,
                'success_rate': success_count / len(recent_executions) if recent_executions else 0
            },
            'registered_tasks': len(self.task_configs),
            'running_futures': len(self.running_futures)
        }
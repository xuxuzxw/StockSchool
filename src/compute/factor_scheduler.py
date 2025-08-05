import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from queue import Empty, PriorityQueue
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算调度器
负责管理和调度因子计算任务
"""


from .base_factor_engine import BaseFactorEngine
from .factor_models import CalculationStatus, CalculationTask, FactorResult, FactorType


@dataclass
class PriorityTask:
    """优先级任务包装器"""
    priority: int
    task: CalculationTask

    def __lt__(self, other):
        """方法描述"""
        return -self.priority < -other.priority


class FactorCalculationScheduler:
    """因子计算调度器"""

    def __init__(self, max_workers: int = 4):
        """
        初始化调度器

        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.task_queue = PriorityQueue()
        self.running_tasks: Dict[str, CalculationTask] = {}
        self.completed_tasks: Dict[str, CalculationTask] = {}
        self.engines: Dict[FactorType, BaseFactorEngine] = {}

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._scheduler_thread = None

        # 任务状态回调
        self._status_callbacks: List[Callable[[CalculationTask], None]] = []

        logger.info(f"因子计算调度器初始化完成，最大工作线程数: {max_workers}")

    def register_engine(self, factor_type: FactorType, engine: BaseFactorEngine):
        """
        注册因子引擎

        Args:
            factor_type: 因子类型
            engine: 因子引擎实例
        """
        self.engines[factor_type] = engine
        logger.info(f"注册{factor_type.value}因子引擎")

    def add_status_callback(self, callback: Callable[[CalculationTask], None]):
        """
        添加任务状态变更回调

        Args:
            callback: 回调函数
        """
        self._status_callbacks.append(callback)

    def submit_task(self,
                   ts_codes: List[str],
                   factor_types: List[FactorType],
                   start_date: Optional[date] = None,
                   end_date: Optional[date] = None,
                   priority: int = 0) -> str:
        """
        提交计算任务

        Args:
            ts_codes: 股票代码列表
            factor_types: 因子类型列表
            start_date: 开始日期
            end_date: 结束日期
            priority: 优先级

        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())

        task = CalculationTask(
            task_id=task_id,
            ts_codes=ts_codes,
            factor_types=factor_types,
            start_date=start_date,
            end_date=end_date,
            priority=priority
        )

        priority_task = PriorityTask(priority=priority, task=task)
        self.task_queue.put(priority_task)

        logger.info(f"提交计算任务 {task_id}，股票数量: {len(ts_codes)}，"
                   f"因子类型: {[ft.value for ft in factor_types]}")

        return task_id

    def start(self):
        """启动调度器"""
        if self._running:
            logger.warning("调度器已经在运行")
            return

        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

        logger.info("因子计算调度器已启动")

    def stop(self):
        """停止调度器"""
        if not self._running:
            return

        self._running = False

        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)

        self._executor.shutdown(wait=True)

        logger.info("因子计算调度器已停止")

    def _scheduler_loop(self):
        """调度器主循环"""
        while self._running:
            try:
                # 获取任务（超时1秒）
                priority_task = self.task_queue.get(timeout=1)
                task = priority_task.task

                # 检查是否有可用的引擎
                available_engines = [ft for ft in task.factor_types if ft in self.engines]
                if not available_engines:
                    logger.error(f"任务 {task.task_id} 没有可用的引擎")
                    task.status = CalculationStatus.FAILED
                    task.error_message = "没有可用的引擎"
                    self.completed_tasks[task.task_id] = task
                    continue

                # 将任务标记为运行中
                task.status = CalculationStatus.RUNNING
                self.running_tasks[task.task_id] = task
                self._notify_status_change(task)

                # 提交任务到线程池
                future = self._executor.submit(self._execute_task, task)

                # 在后台线程中处理任务完成
                def handle_completion(fut):
                    """方法描述"""
                        completed_task = fut.result()
                        self.running_tasks.pop(completed_task.task_id, None)
                        self.completed_tasks[completed_task.task_id] = completed_task
                        self._notify_status_change(completed_task)
                    except Exception as e:
                        logger.error(f"处理任务完成时出错: {e}")

                future.add_done_callback(handle_completion)

            except Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                logger.error(f"调度器循环出错: {e}")

    def _execute_task(self, task: CalculationTask) -> CalculationTask:
        """
        执行计算任务

        Args:
            task: 计算任务

        Returns:
            完成的任务
        """
        start_time = time.time()

        try:
            total_stocks = len(task.ts_codes)
            completed_stocks = 0

            for ts_code in task.ts_codes:
                for factor_type in task.factor_types:
                    if factor_type not in self.engines:
                        logger.warning(f"未找到{factor_type.value}因子引擎")
                        continue

                    engine = self.engines[factor_type]

                    try:
                        # 计算因子
                        result = engine.calculate_factors(
                            ts_code=ts_code,
                            start_date=task.start_date,
                            end_date=task.end_date
                        )

                        task.add_result(result)

                    except Exception as e:
                        logger.error(f"计算股票 {ts_code} 的{factor_type.value}因子时出错: {e}")

                        # 创建失败结果
                        failed_result = FactorResult(
                            ts_code=ts_code,
                            calculation_date=datetime.now(),
                            factor_type=factor_type,
                            status=CalculationStatus.FAILED,
                            error_message=str(e)
                        )
                        task.add_result(failed_result)

                completed_stocks += 1
                task.update_progress(completed_stocks, total_stocks)

                # 通知进度更新
                self._notify_status_change(task)

            # 任务完成
            task.status = CalculationStatus.SUCCESS
            execution_time = time.time() - start_time

            success_count = task.get_success_count()
            failed_count = task.get_failed_count()

            logger.info(f"任务 {task.task_id} 完成，"
                       f"成功: {success_count}，失败: {failed_count}，"
                       f"耗时: {execution_time:.2f}秒")

        except Exception as e:
            task.status = CalculationStatus.FAILED
            task.error_message = str(e)
            logger.error(f"执行任务 {task.task_id} 时出错: {e}")

        return task

    def _notify_status_change(self, task: CalculationTask):
        """通知任务状态变更"""
        for callback in self._status_callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.error(f"执行状态回调时出错: {e}")

    def get_task_status(self, task_id: str) -> Optional[CalculationTask]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            任务对象，如果不存在返回None
        """
        # 检查运行中的任务
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]

        # 检查已完成的任务
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]

        return None

    def list_running_tasks(self) -> List[CalculationTask]:
        """列出所有运行中的任务"""
        return list(self.running_tasks.values())

    def list_completed_tasks(self) -> List[CalculationTask]:
        """列出所有已完成的任务"""
        return list(self.completed_tasks.values())

    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.task_queue.qsize()

    def clear_completed_tasks(self):
        """清理已完成的任务"""
        cleared_count = len(self.completed_tasks)
        self.completed_tasks.clear()
        logger.info(f"清理了 {cleared_count} 个已完成的任务")


if __name__ == "__main__":
    # 测试代码
    print("因子计算调度器测试")

    scheduler = FactorCalculationScheduler(max_workers=2)

    # 添加状态回调
    def status_callback(task: CalculationTask):
        """方法描述"""

    scheduler.add_status_callback(status_callback)

    # 启动调度器
    scheduler.start()

    # 提交测试任务
    task_id = scheduler.submit_task(
        ts_codes=["000001.SZ", "000002.SZ"],
        factor_types=[FactorType.TECHNICAL],
        priority=1
    )

    print(f"提交任务: {task_id}")
    print(f"队列大小: {scheduler.get_queue_size()}")

    # 等待一段时间
    time.sleep(2)

    # 停止调度器
    scheduler.stop()

    print("因子计算调度器测试完成")
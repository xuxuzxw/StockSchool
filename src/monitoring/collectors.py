#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控数据收集器模块

实现各种系统组件的健康状态监控和数据收集
包括数据库、Redis、Celery、API服务、系统资源等

作者: StockSchool Team
创建时间: 2025-01-02
"""

import asyncio
import logging
import psutil
import time
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import requests
from contextlib import asynccontextmanager

from src.utils.config_loader import config
from src.monitoring.models import SyncTaskStatus

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class CollectorConfig:
    """收集器配置"""
    database_url: str = "postgresql://stockschool:stockschool123@localhost:15432/stockschool"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    collection_interval: float = 30.0
    max_retries: int = 3
    timeout: float = 10.0
    api_base_url: str = "http://localhost:8000"
    celery_broker_url: str = "redis://localhost:6379/0"

@dataclass
class HealthMetric:
    """健康状态指标"""
    component: str
    status: str  # healthy, warning, critical, unknown
    value: Optional[float] = None
    unit: Optional[str] = None
    message: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        return data

class BaseCollector:
    """基础收集器类"""
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def collect(self) -> List[HealthMetric]:
        """收集健康状态数据"""
        raise NotImplementedError
    
    def _create_metric(self, component: str, status: str, **kwargs) -> HealthMetric:
        """创建健康指标"""
        return HealthMetric(
            component=component,
            status=status,
            **kwargs
        )

class SystemHealthCollector(BaseCollector):
    """系统健康状态收集器"""
    
    async def collect(self) -> List[HealthMetric]:
        """收集系统健康状态"""
        metrics = []
        
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._get_cpu_status(cpu_percent)
            metrics.append(self._create_metric(
                component="system.cpu",
                status=cpu_status,
                value=cpu_percent,
                unit="percent",
                message=f"CPU使用率: {cpu_percent:.1f}%"
            ))
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_status = self._get_memory_status(memory_percent)
            metrics.append(self._create_metric(
                component="system.memory",
                status=memory_status,
                value=memory_percent,
                unit="percent",
                message=f"内存使用率: {memory_percent:.1f}%",
                metadata={
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used
                }
            ))
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = self._get_disk_status(disk_percent)
            metrics.append(self._create_metric(
                component="system.disk",
                status=disk_status,
                value=disk_percent,
                unit="percent",
                message=f"磁盘使用率: {disk_percent:.1f}%",
                metadata={
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free
                }
            ))
            
            # 网络连接数
            connections = len(psutil.net_connections())
            metrics.append(self._create_metric(
                component="system.network.connections",
                status="healthy",
                value=connections,
                unit="count",
                message=f"网络连接数: {connections}"
            ))
            
        except Exception as e:
            self.logger.error(f"系统健康状态收集失败: {e}")
            metrics.append(self._create_metric(
                component="system",
                status="critical",
                message=f"系统监控失败: {str(e)}"
            ))
        
        return metrics
    
    def _get_cpu_status(self, cpu_percent: float) -> str:
        """获取CPU状态"""
        cpu_threshold = config.get('monitoring_params', {}).get('alerts', {}).get('cpu_threshold', 80.0)
        if cpu_percent >= cpu_threshold:
            return "critical"
        elif cpu_percent >= cpu_threshold * 0.8:
            return "warning"
        return "healthy"
    
    def _get_memory_status(self, memory_percent: float) -> str:
        """获取内存状态"""
        memory_threshold = config.get('monitoring_params', {}).get('alerts', {}).get('memory_threshold', 85.0)
        if memory_percent >= memory_threshold:
            return "critical"
        elif memory_percent >= memory_threshold * 0.8:
            return "warning"
        return "healthy"
    
    def _get_disk_status(self, disk_percent: float) -> str:
        """获取磁盘状态"""
        if disk_percent >= 90:
            return "critical"
        elif disk_percent >= 80:
            return "warning"
        return "healthy"

class DatabaseHealthCollector(BaseCollector):
    """数据库健康状态收集器"""
    
    async def collect(self) -> List[HealthMetric]:
        """收集数据库健康状态"""
        metrics = []
        
        try:
            engine = create_engine(self.config.database_url)
            
            # 测试连接
            start_time = time.time()
            with engine.connect() as conn:
                # 执行简单查询测试连接
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                
                # 连接响应时间
                response_time = (time.time() - start_time) * 1000
                response_threshold = config.get('monitoring_params', {}).get('alerts', {}).get('response_time_threshold', 1000)
                
                response_status = "healthy"
                if response_time >= response_threshold:
                    response_status = "critical"
                elif response_time >= response_threshold * 0.8:
                    response_status = "warning"
                
                metrics.append(self._create_metric(
                    component="database.connection",
                    status="healthy",
                    message="数据库连接正常"
                ))
                
                metrics.append(self._create_metric(
                    component="database.response_time",
                    status=response_status,
                    value=response_time,
                    unit="ms",
                    message=f"数据库响应时间: {response_time:.1f}ms"
                ))
                
                # 获取数据库统计信息
                try:
                    # 活跃连接数
                    active_connections = conn.execute(text(
                        "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                    )).scalar()
                    
                    metrics.append(self._create_metric(
                        component="database.active_connections",
                        status="healthy",
                        value=active_connections,
                        unit="count",
                        message=f"活跃连接数: {active_connections}"
                    ))
                    
                    # 数据库大小
                    db_size = conn.execute(text(
                        "SELECT pg_size_pretty(pg_database_size(current_database()))"
                    )).scalar()
                    
                    metrics.append(self._create_metric(
                        component="database.size",
                        status="healthy",
                        message=f"数据库大小: {db_size}",
                        metadata={"size_pretty": db_size}
                    ))
                    
                except Exception as e:
                    self.logger.warning(f"获取数据库统计信息失败: {e}")
                
        except SQLAlchemyError as e:
            self.logger.error(f"数据库连接失败: {e}")
            metrics.append(self._create_metric(
                component="database.connection",
                status="critical",
                message=f"数据库连接失败: {str(e)}"
            ))
        except Exception as e:
            self.logger.error(f"数据库健康检查失败: {e}")
            metrics.append(self._create_metric(
                component="database",
                status="critical",
                message=f"数据库监控失败: {str(e)}"
            ))
        
        return metrics

class RedisHealthCollector(BaseCollector):
    """Redis健康状态收集器"""
    
    async def collect(self) -> List[HealthMetric]:
        """收集Redis健康状态"""
        metrics = []
        
        try:
            # 创建Redis连接
            redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                socket_timeout=self.config.timeout,
                decode_responses=True
            )
            
            # 测试连接
            start_time = time.time()
            pong = redis_client.ping()
            response_time = (time.time() - start_time) * 1000
            
            if pong:
                metrics.append(self._create_metric(
                    component="redis.connection",
                    status="healthy",
                    message="Redis连接正常"
                ))
                
                metrics.append(self._create_metric(
                    component="redis.response_time",
                    status="healthy",
                    value=response_time,
                    unit="ms",
                    message=f"Redis响应时间: {response_time:.1f}ms"
                ))
                
                # 获取Redis信息
                info = redis_client.info()
                
                # 内存使用
                used_memory = info.get('used_memory', 0)
                used_memory_human = info.get('used_memory_human', 'N/A')
                
                metrics.append(self._create_metric(
                    component="redis.memory",
                    status="healthy",
                    value=used_memory,
                    unit="bytes",
                    message=f"Redis内存使用: {used_memory_human}",
                    metadata={
                        "used_memory_human": used_memory_human,
                        "used_memory_peak": info.get('used_memory_peak', 0)
                    }
                ))
                
                # 连接数
                connected_clients = info.get('connected_clients', 0)
                metrics.append(self._create_metric(
                    component="redis.clients",
                    status="healthy",
                    value=connected_clients,
                    unit="count",
                    message=f"Redis客户端连接数: {connected_clients}"
                ))
                
            else:
                metrics.append(self._create_metric(
                    component="redis.connection",
                    status="critical",
                    message="Redis ping失败"
                ))
                
        except redis.ConnectionError as e:
            self.logger.error(f"Redis连接失败: {e}")
            metrics.append(self._create_metric(
                component="redis.connection",
                status="critical",
                message=f"Redis连接失败: {str(e)}"
            ))
        except Exception as e:
            self.logger.error(f"Redis健康检查失败: {e}")
            metrics.append(self._create_metric(
                component="redis",
                status="critical",
                message=f"Redis监控失败: {str(e)}"
            ))
        
        return metrics

class APIHealthCollector(BaseCollector):
    """API服务健康状态收集器"""
    
    async def collect(self) -> List[HealthMetric]:
        """收集API服务健康状态"""
        metrics = []
        
        try:
            # 测试API健康检查端点
            health_url = f"{self.config.api_base_url}/health"
            
            start_time = time.time()
            response = requests.get(health_url, timeout=self.config.timeout)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                metrics.append(self._create_metric(
                    component="api.health",
                    status="healthy",
                    message="API服务正常"
                ))
                
                metrics.append(self._create_metric(
                    component="api.response_time",
                    status="healthy",
                    value=response_time,
                    unit="ms",
                    message=f"API响应时间: {response_time:.1f}ms"
                ))
                
                # 解析健康检查响应
                try:
                    health_data = response.json()
                    if isinstance(health_data, dict):
                        for key, value in health_data.items():
                            if key != 'status':
                                metrics.append(self._create_metric(
                                    component=f"api.{key}",
                                    status="healthy",
                                    message=f"API {key}: {value}",
                                    metadata={key: value}
                                ))
                except Exception:
                    pass  # 忽略JSON解析错误
                    
            else:
                metrics.append(self._create_metric(
                    component="api.health",
                    status="critical",
                    message=f"API健康检查失败: HTTP {response.status_code}"
                ))
                
        except requests.RequestException as e:
            self.logger.error(f"API健康检查失败: {e}")
            metrics.append(self._create_metric(
                component="api.health",
                status="critical",
                message=f"API连接失败: {str(e)}"
            ))
        except Exception as e:
            self.logger.error(f"API监控失败: {e}")
            metrics.append(self._create_metric(
                component="api",
                status="critical",
                message=f"API监控失败: {str(e)}"
            ))
        
        return metrics

class CeleryHealthCollector(BaseCollector):
    """Celery健康状态收集器"""
    
    async def collect(self) -> List[HealthMetric]:
        """收集Celery健康状态"""
        metrics = []
        
        try:
            # 这里可以添加Celery监控逻辑
            # 由于当前项目可能没有使用Celery，先返回基础状态
            metrics.append(self._create_metric(
                component="celery.status",
                status="unknown",
                message="Celery监控未实现"
            ))
            
        except Exception as e:
            self.logger.error(f"Celery监控失败: {e}")
            metrics.append(self._create_metric(
                component="celery",
                status="critical",
                message=f"Celery监控失败: {str(e)}"
            ))
        
        return metrics

async def collect_system_health(config: Optional[CollectorConfig] = None) -> Dict[str, List[HealthMetric]]:
    """收集所有系统健康状态"""
    if config is None:
        config = CollectorConfig()
    
    collectors = {
        'system': SystemHealthCollector(config),
        'database': DatabaseHealthCollector(config),
        'redis': RedisHealthCollector(config),
        'api': APIHealthCollector(config),
        'celery': CeleryHealthCollector(config)
    }
    
    results = {}
    
    # 并发收集所有健康状态
    tasks = []
    for name, collector in collectors.items():
        task = asyncio.create_task(collector.collect())
        tasks.append((name, task))
    
    for name, task in tasks:
        try:
            metrics = await task
            results[name] = metrics
        except Exception as e:
            logger.error(f"收集器 {name} 执行失败: {e}")
            results[name] = [HealthMetric(
                component=f"{name}.error",
                status="critical",
                message=f"收集器执行失败: {str(e)}"
            )]
    
    return results

if __name__ == "__main__":
    # 测试收集器
    async def test_collectors():
        config = CollectorConfig()
        results = await collect_system_health(config)
        
        for category, metrics in results.items():
            print(f"\n=== {category.upper()} ===")
            for metric in metrics:
                print(f"  {metric.component}: {metric.status} - {metric.message}")
    
    asyncio.run(test_collectors())
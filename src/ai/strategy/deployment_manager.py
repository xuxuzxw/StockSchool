# -*- coding: utf-8 -*-
"""
AI策略系统部署管理器

实现应用的自动化部署、配置管理、环境管理等功能
"""

import json
import logging
import os
import shutil
import subprocess
import time
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import docker
import paramiko
from fabric import Connection
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Text, DateTime, Boolean, JSON, Float
from sqlalchemy.exc import SQLAlchemyError
import requests
import zipfile
import tarfile

logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """部署配置"""
    config_id: str
    config_name: str
    environment: str  # development, staging, production
    app_name: str
    app_version: str
    deployment_type: str  # docker, kubernetes, traditional
    target_servers: List[str]
    database_config: Dict[str, Any]
    redis_config: Dict[str, Any]
    environment_variables: Dict[str, str]
    resource_limits: Dict[str, Any]
    health_check_config: Dict[str, Any]
    rollback_config: Dict[str, Any]
    is_active: bool = True
    created_at: datetime = None

@dataclass
class DeploymentTask:
    """部署任务"""
    task_id: str
    task_name: str
    config_id: str
    deployment_type: str
    status: str  # pending, running, completed, failed, rolled_back
    start_time: datetime = None
    end_time: datetime = None
    logs: List[str] = None
    error_message: str = None
    deployed_version: str = None
    rollback_version: str = None
    created_by: str = None

@dataclass
class ServerInfo:
    """服务器信息"""
    server_id: str
    server_name: str
    ip_address: str
    ssh_port: int
    ssh_user: str
    ssh_key_path: str
    server_type: str  # web, database, cache, worker
    environment: str
    status: str  # active, inactive, maintenance
    cpu_cores: int
    memory_gb: float
    disk_gb: float
    os_type: str
    docker_installed: bool = False
    kubernetes_node: bool = False
    last_health_check: datetime = None

@dataclass
class ApplicationVersion:
    """应用版本"""
    version_id: str
    app_name: str
    version_number: str
    build_number: str
    git_commit: str
    build_time: datetime
    artifact_path: str
    changelog: str
    is_stable: bool = True
    deployment_count: int = 0

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    server_id: str
    check_time: datetime
    status: str  # healthy, unhealthy, unknown
    response_time: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    service_status: Dict[str, str]
    error_message: str = None

class DeploymentManager:
    """部署管理器
    
    提供应用的自动化部署、配置管理、环境管理等功能
    """
    
    def __init__(self, database_url: str = None, docker_client: docker.DockerClient = None):
        from ...utils.db import get_db_manager
        self.engine = get_db_manager().engine
        self.metadata = MetaData()
        self.docker_client = docker_client or docker.from_env()
        
        # 创建数据库表
        self._create_tables()
        
        # 初始化默认配置
        self._init_default_configs()
    
    def _create_tables(self):
        """创建数据库表"""
        try:
            # 部署配置表
            deployment_configs = Table(
                'deployment_configs', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('config_id', String(100), nullable=False, unique=True),
                Column('config_name', String(200), nullable=False),
                Column('environment', String(50), nullable=False),
                Column('app_name', String(100), nullable=False),
                Column('app_version', String(50), nullable=False),
                Column('deployment_type', String(50), nullable=False),
                Column('target_servers', JSON, nullable=False),
                Column('database_config', JSON, nullable=True),
                Column('redis_config', JSON, nullable=True),
                Column('environment_variables', JSON, nullable=True),
                Column('resource_limits', JSON, nullable=True),
                Column('health_check_config', JSON, nullable=True),
                Column('rollback_config', JSON, nullable=True),
                Column('is_active', Boolean, default=True),
                Column('created_at', DateTime, default=datetime.now),
                Column('updated_at', DateTime, default=datetime.now)
            )
            
            # 部署任务表
            deployment_tasks = Table(
                'deployment_tasks', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('task_id', String(100), nullable=False, unique=True),
                Column('task_name', String(200), nullable=False),
                Column('config_id', String(100), nullable=False),
                Column('deployment_type', String(50), nullable=False),
                Column('status', String(20), nullable=False),
                Column('start_time', DateTime, nullable=True),
                Column('end_time', DateTime, nullable=True),
                Column('logs', JSON, nullable=True),
                Column('error_message', Text, nullable=True),
                Column('deployed_version', String(50), nullable=True),
                Column('rollback_version', String(50), nullable=True),
                Column('created_by', String(100), nullable=True),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            # 服务器信息表
            server_info = Table(
                'server_info', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('server_id', String(100), nullable=False, unique=True),
                Column('server_name', String(200), nullable=False),
                Column('ip_address', String(50), nullable=False),
                Column('ssh_port', Integer, default=22),
                Column('ssh_user', String(50), nullable=False),
                Column('ssh_key_path', String(500), nullable=True),
                Column('server_type', String(50), nullable=False),
                Column('environment', String(50), nullable=False),
                Column('status', String(20), default='active'),
                Column('cpu_cores', Integer, nullable=True),
                Column('memory_gb', Float, nullable=True),
                Column('disk_gb', Float, nullable=True),
                Column('os_type', String(50), nullable=True),
                Column('docker_installed', Boolean, default=False),
                Column('kubernetes_node', Boolean, default=False),
                Column('last_health_check', DateTime, nullable=True),
                Column('created_at', DateTime, default=datetime.now),
                Column('updated_at', DateTime, default=datetime.now)
            )
            
            # 应用版本表
            application_versions = Table(
                'application_versions', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('version_id', String(100), nullable=False, unique=True),
                Column('app_name', String(100), nullable=False),
                Column('version_number', String(50), nullable=False),
                Column('build_number', String(50), nullable=False),
                Column('git_commit', String(100), nullable=True),
                Column('build_time', DateTime, nullable=False),
                Column('artifact_path', String(500), nullable=False),
                Column('changelog', Text, nullable=True),
                Column('is_stable', Boolean, default=True),
                Column('deployment_count', Integer, default=0),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            # 健康检查结果表
            health_check_results = Table(
                'health_check_results', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('server_id', String(100), nullable=False),
                Column('check_time', DateTime, nullable=False),
                Column('status', String(20), nullable=False),
                Column('response_time', Float, nullable=True),
                Column('cpu_usage', Float, nullable=True),
                Column('memory_usage', Float, nullable=True),
                Column('disk_usage', Float, nullable=True),
                Column('service_status', JSON, nullable=True),
                Column('error_message', Text, nullable=True)
            )
            
            # 部署历史表
            deployment_history = Table(
                'deployment_history', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('deployment_id', String(100), nullable=False),
                Column('app_name', String(100), nullable=False),
                Column('version_from', String(50), nullable=True),
                Column('version_to', String(50), nullable=False),
                Column('environment', String(50), nullable=False),
                Column('deployment_type', String(50), nullable=False),
                Column('status', String(20), nullable=False),
                Column('duration_seconds', Integer, nullable=True),
                Column('deployed_by', String(100), nullable=True),
                Column('rollback_reason', Text, nullable=True),
                Column('deployed_at', DateTime, default=datetime.now)
            )
            
            self.metadata.create_all(self.engine)
            logger.info("部署管理器数据库表创建成功")
            
        except Exception as e:
            logger.error(f"创建数据库表失败: {e}")
            raise
    
    def _init_default_configs(self):
        """初始化默认配置"""
        try:
            # 默认部署配置
            default_configs = [
                DeploymentConfig(
                    config_id='dev_config',
                    config_name='开发环境配置',
                    environment='development',
                    app_name='stockschool-ai',
                    app_version='latest',
                    deployment_type='docker',
                    target_servers=['dev-server-1'],
                    database_config={
                        'host': 'localhost',
                        'port': 5432,
                        'database': 'stockschool_dev',
                        'user': 'dev_user',
                        'password': 'dev_password'
                    },
                    redis_config={
                        'host': 'localhost',
                        'port': 6379,
                        'database': 0
                    },
                    environment_variables={
                        'FLASK_ENV': 'development',
                        'DEBUG': 'true',
                        'LOG_LEVEL': 'DEBUG'
                    },
                    resource_limits={
                        'cpu': '1',
                        'memory': '2g'
                    },
                    health_check_config={
                        'endpoint': '/health',
                        'interval': 30,
                        'timeout': 10,
                        'retries': 3
                    },
                    rollback_config={
                        'auto_rollback': True,
                        'rollback_threshold': 0.8,
                        'rollback_timeout': 300
                    }
                ),
                DeploymentConfig(
                    config_id='prod_config',
                    config_name='生产环境配置',
                    environment='production',
                    app_name='stockschool-ai',
                    app_version='stable',
                    deployment_type='kubernetes',
                    target_servers=['prod-cluster'],
                    database_config={
                        'host': 'prod-db.internal',
                        'port': 5432,
                        'database': 'stockschool_prod',
                        'user': 'prod_user',
                        'password': '${DB_PASSWORD}'
                    },
                    redis_config={
                        'host': 'prod-redis.internal',
                        'port': 6379,
                        'database': 0
                    },
                    environment_variables={
                        'FLASK_ENV': 'production',
                        'DEBUG': 'false',
                        'LOG_LEVEL': 'INFO'
                    },
                    resource_limits={
                        'cpu': '4',
                        'memory': '8g',
                        'replicas': 3
                    },
                    health_check_config={
                        'endpoint': '/health',
                        'interval': 15,
                        'timeout': 5,
                        'retries': 5
                    },
                    rollback_config={
                        'auto_rollback': True,
                        'rollback_threshold': 0.95,
                        'rollback_timeout': 600
                    }
                )
            ]
            
            for config in default_configs:
                self.save_deployment_config(config)
            
            # 默认服务器信息
            default_servers = [
                ServerInfo(
                    server_id='dev-server-1',
                    server_name='开发服务器1',
                    ip_address='192.168.1.100',
                    ssh_port=22,
                    ssh_user='deploy',
                    ssh_key_path='/home/deploy/.ssh/id_rsa',
                    server_type='web',
                    environment='development',
                    status='active',
                    cpu_cores=4,
                    memory_gb=8.0,
                    disk_gb=100.0,
                    os_type='Ubuntu 20.04',
                    docker_installed=True
                ),
                ServerInfo(
                    server_id='prod-cluster',
                    server_name='生产集群',
                    ip_address='10.0.1.100',
                    ssh_port=22,
                    ssh_user='k8s-admin',
                    ssh_key_path='/home/k8s-admin/.ssh/id_rsa',
                    server_type='kubernetes',
                    environment='production',
                    status='active',
                    cpu_cores=16,
                    memory_gb=64.0,
                    disk_gb=1000.0,
                    os_type='Ubuntu 20.04',
                    kubernetes_node=True
                )
            ]
            
            for server in default_servers:
                self.save_server_info(server)
            
        except Exception as e:
            logger.error(f"初始化默认配置失败: {e}")
    
    def deploy_application(self, config_id: str, version: str = None, 
                          created_by: str = None) -> str:
        """部署应用
        
        Args:
            config_id: 部署配置ID
            version: 应用版本（可选）
            created_by: 部署人员
            
        Returns:
            部署任务ID
        """
        try:
            # 获取部署配置
            config = self.get_deployment_config(config_id)
            if not config:
                raise ValueError(f"部署配置不存在: {config_id}")
            
            # 使用指定版本或配置中的版本
            deploy_version = version or config.app_version
            
            # 创建部署任务
            task_id = f"deploy_{config_id}_{int(time.time())}"
            task = DeploymentTask(
                task_id=task_id,
                task_name=f"部署 {config.app_name} v{deploy_version} 到 {config.environment}",
                config_id=config_id,
                deployment_type=config.deployment_type,
                status='pending',
                deployed_version=deploy_version,
                created_by=created_by,
                logs=[]
            )
            
            self.save_deployment_task(task)
            
            # 异步执行部署
            self._execute_deployment_async(task_id)
            
            return task_id
            
        except Exception as e:
            logger.error(f"部署应用失败: {e}")
            raise
    
    def rollback_application(self, config_id: str, target_version: str,
                           created_by: str = None) -> str:
        """回滚应用
        
        Args:
            config_id: 部署配置ID
            target_version: 目标版本
            created_by: 操作人员
            
        Returns:
            回滚任务ID
        """
        try:
            # 获取部署配置
            config = self.get_deployment_config(config_id)
            if not config:
                raise ValueError(f"部署配置不存在: {config_id}")
            
            # 创建回滚任务
            task_id = f"rollback_{config_id}_{int(time.time())}"
            task = DeploymentTask(
                task_id=task_id,
                task_name=f"回滚 {config.app_name} 到 v{target_version} 在 {config.environment}",
                config_id=config_id,
                deployment_type=config.deployment_type,
                status='pending',
                rollback_version=target_version,
                created_by=created_by,
                logs=[]
            )
            
            self.save_deployment_task(task)
            
            # 异步执行回滚
            self._execute_rollback_async(task_id)
            
            return task_id
            
        except Exception as e:
            logger.error(f"回滚应用失败: {e}")
            raise
    
    def check_deployment_status(self, task_id: str) -> DeploymentTask:
        """检查部署状态
        
        Args:
            task_id: 部署任务ID
            
        Returns:
            部署任务信息
        """
        try:
            return self.get_deployment_task(task_id)
            
        except Exception as e:
            logger.error(f"检查部署状态失败: {e}")
            raise
    
    def health_check_servers(self, environment: str = None) -> List[HealthCheckResult]:
        """健康检查服务器
        
        Args:
            environment: 环境名称（可选）
            
        Returns:
            健康检查结果列表
        """
        try:
            # 获取服务器列表
            servers = self.get_servers_by_environment(environment)
            
            results = []
            for server in servers:
                try:
                    result = self._perform_health_check(server)
                    results.append(result)
                    
                    # 保存健康检查结果
                    self.save_health_check_result(result)
                    
                except Exception as e:
                    logger.error(f"服务器健康检查失败 {server.server_id}: {e}")
                    error_result = HealthCheckResult(
                        server_id=server.server_id,
                        check_time=datetime.now(),
                        status='unknown',
                        response_time=0,
                        cpu_usage=0,
                        memory_usage=0,
                        disk_usage=0,
                        service_status={},
                        error_message=str(e)
                    )
                    results.append(error_result)
                    self.save_health_check_result(error_result)
            
            return results
            
        except Exception as e:
            logger.error(f"健康检查服务器失败: {e}")
            raise
    
    def build_application(self, app_name: str, version: str, 
                         git_commit: str = None, changelog: str = None) -> str:
        """构建应用
        
        Args:
            app_name: 应用名称
            version: 版本号
            git_commit: Git提交哈希
            changelog: 变更日志
            
        Returns:
            版本ID
        """
        try:
            # 创建版本记录
            version_id = f"{app_name}_{version}_{int(time.time())}"
            build_number = f"build_{int(time.time())}"
            
            app_version = ApplicationVersion(
                version_id=version_id,
                app_name=app_name,
                version_number=version,
                build_number=build_number,
                git_commit=git_commit or 'unknown',
                build_time=datetime.now(),
                artifact_path=f"/artifacts/{app_name}/{version}/{build_number}",
                changelog=changelog or 'No changelog provided',
                is_stable=True
            )
            
            # 执行构建
            self._execute_build(app_version)
            
            # 保存版本信息
            self.save_application_version(app_version)
            
            return version_id
            
        except Exception as e:
            logger.error(f"构建应用失败: {e}")
            raise
    
    def get_deployment_history(self, app_name: str = None, 
                             environment: str = None, 
                             limit: int = 50) -> List[Dict[str, Any]]:
        """获取部署历史
        
        Args:
            app_name: 应用名称（可选）
            environment: 环境名称（可选）
            limit: 返回记录数限制
            
        Returns:
            部署历史列表
        """
        try:
            query_sql = """
            SELECT *
            FROM deployment_history
            WHERE 1=1
            """
            
            params = {}
            
            if app_name:
                query_sql += " AND app_name = :app_name"
                params['app_name'] = app_name
            
            if environment:
                query_sql += " AND environment = :environment"
                params['environment'] = environment
            
            query_sql += " ORDER BY deployed_at DESC LIMIT :limit"
            params['limit'] = limit
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                rows = result.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    'deployment_id': row[1],
                    'app_name': row[2],
                    'version_from': row[3],
                    'version_to': row[4],
                    'environment': row[5],
                    'deployment_type': row[6],
                    'status': row[7],
                    'duration_seconds': row[8],
                    'deployed_by': row[9],
                    'rollback_reason': row[10],
                    'deployed_at': row[11]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"获取部署历史失败: {e}")
            return []
    
    def generate_deployment_report(self, environment: str, 
                                 start_date: datetime = None,
                                 end_date: datetime = None) -> Dict[str, Any]:
        """生成部署报告
        
        Args:
            environment: 环境名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            部署报告
        """
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # 获取部署统计
            deployment_stats = self._get_deployment_statistics(
                environment, start_date, end_date
            )
            
            # 获取健康检查统计
            health_stats = self._get_health_check_statistics(
                environment, start_date, end_date
            )
            
            # 获取服务器状态
            server_status = self._get_server_status_summary(environment)
            
            report = {
                'environment': environment,
                'report_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'deployment_statistics': deployment_stats,
                'health_statistics': health_stats,
                'server_status': server_status,
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成部署报告失败: {e}")
            raise
    
    # 保存和获取方法
    def save_deployment_config(self, config: DeploymentConfig):
        """保存部署配置"""
        try:
            # 检查是否已存在
            existing = self._get_deployment_config_exists(config.config_id)
            if existing:
                # 更新现有配置
                update_sql = """
                UPDATE deployment_configs
                SET config_name = :config_name, environment = :environment,
                    app_name = :app_name, app_version = :app_version,
                    deployment_type = :deployment_type, target_servers = :target_servers,
                    database_config = :database_config, redis_config = :redis_config,
                    environment_variables = :environment_variables,
                    resource_limits = :resource_limits,
                    health_check_config = :health_check_config,
                    rollback_config = :rollback_config, is_active = :is_active,
                    updated_at = :updated_at
                WHERE config_id = :config_id
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(update_sql), {
                        'config_id': config.config_id,
                        'config_name': config.config_name,
                        'environment': config.environment,
                        'app_name': config.app_name,
                        'app_version': config.app_version,
                        'deployment_type': config.deployment_type,
                        'target_servers': json.dumps(config.target_servers),
                        'database_config': json.dumps(config.database_config),
                        'redis_config': json.dumps(config.redis_config),
                        'environment_variables': json.dumps(config.environment_variables),
                        'resource_limits': json.dumps(config.resource_limits),
                        'health_check_config': json.dumps(config.health_check_config),
                        'rollback_config': json.dumps(config.rollback_config),
                        'is_active': config.is_active,
                        'updated_at': datetime.now()
                    })
                    conn.commit()
            else:
                # 插入新配置
                insert_sql = """
                INSERT INTO deployment_configs (
                    config_id, config_name, environment, app_name, app_version,
                    deployment_type, target_servers, database_config, redis_config,
                    environment_variables, resource_limits, health_check_config,
                    rollback_config, is_active
                ) VALUES (
                    :config_id, :config_name, :environment, :app_name, :app_version,
                    :deployment_type, :target_servers, :database_config, :redis_config,
                    :environment_variables, :resource_limits, :health_check_config,
                    :rollback_config, :is_active
                )
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(insert_sql), {
                        'config_id': config.config_id,
                        'config_name': config.config_name,
                        'environment': config.environment,
                        'app_name': config.app_name,
                        'app_version': config.app_version,
                        'deployment_type': config.deployment_type,
                        'target_servers': json.dumps(config.target_servers),
                        'database_config': json.dumps(config.database_config),
                        'redis_config': json.dumps(config.redis_config),
                        'environment_variables': json.dumps(config.environment_variables),
                        'resource_limits': json.dumps(config.resource_limits),
                        'health_check_config': json.dumps(config.health_check_config),
                        'rollback_config': json.dumps(config.rollback_config),
                        'is_active': config.is_active
                    })
                    conn.commit()
            
        except Exception as e:
            logger.error(f"保存部署配置失败: {e}")
            raise
    
    def save_deployment_task(self, task: DeploymentTask):
        """保存部署任务"""
        try:
            # 检查是否已存在
            existing = self._get_deployment_task_exists(task.task_id)
            if existing:
                # 更新现有任务
                update_sql = """
                UPDATE deployment_tasks
                SET task_name = :task_name, status = :status,
                    start_time = :start_time, end_time = :end_time,
                    logs = :logs, error_message = :error_message,
                    deployed_version = :deployed_version,
                    rollback_version = :rollback_version
                WHERE task_id = :task_id
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(update_sql), {
                        'task_id': task.task_id,
                        'task_name': task.task_name,
                        'status': task.status,
                        'start_time': task.start_time,
                        'end_time': task.end_time,
                        'logs': json.dumps(task.logs) if task.logs else None,
                        'error_message': task.error_message,
                        'deployed_version': task.deployed_version,
                        'rollback_version': task.rollback_version
                    })
                    conn.commit()
            else:
                # 插入新任务
                insert_sql = """
                INSERT INTO deployment_tasks (
                    task_id, task_name, config_id, deployment_type, status,
                    start_time, end_time, logs, error_message,
                    deployed_version, rollback_version, created_by
                ) VALUES (
                    :task_id, :task_name, :config_id, :deployment_type, :status,
                    :start_time, :end_time, :logs, :error_message,
                    :deployed_version, :rollback_version, :created_by
                )
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(insert_sql), {
                        'task_id': task.task_id,
                        'task_name': task.task_name,
                        'config_id': task.config_id,
                        'deployment_type': task.deployment_type,
                        'status': task.status,
                        'start_time': task.start_time,
                        'end_time': task.end_time,
                        'logs': json.dumps(task.logs) if task.logs else None,
                        'error_message': task.error_message,
                        'deployed_version': task.deployed_version,
                        'rollback_version': task.rollback_version,
                        'created_by': task.created_by
                    })
                    conn.commit()
            
        except Exception as e:
            logger.error(f"保存部署任务失败: {e}")
            raise
    
    def save_server_info(self, server: ServerInfo):
        """保存服务器信息"""
        try:
            # 检查是否已存在
            existing = self._get_server_info_exists(server.server_id)
            if existing:
                # 更新现有服务器信息
                update_sql = """
                UPDATE server_info
                SET server_name = :server_name, ip_address = :ip_address,
                    ssh_port = :ssh_port, ssh_user = :ssh_user,
                    ssh_key_path = :ssh_key_path, server_type = :server_type,
                    environment = :environment, status = :status,
                    cpu_cores = :cpu_cores, memory_gb = :memory_gb,
                    disk_gb = :disk_gb, os_type = :os_type,
                    docker_installed = :docker_installed,
                    kubernetes_node = :kubernetes_node,
                    last_health_check = :last_health_check,
                    updated_at = :updated_at
                WHERE server_id = :server_id
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(update_sql), {
                        'server_id': server.server_id,
                        'server_name': server.server_name,
                        'ip_address': server.ip_address,
                        'ssh_port': server.ssh_port,
                        'ssh_user': server.ssh_user,
                        'ssh_key_path': server.ssh_key_path,
                        'server_type': server.server_type,
                        'environment': server.environment,
                        'status': server.status,
                        'cpu_cores': server.cpu_cores,
                        'memory_gb': server.memory_gb,
                        'disk_gb': server.disk_gb,
                        'os_type': server.os_type,
                        'docker_installed': server.docker_installed,
                        'kubernetes_node': server.kubernetes_node,
                        'last_health_check': server.last_health_check,
                        'updated_at': datetime.now()
                    })
                    conn.commit()
            else:
                # 插入新服务器信息
                insert_sql = """
                INSERT INTO server_info (
                    server_id, server_name, ip_address, ssh_port, ssh_user,
                    ssh_key_path, server_type, environment, status,
                    cpu_cores, memory_gb, disk_gb, os_type,
                    docker_installed, kubernetes_node, last_health_check
                ) VALUES (
                    :server_id, :server_name, :ip_address, :ssh_port, :ssh_user,
                    :ssh_key_path, :server_type, :environment, :status,
                    :cpu_cores, :memory_gb, :disk_gb, :os_type,
                    :docker_installed, :kubernetes_node, :last_health_check
                )
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(insert_sql), {
                        'server_id': server.server_id,
                        'server_name': server.server_name,
                        'ip_address': server.ip_address,
                        'ssh_port': server.ssh_port,
                        'ssh_user': server.ssh_user,
                        'ssh_key_path': server.ssh_key_path,
                        'server_type': server.server_type,
                        'environment': server.environment,
                        'status': server.status,
                        'cpu_cores': server.cpu_cores,
                        'memory_gb': server.memory_gb,
                        'disk_gb': server.disk_gb,
                        'os_type': server.os_type,
                        'docker_installed': server.docker_installed,
                        'kubernetes_node': server.kubernetes_node,
                        'last_health_check': server.last_health_check
                    })
                    conn.commit()
            
        except Exception as e:
            logger.error(f"保存服务器信息失败: {e}")
            raise
    
    def save_application_version(self, version: ApplicationVersion):
        """保存应用版本"""
        try:
            insert_sql = """
            INSERT INTO application_versions (
                version_id, app_name, version_number, build_number,
                git_commit, build_time, artifact_path, changelog,
                is_stable, deployment_count
            ) VALUES (
                :version_id, :app_name, :version_number, :build_number,
                :git_commit, :build_time, :artifact_path, :changelog,
                :is_stable, :deployment_count
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'version_id': version.version_id,
                    'app_name': version.app_name,
                    'version_number': version.version_number,
                    'build_number': version.build_number,
                    'git_commit': version.git_commit,
                    'build_time': version.build_time,
                    'artifact_path': version.artifact_path,
                    'changelog': version.changelog,
                    'is_stable': version.is_stable,
                    'deployment_count': version.deployment_count
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"保存应用版本失败: {e}")
            raise
    
    def save_health_check_result(self, result: HealthCheckResult):
        """保存健康检查结果"""
        try:
            insert_sql = """
            INSERT INTO health_check_results (
                server_id, check_time, status, response_time,
                cpu_usage, memory_usage, disk_usage, service_status,
                error_message
            ) VALUES (
                :server_id, :check_time, :status, :response_time,
                :cpu_usage, :memory_usage, :disk_usage, :service_status,
                :error_message
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'server_id': result.server_id,
                    'check_time': result.check_time,
                    'status': result.status,
                    'response_time': result.response_time,
                    'cpu_usage': result.cpu_usage,
                    'memory_usage': result.memory_usage,
                    'disk_usage': result.disk_usage,
                    'service_status': json.dumps(result.service_status) if result.service_status else None,
                    'error_message': result.error_message
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"保存健康检查结果失败: {e}")
            raise
    
    # 获取方法
    def get_deployment_config(self, config_id: str) -> Optional[DeploymentConfig]:
        """获取部署配置"""
        try:
            query_sql = """
            SELECT *
            FROM deployment_configs
            WHERE config_id = :config_id AND is_active = true
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'config_id': config_id})
                row = result.fetchone()
            
            if row:
                return DeploymentConfig(
                    config_id=row[1],
                    config_name=row[2],
                    environment=row[3],
                    app_name=row[4],
                    app_version=row[5],
                    deployment_type=row[6],
                    target_servers=json.loads(row[7]) if row[7] else [],
                    database_config=json.loads(row[8]) if row[8] else {},
                    redis_config=json.loads(row[9]) if row[9] else {},
                    environment_variables=json.loads(row[10]) if row[10] else {},
                    resource_limits=json.loads(row[11]) if row[11] else {},
                    health_check_config=json.loads(row[12]) if row[12] else {},
                    rollback_config=json.loads(row[13]) if row[13] else {},
                    is_active=row[14],
                    created_at=row[15]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取部署配置失败: {e}")
            return None
    
    def get_deployment_task(self, task_id: str) -> Optional[DeploymentTask]:
        """获取部署任务"""
        try:
            query_sql = """
            SELECT *
            FROM deployment_tasks
            WHERE task_id = :task_id
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'task_id': task_id})
                row = result.fetchone()
            
            if row:
                return DeploymentTask(
                    task_id=row[1],
                    task_name=row[2],
                    config_id=row[3],
                    deployment_type=row[4],
                    status=row[5],
                    start_time=row[6],
                    end_time=row[7],
                    logs=json.loads(row[8]) if row[8] else [],
                    error_message=row[9],
                    deployed_version=row[10],
                    rollback_version=row[11],
                    created_by=row[12]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取部署任务失败: {e}")
            return None
    
    def get_servers_by_environment(self, environment: str = None) -> List[ServerInfo]:
        """根据环境获取服务器列表"""
        try:
            query_sql = """
            SELECT *
            FROM server_info
            WHERE status = 'active'
            """
            
            params = {}
            if environment:
                query_sql += " AND environment = :environment"
                params['environment'] = environment
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                rows = result.fetchall()
            
            servers = []
            for row in rows:
                server = ServerInfo(
                    server_id=row[1],
                    server_name=row[2],
                    ip_address=row[3],
                    ssh_port=row[4],
                    ssh_user=row[5],
                    ssh_key_path=row[6],
                    server_type=row[7],
                    environment=row[8],
                    status=row[9],
                    cpu_cores=row[10],
                    memory_gb=row[11],
                    disk_gb=row[12],
                    os_type=row[13],
                    docker_installed=row[14],
                    kubernetes_node=row[15],
                    last_health_check=row[16]
                )
                servers.append(server)
            
            return servers
            
        except Exception as e:
            logger.error(f"获取服务器列表失败: {e}")
            return []
    
    # 辅助方法
    def _execute_deployment_async(self, task_id: str):
        """异步执行部署"""
        import threading
        
        def deploy_worker():
            try:
                self._execute_deployment(task_id)
            except Exception as e:
                logger.error(f"异步部署失败 {task_id}: {e}")
        
        thread = threading.Thread(target=deploy_worker)
        thread.daemon = True
        thread.start()
    
    def _execute_deployment(self, task_id: str):
        """执行部署"""
        try:
            # 获取任务和配置
            task = self.get_deployment_task(task_id)
            config = self.get_deployment_config(task.config_id)
            
            # 更新任务状态
            task.status = 'running'
            task.start_time = datetime.now()
            task.logs = task.logs or []
            task.logs.append(f"开始部署 {config.app_name} v{task.deployed_version}")
            self.save_deployment_task(task)
            
            # 根据部署类型执行不同的部署逻辑
            if config.deployment_type == 'docker':
                self._deploy_with_docker(task, config)
            elif config.deployment_type == 'kubernetes':
                self._deploy_with_kubernetes(task, config)
            elif config.deployment_type == 'traditional':
                self._deploy_traditional(task, config)
            else:
                raise ValueError(f"不支持的部署类型: {config.deployment_type}")
            
            # 部署成功
            task.status = 'completed'
            task.end_time = datetime.now()
            task.logs.append("部署完成")
            self.save_deployment_task(task)
            
            # 记录部署历史
            self._record_deployment_history(task, config, 'completed')
            
        except Exception as e:
            # 部署失败
            task.status = 'failed'
            task.end_time = datetime.now()
            task.error_message = str(e)
            task.logs.append(f"部署失败: {e}")
            self.save_deployment_task(task)
            
            # 记录部署历史
            self._record_deployment_history(task, config, 'failed')
            
            logger.error(f"部署失败 {task_id}: {e}")
    
    def _execute_rollback_async(self, task_id: str):
        """异步执行回滚"""
        import threading
        
        def rollback_worker():
            try:
                self._execute_rollback(task_id)
            except Exception as e:
                logger.error(f"异步回滚失败 {task_id}: {e}")
        
        thread = threading.Thread(target=rollback_worker)
        thread.daemon = True
        thread.start()
    
    def _execute_rollback(self, task_id: str):
        """执行回滚"""
        try:
            # 获取任务和配置
            task = self.get_deployment_task(task_id)
            config = self.get_deployment_config(task.config_id)
            
            # 更新任务状态
            task.status = 'running'
            task.start_time = datetime.now()
            task.logs = task.logs or []
            task.logs.append(f"开始回滚 {config.app_name} 到 v{task.rollback_version}")
            self.save_deployment_task(task)
            
            # 根据部署类型执行不同的回滚逻辑
            if config.deployment_type == 'docker':
                self._rollback_with_docker(task, config)
            elif config.deployment_type == 'kubernetes':
                self._rollback_with_kubernetes(task, config)
            elif config.deployment_type == 'traditional':
                self._rollback_traditional(task, config)
            else:
                raise ValueError(f"不支持的部署类型: {config.deployment_type}")
            
            # 回滚成功
            task.status = 'completed'
            task.end_time = datetime.now()
            task.logs.append("回滚完成")
            self.save_deployment_task(task)
            
            # 记录部署历史
            self._record_deployment_history(task, config, 'rolled_back')
            
        except Exception as e:
            # 回滚失败
            task.status = 'failed'
            task.end_time = datetime.now()
            task.error_message = str(e)
            task.logs.append(f"回滚失败: {e}")
            self.save_deployment_task(task)
            
            logger.error(f"回滚失败 {task_id}: {e}")
    
    def _deploy_with_docker(self, task: DeploymentTask, config: DeploymentConfig):
        """使用Docker部署"""
        try:
            task.logs.append("使用Docker部署")
            
            # 构建Docker镜像
            image_name = f"{config.app_name}:{task.deployed_version}"
            task.logs.append(f"构建Docker镜像: {image_name}")
            
            # 模拟构建过程
            time.sleep(2)
            
            # 停止旧容器
            try:
                old_container = self.docker_client.containers.get(config.app_name)
                old_container.stop()
                old_container.remove()
                task.logs.append("停止并删除旧容器")
            except docker.errors.NotFound:
                task.logs.append("没有找到旧容器")
            
            # 启动新容器
            environment = config.environment_variables.copy()
            environment.update({
                'DB_HOST': config.database_config.get('host'),
                'DB_PORT': str(config.database_config.get('port')),
                'DB_NAME': config.database_config.get('database'),
                'REDIS_HOST': config.redis_config.get('host'),
                'REDIS_PORT': str(config.redis_config.get('port'))
            })
            
            # 模拟启动容器
            task.logs.append(f"启动新容器: {config.app_name}")
            time.sleep(1)
            
            # 健康检查
            self._perform_deployment_health_check(task, config)
            
        except Exception as e:
            task.logs.append(f"Docker部署失败: {e}")
            raise
    
    def _deploy_with_kubernetes(self, task: DeploymentTask, config: DeploymentConfig):
        """使用Kubernetes部署"""
        try:
            task.logs.append("使用Kubernetes部署")
            
            # 生成Kubernetes配置
            k8s_config = self._generate_kubernetes_config(config, task.deployed_version)
            task.logs.append("生成Kubernetes配置")
            
            # 应用配置
            task.logs.append("应用Kubernetes配置")
            time.sleep(3)  # 模拟部署时间
            
            # 等待Pod就绪
            task.logs.append("等待Pod就绪")
            time.sleep(2)
            
            # 健康检查
            self._perform_deployment_health_check(task, config)
            
        except Exception as e:
            task.logs.append(f"Kubernetes部署失败: {e}")
            raise
    
    def _deploy_traditional(self, task: DeploymentTask, config: DeploymentConfig):
        """传统部署方式"""
        try:
            task.logs.append("使用传统方式部署")
            
            # 连接到目标服务器
            for server_id in config.target_servers:
                server = self._get_server_by_id(server_id)
                if not server:
                    raise ValueError(f"服务器不存在: {server_id}")
                
                task.logs.append(f"连接到服务器: {server.server_name}")
                
                # 使用SSH连接
                with Connection(
                    host=server.ip_address,
                    user=server.ssh_user,
                    port=server.ssh_port,
                    connect_kwargs={'key_filename': server.ssh_key_path}
                ) as conn:
                    # 停止应用
                    task.logs.append("停止应用服务")
                    conn.run(f"sudo systemctl stop {config.app_name}", warn=True)
                    
                    # 备份当前版本
                    task.logs.append("备份当前版本")
                    conn.run(f"sudo cp -r /opt/{config.app_name} /opt/{config.app_name}.backup")
                    
                    # 部署新版本
                    task.logs.append("部署新版本")
                    # 这里应该上传新的应用文件
                    
                    # 更新配置
                    self._update_server_config(conn, config)
                    
                    # 启动应用
                    task.logs.append("启动应用服务")
                    conn.run(f"sudo systemctl start {config.app_name}")
                    
                    # 检查服务状态
                    result = conn.run(f"sudo systemctl is-active {config.app_name}")
                    if result.stdout.strip() != 'active':
                        raise Exception(f"服务启动失败: {result.stderr}")
            
            # 健康检查
            self._perform_deployment_health_check(task, config)
            
        except Exception as e:
            task.logs.append(f"传统部署失败: {e}")
            raise
    
    def _rollback_with_docker(self, task: DeploymentTask, config: DeploymentConfig):
        """使用Docker回滚"""
        try:
            task.logs.append("使用Docker回滚")
            
            # 停止当前容器
            try:
                current_container = self.docker_client.containers.get(config.app_name)
                current_container.stop()
                current_container.remove()
                task.logs.append("停止当前容器")
            except docker.errors.NotFound:
                task.logs.append("没有找到当前容器")
            
            # 启动回滚版本容器
            image_name = f"{config.app_name}:{task.rollback_version}"
            task.logs.append(f"启动回滚版本容器: {image_name}")
            
            # 模拟启动容器
            time.sleep(1)
            
            # 健康检查
            self._perform_deployment_health_check(task, config)
            
        except Exception as e:
            task.logs.append(f"Docker回滚失败: {e}")
            raise
    
    def _rollback_with_kubernetes(self, task: DeploymentTask, config: DeploymentConfig):
        """使用Kubernetes回滚"""
        try:
            task.logs.append("使用Kubernetes回滚")
            
            # 执行回滚命令
            task.logs.append(f"回滚到版本: {task.rollback_version}")
            time.sleep(2)  # 模拟回滚时间
            
            # 等待Pod就绪
            task.logs.append("等待Pod就绪")
            time.sleep(2)
            
            # 健康检查
            self._perform_deployment_health_check(task, config)
            
        except Exception as e:
            task.logs.append(f"Kubernetes回滚失败: {e}")
            raise
    
    def _rollback_traditional(self, task: DeploymentTask, config: DeploymentConfig):
        """传统方式回滚"""
        try:
            task.logs.append("使用传统方式回滚")
            
            # 连接到目标服务器
            for server_id in config.target_servers:
                server = self._get_server_by_id(server_id)
                if not server:
                    raise ValueError(f"服务器不存在: {server_id}")
                
                task.logs.append(f"连接到服务器: {server.server_name}")
                
                # 使用SSH连接
                with Connection(
                    host=server.ip_address,
                    user=server.ssh_user,
                    port=server.ssh_port,
                    connect_kwargs={'key_filename': server.ssh_key_path}
                ) as conn:
                    # 停止应用
                    task.logs.append("停止应用服务")
                    conn.run(f"sudo systemctl stop {config.app_name}", warn=True)
                    
                    # 恢复备份版本
                    task.logs.append("恢复备份版本")
                    conn.run(f"sudo rm -rf /opt/{config.app_name}")
                    conn.run(f"sudo mv /opt/{config.app_name}.backup /opt/{config.app_name}")
                    
                    # 启动应用
                    task.logs.append("启动应用服务")
                    conn.run(f"sudo systemctl start {config.app_name}")
                    
                    # 检查服务状态
                    result = conn.run(f"sudo systemctl is-active {config.app_name}")
                    if result.stdout.strip() != 'active':
                        raise Exception(f"服务启动失败: {result.stderr}")
            
            # 健康检查
            self._perform_deployment_health_check(task, config)
            
        except Exception as e:
            task.logs.append(f"传统回滚失败: {e}")
            raise
    
    def _perform_health_check(self, server: ServerInfo) -> HealthCheckResult:
        """执行服务器健康检查"""
        try:
            start_time = time.time()
            
            # 基本连接测试
            response = requests.get(f"http://{server.ip_address}/health", timeout=10)
            response_time = time.time() - start_time
            
            # 获取系统资源使用情况
            with Connection(
                host=server.ip_address,
                user=server.ssh_user,
                port=server.ssh_port,
                connect_kwargs={'key_filename': server.ssh_key_path}
            ) as conn:
                # CPU使用率
                cpu_result = conn.run("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1")
                cpu_usage = float(cpu_result.stdout.strip()) if cpu_result.stdout.strip() else 0
                
                # 内存使用率
                mem_result = conn.run("free | grep Mem | awk '{printf \"%.2f\", $3/$2 * 100.0}'")
                memory_usage = float(mem_result.stdout.strip()) if mem_result.stdout.strip() else 0
                
                # 磁盘使用率
                disk_result = conn.run("df -h / | awk 'NR==2{print $5}' | cut -d'%' -f1")
                disk_usage = float(disk_result.stdout.strip()) if disk_result.stdout.strip() else 0
                
                # 服务状态检查
                service_status = {}
                services = ['nginx', 'postgresql', 'redis']
                for service in services:
                    status_result = conn.run(f"systemctl is-active {service}", warn=True)
                    service_status[service] = status_result.stdout.strip()
            
            # 判断健康状态
            status = 'healthy'
            if response.status_code != 200:
                status = 'unhealthy'
            elif cpu_usage > 90 or memory_usage > 90 or disk_usage > 90:
                status = 'unhealthy'
            
            return HealthCheckResult(
                server_id=server.server_id,
                check_time=datetime.now(),
                status=status,
                response_time=response_time,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                service_status=service_status
            )
            
        except Exception as e:
            return HealthCheckResult(
                server_id=server.server_id,
                check_time=datetime.now(),
                status='unknown',
                response_time=0,
                cpu_usage=0,
                memory_usage=0,
                disk_usage=0,
                service_status={},
                error_message=str(e)
            )
    
    def _perform_deployment_health_check(self, task: DeploymentTask, config: DeploymentConfig):
        """执行部署后健康检查"""
        try:
            task.logs.append("执行健康检查")
            
            health_config = config.health_check_config
            endpoint = health_config.get('endpoint', '/health')
            timeout = health_config.get('timeout', 10)
            retries = health_config.get('retries', 3)
            
            for server_id in config.target_servers:
                server = self._get_server_by_id(server_id)
                if not server:
                    continue
                
                # 重试机制
                for attempt in range(retries):
                    try:
                        response = requests.get(
                            f"http://{server.ip_address}{endpoint}",
                            timeout=timeout
                        )
                        
                        if response.status_code == 200:
                            task.logs.append(f"服务器 {server.server_name} 健康检查通过")
                            break
                        else:
                            task.logs.append(f"服务器 {server.server_name} 健康检查失败: HTTP {response.status_code}")
                    
                    except Exception as e:
                        task.logs.append(f"服务器 {server.server_name} 健康检查异常 (尝试 {attempt + 1}/{retries}): {e}")
                        
                        if attempt == retries - 1:
                            raise Exception(f"健康检查失败: {e}")
                        
                        time.sleep(5)  # 等待5秒后重试
            
        except Exception as e:
            task.logs.append(f"健康检查失败: {e}")
            raise
    
    def _execute_build(self, version: ApplicationVersion):
        """执行应用构建"""
        try:
            # 创建构建目录
            build_dir = Path(version.artifact_path)
            build_dir.mkdir(parents=True, exist_ok=True)
            
            # 模拟构建过程
            logger.info(f"开始构建 {version.app_name} v{version.version_number}")
            
            # 这里应该执行实际的构建命令
            # 例如: docker build, npm build, maven package 等
            
            # 创建构建产物
            artifact_file = build_dir / f"{version.app_name}-{version.version_number}.tar.gz"
            
            # 模拟创建tar.gz文件
            with tarfile.open(artifact_file, 'w:gz') as tar:
                # 这里应该添加实际的应用文件
                pass
            
            logger.info(f"构建完成: {artifact_file}")
            
        except Exception as e:
            logger.error(f"构建失败: {e}")
            raise
    
    def _generate_kubernetes_config(self, config: DeploymentConfig, version: str) -> Dict[str, Any]:
        """生成Kubernetes配置"""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': config.app_name,
                'namespace': config.environment
            },
            'spec': {
                'replicas': config.resource_limits.get('replicas', 1),
                'selector': {
                    'matchLabels': {
                        'app': config.app_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': config.app_name,
                            'version': version
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': config.app_name,
                            'image': f"{config.app_name}:{version}",
                            'ports': [{
                                'containerPort': 8000
                            }],
                            'env': [{
                                'name': key,
                                'value': value
                            } for key, value in config.environment_variables.items()],
                            'resources': {
                                'limits': {
                                    'cpu': config.resource_limits.get('cpu', '1'),
                                    'memory': config.resource_limits.get('memory', '1Gi')
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': config.health_check_config.get('endpoint', '/health'),
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            }
                        }]
                    }
                }
            }
        }
    
    def _update_server_config(self, conn: Connection, config: DeploymentConfig):
        """更新服务器配置"""
        try:
            # 更新环境变量
            env_content = '\n'.join([
                f"export {key}={value}"
                for key, value in config.environment_variables.items()
            ])
            
            conn.run(f"echo '{env_content}' > /opt/{config.app_name}/.env")
            
            # 更新数据库配置
            db_config = config.database_config
            db_content = f"""
[database]
host = {db_config.get('host')}
port = {db_config.get('port')}
database = {db_config.get('database')}
user = {db_config.get('user')}
password = {db_config.get('password')}
"""
            
            conn.run(f"echo '{db_content}' > /opt/{config.app_name}/config/database.conf")
            
        except Exception as e:
            logger.error(f"更新服务器配置失败: {e}")
            raise
    
    def _record_deployment_history(self, task: DeploymentTask, config: DeploymentConfig, status: str):
        """记录部署历史"""
        try:
            duration = None
            if task.start_time and task.end_time:
                duration = int((task.end_time - task.start_time).total_seconds())
            
            insert_sql = """
            INSERT INTO deployment_history (
                deployment_id, app_name, version_to, environment,
                deployment_type, status, duration_seconds,
                deployed_by, deployed_at
            ) VALUES (
                :deployment_id, :app_name, :version_to, :environment,
                :deployment_type, :status, :duration_seconds,
                :deployed_by, :deployed_at
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'deployment_id': task.task_id,
                    'app_name': config.app_name,
                    'version_to': task.deployed_version or task.rollback_version,
                    'environment': config.environment,
                    'deployment_type': config.deployment_type,
                    'status': status,
                    'duration_seconds': duration,
                    'deployed_by': task.created_by,
                    'deployed_at': task.start_time or datetime.now()
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"记录部署历史失败: {e}")
    
    def _get_deployment_statistics(self, environment: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """获取部署统计"""
        try:
            query_sql = """
            SELECT 
                COUNT(*) as total_deployments,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_deployments,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_deployments,
                COUNT(CASE WHEN status = 'rolled_back' THEN 1 END) as rollbacks,
                AVG(duration_seconds) as avg_duration
            FROM deployment_history
            WHERE environment = :environment
            AND deployed_at BETWEEN :start_date AND :end_date
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {
                    'environment': environment,
                    'start_date': start_date,
                    'end_date': end_date
                })
                row = result.fetchone()
            
            if row:
                total = row[0] or 0
                success_rate = (row[1] / total * 100) if total > 0 else 0
                
                return {
                    'total_deployments': total,
                    'successful_deployments': row[1] or 0,
                    'failed_deployments': row[2] or 0,
                    'rollbacks': row[3] or 0,
                    'success_rate': round(success_rate, 2),
                    'average_duration_seconds': round(row[4] or 0, 2)
                }
            
            return {
                'total_deployments': 0,
                'successful_deployments': 0,
                'failed_deployments': 0,
                'rollbacks': 0,
                'success_rate': 0,
                'average_duration_seconds': 0
            }
            
        except Exception as e:
            logger.error(f"获取部署统计失败: {e}")
            return {}
    
    def _get_health_check_statistics(self, environment: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """获取健康检查统计"""
        try:
            query_sql = """
            SELECT 
                COUNT(*) as total_checks,
                COUNT(CASE WHEN status = 'healthy' THEN 1 END) as healthy_checks,
                COUNT(CASE WHEN status = 'unhealthy' THEN 1 END) as unhealthy_checks,
                AVG(response_time) as avg_response_time,
                AVG(cpu_usage) as avg_cpu_usage,
                AVG(memory_usage) as avg_memory_usage
            FROM health_check_results hcr
            JOIN server_info si ON hcr.server_id = si.server_id
            WHERE si.environment = :environment
            AND hcr.check_time BETWEEN :start_date AND :end_date
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {
                    'environment': environment,
                    'start_date': start_date,
                    'end_date': end_date
                })
                row = result.fetchone()
            
            if row:
                total = row[0] or 0
                health_rate = (row[1] / total * 100) if total > 0 else 0
                
                return {
                    'total_checks': total,
                    'healthy_checks': row[1] or 0,
                    'unhealthy_checks': row[2] or 0,
                    'health_rate': round(health_rate, 2),
                    'average_response_time': round(row[3] or 0, 3),
                    'average_cpu_usage': round(row[4] or 0, 2),
                    'average_memory_usage': round(row[5] or 0, 2)
                }
            
            return {
                'total_checks': 0,
                'healthy_checks': 0,
                'unhealthy_checks': 0,
                'health_rate': 0,
                'average_response_time': 0,
                'average_cpu_usage': 0,
                'average_memory_usage': 0
            }
            
        except Exception as e:
            logger.error(f"获取健康检查统计失败: {e}")
            return {}
    
    def _get_server_status_summary(self, environment: str) -> Dict[str, Any]:
        """获取服务器状态摘要"""
        try:
            query_sql = """
            SELECT 
                COUNT(*) as total_servers,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active_servers,
                COUNT(CASE WHEN status = 'inactive' THEN 1 END) as inactive_servers,
                COUNT(CASE WHEN status = 'maintenance' THEN 1 END) as maintenance_servers
            FROM server_info
            WHERE environment = :environment
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'environment': environment})
                row = result.fetchone()
            
            if row:
                return {
                    'total_servers': row[0] or 0,
                    'active_servers': row[1] or 0,
                    'inactive_servers': row[2] or 0,
                    'maintenance_servers': row[3] or 0
                }
            
            return {
                'total_servers': 0,
                'active_servers': 0,
                'inactive_servers': 0,
                'maintenance_servers': 0
            }
            
        except Exception as e:
            logger.error(f"获取服务器状态摘要失败: {e}")
            return {}
    
    # 辅助查询方法
    def _get_deployment_config_exists(self, config_id: str) -> bool:
        """检查部署配置是否存在"""
        try:
            query_sql = "SELECT 1 FROM deployment_configs WHERE config_id = :config_id"
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'config_id': config_id})
                return result.fetchone() is not None
        except Exception:
            return False
    
    def _get_deployment_task_exists(self, task_id: str) -> bool:
        """检查部署任务是否存在"""
        try:
            query_sql = "SELECT 1 FROM deployment_tasks WHERE task_id = :task_id"
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'task_id': task_id})
                return result.fetchone() is not None
        except Exception:
            return False
    
    def _get_server_info_exists(self, server_id: str) -> bool:
        """检查服务器信息是否存在"""
        try:
            query_sql = "SELECT 1 FROM server_info WHERE server_id = :server_id"
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'server_id': server_id})
                return result.fetchone() is not None
        except Exception:
            return False
    
    def _get_server_by_id(self, server_id: str) -> Optional[ServerInfo]:
        """根据ID获取服务器信息"""
        try:
            query_sql = "SELECT * FROM server_info WHERE server_id = :server_id"
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'server_id': server_id})
                row = result.fetchone()
            
            if row:
                return ServerInfo(
                    server_id=row[1],
                    server_name=row[2],
                    ip_address=row[3],
                    ssh_port=row[4],
                    ssh_user=row[5],
                    ssh_key_path=row[6],
                    server_type=row[7],
                    environment=row[8],
                    status=row[9],
                    cpu_cores=row[10],
                    memory_gb=row[11],
                    disk_gb=row[12],
                    os_type=row[13],
                    docker_installed=row[14],
                    kubernetes_node=row[15],
                    last_health_check=row[16]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取服务器信息失败: {e}")
            return None
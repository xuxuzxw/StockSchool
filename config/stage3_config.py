# -*- coding: utf-8 -*-
"""
第三阶段配置文件

包含所有第三阶段模块的配置参数
"""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelMonitorConfig:
    """模型监控配置"""
    # 监控间隔（秒）
    monitoring_interval: int = 1800  # 30分钟
    
    # 数据漂移检测阈值
    drift_threshold: float = 0.1
    
    # 性能下降阈值
    performance_threshold: float = 0.05
    
    # 告警配置
    alert_email_enabled: bool = True
    alert_webhook_enabled: bool = False
    alert_webhook_url: str = ""
    
    # 自动重训练配置
    auto_retrain_enabled: bool = True
    retrain_threshold: float = 0.15
    
    # 监控指标保留天数
    metrics_retention_days: int = 30

@dataclass
class SystemOptimizerConfig:
    """系统优化配置"""
    # 优化任务执行间隔（秒）
    optimization_interval: int = 3600  # 1小时
    
    # 缓存清理配置
    cache_cleanup_enabled: bool = True
    cache_memory_threshold: float = 0.8  # 80%
    
    # 数据库优化配置
    db_optimization_enabled: bool = True
    slow_query_threshold: float = 1.0  # 1秒
    
    # 日志归档配置
    log_archival_enabled: bool = True
    log_retention_days: int = 7
    
    # 性能监控配置
    performance_monitoring_enabled: bool = True
    cpu_threshold: float = 0.8  # 80%
    memory_threshold: float = 0.8  # 80%
    disk_threshold: float = 0.9  # 90%
    
    # 告警配置
    alert_enabled: bool = True
    alert_cooldown_minutes: int = 30

@dataclass
class DocumentGeneratorConfig:
    """文档生成配置"""
    # 文档输出目录
    output_directory: str = "docs/generated"
    
    # API文档配置
    api_doc_enabled: bool = True
    api_doc_format: str = "markdown"  # markdown, html, pdf
    
    # 系统文档配置
    system_doc_enabled: bool = True
    include_database_schema: bool = True
    include_api_endpoints: bool = True
    
    # 用户手册配置
    user_manual_enabled: bool = True
    include_screenshots: bool = False
    
    # 部署指南配置
    deployment_guide_enabled: bool = True
    include_docker_config: bool = True
    include_k8s_config: bool = True
    
    # 文档更新配置
    auto_update_enabled: bool = True
    update_interval_hours: int = 24

@dataclass
class TestFrameworkConfig:
    """测试框架配置"""
    # 测试执行配置
    parallel_execution: bool = True
    max_workers: int = 4
    test_timeout_seconds: int = 300  # 5分钟
    
    # 单元测试配置
    unit_test_enabled: bool = True
    unit_test_coverage_threshold: float = 0.8  # 80%
    
    # 集成测试配置
    integration_test_enabled: bool = True
    integration_test_timeout: int = 600  # 10分钟
    
    # 性能测试配置
    performance_test_enabled: bool = True
    load_test_users: int = 100
    load_test_duration: int = 300  # 5分钟
    
    # UI测试配置
    ui_test_enabled: bool = False
    ui_test_browser: str = "chrome"
    ui_test_headless: bool = True
    
    # 测试报告配置
    report_format: str = "html"  # html, json, xml
    report_directory: str = "reports/tests"
    
    # 自动化测试配置
    auto_test_on_deploy: bool = True
    auto_test_schedule: str = "0 2 * * *"  # 每天凌晨2点

@dataclass
class DeploymentManagerConfig:
    """部署管理配置"""
    # 部署配置
    default_environment: str = "development"
    supported_environments: list = None
    
    # Docker配置
    docker_enabled: bool = True
    docker_registry: str = "localhost:5000"
    docker_image_prefix: str = "stockschool"
    
    # Kubernetes配置
    k8s_enabled: bool = False
    k8s_namespace: str = "stockschool"
    k8s_config_path: str = "~/.kube/config"
    
    # 健康检查配置
    health_check_enabled: bool = True
    health_check_timeout: int = 30
    health_check_retries: int = 3
    health_check_interval: int = 10
    
    # 回滚配置
    auto_rollback_enabled: bool = True
    rollback_on_health_check_fail: bool = True
    
    # 部署历史配置
    deployment_history_retention: int = 10
    
    # 通知配置
    notification_enabled: bool = True
    notification_webhook: str = ""
    
    def __post_init__(self):
        if self.supported_environments is None:
            self.supported_environments = ["development", "staging", "production"]

@dataclass
class DatabaseConfig:
    """数据库配置"""
    # 连接配置
    host: str = "localhost"
    port: int = 5432
    database: str = "stockschool"
    username: str = "postgres"
    password: str = "password"
    
    # 连接池配置
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    
    # 查询配置
    query_timeout: int = 30
    slow_query_threshold: float = 1.0
    
    # 备份配置
    backup_enabled: bool = True
    backup_interval_hours: int = 6
    backup_retention_days: int = 7

@dataclass
class RedisConfig:
    """Redis配置"""
    # 连接配置
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: str = ""
    
    # 连接池配置
    max_connections: int = 50
    connection_timeout: int = 5
    
    # 缓存配置
    default_ttl: int = 3600  # 1小时
    max_memory_policy: str = "allkeys-lru"

@dataclass
class LoggingConfig:
    """日志配置"""
    # 日志级别
    level: str = "INFO"
    
    # 日志格式
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 文件日志配置
    file_enabled: bool = True
    file_path: str = "logs/stage3.log"
    file_max_size: str = "10MB"
    file_backup_count: int = 5
    
    # 控制台日志配置
    console_enabled: bool = True
    
    # 日志轮转配置
    rotation_enabled: bool = True
    rotation_when: str = "midnight"
    rotation_interval: int = 1

@dataclass
class SecurityConfig:
    """安全配置"""
    # API安全配置
    api_key_required: bool = True
    api_key_header: str = "X-API-Key"
    
    # JWT配置
    jwt_enabled: bool = True
    jwt_secret_key: str = "your-secret-key-here"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # CORS配置
    cors_enabled: bool = True
    cors_origins: list = None
    
    # 速率限制配置
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1小时
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]

class Stage3Config:
    """第三阶段总配置"""
    
    def __init__(self):
        # 从环境变量加载配置
        self.model_monitor = ModelMonitorget_config()
        self.system_optimizer = SystemOptimizerget_config()
        self.doc_generator = DocumentGeneratorget_config()
        self.test_framework = TestFrameworkget_config()
        self.deployment_manager = DeploymentManagerget_config()
        self.database = Databaseget_config()
        self.redis = Redisget_config()
        self.logging = Loggingget_config()
        self.security = Securityget_config()
        
        # 加载环境变量覆盖
        self._load_from_env()
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # 数据库配置
        self.database.host = os.getenv('DB_HOST', self.database.host)
        self.database.port = int(os.getenv('DB_PORT', self.database.port))
        self.database.database = os.getenv('DB_NAME', self.database.database)
        self.database.username = os.getenv('DB_USER', self.database.username)
        self.database.password = os.getenv('DB_PASSWORD', self.database.password)
        
        # Redis配置
        self.redis.host = os.getenv('REDIS_HOST', self.redis.host)
        self.redis.port = int(os.getenv('REDIS_PORT', self.redis.port))
        self.redis.password = os.getenv('REDIS_PASSWORD', self.redis.password)
        
        # 安全配置
        self.security.jwt_secret_key = os.getenv('JWT_SECRET_KEY', self.security.jwt_secret_key)
        
        # 部署配置
        self.deployment_manager.default_environment = os.getenv('DEPLOY_ENV', self.deployment_manager.default_environment)
        
        # 日志配置
        self.logging.level = os.getenv('LOG_LEVEL', self.logging.level)
    
    def get_database_url(self) -> str:
        """获取数据库连接URL"""
        return f"postgresql://{self.database.username}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.database}"
    
    def get_redis_url(self) -> str:
        """获取Redis连接URL"""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.database}"
        else:
            return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.database}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_monitor': self.model_monitor.__dict__,
            'system_optimizer': self.system_optimizer.__dict__,
            'doc_generator': self.doc_generator.__dict__,
            'test_framework': self.test_framework.__dict__,
            'deployment_manager': self.deployment_manager.__dict__,
            'database': self.database.__dict__,
            'redis': self.redis.__dict__,
            'logging': self.logging.__dict__,
            'security': self.security.__dict__
        }
    
    def validate(self) -> bool:
        """验证配置"""
        try:
            # 验证必要的配置项
            assert self.database.host, "数据库主机不能为空"
            assert self.database.database, "数据库名称不能为空"
            assert self.database.username, "数据库用户名不能为空"
            
            assert self.redis.host, "Redis主机不能为空"
            
            assert self.security.jwt_secret_key, "JWT密钥不能为空"
            
            # 验证数值范围
            assert 0 < self.model_monitor.monitoring_interval <= 86400, "监控间隔必须在1秒到1天之间"
            assert 0 < self.model_monitor.drift_threshold <= 1, "漂移阈值必须在0到1之间"
            
            assert 0 < self.system_optimizer.optimization_interval <= 86400, "优化间隔必须在1秒到1天之间"
            assert 0 < self.system_optimizer.cpu_threshold <= 1, "CPU阈值必须在0到1之间"
            
            assert self.test_framework.max_workers > 0, "最大工作线程数必须大于0"
            assert self.test_framework.test_timeout_seconds > 0, "测试超时时间必须大于0"
            
            return True
            
        except AssertionError as e:
            print(f"配置验证失败: {e}")
            return False
        except Exception as e:
            print(f"配置验证出错: {e}")
            return False

# 全局配置实例
config = Stage3get_config()

# 配置验证
if not config.validate():
    raise ValueError("第三阶段配置验证失败")

# 导出常用配置
MODEL_MONITOR_CONFIG = config.model_monitor
SYSTEM_OPTIMIZER_CONFIG = config.system_optimizer
DOC_GENERATOR_CONFIG = config.doc_generator
TEST_FRAMEWORK_CONFIG = config.test_framework
DEPLOYMENT_MANAGER_CONFIG = config.deployment_manager
DATABASE_CONFIG = config.database
REDIS_CONFIG = config.redis
LOGGING_CONFIG = config.logging
SECURITY_CONFIG = config.security

# 环境配置映射
ENVIRONMENT_CONFIGS = {
    'development': {
        'debug': True,
        'testing': False,
        'log_level': 'DEBUG',
        'api_rate_limit': 1000
    },
    'staging': {
        'debug': False,
        'testing': True,
        'log_level': 'INFO',
        'api_rate_limit': 500
    },
    'production': {
        'debug': False,
        'testing': False,
        'log_level': 'WARNING',
        'api_rate_limit': 100
    }
}

def get_environment_config(env: str = None) -> Dict[str, Any]:
    """获取环境配置"""
    env = env or os.getenv('ENVIRONMENT', 'development')
    return ENVIRONMENT_CONFIGS.get(env, ENVIRONMENT_CONFIGS['development'])
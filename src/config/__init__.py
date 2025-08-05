"""
配置管理模块

提供完整的配置管理功能，包括：
- 配置文件管理
- 多环境支持
- 配置验证
- 热更新
- 配置回滚
"""

from .compatibility import (
    CompatibilityIssue,
    CompatibilityLevel,
    CompatibilityReport,
    CompatibilityRule,
    ConfigCompatibilityChecker,
    check_config_compatibility,
    create_compatibility_checker,
)
from .diagnostics import (
    ConfigDiagnostics,
    DiagnosticCategory,
    DiagnosticIssue,
    DiagnosticLevel,
    DiagnosticReport,
    create_config_diagnostics,
    diagnose_config_file,
)
from .hot_reload import (
    ChangeImpact,
    ChangeImpactAnalysis,
    ConfigImpactRule,
    HotReloadManager,
    create_hot_reload_manager,
)
from .manager import ConfigChange, ConfigEnvironment, ConfigManager, get_config_manager, initialize_config_manager
from .rollback import ConfigRollbackManager, ConfigSnapshot, RollbackPlan, RollbackType, create_rollback_manager
from .templates import (
    create_config_files,
    get_base_config_template,
    get_config_template_by_environment,
    get_development_config_template,
    get_production_config_template,
    get_testing_config_template,
)
from .utils import (
    backup_config,
    cleanup_old_backups,
    create_config_template,
    diff_configs,
    export_config_documentation,
    flatten_config,
    get_config_health_check,
    get_config_size,
    merge_configs,
    migrate_config,
    restore_config,
    substitute_env_vars,
    unflatten_config,
    validate_config_schema,
)
from .validators import ConfigValidator, ValidationResult, ValidationType, create_config_validator, validate_config_file

# 配置验证规则已整合到validators模块中


# 版本信息
__version__ = "1.0.0"

# 导出的主要类和函数
__all__ = [
    # 核心管理器
    "ConfigManager",
    "get_config_manager",
    "initialize_config_manager",
    # 枚举和数据类
    "ConfigEnvironment",
    "ConfigChange",
    "ValidationResult",
    "ValidationType",
    # 验证器
    "ConfigValidator",
    "create_config_validator",
    "validate_config_file",
    # 模板函数
    "get_base_config_template",
    "get_development_config_template",
    "get_testing_config_template",
    "get_production_config_template",
    "get_config_template_by_environment",
    "create_config_files",
    # 工具函数
    "backup_config",
    "restore_config",
    "merge_configs",
    "diff_configs",
    "flatten_config",
    "unflatten_config",
    "substitute_env_vars",
    "validate_config_schema",
    "get_config_size",
    "export_config_documentation",
    "create_config_template",
    "migrate_config",
    "cleanup_old_backups",
    "get_config_health_check",
    # 热更新
    "HotReloadManager",
    "ChangeImpact",
    "ConfigImpactRule",
    "ChangeImpactAnalysis",
    "create_hot_reload_manager",
    # 回滚管理
    "ConfigRollbackManager",
    "ConfigSnapshot",
    "RollbackPlan",
    "RollbackType",
    "create_rollback_manager",
    # 诊断系统
    "ConfigDiagnostics",
    "DiagnosticIssue",
    "DiagnosticReport",
    "DiagnosticLevel",
    "DiagnosticCategory",
    "create_config_diagnostics",
    "diagnose_config_file",
    # 兼容性检查
    "ConfigCompatibilityChecker",
    "CompatibilityRule",
    "CompatibilityIssue",
    "CompatibilityReport",
    "CompatibilityLevel",
    "create_compatibility_checker",
    "check_config_compatibility",
]


def setup_config_system(
    config_dir: str = "config", environment: str = None, enable_hot_reload: bool = True, create_templates: bool = False
) -> ConfigManager:
    """
    设置配置系统

    Args:
        config_dir: 配置文件目录
        environment: 环境名称
        enable_hot_reload: 是否启用热更新
        create_templates: 是否创建模板文件

    Returns:
        ConfigManager: 配置管理器实例
    """
    # 创建模板文件（如果需要）
    if create_templates:
        create_config_files(config_dir)

    # 初始化配置管理器
    config_manager = initialize_config_manager(
        config_dir=config_dir, environment=environment, enable_hot_reload=enable_hot_reload
    )

    # 验证配置
    validator = create_config_validator()
    validation_result = validator.validate_config(config_manager._config)

    if not validation_result.is_valid:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("配置验证发现问题:")
        for error in validation_result.errors:
            logger.error(f"  - {error}")
        for warning in validation_result.warnings:
            logger.warning(f"  - {warning}")

    return config_manager


# 便捷函数，用于快速访问配置
def get_config(key_path: str = None, default=None):
    """
    获取配置值的便捷函数

    Args:
        key_path: 配置键路径，如 'database.host'
        default: 默认值

    Returns:
        配置值或默认值
    """
    manager = get_config_manager()
    return manager.get(key_path, default)


def set_config(key_path: str, value, source: str = "runtime"):
    """
    设置配置值的便捷函数

    Args:
        key_path: 配置键路径
        value: 配置值
        source: 变更来源
    """
    manager = get_config_manager()
    manager.set(key_path, value, source=source)


def has_config(key_path: str) -> bool:
    """
    检查配置项是否存在的便捷函数

    Args:
        key_path: 配置键路径

    Returns:
        是否存在
    """
    manager = get_config_manager()
    return manager.has(key_path)


def reload_config():
    """重新加载配置的便捷函数"""
    manager = get_config_manager()
    manager._reload_config()


def validate_current_config():
    """验证当前配置的便捷函数"""
    manager = get_config_manager()
    validator = create_config_validator()
    return validator.validate_config(manager._config)


# 向后兼容性支持
def get_factor_params() -> dict:
    """获取因子参数配置（向后兼容）"""
    return get_config("factor_params", {})


def get_training_params() -> dict:
    """获取训练参数配置（向后兼容）"""
    return get_config("training_params", {})


def get_monitoring_params() -> dict:
    """获取监控参数配置（向后兼容）"""
    return get_config("monitoring_params", {})


def get_database_params() -> dict:
    """获取数据库参数配置（向后兼容）"""
    return get_config("database_params", {})


def get_api_params() -> dict:
    """获取API参数配置（向后兼容）"""
    return get_config("api_params", {})

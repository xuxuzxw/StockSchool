import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from packaging import version

"""
配置兼容性检查系统

提供配置版本兼容性检查和迁移建议
"""


logger = logging.getLogger(__name__)


class CompatibilityLevel(Enum):
    """兼容性级别"""

    COMPATIBLE = "compatible"  # 完全兼容
    DEPRECATED = "deprecated"  # 已弃用但兼容
    BREAKING = "breaking"  # 破坏性变更
    REMOVED = "removed"  # 已移除


@dataclass
class CompatibilityRule:
    """兼容性规则"""

    config_path: str
    introduced_version: str
    deprecated_version: Optional[str] = None
    removed_version: Optional[str] = None
    replacement_path: Optional[str] = None
    migration_function: Optional[str] = None
    description: str = ""
    breaking_changes: List[str] = field(default_factory=list)


@dataclass
class CompatibilityIssue:
    """兼容性问题"""

    level: CompatibilityLevel
    config_path: str
    current_version: str
    target_version: str
    message: str
    description: str = ""
    migration_steps: List[str] = field(default_factory=list)
    replacement_suggestion: Optional[str] = None
    auto_migration_available: bool = False


@dataclass
class CompatibilityReport:
    """兼容性报告"""

    current_version: str
    target_version: str
    overall_compatibility: CompatibilityLevel
    total_issues: int
    issues_by_level: Dict[str, int]
    issues: List[CompatibilityIssue]
    migration_required: bool
    estimated_migration_effort: str  # "low", "medium", "high"


class ConfigCompatibilityChecker:
    """配置兼容性检查器"""

    def __init__(self):
        """方法描述"""
        self.migration_functions = {}
        self._setup_compatibility_rules()
        self._setup_migration_functions()

    def _setup_compatibility_rules(self):
        """设置兼容性规则"""

        # 数据同步参数变更历史
        self.add_compatibility_rule(
            CompatibilityRule(
                config_path="data_sync_params.sleep_time",
                introduced_version="1.0.0",
                deprecated_version="1.2.0",
                removed_version="2.0.0",
                replacement_path="data_sync_params.sleep_interval",
                migration_function="migrate_sleep_time_to_interval",
                description="sleep_time参数已重命名为sleep_interval",
            )
        )

        self.add_compatibility_rule(
            CompatibilityRule(
                config_path="data_sync_params.thread_count",
                introduced_version="1.0.0",
                deprecated_version="1.3.0",
                removed_version="2.0.0",
                replacement_path="data_sync_params.max_workers",
                migration_function="migrate_thread_count_to_max_workers",
                description="thread_count参数已重命名为max_workers",
            )
        )

        # API参数变更
        self.add_compatibility_rule(
            CompatibilityRule(
                config_path="api_params.enable_cors",
                introduced_version="1.0.0",
                deprecated_version="1.4.0",
                removed_version="2.0.0",
                replacement_path="api_params.cors_origins",
                migration_function="migrate_enable_cors_to_origins",
                description="enable_cors布尔参数已替换为cors_origins列表",
                breaking_changes=["配置格式从布尔值改为列表"],
            )
        )

        # 因子参数结构变更
        self.add_compatibility_rule(
            CompatibilityRule(
                config_path="factor_params.technical_indicators",
                introduced_version="1.0.0",
                deprecated_version="1.5.0",
                removed_version="2.0.0",
                replacement_path="factor_params",
                migration_function="migrate_technical_indicators_structure",
                description="技术指标参数结构已扁平化",
                breaking_changes=["配置结构从嵌套改为扁平化"],
            )
        )

        # 数据库参数变更
        self.add_compatibility_rule(
            CompatibilityRule(
                config_path="database_params.pool_config",
                introduced_version="1.1.0",
                deprecated_version="1.6.0",
                removed_version="2.0.0",
                replacement_path="database_params",
                migration_function="migrate_pool_config_structure",
                description="数据库连接池配置结构已简化",
            )
        )

        # 监控参数新增
        self.add_compatibility_rule(
            CompatibilityRule(
                config_path="monitoring_params.alerts",
                introduced_version="1.3.0",
                description="告警配置在1.3.0版本中新增",
            )
        )

        # 特征工程参数变更
        self.add_compatibility_rule(
            CompatibilityRule(
                config_path="feature_params.gpu_config",
                introduced_version="1.2.0",
                deprecated_version="1.7.0",
                removed_version="2.0.0",
                replacement_path="feature_params.use_cuda",
                migration_function="migrate_gpu_config_to_use_cuda",
                description="GPU配置已简化为use_cuda布尔参数",
            )
        )

        # 新版本功能
        self.add_compatibility_rule(
            CompatibilityRule(
                config_path="data_quality", introduced_version="1.8.0", description="数据质量监控配置在1.8.0版本中新增"
            )
        )

        self.add_compatibility_rule(
            CompatibilityRule(
                config_path="sync_strategy", introduced_version="1.9.0", description="同步策略配置在1.9.0版本中新增"
            )
        )

    def _setup_migration_functions(self):
        """设置迁移函数"""
        self.migration_functions = {
            "migrate_sleep_time_to_interval": self._migrate_sleep_time_to_interval,
            "migrate_thread_count_to_max_workers": self._migrate_thread_count_to_max_workers,
            "migrate_enable_cors_to_origins": self._migrate_enable_cors_to_origins,
            "migrate_technical_indicators_structure": self._migrate_technical_indicators_structure,
            "migrate_pool_config_structure": self._migrate_pool_config_structure,
            "migrate_gpu_config_to_use_cuda": self._migrate_gpu_config_to_use_cuda,
        }

    def add_compatibility_rule(self, rule: CompatibilityRule):
        """添加兼容性规则"""
        self.compatibility_rules.append(rule)

    def check_compatibility(
        self, config: Dict[str, Any], current_version: str, target_version: str
    ) -> CompatibilityReport:
        """检查配置兼容性"""

        issues = []
        flat_config = self._flatten_config(config)

        # 检查每个配置项
        for config_path, value in flat_config.items():
            issue = self._check_config_path_compatibility(config_path, value, current_version, target_version)
            if issue:
                issues.append(issue)

        # 检查缺失的新配置项
        missing_issues = self._check_missing_new_configs(flat_config, current_version, target_version)
        issues.extend(missing_issues)

        # 生成报告
        return self._generate_compatibility_report(issues, current_version, target_version)

    def _check_config_path_compatibility(
        self, config_path: str, value: Any, current_version: str, target_version: str
    ) -> Optional[CompatibilityIssue]:
        """检查单个配置项的兼容性"""

        # 查找匹配的规则
        matching_rule = None
        for rule in self.compatibility_rules:
            if self._path_matches(config_path, rule.config_path):
                matching_rule = rule
                break

        if not matching_rule:
            return None

        # 检查版本兼容性
        current_ver = version.parse(current_version)
        target_ver = version.parse(target_version)
        introduced_ver = version.parse(matching_rule.introduced_version)

        # 检查是否在目标版本中被移除
        if matching_rule.removed_version and target_ver >= version.parse(matching_rule.removed_version):
            return CompatibilityIssue(
                level=CompatibilityLevel.REMOVED,
                config_path=config_path,
                current_version=current_version,
                target_version=target_version,
                message=f"配置项 {config_path} 在版本 {target_version} 中已被移除",
                description=matching_rule.description,
                replacement_suggestion=matching_rule.replacement_path,
                migration_steps=self._generate_migration_steps(matching_rule),
                auto_migration_available=matching_rule.migration_function is not None,
            )

        # 检查是否在目标版本中被弃用
        if matching_rule.deprecated_version and target_ver >= version.parse(matching_rule.deprecated_version):
            return CompatibilityIssue(
                level=CompatibilityLevel.DEPRECATED,
                config_path=config_path,
                current_version=current_version,
                target_version=target_version,
                message=f"配置项 {config_path} 在版本 {target_version} 中已被弃用",
                description=matching_rule.description,
                replacement_suggestion=matching_rule.replacement_path,
                migration_steps=self._generate_migration_steps(matching_rule),
                auto_migration_available=matching_rule.migration_function is not None,
            )

        # 检查破坏性变更
        if matching_rule.breaking_changes:
            return CompatibilityIssue(
                level=CompatibilityLevel.BREAKING,
                config_path=config_path,
                current_version=current_version,
                target_version=target_version,
                message=f"配置项 {config_path} 存在破坏性变更",
                description=matching_rule.description,
                migration_steps=matching_rule.breaking_changes,
                auto_migration_available=matching_rule.migration_function is not None,
            )

        return None

    def _check_missing_new_configs(
        self, flat_config: Dict[str, Any], current_version: str, target_version: str
    ) -> List[CompatibilityIssue]:
        """检查缺失的新配置项"""
        issues = []
        current_ver = version.parse(current_version)
        target_ver = version.parse(target_version)

        for rule in self.compatibility_rules:
            introduced_ver = version.parse(rule.introduced_version)

            # 如果配置项在目标版本中新增，但当前配置中不存在
            if (
                introduced_ver > current_ver
                and introduced_ver <= target_ver
                and not any(self._path_matches(path, rule.config_path) for path in flat_config.keys())
            ):

                issues.append(
                    CompatibilityIssue(
                        level=CompatibilityLevel.COMPATIBLE,
                        config_path=rule.config_path,
                        current_version=current_version,
                        target_version=target_version,
                        message=f"新配置项 {rule.config_path} 在版本 {rule.introduced_version} 中新增",
                        description=rule.description,
                        migration_steps=[f"考虑添加配置项 {rule.config_path}"],
                    )
                )

        return issues

    def _path_matches(self, config_path: str, rule_path: str) -> bool:
        """检查配置路径是否匹配规则"""
        # 支持通配符匹配
        import fnmatch

        return fnmatch.fnmatch(config_path, rule_path)

    def _generate_migration_steps(self, rule: CompatibilityRule) -> List[str]:
        """生成迁移步骤"""
        steps = []

        if rule.replacement_path:
            steps.append(f"将 {rule.config_path} 重命名为 {rule.replacement_path}")

        if rule.migration_function:
            steps.append("可以使用自动迁移功能")

        if rule.breaking_changes:
            steps.extend(rule.breaking_changes)

        return steps

    def _generate_compatibility_report(
        self, issues: List[CompatibilityIssue], current_version: str, target_version: str
    ) -> CompatibilityReport:
        """生成兼容性报告"""

        # 统计问题
        issues_by_level = {level.value: 0 for level in CompatibilityLevel}
        for issue in issues:
            issues_by_level[issue.level.value] += 1

        # 确定整体兼容性级别
        if issues_by_level[CompatibilityLevel.REMOVED.value] > 0:
            overall_compatibility = CompatibilityLevel.REMOVED
        elif issues_by_level[CompatibilityLevel.BREAKING.value] > 0:
            overall_compatibility = CompatibilityLevel.BREAKING
        elif issues_by_level[CompatibilityLevel.DEPRECATED.value] > 0:
            overall_compatibility = CompatibilityLevel.DEPRECATED
        else:
            overall_compatibility = CompatibilityLevel.COMPATIBLE

        # 判断是否需要迁移
        migration_required = (
            issues_by_level[CompatibilityLevel.REMOVED.value] > 0
            or issues_by_level[CompatibilityLevel.BREAKING.value] > 0
        )

        # 估算迁移工作量
        effort_score = (
            issues_by_level[CompatibilityLevel.REMOVED.value] * 3
            + issues_by_level[CompatibilityLevel.BREAKING.value] * 2
            + issues_by_level[CompatibilityLevel.DEPRECATED.value] * 1
        )

        if effort_score == 0:
            estimated_effort = "none"
        elif effort_score <= 3:
            estimated_effort = "low"
        elif effort_score <= 10:
            estimated_effort = "medium"
        else:
            estimated_effort = "high"

        return CompatibilityReport(
            current_version=current_version,
            target_version=target_version,
            overall_compatibility=overall_compatibility,
            total_issues=len(issues),
            issues_by_level=issues_by_level,
            issues=issues,
            migration_required=migration_required,
            estimated_migration_effort=estimated_effort,
        )

    def migrate_config(self, config: Dict[str, Any], current_version: str, target_version: str) -> Dict[str, Any]:
        """自动迁移配置"""

        migrated_config = config.copy()

        # 获取兼容性报告
        report = self.check_compatibility(config, current_version, target_version)

        # 执行自动迁移
        for issue in report.issues:
            if issue.auto_migration_available:
                # 查找对应的规则
                matching_rule = None
                for rule in self.compatibility_rules:
                    if self._path_matches(issue.config_path, rule.config_path):
                        matching_rule = rule
                        break

                if matching_rule and matching_rule.migration_function:
                    migration_func = self.migration_functions.get(matching_rule.migration_function)
                    if migration_func:
                        try:
                            migrated_config = migration_func(migrated_config, matching_rule)
                            logger.info(f"自动迁移配置项: {issue.config_path}")
                        except Exception as e:
                            logger.error(f"自动迁移失败 {issue.config_path}: {e}")

        return migrated_config

    def _flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """扁平化配置"""
        result = {}

        for key, value in config.items():
            new_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                result.update(self._flatten_config(value, new_key))
            else:
                result[new_key] = value

        return result

    # 迁移函数
    def _migrate_sleep_time_to_interval(self, config: Dict[str, Any], rule: CompatibilityRule) -> Dict[str, Any]:
        """迁移sleep_time到sleep_interval"""
        if "data_sync_params" in config and "sleep_time" in config["data_sync_params"]:
            sleep_time = config["data_sync_params"].pop("sleep_time")
            config["data_sync_params"]["sleep_interval"] = sleep_time
        return config

    def _migrate_thread_count_to_max_workers(self, config: Dict[str, Any], rule: CompatibilityRule) -> Dict[str, Any]:
        """迁移thread_count到max_workers"""
        if "data_sync_params" in config and "thread_count" in config["data_sync_params"]:
            thread_count = config["data_sync_params"].pop("thread_count")
            config["data_sync_params"]["max_workers"] = thread_count
        return config

    def _migrate_enable_cors_to_origins(self, config: Dict[str, Any], rule: CompatibilityRule) -> Dict[str, Any]:
        """迁移enable_cors到cors_origins"""
        if "api_params" in config and "enable_cors" in config["api_params"]:
            enable_cors = config["api_params"].pop("enable_cors")
            if enable_cors:
                config["api_params"]["cors_origins"] = ["*"]
            else:
                config["api_params"]["cors_origins"] = []
        return config

    def _migrate_technical_indicators_structure(
        self, config: Dict[str, Any], rule: CompatibilityRule
    ) -> Dict[str, Any]:
        """迁移技术指标结构"""
        if "factor_params" in config and "technical_indicators" in config["factor_params"]:
            technical_indicators = config["factor_params"].pop("technical_indicators")
            # 将嵌套结构扁平化到factor_params
            config["factor_params"].update(technical_indicators)
        return config

    def _migrate_pool_config_structure(self, config: Dict[str, Any], rule: CompatibilityRule) -> Dict[str, Any]:
        """迁移数据库连接池配置结构"""
        if "database_params" in config and "pool_config" in config["database_params"]:
            pool_config = config["database_params"].pop("pool_config")
            # 将pool_config的内容合并到database_params
            config["database_params"].update(pool_config)
        return config

    def _migrate_gpu_config_to_use_cuda(self, config: Dict[str, Any], rule: CompatibilityRule) -> Dict[str, Any]:
        """迁移GPU配置到use_cuda"""
        if "feature_params" in config and "gpu_config" in config["feature_params"]:
            gpu_config = config["feature_params"].pop("gpu_config")
            # 简化为布尔值
            config["feature_params"]["use_cuda"] = gpu_config.get("enabled", False)
        return config


def create_compatibility_checker() -> ConfigCompatibilityChecker:
    """创建兼容性检查器"""
    return ConfigCompatibilityChecker()


def check_config_compatibility(
    config: Dict[str, Any], current_version: str, target_version: str
) -> CompatibilityReport:
    """检查配置兼容性"""
    checker = create_compatibility_checker()
    return checker.check_compatibility(config, current_version, target_version)


if __name__ == "__main__":
    # 测试兼容性检查
    checker = create_compatibility_checker()

    # 测试配置（包含旧版本参数）
    test_config = {
        "data_sync_params": {"sleep_time": 1.0, "thread_count": 4, "batch_size": 1000},  # 旧参数  # 旧参数
        "api_params": {"enable_cors": True, "port": 8000},  # 旧参数
        "factor_params": {
            "technical_indicators": {"rsi": {"window": 14}, "macd": {"fast_period": 12, "slow_period": 26}}  # 旧结构
        },
    }

    # 检查从1.0.0到2.0.0的兼容性
    report = checker.check_compatibility(test_config, "1.0.0", "2.0.0")

    print(f"兼容性检查结果:")
    print(f"整体兼容性: {report.overall_compatibility.value}")
    print(f"总问题数: {report.total_issues}")
    print(f"需要迁移: {report.migration_required}")
    print(f"迁移工作量: {report.estimated_migration_effort}")

    print(f"\n问题详情:")
    for issue in report.issues:
        print(f"- {issue.level.value.upper()}: {issue.message}")
        if issue.migration_steps:
            for step in issue.migration_steps:
                print(f"  迁移步骤: {step}")

    # 测试自动迁移
    print(f"\n执行自动迁移...")
    migrated_config = checker.migrate_config(test_config, "1.0.0", "2.0.0")

    import json

    print("迁移后的配置:")
    print(json.dumps(migrated_config, indent=2, ensure_ascii=False))

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

"""
配置验证系统

提供配置参数的验证功能
"""


logger = logging.getLogger(__name__)


class ValidationType(Enum):
    """验证类型枚举"""
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    ENUM_CHECK = "enum_check"
    REGEX_CHECK = "regex_check"
    CUSTOM_CHECK = "custom_check"
    DEPENDENCY_CHECK = "dependency_check"


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    path: str
    value: Any


class ConfigValidator:
    """配置验证器"""

    def __init__(self):
        """方法描述"""
        self._setup_default_rules()

    def _setup_default_rules(self):
        """设置默认验证规则"""

        # 数据同步参数验证
        self.add_rule("data_sync_params.batch_size", [
            self._required_validator(),
            self._type_validator(int),
            self._range_validator(1, 10000)
        ])

        self.add_rule("data_sync_params.retry_times", [
            self._required_validator(),
            self._type_validator(int),
            self._range_validator(0, 10)
        ])

        self.add_rule("data_sync_params.retry_delay", [
            self._required_validator(),
            self._type_validator((int, float)),
            self._range_validator(0.1, 60)
        ])

        self.add_rule("data_sync_params.max_workers", [
            self._required_validator(),
            self._type_validator(int),
            self._range_validator(1, 50)
        ])

        self.add_rule("data_sync_params.sleep_interval", [
            self._required_validator(),
            self._type_validator((int, float)),
            self._range_validator(0.1, 10)
        ])

        # Tushare配置验证
        self.add_rule("data_sync_params.tushare.enabled", [
            self._required_validator(),
            self._type_validator(bool)
        ])

        self.add_rule("data_sync_params.tushare.api_limit", [
            self._type_validator(int),
            self._range_validator(1, 1000)
        ])

        # Akshare配置验证
        self.add_rule("data_sync_params.akshare.enabled", [
            self._required_validator(),
            self._type_validator(bool)
        ])

        self.add_rule("data_sync_params.akshare.api_limit", [
            self._type_validator(int),
            self._range_validator(1, 1000)
        ])

        # 因子参数验证
        self.add_rule("factor_params.min_data_days", [
            self._required_validator(),
            self._type_validator(int),
            self._range_validator(10, 1000)
        ])

        self.add_rule("factor_params.rsi.window", [
            self._required_validator(),
            self._type_validator(int),
            self._range_validator(2, 100)
        ])

        self.add_rule("factor_params.ma.windows", [
            self._required_validator(),
            self._type_validator(list),
            self._custom_validator(self._validate_ma_windows)
        ])

        self.add_rule("factor_params.macd.fast_period", [
            self._required_validator(),
            self._type_validator(int),
            self._range_validator(1, 50)
        ])

        self.add_rule("factor_params.macd.slow_period", [
            self._required_validator(),
            self._type_validator(int),
            self._range_validator(1, 100)
        ])

        # 数据库参数验证
        self.add_rule("database_params.connection_pool_size", [
            self._required_validator(),
            self._type_validator(int),
            self._range_validator(1, 100)
        ])

        self.add_rule("database_params.max_overflow", [
            self._required_validator(),
            self._type_validator(int),
            self._range_validator(0, 200)
        ])

        self.add_rule("database_params.pool_timeout", [
            self._required_validator(),
            self._type_validator((int, float)),
            self._range_validator(1, 300)
        ])

        # API参数验证
        self.add_rule("api_params.host", [
            self._required_validator(),
            self._type_validator(str),
            self._regex_validator(r'^(\d{1,3}\.){3}\d{1,3}$|^localhost$|^0\.0\.0\.0$')
        ])

        self.add_rule("api_params.port", [
            self._required_validator(),
            self._type_validator(int),
            self._range_validator(1000, 65535)
        ])

        self.add_rule("api_params.workers", [
            self._required_validator(),
            self._type_validator(int),
            self._range_validator(1, 20)
        ])

        self.add_rule("api_params.log_level", [
            self._required_validator(),
            self._type_validator(str),
            self._enum_validator(["debug", "info", "warning", "error", "critical"])
        ])

        # 监控参数验证
        self.add_rule("monitoring_params.collection_interval", [
            self._required_validator(),
            self._type_validator(int),
            self._range_validator(10, 3600)
        ])

        self.add_rule("monitoring_params.alerts.cpu_threshold", [
            self._type_validator((int, float)),
            self._range_validator(0, 100)
        ])

        self.add_rule("monitoring_params.alerts.memory_threshold", [
            self._type_validator((int, float)),
            self._range_validator(0, 100)
        ])

        # 数据质量参数验证
        self.add_rule("data_quality.outlier_detection.threshold", [
            self._type_validator((int, float)),
            self._range_validator(1, 5)
        ])

        self.add_rule("data_quality.anomaly_alert.alert_threshold", [
            self._type_validator((int, float)),
            self._range_validator(0, 1)
        ])

        # 特征工程参数验证
        self.add_rule("feature_params.use_cuda", [
            self._type_validator(bool)
        ])

        self.add_rule("feature_params.shap_batch_size", [
            self._type_validator(int),
            self._range_validator(1, 10000)
        ])

        self.add_rule("feature_params.max_cache_size", [
            self._type_validator(int),
            self._range_validator(100, 1000000)  # MB
        ])

        # 依赖关系验证
        self.add_dependency_rule("factor_params.macd.fast_period", "factor_params.macd.slow_period",
                               self._validate_macd_periods)

    def add_rule(self, path: str, validators: List[Callable]):
        """添加验证规则"""
        self.validation_rules[path] = validators

    def add_dependency_rule(self, path1: str, path2: str, validator: Callable):
        """添加依赖关系验证规则"""
        dependency_key = f"__dependency__{path1}__{path2}"
        self.validation_rules[dependency_key] = [validator]

    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """验证整个配置"""
        all_errors = []
        all_warnings = []

        # 验证单个配置项
        for path, validators in self.validation_rules.items():
            if path.startswith("__dependency__"):
                # 处理依赖关系验证
                parts = path.split("__")
                path1, path2 = parts[2], parts[3]
                value1 = self._get_config_value(config, path1)
                value2 = self._get_config_value(config, path2)

                for validator in validators:
                    try:
                        result = validator(value1, value2, path1, path2)
                        if not result.is_valid:
                            all_errors.extend(result.errors)
                        all_warnings.extend(result.warnings)
                    except Exception as e:
                        all_errors.append(f"依赖验证失败 {path1} <-> {path2}: {e}")
            else:
                # 处理普通验证
                value = self._get_config_value(config, path)

                for validator in validators:
                    try:
                        result = validator(value, path)
                        if not result.is_valid:
                            all_errors.extend(result.errors)
                        all_warnings.extend(result.warnings)
                    except Exception as e:
                        all_errors.append(f"验证失败 {path}: {e}")

        # 执行自定义全局验证
        global_result = self._validate_global_config(config)
        all_errors.extend(global_result.errors)
        all_warnings.extend(global_result.warnings)

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            path="global",
            value=config
        )

    def validate_single_config(self, config: Dict[str, Any], path: str) -> ValidationResult:
        """验证单个配置项"""
        value = self._get_config_value(config, path)
        errors = []
        warnings = []

        if path in self.validation_rules:
            for validator in self.validation_rules[path]:
                try:
                    result = validator(value, path)
                    if not result.is_valid:
                        errors.extend(result.errors)
                    warnings.extend(result.warnings)
                except Exception as e:
                    errors.append(f"验证失败 {path}: {e}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            path=path,
            value=value
        )

    def _get_config_value(self, config: Dict[str, Any], path: str) -> Any:
        """获取配置值"""
        keys = path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def _required_validator(self):
        """必需项验证器"""
        def validator(value: Any, path: str) -> ValidationResult:
            """方法描述"""
            is_valid = value is not None
            errors = [] if is_valid else [f"必需配置项缺失: {path}"]
            return ValidationResult(is_valid, errors, [], path, value)
        return validator

    def _type_validator(self, expected_type: Union[type, tuple]):
        """类型验证器"""
        def validator(value: Any, path: str) -> ValidationResult:
            """方法描述"""
            is_valid = isinstance(value, expected_type)
            errors = []
            if not is_valid:
                if isinstance(expected_type, tuple):
                    type_names = [t.__name__ for t in expected_type]
                    expected_str = " 或 ".join(type_names)
                else:
                    expected_str = expected_type.__name__
                errors.append(f"配置项 {path} 类型错误，期望 {expected_str}，实际 {type(value).__name__}")

            return ValidationResult(is_valid, errors, [], path, value)
        return validator

    def _range_validator(self, min_value: Union[int, float], max_value: Union[int, float]):
        """范围验证器"""
        def validator(value: Any, path: str) -> ValidationResult:
            errors = []
            warnings = []

            if not isinstance(value, (int, float)):
                return ValidationResult(True, [], [], path, value)

            if value < min_value:
                errors.append(f"配置项 {path} 值 {value} 小于最小值 {min_value}")
            elif value > max_value:
                errors.append(f"配置项 {path} 值 {value} 大于最大值 {max_value}")

            # 警告：接近边界值
            range_size = max_value - min_value
            if value <= min_value + range_size * 0.1:
                warnings.append(f"配置项 {path} 值 {value} 接近最小值")
            elif value >= max_value - range_size * 0.1:
                warnings.append(f"配置项 {path} 值 {value} 接近最大值")

            return ValidationResult(len(errors) == 0, errors, warnings, path, value)
        return validator

    def _enum_validator(self, allowed_values: List[Any]):
        """枚举值验证器"""
        def validator(value: Any, path: str) -> ValidationResult:
            is_valid = value in allowed_values
            errors = []
            if not is_valid:
                errors.append(f"配置项 {path} 值 {value} 不在允许值列表中: {allowed_values}")

            return ValidationResult(is_valid, errors, [], path, value)
        return validator

    def _regex_validator(self, pattern: str):
        """正则表达式验证器"""
        def validator(value: Any, path: str) -> ValidationResult:
            if not isinstance(value, str):
                return ValidationResult(True, [], [], path, value)

            is_valid = bool(re.match(pattern, value))
            errors = []
            if not is_valid:
                errors.append(f"配置项 {path} 值 {value} 不匹配正则表达式 {pattern}")

            return ValidationResult(is_valid, errors, [], path, value)
        return validator

    def _custom_validator(self, validator_func: Callable):
        """自定义验证器"""
        def validator(value: Any, path: str) -> ValidationResult:
            try:
                return validator_func(value, path)
            except Exception as e:
                return ValidationResult(False, [f"自定义验证失败 {path}: {e}"], [], path, value)
        return validator

    def _validate_ma_windows(self, value: Any, path: str) -> ValidationResult:
        """验证移动平均线窗口期"""
        errors = []
        warnings = []

        if not isinstance(value, list):
            return ValidationResult(True, [], [], path, value)

        if len(value) == 0:
            errors.append(f"配置项 {path} 不能为空列表")

        for i, window in enumerate(value):
            if not isinstance(window, int):
                errors.append(f"配置项 {path}[{i}] 必须为整数")
            elif window < 1:
                errors.append(f"配置项 {path}[{i}] 必须大于0")
            elif window > 250:
                warnings.append(f"配置项 {path}[{i}] 值 {window} 可能过大")

        # 检查重复值
        if len(value) != len(set(value)):
            warnings.append(f"配置项 {path} 包含重复值")

        # 检查排序
        if value != sorted(value):
            warnings.append(f"配置项 {path} 建议按升序排列")

        return ValidationResult(len(errors) == 0, errors, warnings, path, value)

    def _validate_macd_periods(self, fast_period: Any, slow_period: Any, path1: str, path2: str) -> ValidationResult:
        """验证MACD周期参数"""
        errors = []
        warnings = []

        if fast_period is None or slow_period is None:
            return ValidationResult(True, [], [], f"{path1},{path2}", (fast_period, slow_period))

        if not isinstance(fast_period, int) or not isinstance(slow_period, int):
            return ValidationResult(True, [], [], f"{path1},{path2}", (fast_period, slow_period))

        if fast_period >= slow_period:
            errors.append(f"MACD快周期 {fast_period} 必须小于慢周期 {slow_period}")

        if slow_period - fast_period < 5:
            warnings.append(f"MACD快慢周期差值 {slow_period - fast_period} 可能过小")

        return ValidationResult(len(errors) == 0, errors, warnings, f"{path1},{path2}", (fast_period, slow_period))

    def _validate_global_config(self, config: Dict[str, Any]) -> ValidationResult:
        """全局配置验证"""
        errors = []
        warnings = []

        # 检查环境变量依赖
        required_env_vars = []

        # 检查Tushare配置
        if self._get_config_value(config, "data_sync_params.tushare.enabled"):
            if not os.getenv("TUSHARE_TOKEN"):
                required_env_vars.append("TUSHARE_TOKEN")

        # 检查数据库配置
        if not os.getenv("POSTGRES_PASSWORD"):
            required_env_vars.append("POSTGRES_PASSWORD")

        for env_var in required_env_vars:
            errors.append(f"缺少必需的环境变量: {env_var}")

        # 检查配置一致性
        api_workers = self._get_config_value(config, "api_params.workers")
        db_pool_size = self._get_config_value(config, "database_params.connection_pool_size")

        if api_workers and db_pool_size and api_workers > db_pool_size:
            warnings.append(f"API工作进程数 {api_workers} 大于数据库连接池大小 {db_pool_size}，可能导致连接不足")

        # 检查CUDA配置
        use_cuda = self._get_config_value(config, "feature_params.use_cuda")
        if use_cuda:
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings.append("配置启用CUDA但系统不支持，将自动回退到CPU")
            except ImportError:
                warnings.append("配置启用CUDA但未安装PyTorch")

        return ValidationResult(len(errors) == 0, errors, warnings, "global", config)


def create_config_validator() -> ConfigValidator:
    """创建配置验证器"""
    return ConfigValidator()


def validate_config_file(file_path: str) -> ValidationResult:
    """验证配置文件"""
    import yaml

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        validator = create_config_validator()
        return validator.validate_config(config)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            errors=[f"读取配置文件失败: {e}"],
            warnings=[],
            path=file_path,
            value=None
        )


if __name__ == "__main__":
    # 测试配置验证
    validator = create_config_validator()

    # 测试配置
    test_config = {
        "data_sync_params": {
            "batch_size": 1000,
            "retry_times": 3,
            "max_workers": 4
        },
        "factor_params": {
            "rsi": {"window": 14},
            "macd": {"fast_period": 12, "slow_period": 26}
        }
    }

    result = validator.validate_config(test_config)
    print(f"验证结果: {'通过' if result.is_valid else '失败'}")
    if result.errors:
        print("错误:")
        for error in result.errors:
            print(f"  - {error}")
    if result.warnings:
        print("警告:")
        for warning in result.warnings:
            print(f"  - {warning}")
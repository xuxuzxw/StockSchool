"""
配置验证器 - 专门负责配置验证逻辑
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """验证规则"""
    path: str
    required: bool = False
    data_type: type = str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    validator: Optional[callable] = None
    description: str = ""


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._setup_default_rules()
    
    def add_rule(self, rule: ValidationRule) -> None:
        """添加验证规则"""
        self.rules.append(rule)
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """验证配置"""
        errors = []
        
        for rule in self.rules:
            try:
                value = self._get_nested_value(config, rule.path)
                error = self._validate_single_rule(rule, value)
                if error:
                    errors.append(error)
            except Exception as e:
                errors.append(f"验证配置项 {rule.path} 时出错: {e}")
        
        return errors
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """获取嵌套配置值"""
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _validate_single_rule(self, rule: ValidationRule, value: Any) -> Optional[str]:
        """验证单个规则"""
        # 检查必需项
        if rule.required and value is None:
            return f"必需配置项缺失: {rule.path}"
        
        if value is None:
            return None
        
        # 检查数据类型
        if not isinstance(value, rule.data_type):
            return f"配置项 {rule.path} 类型错误，期望 {rule.data_type.__name__}，实际 {type(value).__name__}"
        
        # 检查数值范围
        if rule.min_value is not None and isinstance(value, (int, float)) and value < rule.min_value:
            return f"配置项 {rule.path} 值 {value} 小于最小值 {rule.min_value}"
        
        if rule.max_value is not None and isinstance(value, (int, float)) and value > rule.max_value:
            return f"配置项 {rule.path} 值 {value} 大于最大值 {rule.max_value}"
        
        # 检查允许值
        if rule.allowed_values is not None and value not in rule.allowed_values:
            return f"配置项 {rule.path} 值 {value} 不在允许值列表中: {rule.allowed_values}"
        
        # 自定义验证器
        if rule.validator is not None and not rule.validator(value):
            return f"配置项 {rule.path} 自定义验证失败"
        
        return None
    
    def _setup_default_rules(self) -> None:
        """设置默认验证规则"""
        default_rules = [
            ValidationRule("data_sync_params.batch_size", True, int, 1, 10000, description="数据同步批次大小"),
            ValidationRule("data_sync_params.retry_times", True, int, 0, 10, description="重试次数"),
            ValidationRule("data_sync_params.max_workers", True, int, 1, 20, description="最大工作线程数"),
            ValidationRule("factor_params.rsi.window", True, int, 2, 100, description="RSI窗口期"),
            ValidationRule("factor_params.ma.windows", True, list, description="移动平均线窗口期列表"),
            ValidationRule("monitoring_params.collection_interval", True, int, 10, 3600, description="监控数据收集间隔"),
            ValidationRule("api_params.port", True, int, 1000, 65535, description="API服务端口"),
            ValidationRule("database_params.connection_pool_size", True, int, 1, 100, description="数据库连接池大小"),
        ]
        
        self.rules.extend(default_rules)
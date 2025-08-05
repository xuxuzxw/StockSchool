import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

"""
配置模式定义和验证
"""


@dataclass
class ConfigSchema:
    """配置模式定义"""

    name: str
    version: str
    properties: Dict[str, Any]
    required: List[str] = None

    def __post_init__(self):
        """方法描述"""

    self.required = []


class SchemaValidator:
    """基于模式的配置验证器"""

    def __init__(self, schema: ConfigSchema):
        """方法描述"""

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """根据模式验证配置"""
        errors = []

        # 检查必需属性
        for required_prop in self.schema.required:
            if required_prop not in config:
                errors.append(f"缺少必需属性: {required_prop}")

        # 验证属性类型和约束
        for prop_name, prop_schema in self.schema.properties.items():
            if prop_name in config:
                prop_errors = self._validate_property(prop_name, config[prop_name], prop_schema)
                errors.extend(prop_errors)

        return errors

    def _validate_property(self, name: str, value: Any, schema: Dict[str, Any]) -> List[str]:
        """验证单个属性"""
        errors = []

        # 类型检查
        expected_type = schema.get("type")
        if expected_type and not self._check_type(value, expected_type):
            errors.append(f"属性 {name} 类型错误，期望 {expected_type}")

        # 范围检查
        if "minimum" in schema and isinstance(value, (int, float)):
            if value < schema["minimum"]:
                errors.append(f"属性 {name} 值 {value} 小于最小值 {schema['minimum']}")

        if "maximum" in schema and isinstance(value, (int, float)):
            if value > schema["maximum"]:
                errors.append(f"属性 {name} 值 {value} 大于最大值 {schema['maximum']}")

        # 枚举值检查
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"属性 {name} 值 {value} 不在允许值列表中: {schema['enum']}")

        return errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """检查值类型"""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)

        return True


# 预定义的配置模式
STOCKSCHOOL_CONFIG_SCHEMA = ConfigSchema(
    name="stockschool",
    version="1.0",
    properties={
        "data_sync_params": {
            "type": "object",
            "properties": {
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 10000},
                "retry_times": {"type": "integer", "minimum": 0, "maximum": 10},
                "max_workers": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["batch_size", "retry_times", "max_workers"],
        },
        "factor_params": {
            "type": "object",
            "properties": {
                "rsi": {"type": "object", "properties": {"window": {"type": "integer", "minimum": 2, "maximum": 100}}},
                "ma": {"type": "object", "properties": {"windows": {"type": "array"}}},
            },
        },
        "api_params": {
            "type": "object",
            "properties": {"port": {"type": "integer", "minimum": 1000, "maximum": 65535}},
        },
    },
    required=["data_sync_params", "factor_params", "api_params"],
)

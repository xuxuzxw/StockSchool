import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

"""
配置工具函数

提供配置相关的实用工具
"""


logger = logging.getLogger(__name__)


def backup_config(config_path: str, backup_dir: str = "config_backups") -> str:
    """备份配置文件"""
    try:
        config_path = Path(config_path)
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{config_path.stem}_{timestamp}{config_path.suffix}"
        backup_path = backup_dir / backup_filename

        shutil.copy2(config_path, backup_path)
        logger.info(f"配置文件已备份: {backup_path}")

        return str(backup_path)

    except Exception as e:
        logger.error(f"备份配置文件失败: {e}")
        raise


def restore_config(backup_path: str, target_path: str) -> bool:
    """恢复配置文件"""
    try:
        shutil.copy2(backup_path, target_path)
        logger.info(f"配置文件已恢复: {target_path}")
        return True

    except Exception as e:
        logger.error(f"恢复配置文件失败: {e}")
        return False


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """合并多个配置字典"""
    result = {}

    def merge_dict(base: Dict, update: Dict):
        """方法描述"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dict(base[key], value)
            else:
                base[key] = value

    for config in configs:
        if config:
            merge_dict(result, config)

    return result


def diff_configs(config1: Dict[str, Any], config2: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
    """比较两个配置的差异"""
    differences = []

    def compare_dict(dict1: Dict, dict2: Dict, current_path: str):
        """方法描述"""

        for key in all_keys:
            key_path = f"{current_path}.{key}" if current_path else key

            if key not in dict1:
                differences.append({
                    "type": "added",
                    "path": key_path,
                    "value": dict2[key]
                })
            elif key not in dict2:
                differences.append({
                    "type": "removed",
                    "path": key_path,
                    "value": dict1[key]
                })
            elif dict1[key] != dict2[key]:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    compare_dict(dict1[key], dict2[key], key_path)
                else:
                    differences.append({
                        "type": "modified",
                        "path": key_path,
                        "old_value": dict1[key],
                        "new_value": dict2[key]
                    })

    compare_dict(config1, config2, path)
    return differences


def flatten_config(config: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """扁平化配置字典"""
    result = {}

    def flatten_dict(d: Dict, prefix: str = ""):
        """方法描述"""
        for key, value in d.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key

            if isinstance(value, dict):
                flatten_dict(value, new_key)
            else:
                result[new_key] = value

    def substitute_value(value):
        """方法描述"""
        if isinstance(value, str):
            # 替换 ${VAR_NAME} 格式的环境变量
            pattern = r'\$\{([^}]+)\}'

            def replace_env_var(match):
                """方法描述"""
                env_var = match.group(1)
                default_value = None

                # 支持默认值语法: ${VAR_NAME:default_value}
                if ':' in env_var:
                    env_var, default_value = env_var.split(':', 1)

                return os.getenv(env_var, default_value or match.group(0))

            return re.sub(pattern, replace_env_var, value)
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        else:
            return value

    flatten_dict(config)
    return result


def unflatten_config(flat_config: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """反扁平化配置字典"""
    result = {}

    for key, value in flat_config.items():
        keys = key.split(separator)
        current = result

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    return result


def substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """替换配置中的环境变量"""
    import re

    def substitute_value(value):
        """方法描述"""
        if isinstance(value, str):
            # 替换 ${VAR_NAME} 格式的环境变量
            pattern = r'\$\{([^}]+)\}'

            def replace_env_var(match):
                """方法描述"""
                env_var = match.group(1)
                default_value = None

                # 支持默认值语法: ${VAR_NAME:default_value}
                if ':' in env_var:
                    env_var, default_value = env_var.split(':', 1)

                return os.getenv(env_var, default_value or match.group(0))

            return re.sub(pattern, replace_env_var, value)
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        else:
            return value

    return substitute_value(config)


def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """根据模式验证配置"""
    errors = []

    def validate_dict(data: Dict, schema_dict: Dict, path: str = ""):
        """方法描述"""
        for key, schema_value in schema_dict.items():
            current_path = f"{path}.{key}" if path else key

            if key not in data:
                if isinstance(schema_value, dict) and schema_value.get("required", False):
                    errors.append(f"必需配置项缺失: {current_path}")
                continue

            data_value = data[key]

            if isinstance(schema_value, dict):
                if "type" in schema_value:
                    expected_type = schema_value["type"]
                    if expected_type == "dict" and not isinstance(data_value, dict):
                        errors.append(f"配置项 {current_path} 应为字典类型")
                    elif expected_type == "list" and not isinstance(data_value, list):
                        errors.append(f"配置项 {current_path} 应为列表类型")
                    elif expected_type == "int" and not isinstance(data_value, int):
                        errors.append(f"配置项 {current_path} 应为整数类型")
                    elif expected_type == "float" and not isinstance(data_value, (int, float)):
                        errors.append(f"配置项 {current_path} 应为数值类型")
                    elif expected_type == "str" and not isinstance(data_value, str):
                        errors.append(f"配置项 {current_path} 应为字符串类型")
                    elif expected_type == "bool" and not isinstance(data_value, bool):
                        errors.append(f"配置项 {current_path} 应为布尔类型")

                if "properties" in schema_value and isinstance(data_value, dict):
                    validate_dict(data_value, schema_value["properties"], current_path)

    validate_dict(config, schema)
    return errors


def get_config_size(config: Dict[str, Any]) -> Dict[str, int]:
    """获取配置大小统计"""
    stats = {
        "total_keys": 0,
        "dict_keys": 0,
        "list_keys": 0,
        "string_keys": 0,
        "number_keys": 0,
        "boolean_keys": 0,
        "null_keys": 0
    }

    def count_items(data):
        """方法描述"""
        if isinstance(data, dict):
            stats["total_keys"] += len(data)
            stats["dict_keys"] += 1
            for value in data.values():
                count_items(value)
        elif isinstance(data, list):
            stats["list_keys"] += 1
            for item in data:
                count_items(item)
        elif isinstance(data, str):
            stats["string_keys"] += 1
        elif isinstance(data, (int, float)):
            stats["number_keys"] += 1
        elif isinstance(data, bool):
            stats["boolean_keys"] += 1
        elif data is None:
            stats["null_keys"] += 1

    count_items(config)
    return stats


def export_config_documentation(config: Dict[str, Any], output_path: str):
    """导出配置文档"""
    doc_lines = ["# 配置文档\n"]

    def document_dict(data: Dict, level: int = 1):
        """方法描述"""
        for key, value in data.items():
            indent = "  " * level
            doc_lines.append(f"{indent}- **{key}**")

            if isinstance(value, dict):
                doc_lines.append(f"{indent}  - 类型: 字典")
                doc_lines.append(f"{indent}  - 子项:")
                document_dict(value, level + 1)
            elif isinstance(value, list):
                doc_lines.append(f"{indent}  - 类型: 列表")
                if value and not isinstance(value[0], (dict, list)):
                    doc_lines.append(f"{indent}  - 示例值: {value[:3]}")
            else:
                doc_lines.append(f"{indent}  - 类型: {type(value).__name__}")
                doc_lines.append(f"{indent}  - 当前值: {value}")

            doc_lines.append("")

    document_dict(config)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(doc_lines))

    logger.info(f"配置文档已导出: {output_path}")


def create_config_template(template_name: str, output_path: str):
    """创建配置模板文件"""
    from .templates import get_config_template_by_environment

    try:
        template_config = get_config_template_by_environment(template_name)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(template_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"配置模板已创建: {output_path}")

    except Exception as e:
        logger.error(f"创建配置模板失败: {e}")
        raise


def migrate_config(old_config_path: str, new_config_path: str, migration_rules: Dict[str, str]):
    """迁移配置文件"""
    try:
        # 读取旧配置
        with open(old_config_path, 'r', encoding='utf-8') as f:
            old_config = yaml.safe_load(f)

        # 应用迁移规则
        new_config = {}
        flat_old = flatten_config(old_config)

        for old_key, new_key in migration_rules.items():
            if old_key in flat_old:
                new_config[new_key] = flat_old[old_key]

        # 反扁平化
        new_config = unflatten_config(new_config)

        # 保存新配置
        with open(new_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"配置迁移完成: {old_config_path} -> {new_config_path}")

    except Exception as e:
        logger.error(f"配置迁移失败: {e}")
        raise


def cleanup_old_backups(backup_dir: str, keep_count: int = 10):
    """清理旧的备份文件"""
    try:
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            return

        # 获取所有备份文件，按修改时间排序
        backup_files = []
        for file_path in backup_path.glob("*.yml"):
            backup_files.append((file_path.stat().st_mtime, file_path))

        backup_files.sort(reverse=True)  # 最新的在前

        # 删除多余的备份文件
        for _, file_path in backup_files[keep_count:]:
            file_path.unlink()
            logger.info(f"删除旧备份文件: {file_path}")

        logger.info(f"备份清理完成，保留 {min(len(backup_files), keep_count)} 个文件")

    except Exception as e:
        logger.error(f"清理备份文件失败: {e}")


def get_config_health_check(config: Dict[str, Any]) -> Dict[str, Any]:
    """配置健康检查"""
    health = {
        "status": "healthy",
        "issues": [],
        "warnings": [],
        "recommendations": []
    }

    # 检查必要的配置项
    required_sections = [
        "data_sync_params",
        "factor_params",
        "database_params",
        "api_params"
    ]

    for section in required_sections:
        if section not in config:
            health["issues"].append(f"缺少必要配置段: {section}")
            health["status"] = "unhealthy"

    # 检查环境变量
    if not os.getenv("TUSHARE_TOKEN"):
        health["warnings"].append("未设置TUSHARE_TOKEN环境变量")

    if not os.getenv("POSTGRES_PASSWORD"):
        health["issues"].append("未设置POSTGRES_PASSWORD环境变量")
        health["status"] = "unhealthy"

    # 性能建议
    if config.get("data_sync_params", {}).get("batch_size", 0) > 5000:
        health["recommendations"].append("批次大小过大可能影响性能，建议调整到1000-2000")

    if config.get("database_params", {}).get("connection_pool_size", 0) < 5:
        health["recommendations"].append("数据库连接池过小可能影响并发性能")

    return health


if __name__ == "__main__":
    # 测试工具函数
    test_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "settings": {
                "pool_size": 10,
                "timeout": 30
            }
        },
        "api": {
            "port": 8000,
            "debug": True
        }
    }

    # 测试扁平化
    flat = flatten_config(test_config)
    print("扁平化配置:", flat)

    # 测试反扁平化
    unflat = unflatten_config(flat)
    print("反扁平化配置:", unflat)

    # 测试配置统计
    stats = get_config_size(test_config)
    print("配置统计:", stats)
"""
配置诊断系统

提供配置错误诊断和修复建议功能
"""

import os
import re
import sys
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DiagnosticLevel(Enum):
    """诊断级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DiagnosticCategory(Enum):
    """诊断类别"""
    SYNTAX = "syntax"                    # 语法错误
    TYPE = "type"                       # 类型错误
    VALUE = "value"                     # 值错误
    DEPENDENCY = "dependency"           # 依赖错误
    COMPATIBILITY = "compatibility"     # 兼容性错误
    PERFORMANCE = "performance"         # 性能问题
    SECURITY = "security"              # 安全问题
    ENVIRONMENT = "environment"         # 环境问题


@dataclass
class DiagnosticIssue:
    """诊断问题"""
    level: DiagnosticLevel
    category: DiagnosticCategory
    path: str
    message: str
    description: str = ""
    suggestions: List[str] = field(default_factory=list)
    auto_fix_available: bool = False
    auto_fix_function: Optional[str] = None
    related_paths: List[str] = field(default_factory=list)
    documentation_url: str = ""


@dataclass
class DiagnosticReport:
    """诊断报告"""
    timestamp: str
    config_path: str
    total_issues: int
    issues_by_level: Dict[str, int]
    issues_by_category: Dict[str, int]
    issues: List[DiagnosticIssue]
    auto_fixable_count: int
    health_score: float  # 0-100


class ConfigDiagnostics:
    """配置诊断器"""
    
    def __init__(self):
        self.diagnostic_rules = []
        self.auto_fix_functions = {}
        self._setup_diagnostic_rules()
        self._setup_auto_fix_functions()
    
    def _setup_diagnostic_rules(self):
        """设置诊断规则"""
        
        # 语法检查规则
        self.add_diagnostic_rule(
            path_pattern="*",
            check_function=self._check_yaml_syntax,
            level=DiagnosticLevel.ERROR,
            category=DiagnosticCategory.SYNTAX
        )
        
        # 类型检查规则
        self.add_diagnostic_rule(
            path_pattern="*.batch_size",
            check_function=self._check_positive_integer,
            level=DiagnosticLevel.ERROR,
            category=DiagnosticCategory.TYPE
        )
        
        self.add_diagnostic_rule(
            path_pattern="*.port",
            check_function=self._check_port_number,
            level=DiagnosticLevel.ERROR,
            category=DiagnosticCategory.VALUE
        )
        
        # 环境依赖检查
        self.add_diagnostic_rule(
            path_pattern="data_sync_params.tushare.enabled",
            check_function=self._check_tushare_token,
            level=DiagnosticLevel.ERROR,
            category=DiagnosticCategory.ENVIRONMENT
        )
        
        # 性能检查规则
        self.add_diagnostic_rule(
            path_pattern="data_sync_params.batch_size",
            check_function=self._check_batch_size_performance,
            level=DiagnosticLevel.WARNING,
            category=DiagnosticCategory.PERFORMANCE
        )
        
        # 兼容性检查规则
        self.add_diagnostic_rule(
            path_pattern="feature_params.use_cuda",
            check_function=self._check_cuda_compatibility,
            level=DiagnosticLevel.WARNING,
            category=DiagnosticCategory.COMPATIBILITY
        )
        
        # 安全检查规则
        self.add_diagnostic_rule(
            path_pattern="api_params.cors_origins",
            check_function=self._check_cors_security,
            level=DiagnosticLevel.WARNING,
            category=DiagnosticCategory.SECURITY
        )
        
        # 依赖关系检查
        self.add_diagnostic_rule(
            path_pattern="factor_params.macd.*",
            check_function=self._check_macd_periods,
            level=DiagnosticLevel.ERROR,
            category=DiagnosticCategory.DEPENDENCY
        )
    
    def _setup_auto_fix_functions(self):
        """设置自动修复函数"""
        self.auto_fix_functions = {
            "fix_batch_size": self._fix_batch_size,
            "fix_port_number": self._fix_port_number,
            "fix_cors_origins": self._fix_cors_origins,
            "fix_macd_periods": self._fix_macd_periods
        }
    
    def add_diagnostic_rule(self, path_pattern: str, check_function: callable,
                          level: DiagnosticLevel, category: DiagnosticCategory):
        """添加诊断规则"""
        self.diagnostic_rules.append({
            "path_pattern": path_pattern,
            "check_function": check_function,
            "level": level,
            "category": category
        })
    
    def diagnose_config(self, config: Dict[str, Any], config_path: str = "") -> DiagnosticReport:
        """诊断配置"""
        from datetime import datetime
        import fnmatch
        
        issues = []
        
        # 扁平化配置以便检查
        flat_config = self._flatten_config(config)
        
        # 应用诊断规则
        for rule in self.diagnostic_rules:
            pattern = rule["path_pattern"]
            check_function = rule["check_function"]
            level = rule["level"]
            category = rule["category"]
            
            # 匹配路径
            matching_paths = []
            if pattern == "*":
                matching_paths = ["__root__"]
            else:
                matching_paths = [path for path in flat_config.keys() 
                                if fnmatch.fnmatch(path, pattern)]
            
            # 执行检查
            for path in matching_paths:
                try:
                    if path == "__root__":
                        result = check_function(config, "")
                    else:
                        value = flat_config[path]
                        result = check_function(value, path)
                    
                    if result:
                        issue = DiagnosticIssue(
                            level=level,
                            category=category,
                            path=path,
                            message=result["message"],
                            description=result.get("description", ""),
                            suggestions=result.get("suggestions", []),
                            auto_fix_available=result.get("auto_fix_available", False),
                            auto_fix_function=result.get("auto_fix_function"),
                            related_paths=result.get("related_paths", []),
                            documentation_url=result.get("documentation_url", "")
                        )
                        issues.append(issue)
                        
                except Exception as e:
                    logger.error(f"诊断规则执行失败 {path}: {e}")
        
        # 生成报告
        return self._generate_report(issues, config_path)
    
    def _generate_report(self, issues: List[DiagnosticIssue], config_path: str) -> DiagnosticReport:
        """生成诊断报告"""
        from datetime import datetime
        
        # 统计问题
        issues_by_level = {level.value: 0 for level in DiagnosticLevel}
        issues_by_category = {category.value: 0 for category in DiagnosticCategory}
        auto_fixable_count = 0
        
        for issue in issues:
            issues_by_level[issue.level.value] += 1
            issues_by_category[issue.category.value] += 1
            if issue.auto_fix_available:
                auto_fixable_count += 1
        
        # 计算健康分数
        health_score = self._calculate_health_score(issues)
        
        return DiagnosticReport(
            timestamp=datetime.now().isoformat(),
            config_path=config_path,
            total_issues=len(issues),
            issues_by_level=issues_by_level,
            issues_by_category=issues_by_category,
            issues=issues,
            auto_fixable_count=auto_fixable_count,
            health_score=health_score
        )
    
    def _calculate_health_score(self, issues: List[DiagnosticIssue]) -> float:
        """计算健康分数"""
        if not issues:
            return 100.0
        
        # 权重分配
        weights = {
            DiagnosticLevel.CRITICAL: 25,
            DiagnosticLevel.ERROR: 15,
            DiagnosticLevel.WARNING: 5,
            DiagnosticLevel.INFO: 1
        }
        
        total_penalty = sum(weights.get(issue.level, 0) for issue in issues)
        max_score = 100
        
        # 计算分数
        score = max(0, max_score - total_penalty)
        return round(score, 1)
    
    def auto_fix_issues(self, config: Dict[str, Any], issues: List[DiagnosticIssue]) -> Dict[str, Any]:
        """自动修复问题"""
        fixed_config = config.copy()
        fixed_count = 0
        
        for issue in issues:
            if issue.auto_fix_available and issue.auto_fix_function:
                try:
                    fix_function = self.auto_fix_functions.get(issue.auto_fix_function)
                    if fix_function:
                        fixed_config = fix_function(fixed_config, issue)
                        fixed_count += 1
                        logger.info(f"自动修复问题: {issue.path}")
                except Exception as e:
                    logger.error(f"自动修复失败 {issue.path}: {e}")
        
        logger.info(f"自动修复了 {fixed_count} 个问题")
        return fixed_config
    
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
    
    # 检查函数
    def _check_yaml_syntax(self, config: Dict[str, Any], path: str) -> Optional[Dict[str, Any]]:
        """检查YAML语法"""
        # 这里主要检查配置结构的合理性
        if not isinstance(config, dict):
            return {
                "message": "配置根节点必须是字典类型",
                "description": "YAML配置文件的根节点应该是键值对结构",
                "suggestions": ["检查YAML文件格式", "确保正确的缩进"]
            }
        return None
    
    def _check_positive_integer(self, value: Any, path: str) -> Optional[Dict[str, Any]]:
        """检查正整数"""
        if not isinstance(value, int) or value <= 0:
            return {
                "message": f"配置项 {path} 必须是正整数",
                "description": "批次大小等参数必须是大于0的整数",
                "suggestions": ["设置为正整数值", "推荐值范围: 100-5000"],
                "auto_fix_available": True,
                "auto_fix_function": "fix_batch_size"
            }
        return None
    
    def _check_port_number(self, value: Any, path: str) -> Optional[Dict[str, Any]]:
        """检查端口号"""
        if not isinstance(value, int) or not (1000 <= value <= 65535):
            return {
                "message": f"配置项 {path} 必须是有效的端口号 (1000-65535)",
                "description": "端口号必须在有效范围内",
                "suggestions": ["使用1000-65535范围内的端口", "避免使用系统保留端口"],
                "auto_fix_available": True,
                "auto_fix_function": "fix_port_number"
            }
        return None
    
    def _check_tushare_token(self, value: Any, path: str) -> Optional[Dict[str, Any]]:
        """检查Tushare令牌"""
        if value is True and not os.getenv("TUSHARE_TOKEN"):
            return {
                "message": "启用Tushare但未设置TUSHARE_TOKEN环境变量",
                "description": "使用Tushare数据源需要有效的API令牌",
                "suggestions": [
                    "设置TUSHARE_TOKEN环境变量",
                    "或将data_sync_params.tushare.enabled设置为false"
                ],
                "documentation_url": "https://tushare.pro/document/1?doc_id=39"
            }
        return None
    
    def _check_batch_size_performance(self, value: Any, path: str) -> Optional[Dict[str, Any]]:
        """检查批次大小性能"""
        if isinstance(value, int):
            if value > 5000:
                return {
                    "message": f"批次大小 {value} 可能过大，影响性能",
                    "description": "过大的批次大小可能导致内存占用过高和API超时",
                    "suggestions": ["建议设置为1000-2000", "根据系统内存调整"],
                    "auto_fix_available": True,
                    "auto_fix_function": "fix_batch_size"
                }
            elif value < 100:
                return {
                    "message": f"批次大小 {value} 可能过小，影响效率",
                    "description": "过小的批次大小会增加API调用次数",
                    "suggestions": ["建议设置为500-1000", "平衡效率和性能"]
                }
        return None
    
    def _check_cuda_compatibility(self, value: Any, path: str) -> Optional[Dict[str, Any]]:
        """检查CUDA兼容性"""
        if value is True:
            try:
                import torch
                if not torch.cuda.is_available():
                    return {
                        "message": "启用CUDA但系统不支持",
                        "description": "系统未检测到可用的CUDA设备",
                        "suggestions": [
                            "安装CUDA驱动程序",
                            "或将use_cuda设置为false",
                            "检查PyTorch CUDA版本"
                        ]
                    }
            except ImportError:
                return {
                    "message": "启用CUDA但未安装PyTorch",
                    "description": "使用CUDA功能需要安装PyTorch",
                    "suggestions": [
                        "安装PyTorch: pip install torch",
                        "或将use_cuda设置为false"
                    ]
                }
        return None
    
    def _check_cors_security(self, value: Any, path: str) -> Optional[Dict[str, Any]]:
        """检查CORS安全性"""
        if isinstance(value, list) and "*" in value:
            return {
                "message": "CORS配置允许所有来源，存在安全风险",
                "description": "允许所有来源的CORS配置在生产环境中不安全",
                "suggestions": [
                    "指定具体的允许来源",
                    "使用域名而不是通配符",
                    "在开发环境中可以保持当前设置"
                ],
                "auto_fix_available": True,
                "auto_fix_function": "fix_cors_origins"
            }
        return None
    
    def _check_macd_periods(self, value: Any, path: str) -> Optional[Dict[str, Any]]:
        """检查MACD周期参数"""
        if path.endswith("fast_period"):
            # 需要检查与slow_period的关系，这里简化处理
            if isinstance(value, int) and value >= 26:  # 假设slow_period默认为26
                return {
                    "message": "MACD快周期不应大于等于慢周期",
                    "description": "快周期必须小于慢周期才能正确计算MACD",
                    "suggestions": ["设置快周期小于慢周期", "推荐快周期12，慢周期26"],
                    "related_paths": ["factor_params.macd.slow_period"],
                    "auto_fix_available": True,
                    "auto_fix_function": "fix_macd_periods"
                }
        return None
    
    # 自动修复函数
    def _fix_batch_size(self, config: Dict[str, Any], issue: DiagnosticIssue) -> Dict[str, Any]:
        """修复批次大小"""
        path_parts = issue.path.split('.')
        current = config
        
        # 导航到目标位置
        for part in path_parts[:-1]:
            current = current[part]
        
        # 修复值
        current_value = current[path_parts[-1]]
        if isinstance(current_value, int):
            if current_value > 5000:
                current[path_parts[-1]] = 2000
            elif current_value <= 0:
                current[path_parts[-1]] = 1000
            elif current_value < 100:
                current[path_parts[-1]] = 500
        
        return config
    
    def _fix_port_number(self, config: Dict[str, Any], issue: DiagnosticIssue) -> Dict[str, Any]:
        """修复端口号"""
        path_parts = issue.path.split('.')
        current = config
        
        for part in path_parts[:-1]:
            current = current[part]
        
        # 设置默认端口
        current[path_parts[-1]] = 8000
        
        return config
    
    def _fix_cors_origins(self, config: Dict[str, Any], issue: DiagnosticIssue) -> Dict[str, Any]:
        """修复CORS配置"""
        path_parts = issue.path.split('.')
        current = config
        
        for part in path_parts[:-1]:
            current = current[part]
        
        # 设置更安全的默认值
        current[path_parts[-1]] = ["http://localhost:3000", "http://localhost:8501"]
        
        return config
    
    def _fix_macd_periods(self, config: Dict[str, Any], issue: DiagnosticIssue) -> Dict[str, Any]:
        """修复MACD周期"""
        # 设置标准的MACD参数
        if "factor_params" in config and "macd" in config["factor_params"]:
            config["factor_params"]["macd"]["fast_period"] = 12
            config["factor_params"]["macd"]["slow_period"] = 26
        
        return config


def create_config_diagnostics() -> ConfigDiagnostics:
    """创建配置诊断器"""
    return ConfigDiagnostics()


def diagnose_config_file(file_path: str) -> DiagnosticReport:
    """诊断配置文件"""
    import yaml
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        diagnostics = create_config_diagnostics()
        return diagnostics.diagnose_config(config, file_path)
        
    except Exception as e:
        # 创建错误报告
        error_issue = DiagnosticIssue(
            level=DiagnosticLevel.CRITICAL,
            category=DiagnosticCategory.SYNTAX,
            path="__file__",
            message=f"无法读取配置文件: {e}",
            description="配置文件可能存在语法错误或文件不存在"
        )
        
        return DiagnosticReport(
            timestamp=datetime.now().isoformat(),
            config_path=file_path,
            total_issues=1,
            issues_by_level={level.value: 1 if level == DiagnosticLevel.CRITICAL else 0 for level in DiagnosticLevel},
            issues_by_category={category.value: 1 if category == DiagnosticCategory.SYNTAX else 0 for category in DiagnosticCategory},
            issues=[error_issue],
            auto_fixable_count=0,
            health_score=0.0
        )


if __name__ == "__main__":
    # 测试配置诊断
    diagnostics = create_config_diagnostics()
    
    # 测试配置
    test_config = {
        "data_sync_params": {
            "batch_size": -100,  # 错误：负数
            "tushare": {"enabled": True}  # 可能错误：缺少环境变量
        },
        "api_params": {
            "port": 99999,  # 错误：端口号超出范围
            "cors_origins": ["*"]  # 警告：安全问题
        },
        "factor_params": {
            "macd": {
                "fast_period": 30,  # 错误：快周期大于慢周期
                "slow_period": 26
            }
        }
    }
    
    report = diagnostics.diagnose_config(test_config)
    print(f"诊断完成，发现 {report.total_issues} 个问题")
    print(f"健康分数: {report.health_score}")
    
    for issue in report.issues:
        print(f"- {issue.level.value.upper()}: {issue.message}")
        if issue.suggestions:
            for suggestion in issue.suggestions:
                print(f"  建议: {suggestion}")
    
    # 测试自动修复
    if report.auto_fixable_count > 0:
        fixed_config = diagnostics.auto_fix_issues(test_config, report.issues)
        print(f"\n自动修复后的配置:")
        import json
        print(json.dumps(fixed_config, indent=2, ensure_ascii=False))
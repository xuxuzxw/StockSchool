import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

"""
配置热更新系统

提供配置的动态更新和影响分析功能
"""


logger = logging.getLogger(__name__)


class ChangeImpact(Enum):
    """配置变更影响级别"""

    LOW = "low"  # 低影响，无需重启
    MEDIUM = "medium"  # 中等影响，需要重新初始化某些组件
    HIGH = "high"  # 高影响，需要重启服务
    CRITICAL = "critical"  # 关键影响，需要立即处理


@dataclass
class ConfigImpactRule:
    """配置影响规则"""

    path_pattern: str
    impact_level: ChangeImpact
    affected_components: List[str] = field(default_factory=list)
    restart_required: bool = False
    custom_handler: Optional[Callable] = None
    description: str = ""


@dataclass
class ChangeImpactAnalysis:
    """变更影响分析结果"""

    path: str
    old_value: Any
    new_value: Any
    impact_level: ChangeImpact
    affected_components: List[str]
    restart_required: bool
    recommendations: List[str]
    warnings: List[str]


class HotReloadManager:
    """热更新管理器"""

    def __init__(self, config_manager):
        """方法描述"""
        self.impact_rules: List[ConfigImpactRule] = []
        self.component_handlers: Dict[str, Callable] = {}
        self.reload_lock = threading.RLock()
        self.reload_history: List[Dict[str, Any]] = []

        # 设置默认影响规则
        self._setup_default_impact_rules()

        # 注册配置变更回调
        self.config_manager.add_change_callback(self._handle_config_change)

    def _setup_default_impact_rules(self):
        """设置默认影响规则"""

        # 数据库配置变更 - 高影响
        self.add_impact_rule(
            ConfigImpactRule(
                path_pattern="database_params.*",
                impact_level=ChangeImpact.HIGH,
                affected_components=["database", "data_sync", "api"],
                restart_required=True,
                description="数据库配置变更需要重启服务",
            )
        )

        # API配置变更 - 中等影响
        self.add_impact_rule(
            ConfigImpactRule(
                path_pattern="api_params.port",
                impact_level=ChangeImpact.HIGH,
                affected_components=["api"],
                restart_required=True,
                description="API端口变更需要重启API服务",
            )
        )

        self.add_impact_rule(
            ConfigImpactRule(
                path_pattern="api_params.workers",
                impact_level=ChangeImpact.HIGH,
                affected_components=["api"],
                restart_required=True,
                description="API工作进程数变更需要重启API服务",
            )
        )

        self.add_impact_rule(
            ConfigImpactRule(
                path_pattern="api_params.log_level",
                impact_level=ChangeImpact.LOW,
                affected_components=["logging"],
                restart_required=False,
                description="日志级别变更可以动态应用",
            )
        )

        # 数据同步配置变更 - 中等影响
        self.add_impact_rule(
            ConfigImpactRule(
                path_pattern="data_sync_params.batch_size",
                impact_level=ChangeImpact.MEDIUM,
                affected_components=["data_sync"],
                restart_required=False,
                description="批次大小变更需要重新初始化同步器",
            )
        )

        self.add_impact_rule(
            ConfigImpactRule(
                path_pattern="data_sync_params.max_workers",
                impact_level=ChangeImpact.MEDIUM,
                affected_components=["data_sync"],
                restart_required=False,
                description="最大工作线程数变更需要重新初始化线程池",
            )
        )

        # 因子计算配置变更 - 低影响
        self.add_impact_rule(
            ConfigImpactRule(
                path_pattern="factor_params.*",
                impact_level=ChangeImpact.LOW,
                affected_components=["factor_engine"],
                restart_required=False,
                description="因子参数变更在下次计算时生效",
            )
        )

        # 监控配置变更 - 低影响
        self.add_impact_rule(
            ConfigImpactRule(
                path_pattern="monitoring_params.collection_interval",
                impact_level=ChangeImpact.MEDIUM,
                affected_components=["monitoring"],
                restart_required=False,
                description="监控间隔变更需要重新调度监控任务",
            )
        )

        self.add_impact_rule(
            ConfigImpactRule(
                path_pattern="monitoring_params.alerts.*",
                impact_level=ChangeImpact.LOW,
                affected_components=["monitoring"],
                restart_required=False,
                description="告警阈值变更立即生效",
            )
        )

        # 特征工程配置变更 - 中等影响
        self.add_impact_rule(
            ConfigImpactRule(
                path_pattern="feature_params.use_cuda",
                impact_level=ChangeImpact.HIGH,
                affected_components=["ai_model", "feature_engine"],
                restart_required=True,
                description="CUDA设置变更需要重启相关服务",
            )
        )

        self.add_impact_rule(
            ConfigImpactRule(
                path_pattern="feature_params.shap_batch_size",
                impact_level=ChangeImpact.LOW,
                affected_components=["ai_model"],
                restart_required=False,
                description="SHAP批次大小变更在下次计算时生效",
            )
        )

        # 数据质量配置变更 - 低影响
        self.add_impact_rule(
            ConfigImpactRule(
                path_pattern="data_quality.*",
                impact_level=ChangeImpact.LOW,
                affected_components=["data_quality"],
                restart_required=False,
                description="数据质量配置变更立即生效",
            )
        )

    def add_impact_rule(self, rule: ConfigImpactRule):
        """添加影响规则"""
        self.impact_rules.append(rule)

    def register_component_handler(self, component: str, handler: Callable):
        """注册组件处理器"""
        self.component_handlers[component] = handler

    def _handle_config_change(self, path: str, old_value: Any, new_value: Any):
        """处理配置变更"""
        try:
            with self.reload_lock:
                # 分析变更影响
                impact_analysis = self.analyze_change_impact(path, old_value, new_value)

                # 记录变更历史
                self.reload_history.append(
                    {
                        "timestamp": datetime.now(),
                        "path": path,
                        "old_value": old_value,
                        "new_value": new_value,
                        "impact_analysis": impact_analysis,
                    }
                )

                # 应用变更
                self._apply_config_change(impact_analysis)

                logger.info(f"配置热更新完成: {path} = {new_value}, 影响级别: {impact_analysis.impact_level.value}")

        except Exception as e:
            logger.error(f"处理配置变更失败 {path}: {e}")

    def analyze_change_impact(self, path: str, old_value: Any, new_value: Any) -> ChangeImpactAnalysis:
        """分析配置变更影响"""
        import fnmatch

        # 默认影响分析
        impact_analysis = ChangeImpactAnalysis(
            path=path,
            old_value=old_value,
            new_value=new_value,
            impact_level=ChangeImpact.LOW,
            affected_components=[],
            restart_required=False,
            recommendations=[],
            warnings=[],
        )

        # 匹配影响规则
        matched_rules = []
        for rule in self.impact_rules:
            if fnmatch.fnmatch(path, rule.path_pattern):
                matched_rules.append(rule)

        if matched_rules:
            # 取最高影响级别
            max_impact_rule = max(matched_rules, key=lambda r: list(ChangeImpact).index(r.impact_level))

            impact_analysis.impact_level = max_impact_rule.impact_level
            impact_analysis.restart_required = max_impact_rule.restart_required

            # 合并受影响的组件
            for rule in matched_rules:
                impact_analysis.affected_components.extend(rule.affected_components)
            impact_analysis.affected_components = list(set(impact_analysis.affected_components))

        # 生成建议和警告
        self._generate_recommendations(impact_analysis)

        return impact_analysis

    def _generate_recommendations(self, analysis: ChangeImpactAnalysis):
        """生成建议和警告"""

        # 根据影响级别生成建议
        if analysis.impact_level == ChangeImpact.CRITICAL:
            analysis.recommendations.append("立即检查系统状态，可能需要紧急处理")
            analysis.warnings.append("关键配置变更，请谨慎操作")
        elif analysis.impact_level == ChangeImpact.HIGH:
            analysis.recommendations.append("建议在维护窗口期间重启相关服务")
            analysis.warnings.append("高影响变更，可能影响系统可用性")
        elif analysis.impact_level == ChangeImpact.MEDIUM:
            analysis.recommendations.append("建议重新初始化相关组件")

        # 根据受影响组件生成建议
        if "database" in analysis.affected_components:
            analysis.recommendations.append("检查数据库连接状态")

        if "api" in analysis.affected_components:
            analysis.recommendations.append("验证API服务可用性")

        if "data_sync" in analysis.affected_components:
            analysis.recommendations.append("检查数据同步任务状态")

        # 特殊值变更警告
        if analysis.path.endswith("batch_size") and isinstance(analysis.new_value, int):
            if analysis.new_value > 5000:
                analysis.warnings.append("批次大小过大可能影响性能")
            elif analysis.new_value < 100:
                analysis.warnings.append("批次大小过小可能影响效率")

        if analysis.path.endswith("max_workers") and isinstance(analysis.new_value, int):
            if analysis.new_value > 20:
                analysis.warnings.append("工作线程数过多可能导致资源竞争")
            elif analysis.new_value < 1:
                analysis.warnings.append("工作线程数不能小于1")

    def _apply_config_change(self, analysis: ChangeImpactAnalysis):
        """应用配置变更"""

        # 如果需要重启，记录但不自动重启
        if analysis.restart_required:
            logger.warning(f"配置变更 {analysis.path} 需要重启服务才能生效")
            return

        # 调用组件处理器
        for component in analysis.affected_components:
            if component in self.component_handlers:
                try:
                    handler = self.component_handlers[component]
                    handler(analysis.path, analysis.old_value, analysis.new_value)
                    logger.info(f"组件 {component} 配置更新完成")
                except Exception as e:
                    logger.error(f"组件 {component} 配置更新失败: {e}")

        # 特殊处理
        self._handle_special_config_changes(analysis)

    def _handle_special_config_changes(self, analysis: ChangeImpactAnalysis):
        """处理特殊配置变更"""

        # 日志级别变更
        if analysis.path == "api_params.log_level":
            try:
                import logging

                new_level = getattr(logging, analysis.new_value.upper())
                logging.getLogger().setLevel(new_level)
                logger.info(f"日志级别已更新为: {analysis.new_value}")
            except Exception as e:
                logger.error(f"更新日志级别失败: {e}")

        # 监控间隔变更
        elif analysis.path == "monitoring_params.collection_interval":
            # 这里可以通知监控系统更新收集间隔
            logger.info(f"监控收集间隔已更新为: {analysis.new_value}秒")

        # 告警阈值变更
        elif analysis.path.startswith("monitoring_params.alerts."):
            # 这里可以通知告警系统更新阈值
            alert_type = analysis.path.split(".")[-1]
            logger.info(f"告警阈值 {alert_type} 已更新为: {analysis.new_value}")

    def get_reload_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取热更新历史"""
        return self.reload_history[-limit:]

    def get_pending_restarts(self) -> List[Dict[str, Any]]:
        """获取需要重启的配置变更"""
        pending = []

        for record in self.reload_history:
            impact_analysis = record["impact_analysis"]
            if impact_analysis.restart_required:
                pending.append(
                    {
                        "timestamp": record["timestamp"],
                        "path": record["path"],
                        "new_value": record["new_value"],
                        "affected_components": impact_analysis.affected_components,
                        "recommendations": impact_analysis.recommendations,
                    }
                )

        return pending

    def clear_restart_flags(self):
        """清除重启标记（在重启后调用）"""
        self.reload_history = [
            record for record in self.reload_history if not record["impact_analysis"].restart_required
        ]
        logger.info("重启标记已清除")

    def simulate_config_change(self, path: str, new_value: Any) -> ChangeImpactAnalysis:
        """模拟配置变更（不实际应用）"""
        old_value = self.config_manager.get(path)
        return self.analyze_change_impact(path, old_value, new_value)

    def batch_update_config(self, updates: Dict[str, Any], dry_run: bool = False) -> List[ChangeImpactAnalysis]:
        """批量更新配置"""
        analyses = []

        # 分析所有变更
        for path, new_value in updates.items():
            old_value = self.config_manager.get(path)
            analysis = self.analyze_change_impact(path, old_value, new_value)
            analyses.append(analysis)

        if dry_run:
            return analyses

        # 按影响级别排序，先处理低影响的变更
        analyses.sort(key=lambda a: list(ChangeImpact).index(a.impact_level))

        # 应用变更
        with self.reload_lock:
            for analysis in analyses:
                try:
                    self.config_manager.set(analysis.path, analysis.new_value, source="batch_update")
                except Exception as e:
                    logger.error(f"批量更新配置失败 {analysis.path}: {e}")

        return analyses

    def rollback_to_timestamp(self, timestamp: datetime) -> bool:
        """回滚到指定时间点"""
        try:
            # 找到需要回滚的变更
            changes_to_rollback = [record for record in self.reload_history if record["timestamp"] > timestamp]

            if not changes_to_rollback:
                logger.info("没有需要回滚的配置变更")
                return True

            # 按时间倒序回滚
            changes_to_rollback.sort(key=lambda x: x["timestamp"], reverse=True)

            rollback_updates = {}
            for record in changes_to_rollback:
                path = record["path"]
                old_value = record["old_value"]
                rollback_updates[path] = old_value

            # 批量回滚
            self.batch_update_config(rollback_updates)

            logger.info(f"配置回滚完成，回滚了 {len(changes_to_rollback)} 个变更")
            return True

        except Exception as e:
            logger.error(f"配置回滚失败: {e}")
            return False

    def get_impact_summary(self) -> Dict[str, Any]:
        """获取影响摘要"""
        summary = {
            "total_changes": len(self.reload_history),
            "impact_levels": {level.value: 0 for level in ChangeImpact},
            "affected_components": {},
            "pending_restarts": len(self.get_pending_restarts()),
            "recent_changes": [],
        }

        # 统计影响级别
        for record in self.reload_history:
            impact_level = record["impact_analysis"].impact_level.value
            summary["impact_levels"][impact_level] += 1

            # 统计受影响组件
            for component in record["impact_analysis"].affected_components:
                summary["affected_components"][component] = summary["affected_components"].get(component, 0) + 1

        # 最近的变更
        summary["recent_changes"] = [
            {
                "timestamp": record["timestamp"],
                "path": record["path"],
                "impact_level": record["impact_analysis"].impact_level.value,
            }
            for record in self.reload_history[-10:]
        ]

        return summary


def create_hot_reload_manager(config_manager) -> HotReloadManager:
    """创建热更新管理器"""
    return HotReloadManager(config_manager)


# 组件处理器示例
def database_config_handler(path: str, old_value: Any, new_value: Any):
    """数据库配置处理器"""
    logger.info(f"数据库配置变更: {path} = {new_value}")
    # 这里可以重新初始化数据库连接池等


def data_sync_config_handler(path: str, old_value: Any, new_value: Any):
    """数据同步配置处理器"""
    logger.info(f"数据同步配置变更: {path} = {new_value}")
    # 这里可以重新初始化同步器参数


def monitoring_config_handler(path: str, old_value: Any, new_value: Any):
    """监控配置处理器"""
    logger.info(f"监控配置变更: {path} = {new_value}")
    # 这里可以更新监控参数


if __name__ == "__main__":
    # 测试热更新管理器
    from .manager import ConfigManager

    config_manager = ConfigManager()
    hot_reload_manager = create_hot_reload_manager(config_manager)

    # 注册组件处理器
    hot_reload_manager.register_component_handler("database", database_config_handler)
    hot_reload_manager.register_component_handler("data_sync", data_sync_config_handler)
    hot_reload_manager.register_component_handler("monitoring", monitoring_config_handler)

    # 模拟配置变更
    analysis = hot_reload_manager.simulate_config_change("data_sync_params.batch_size", 2000)
    print(f"变更影响分析: {analysis}")

    # 获取影响摘要
    summary = hot_reload_manager.get_impact_summary()
    print(f"影响摘要: {summary}")

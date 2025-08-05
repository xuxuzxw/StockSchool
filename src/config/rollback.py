import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

"""
配置回滚系统

提供配置的回滚和恢复功能
"""


logger = logging.getLogger(__name__)


class RollbackType(Enum):
    """回滚类型"""

    TIMESTAMP = "timestamp"  # 基于时间戳回滚
    VERSION = "version"  # 基于版本回滚
    SNAPSHOT = "snapshot"  # 基于快照回滚
    SELECTIVE = "selective"  # 选择性回滚


@dataclass
class ConfigSnapshot:
    """配置快照"""

    id: str
    timestamp: datetime
    config: Dict[str, Any]
    description: str = ""
    tags: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """方法描述"""
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RollbackPlan:
    """回滚计划"""

    rollback_type: RollbackType
    target_snapshot: Optional[ConfigSnapshot]
    target_timestamp: Optional[datetime]
    changes_to_rollback: List[Dict[str, Any]]
    affected_paths: List[str]
    impact_analysis: Dict[str, Any]
    estimated_duration: int  # 秒
    risks: List[str]
    prerequisites: List[str]


class ConfigRollbackManager:
    """配置回滚管理器"""

    def __init__(self, config_manager, snapshot_dir: str = "config_snapshots"):
        """方法描述"""
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.snapshots: List[ConfigSnapshot] = []
        self.rollback_history: List[Dict[str, Any]] = []

        # 加载已有快照
        self._load_snapshots()

        # 自动创建初始快照
        self.create_snapshot("initial", "系统启动时的初始配置")

    def _load_snapshots(self):
        """加载已有快照"""
        try:
            for snapshot_file in self.snapshot_dir.glob("*.json"):
                with open(snapshot_file, "r", encoding="utf-8") as f:
                    snapshot_data = json.load(f)

                    snapshot = ConfigSnapshot(
                        id=snapshot_data["id"],
                        timestamp=datetime.fromisoformat(snapshot_data["timestamp"]),
                        config=snapshot_data["config"],
                        description=snapshot_data.get("description", ""),
                        tags=snapshot_data.get("tags", []),
                        metadata=snapshot_data.get("metadata", {}),
                    )
                    self.snapshots.append(snapshot)

            # 按时间排序
            self.snapshots.sort(key=lambda s: s.timestamp)
            logger.info(f"加载了 {len(self.snapshots)} 个配置快照")

        except Exception as e:
            logger.error(f"加载配置快照失败: {e}")

    def create_snapshot(
        self, snapshot_id: str, description: str = "", tags: List[str] = None, metadata: Dict[str, Any] = None
    ) -> ConfigSnapshot:
        """创建配置快照"""
        try:
            # 获取当前配置
            current_config = self.config_manager._config.copy()

            # 创建快照
            snapshot = ConfigSnapshot(
                id=snapshot_id,
                timestamp=datetime.now(),
                config=current_config,
                description=description,
                tags=tags or [],
                metadata=metadata or {},
            )

            # 保存到文件
            snapshot_file = self.snapshot_dir / f"{snapshot_id}_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            with open(snapshot_file, "w", encoding="utf-8") as f:
                json.dump(asdict(snapshot), f, indent=2, ensure_ascii=False, default=str)

            # 添加到列表
            self.snapshots.append(snapshot)

            logger.info(f"配置快照已创建: {snapshot_id}")
            return snapshot

        except Exception as e:
            logger.error(f"创建配置快照失败: {e}")
            raise

    def list_snapshots(self, tags: List[str] = None, limit: int = None) -> List[ConfigSnapshot]:
        """列出配置快照"""
        snapshots = self.snapshots

        # 按标签过滤
        if tags:
            snapshots = [s for s in snapshots if any(tag in s.tags for tag in tags)]

        # 限制数量
        if limit:
            snapshots = snapshots[-limit:]

        return snapshots

    def get_snapshot(self, snapshot_id: str) -> Optional[ConfigSnapshot]:
        """获取指定快照"""
        for snapshot in self.snapshots:
            if snapshot.id == snapshot_id:
                return snapshot
        return None

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """删除配置快照"""
        try:
            snapshot = self.get_snapshot(snapshot_id)
            if not snapshot:
                logger.warning(f"快照不存在: {snapshot_id}")
                return False

            # 删除文件
            snapshot_files = list(self.snapshot_dir.glob(f"{snapshot_id}_*.json"))
            for file_path in snapshot_files:
                file_path.unlink()

            # 从列表中移除
            self.snapshots = [s for s in self.snapshots if s.id != snapshot_id]

            logger.info(f"配置快照已删除: {snapshot_id}")
            return True

        except Exception as e:
            logger.error(f"删除配置快照失败: {e}")
            return False

    def create_rollback_plan(
        self,
        rollback_type: RollbackType,
        target_snapshot_id: str = None,
        target_timestamp: datetime = None,
        selective_paths: List[str] = None,
    ) -> RollbackPlan:
        """创建回滚计划"""

        current_config = self.config_manager._config
        changes_to_rollback = []
        affected_paths = []

        if rollback_type == RollbackType.SNAPSHOT:
            if not target_snapshot_id:
                raise ValueError("快照回滚需要指定目标快照ID")

            target_snapshot = self.get_snapshot(target_snapshot_id)
            if not target_snapshot:
                raise ValueError(f"快照不存在: {target_snapshot_id}")

            # 比较当前配置和目标快照
            changes_to_rollback = self._compare_configs(current_config, target_snapshot.config)
            affected_paths = [change["path"] for change in changes_to_rollback]

        elif rollback_type == RollbackType.TIMESTAMP:
            if not target_timestamp:
                raise ValueError("时间戳回滚需要指定目标时间")

            # 找到最接近目标时间的快照
            target_snapshot = self._find_closest_snapshot(target_timestamp)
            if not target_snapshot:
                raise ValueError(f"未找到接近时间 {target_timestamp} 的快照")

            changes_to_rollback = self._compare_configs(current_config, target_snapshot.config)
            affected_paths = [change["path"] for change in changes_to_rollback]

        elif rollback_type == RollbackType.SELECTIVE:
            if not selective_paths:
                raise ValueError("选择性回滚需要指定路径列表")

            # 找到最近的快照作为参考
            if self.snapshots:
                reference_snapshot = self.snapshots[-1]
                all_changes = self._compare_configs(current_config, reference_snapshot.config)

                # 只回滚指定路径
                changes_to_rollback = [
                    change for change in all_changes if any(change["path"].startswith(path) for path in selective_paths)
                ]
                affected_paths = selective_paths

        # 分析影响
        impact_analysis = self._analyze_rollback_impact(changes_to_rollback)

        # 评估风险
        risks = self._assess_rollback_risks(changes_to_rollback, impact_analysis)

        # 生成先决条件
        prerequisites = self._generate_prerequisites(changes_to_rollback, impact_analysis)

        # 估算持续时间
        estimated_duration = self._estimate_rollback_duration(changes_to_rollback)

        return RollbackPlan(
            rollback_type=rollback_type,
            target_snapshot=(
                target_snapshot if rollback_type in [RollbackType.SNAPSHOT, RollbackType.TIMESTAMP] else None
            ),
            target_timestamp=target_timestamp,
            changes_to_rollback=changes_to_rollback,
            affected_paths=affected_paths,
            impact_analysis=impact_analysis,
            estimated_duration=estimated_duration,
            risks=risks,
            prerequisites=prerequisites,
        )

    def execute_rollback_plan(self, plan: RollbackPlan, dry_run: bool = False) -> Dict[str, Any]:
        """执行回滚计划"""

        if dry_run:
            return {
                "success": True,
                "message": "干运行模式，未实际执行回滚",
                "plan": asdict(plan),
                "changes_count": len(plan.changes_to_rollback),
            }

        try:
            # 创建回滚前快照
            pre_rollback_snapshot = self.create_snapshot(
                f"pre_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "回滚前的配置快照", tags=["pre_rollback"]
            )

            # 执行回滚
            rollback_start = datetime.now()

            for change in plan.changes_to_rollback:
                try:
                    path = change["path"]
                    target_value = change["target_value"]

                    self.config_manager.set(path, target_value, source="rollback")
                    logger.debug(f"回滚配置项: {path} = {target_value}")

                except Exception as e:
                    logger.error(f"回滚配置项失败 {path}: {e}")
                    # 继续执行其他回滚操作

            rollback_end = datetime.now()
            duration = (rollback_end - rollback_start).total_seconds()

            # 记录回滚历史
            rollback_record = {
                "timestamp": rollback_start,
                "rollback_type": plan.rollback_type.value,
                "target_snapshot_id": plan.target_snapshot.id if plan.target_snapshot else None,
                "target_timestamp": plan.target_timestamp,
                "changes_count": len(plan.changes_to_rollback),
                "duration": duration,
                "pre_rollback_snapshot_id": pre_rollback_snapshot.id,
                "success": True,
            }
            self.rollback_history.append(rollback_record)

            logger.info(f"配置回滚完成，耗时 {duration:.2f} 秒，回滚了 {len(plan.changes_to_rollback)} 个配置项")

            return {
                "success": True,
                "message": "回滚执行成功",
                "duration": duration,
                "changes_count": len(plan.changes_to_rollback),
                "pre_rollback_snapshot_id": pre_rollback_snapshot.id,
            }

        except Exception as e:
            logger.error(f"执行回滚计划失败: {e}")

            # 记录失败的回滚
            rollback_record = {
                "timestamp": datetime.now(),
                "rollback_type": plan.rollback_type.value,
                "error": str(e),
                "success": False,
            }
            self.rollback_history.append(rollback_record)

            return {"success": False, "message": f"回滚执行失败: {e}", "error": str(e)}

    def _compare_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """比较两个配置"""
        changes = []

        def compare_dict(dict1: Dict, dict2: Dict, path: str = ""):
            """方法描述"""

            for key in all_keys:
                current_path = f"{path}.{key}" if path else key

                if key not in dict1:
                    # 新增项，回滚时需要删除
                    changes.append(
                        {"type": "delete", "path": current_path, "current_value": dict2[key], "target_value": None}
                    )
                elif key not in dict2:
                    # 删除项，回滚时需要添加
                    changes.append(
                        {"type": "add", "path": current_path, "current_value": None, "target_value": dict1[key]}
                    )
                elif dict1[key] != dict2[key]:
                    if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                        compare_dict(dict1[key], dict2[key], current_path)
                    else:
                        # 修改项
                        changes.append(
                            {
                                "type": "modify",
                                "path": current_path,
                                "current_value": dict2[key],
                                "target_value": dict1[key],
                            }
                        )

        compare_dict(config2, config1)  # 注意参数顺序
        return changes

    def _find_closest_snapshot(self, target_timestamp: datetime) -> Optional[ConfigSnapshot]:
        """找到最接近目标时间的快照"""
        if not self.snapshots:
            return None

        closest_snapshot = None
        min_diff = float("inf")

        for snapshot in self.snapshots:
            diff = abs((snapshot.timestamp - target_timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_snapshot = snapshot

        return closest_snapshot

    def _analyze_rollback_impact(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析回滚影响"""
        impact = {
            "total_changes": len(changes),
            "change_types": {"add": 0, "modify": 0, "delete": 0},
            "affected_sections": set(),
            "critical_changes": [],
            "restart_required": False,
        }

        for change in changes:
            # 统计变更类型
            change_type = change["type"]
            impact["change_types"][change_type] += 1

            # 分析受影响的配置段
            path = change["path"]
            section = path.split(".")[0]
            impact["affected_sections"].add(section)

            # 识别关键变更
            if self._is_critical_config(path):
                impact["critical_changes"].append(change)
                impact["restart_required"] = True

        impact["affected_sections"] = list(impact["affected_sections"])
        return impact

    def _is_critical_config(self, path: str) -> bool:
        """判断是否为关键配置"""
        critical_patterns = ["database_params", "api_params.port", "api_params.workers", "feature_params.use_cuda"]

        return any(path.startswith(pattern) for pattern in critical_patterns)

    def _assess_rollback_risks(self, changes: List[Dict[str, Any]], impact: Dict[str, Any]) -> List[str]:
        """评估回滚风险"""
        risks = []

        if impact["restart_required"]:
            risks.append("回滚涉及关键配置，需要重启服务")

        if impact["total_changes"] > 50:
            risks.append("回滚变更数量较多，可能影响系统稳定性")

        if "database_params" in impact["affected_sections"]:
            risks.append("数据库配置回滚可能导致连接中断")

        if "api_params" in impact["affected_sections"]:
            risks.append("API配置回滚可能影响服务可用性")

        return risks

    def _generate_prerequisites(self, changes: List[Dict[str, Any]], impact: Dict[str, Any]) -> List[str]:
        """生成先决条件"""
        prerequisites = []

        if impact["restart_required"]:
            prerequisites.append("确保在维护窗口期间执行")
            prerequisites.append("通知相关用户服务将重启")

        if "database_params" in impact["affected_sections"]:
            prerequisites.append("备份当前数据库配置")
            prerequisites.append("确认数据库连接可用")

        if impact["total_changes"] > 20:
            prerequisites.append("创建当前配置的完整备份")

        prerequisites.append("验证目标配置的有效性")
        prerequisites.append("准备回滚失败时的恢复方案")

        return prerequisites

    def _estimate_rollback_duration(self, changes: List[Dict[str, Any]]) -> int:
        """估算回滚持续时间（秒）"""
        base_time = 10  # 基础时间
        change_time = len(changes) * 0.5  # 每个变更0.5秒

        # 关键配置需要更多时间
        critical_changes = sum(1 for change in changes if self._is_critical_config(change["path"]))
        critical_time = critical_changes * 5

        return int(base_time + change_time + critical_time)

    def get_rollback_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取回滚历史"""
        return self.rollback_history[-limit:]

    def cleanup_old_snapshots(self, keep_days: int = 30, keep_count: int = 10):
        """清理旧快照"""
        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)

            # 按时间过滤
            old_snapshots = [s for s in self.snapshots if s.timestamp < cutoff_date]

            # 保留最近的快照
            if len(old_snapshots) > keep_count:
                snapshots_to_delete = old_snapshots[:-keep_count]

                for snapshot in snapshots_to_delete:
                    self.delete_snapshot(snapshot.id)

                logger.info(f"清理了 {len(snapshots_to_delete)} 个旧快照")

        except Exception as e:
            logger.error(f"清理旧快照失败: {e}")

    def export_snapshot(self, snapshot_id: str, export_path: str):
        """导出快照"""
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            raise ValueError(f"快照不存在: {snapshot_id}")

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(asdict(snapshot), f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"快照已导出: {export_path}")

    def import_snapshot(self, import_path: str) -> ConfigSnapshot:
        """导入快照"""
        with open(import_path, "r", encoding="utf-8") as f:
            snapshot_data = json.load(f)

        snapshot = ConfigSnapshot(
            id=snapshot_data["id"],
            timestamp=datetime.fromisoformat(snapshot_data["timestamp"]),
            config=snapshot_data["config"],
            description=snapshot_data.get("description", ""),
            tags=snapshot_data.get("tags", []),
            metadata=snapshot_data.get("metadata", {}),
        )

        # 保存快照
        snapshot_file = self.snapshot_dir / f"{snapshot.id}_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(snapshot_file, "w", encoding="utf-8") as f:
            json.dump(asdict(snapshot), f, indent=2, ensure_ascii=False, default=str)

        self.snapshots.append(snapshot)

        logger.info(f"快照已导入: {snapshot.id}")
        return snapshot


def create_rollback_manager(config_manager, snapshot_dir: str = "config_snapshots") -> ConfigRollbackManager:
    """创建回滚管理器"""
    return ConfigRollbackManager(config_manager, snapshot_dir)


if __name__ == "__main__":
    # 测试回滚管理器
    from .manager import ConfigManager

    config_manager = ConfigManager()
    rollback_manager = create_rollback_manager(config_manager)

    # 创建测试快照
    snapshot = rollback_manager.create_snapshot("test", "测试快照")
    print(f"创建快照: {snapshot.id}")

    # 列出快照
    snapshots = rollback_manager.list_snapshots()
    print(f"快照列表: {[s.id for s in snapshots]}")

    # 创建回滚计划
    plan = rollback_manager.create_rollback_plan(RollbackType.SNAPSHOT, target_snapshot_id="test")
    print(f"回滚计划: {len(plan.changes_to_rollback)} 个变更")

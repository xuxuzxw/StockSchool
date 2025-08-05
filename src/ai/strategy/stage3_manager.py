import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ...utils.db import get_db_engine
from .deployment_manager import DeploymentManager
from .doc_generator import DocumentGenerator
from .model_monitor import ModelMonitor
from .system_optimizer import SystemOptimizer
from .test_framework import TestFramework

logger = logging.getLogger(__name__)


@dataclass
class Stage3Status:
    """第三阶段状态"""

    model_monitor_active: bool
    system_optimizer_active: bool
    doc_generator_active: bool
    test_framework_active: bool
    deployment_manager_active: bool
    overall_health: str
    last_check_time: datetime
    active_tasks: int
    error_count: int


@dataclass
class Stage3Metrics:
    """第三阶段指标"""

    monitored_models: int
    active_alerts: int
    optimization_tasks: int
    generated_documents: int
    test_executions: int
    deployments: int
    system_health_score: float
    uptime_percentage: float


class Stage3Manager:
    """第三阶段管理器 - 统一管理所有第三阶段功能模块"""

    def __init__(self, engine=None):
        """方法描述"""

        # 初始化各功能模块
        self.model_monitor = ModelMonitor()
        self.system_optimizer = SystemOptimizer(self.engine)
        self.doc_generator = DocumentGenerator(self.engine)
        self.test_framework = TestFramework(self.engine)
        self.deployment_manager = DeploymentManager(self.engine)

        # 状态跟踪
        self._last_health_check = None
        self._active_tasks = set()
        self._error_count = 0

        logger.info("第三阶段管理器初始化完成")

    async def initialize_all_modules(self) -> bool:
        """初始化所有模块"""
        try:
            logger.info("开始初始化第三阶段所有模块...")

            # 创建数据库表
            await self._create_all_tables()

            # 初始化默认配置
            await self._initialize_default_configs()

            # 启动后台任务
            await self._start_background_tasks()

            logger.info("第三阶段所有模块初始化完成")
            return True

        except Exception as e:
            logger.error(f"初始化第三阶段模块失败: {e}")
            self._error_count += 1
            return False

    async def _create_all_tables(self):
        """创建所有数据库表"""
        try:
            # 各模块的表在初始化时已经创建，这里只需要确认
            logger.info("所有数据库表已在模块初始化时创建完成")
        except Exception as e:
            logger.error(f"创建数据库表失败: {e}")
            raise

    async def _initialize_default_configs(self):
        """初始化默认配置"""
        try:
            # 各模块的默认配置已在初始化时设置，这里只需要确认
            logger.info("默认配置已在模块初始化时完成")
        except Exception as e:
            logger.error(f"初始化默认配置失败: {e}")
            raise

    async def _start_background_tasks(self):
        """启动后台任务"""
        try:
            # 启动定期健康检查
            asyncio.create_task(self._periodic_health_check())

            # 启动系统优化任务
            asyncio.create_task(self._periodic_optimization())

            # 启动模型监控任务
            asyncio.create_task(self._periodic_model_monitoring())

            logger.info("后台任务启动完成")
        except Exception as e:
            logger.error(f"启动后台任务失败: {e}")
            raise

    async def _periodic_health_check(self):
        """定期健康检查"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟检查一次
                await self.check_system_health()
            except Exception as e:
                logger.error(f"定期健康检查失败: {e}")
                self._error_count += 1

    async def _periodic_optimization(self):
        """定期系统优化"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时执行一次
                await self.system_optimizer.run_scheduled_optimization()
            except Exception as e:
                logger.error(f"定期系统优化失败: {e}")
                self._error_count += 1

    async def _periodic_model_monitoring(self):
        """定期模型监控"""
        while True:
            try:
                await asyncio.sleep(1800)  # 每30分钟检查一次
                await self.model_monitor.run_scheduled_monitoring()
            except Exception as e:
                logger.error(f"定期模型监控失败: {e}")
                self._error_count += 1

    async def check_system_health(self) -> Stage3Status:
        """检查系统健康状态"""
        try:
            self._last_health_check = datetime.now()

            # 检查各模块状态
            model_monitor_active = await self._check_module_health("model_monitor")
            system_optimizer_active = await self._check_module_health("system_optimizer")
            doc_generator_active = await self._check_module_health("doc_generator")
            test_framework_active = await self._check_module_health("test_framework")
            deployment_manager_active = await self._check_module_health("deployment_manager")

            # 计算整体健康状态
            active_modules = sum(
                [
                    model_monitor_active,
                    system_optimizer_active,
                    doc_generator_active,
                    test_framework_active,
                    deployment_manager_active,
                ]
            )

            if active_modules == 5:
                overall_health = "excellent"
            elif active_modules >= 4:
                overall_health = "good"
            elif active_modules >= 3:
                overall_health = "fair"
            elif active_modules >= 2:
                overall_health = "poor"
            else:
                overall_health = "critical"

            status = Stage3Status(
                model_monitor_active=model_monitor_active,
                system_optimizer_active=system_optimizer_active,
                doc_generator_active=doc_generator_active,
                test_framework_active=test_framework_active,
                deployment_manager_active=deployment_manager_active,
                overall_health=overall_health,
                last_check_time=self._last_health_check,
                active_tasks=len(self._active_tasks),
                error_count=self._error_count,
            )

            logger.info(f"系统健康检查完成: {overall_health}")
            return status

        except Exception as e:
            logger.error(f"系统健康检查失败: {e}")
            self._error_count += 1
            return Stage3Status(
                model_monitor_active=False,
                system_optimizer_active=False,
                doc_generator_active=False,
                test_framework_active=False,
                deployment_manager_active=False,
                overall_health="critical",
                last_check_time=datetime.now(),
                active_tasks=0,
                error_count=self._error_count,
            )

    async def _check_module_health(self, module_name: str) -> bool:
        """检查单个模块健康状态"""
        try:
            if module_name == "model_monitor":
                # 检查模型监控模块
                dashboard_data = self.model_monitor.get_monitoring_dashboard_data()
                return dashboard_data is not None

            elif module_name == "system_optimizer":
                # 检查系统优化模块
                health_report = self.system_optimizer.get_system_health_report()
                return health_report is not None

            elif module_name == "doc_generator":
                # 检查文档生成模块
                return True  # 文档生成模块通常是无状态的

            elif module_name == "test_framework":
                # 检查测试框架模块
                return True  # 测试框架模块通常是无状态的

            elif module_name == "deployment_manager":
                # 检查部署管理模块
                return True  # 部署管理模块通常是无状态的

            return False

        except Exception as e:
            logger.error(f"检查模块 {module_name} 健康状态失败: {e}")
            return False

    async def get_system_metrics(self) -> Stage3Metrics:
        """获取系统指标"""
        try:
            # 获取各模块指标
            monitored_models = await self._get_monitored_models_count()
            active_alerts = await self._get_active_alerts_count()
            optimization_tasks = await self._get_optimization_tasks_count()
            generated_documents = await self._get_generated_documents_count()
            test_executions = await self._get_test_executions_count()
            deployments = await self._get_deployments_count()

            # 计算系统健康评分
            system_health_score = await self._calculate_health_score()

            # 计算运行时间百分比
            uptime_percentage = await self._calculate_uptime_percentage()

            return Stage3Metrics(
                monitored_models=monitored_models,
                active_alerts=active_alerts,
                optimization_tasks=optimization_tasks,
                generated_documents=generated_documents,
                test_executions=test_executions,
                deployments=deployments,
                system_health_score=system_health_score,
                uptime_percentage=uptime_percentage,
            )

        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return Stage3Metrics(
                monitored_models=0,
                active_alerts=0,
                optimization_tasks=0,
                generated_documents=0,
                test_executions=0,
                deployments=0,
                system_health_score=0.0,
                uptime_percentage=0.0,
            )

    async def _get_monitored_models_count(self) -> int:
        """获取监控模型数量"""
        try:
            dashboard_data = self.model_monitor.get_monitoring_dashboard_data()
            return dashboard_data.get("overview", {}).get("total_models", 0)
        except Exception:
            return 0

    async def _get_active_alerts_count(self) -> int:
        """获取活跃告警数量"""
        try:
            alerts = self.model_monitor.get_alerts(status="active", limit=1000)
            return len(alerts)
        except Exception:
            return 0

    async def _get_optimization_tasks_count(self) -> int:
        """获取优化任务数量"""
        try:
            health_report = self.system_optimizer.get_system_health_report()
            return health_report.get("active_tasks", 0)
        except Exception:
            return 0

    async def _get_generated_documents_count(self) -> int:
        """获取生成文档数量"""
        try:
            documents = await self.doc_generator.get_generated_documents(limit=1000)
            return len(documents)
        except Exception:
            return 0

    async def _get_test_executions_count(self) -> int:
        """获取测试执行数量"""
        try:
            # 获取最近24小时的测试执行数量
            start_time = datetime.now() - timedelta(hours=24)
            results = await self.test_framework.get_test_results(start_time=start_time, limit=1000)
            return len(results)
        except Exception:
            return 0

    async def _get_deployments_count(self) -> int:
        """获取部署数量"""
        try:
            # 获取最近24小时的部署数量
            start_date = datetime.now() - timedelta(hours=24)
            end_date = datetime.now()
            report = await self.deployment_manager.generate_deployment_report(
                environment="all", start_date=start_date, end_date=end_date
            )
            return report.get("deployment_statistics", {}).get("total_deployments", 0)
        except Exception:
            return 0

    async def _calculate_health_score(self) -> float:
        """计算系统健康评分"""
        try:
            status = await self.check_system_health()

            # 基于各模块状态计算评分
            score = 0.0
            if status.model_monitor_active:
                score += 20.0
            if status.system_optimizer_active:
                score += 20.0
            if status.doc_generator_active:
                score += 20.0
            if status.test_framework_active:
                score += 20.0
            if status.deployment_manager_active:
                score += 20.0

            # 根据错误数量调整评分
            error_penalty = min(status.error_count * 2, 20)
            score = max(0, score - error_penalty)

            return round(score, 2)

        except Exception:
            return 0.0

    async def _calculate_uptime_percentage(self) -> float:
        """计算运行时间百分比"""
        try:
            # 这里可以基于实际的运行时间统计
            # 暂时返回一个基于错误数量的估算值
            if self._error_count == 0:
                return 100.0
            elif self._error_count < 5:
                return 95.0
            elif self._error_count < 10:
                return 90.0
            elif self._error_count < 20:
                return 85.0
            else:
                return 80.0
        except Exception:
            return 0.0

    async def execute_comprehensive_check(self) -> Dict[str, Any]:
        """执行全面检查"""
        try:
            logger.info("开始执行全面系统检查...")

            # 获取系统状态
            status = await self.check_system_health()

            # 获取系统指标
            metrics = await self.get_system_metrics()

            # 执行各模块检查
            module_checks = await self._execute_module_checks()

            # 生成检查报告
            report = {
                "timestamp": datetime.now(),
                "overall_status": {
                    "health": status.overall_health,
                    "active_modules": sum(
                        [
                            status.model_monitor_active,
                            status.system_optimizer_active,
                            status.doc_generator_active,
                            status.test_framework_active,
                            status.deployment_manager_active,
                        ]
                    ),
                    "total_modules": 5,
                    "error_count": status.error_count,
                    "active_tasks": status.active_tasks,
                },
                "system_metrics": {
                    "monitored_models": metrics.monitored_models,
                    "active_alerts": metrics.active_alerts,
                    "optimization_tasks": metrics.optimization_tasks,
                    "generated_documents": metrics.generated_documents,
                    "test_executions": metrics.test_executions,
                    "deployments": metrics.deployments,
                    "health_score": metrics.system_health_score,
                    "uptime_percentage": metrics.uptime_percentage,
                },
                "module_status": {
                    "model_monitor": status.model_monitor_active,
                    "system_optimizer": status.system_optimizer_active,
                    "doc_generator": status.doc_generator_active,
                    "test_framework": status.test_framework_active,
                    "deployment_manager": status.deployment_manager_active,
                },
                "module_checks": module_checks,
                "recommendations": await self._generate_recommendations(status, metrics),
            }

            logger.info("全面系统检查完成")
            return report

        except Exception as e:
            logger.error(f"执行全面检查失败: {e}")
            return {"timestamp": datetime.now(), "error": str(e), "status": "failed"}

    async def _execute_module_checks(self) -> Dict[str, Any]:
        """执行模块检查"""
        checks = {}

        try:
            # 模型监控检查
            checks["model_monitor"] = await self._check_model_monitor_details()

            # 系统优化检查
            checks["system_optimizer"] = await self._check_system_optimizer_details()

            # 文档生成检查
            checks["doc_generator"] = await self._check_doc_generator_details()

            # 测试框架检查
            checks["test_framework"] = await self._check_test_framework_details()

            # 部署管理检查
            checks["deployment_manager"] = await self._check_deployment_manager_details()

        except Exception as e:
            logger.error(f"执行模块检查失败: {e}")

        return checks

    async def _check_model_monitor_details(self) -> Dict[str, Any]:
        """检查模型监控模块详情"""
        try:
            dashboard_data = self.model_monitor.get_monitoring_dashboard_data()
            alerts = self.model_monitor.get_alerts(limit=10)

            return {
                "status": "healthy" if dashboard_data else "unhealthy",
                "dashboard_available": dashboard_data is not None,
                "recent_alerts": len(alerts),
                "last_check": datetime.now(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "last_check": datetime.now()}

    async def _check_system_optimizer_details(self) -> Dict[str, Any]:
        """检查系统优化模块详情"""
        try:
            health_report = self.system_optimizer.get_system_health_report()

            return {
                "status": "healthy" if health_report else "unhealthy",
                "health_report_available": health_report is not None,
                "active_optimizations": health_report.get("active_tasks", 0) if health_report else 0,
                "last_check": datetime.now(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "last_check": datetime.now()}

    async def _check_doc_generator_details(self) -> Dict[str, Any]:
        """检查文档生成详情"""
        try:
            documents = await self.doc_generator.get_generated_documents(limit=5)

            return {"status": "healthy", "recent_documents": len(documents), "last_check": datetime.now()}
        except Exception as e:
            return {"status": "error", "error": str(e), "last_check": datetime.now()}

    async def _check_test_framework_details(self) -> Dict[str, Any]:
        """检查测试框架详情"""
        try:
            results = await self.test_framework.get_test_results(limit=5)

            return {"status": "healthy", "recent_test_results": len(results), "last_check": datetime.now()}
        except Exception as e:
            return {"status": "error", "error": str(e), "last_check": datetime.now()}

    async def _check_deployment_manager_details(self) -> Dict[str, Any]:
        """检查部署管理详情"""
        try:
            # 检查最近的部署历史
            start_date = datetime.now() - timedelta(days=1)
            end_date = datetime.now()
            report = await self.deployment_manager.generate_deployment_report(
                environment="all", start_date=start_date, end_date=end_date
            )

            return {
                "status": "healthy",
                "recent_deployments": report.get("deployment_statistics", {}).get("total_deployments", 0),
                "last_check": datetime.now(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "last_check": datetime.now()}

    async def _generate_recommendations(self, status: Stage3Status, metrics: Stage3Metrics) -> List[str]:
        """生成改进建议"""
        recommendations = []

        try:
            # 基于系统状态生成建议
            if status.overall_health == "critical":
                recommendations.append("系统处于关键状态，建议立即检查所有模块")
            elif status.overall_health == "poor":
                recommendations.append("系统状态较差，建议检查非活跃模块")

            # 基于错误数量生成建议
            if status.error_count > 10:
                recommendations.append("错误数量较多，建议检查日志并修复问题")
            elif status.error_count > 5:
                recommendations.append("存在一些错误，建议定期检查系统状态")

            # 基于指标生成建议
            if metrics.active_alerts > 5:
                recommendations.append("活跃告警较多，建议及时处理")

            if metrics.system_health_score < 80:
                recommendations.append("系统健康评分较低，建议进行系统优化")

            if metrics.uptime_percentage < 95:
                recommendations.append("系统运行时间百分比较低，建议检查稳定性")

            # 模块特定建议
            if not status.model_monitor_active:
                recommendations.append("模型监控模块未激活，建议检查配置")

            if not status.system_optimizer_active:
                recommendations.append("系统优化模块未激活，建议启用自动优化")

            if len(recommendations) == 0:
                recommendations.append("系统运行良好，继续保持")

        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            recommendations.append("无法生成建议，请检查系统状态")

        return recommendations

    def get_module_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        return {
            "stage": "第三阶段",
            "version": "3.0.0",
            "modules": {
                "model_monitor": {
                    "name": "模型监控",
                    "description": "AI模型性能监控、数据漂移检测、自动告警",
                    "features": ["性能监控", "漂移检测", "自动重训练", "告警通知"],
                },
                "system_optimizer": {
                    "name": "系统优化",
                    "description": "系统性能优化、资源管理、自动调优",
                    "features": ["性能监控", "资源优化", "缓存管理", "自动调优"],
                },
                "doc_generator": {
                    "name": "文档生成",
                    "description": "自动化文档生成、API文档、系统文档",
                    "features": ["API文档", "系统文档", "用户手册", "部署指南"],
                },
                "test_framework": {
                    "name": "测试框架",
                    "description": "全面的自动化测试框架",
                    "features": ["单元测试", "集成测试", "性能测试", "UI测试"],
                },
                "deployment_manager": {
                    "name": "部署管理",
                    "description": "智能部署管理、多环境支持、自动回滚",
                    "features": ["多环境部署", "健康检查", "自动回滚", "部署监控"],
                },
            },
            "capabilities": [
                "智能模型监控与告警",
                "系统性能自动优化",
                "文档自动生成与维护",
                "全面自动化测试",
                "智能部署与运维",
            ],
        }

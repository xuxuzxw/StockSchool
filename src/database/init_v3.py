import logging
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from src.ai.strategy.deployment_manager import DeploymentManager
from src.ai.strategy.doc_generator import DocumentGenerator
from src.ai.strategy.model_monitor import ModelMonitor
from src.ai.strategy.system_optimizer import SystemOptimizer
from src.ai.strategy.test_framework import TestFramework
from src.utils.db import get_db_engine

logger = logging.getLogger(__name__)


class DatabaseInitializerV3:
    """第三阶段数据库初始化器"""

    def __init__(self, engine=None):
        """方法描述"""

        # 初始化各模块
        self.model_monitor = ModelMonitor()
        self.system_optimizer = SystemOptimizer()
        self.doc_generator = DocumentGenerator()
        self.test_framework = TestFramework()
        self.deployment_manager = DeploymentManager()

    async def initialize_all_tables(self) -> bool:
        """初始化所有第三阶段数据库表"""
        try:
            logger.info("开始初始化第三阶段数据库表...")

            # 初始化模型监控相关表
            await self._initialize_model_monitor_tables()

            # 初始化系统优化相关表
            await self._initialize_system_optimizer_tables()

            # 初始化文档生成相关表
            await self._initialize_doc_generator_tables()

            # 初始化测试框架相关表
            await self._initialize_test_framework_tables()

            # 初始化部署管理相关表
            await self._initialize_deployment_manager_tables()

            # 初始化默认数据
            await self._initialize_default_data()

            logger.info("第三阶段数据库表初始化完成")
            return True

        except Exception as e:
            logger.error(f"初始化第三阶段数据库表失败: {e}")
            return False

    async def _initialize_model_monitor_tables(self):
        """初始化模型监控相关表"""
        try:
            logger.info("初始化模型监控相关表...")
            # ModelMonitor的表在初始化时已经创建
            logger.info("模型监控表创建完成")
        except Exception as e:
            logger.error(f"创建模型监控表失败: {e}")
            raise

    async def _initialize_system_optimizer_tables(self):
        """初始化系统优化相关表"""
        try:
            logger.info("初始化系统优化相关表...")
            # SystemOptimizer的表在初始化时已经创建
            logger.info("系统优化表创建完成")
        except Exception as e:
            logger.error(f"创建系统优化表失败: {e}")
            raise

    async def _initialize_doc_generator_tables(self):
        """初始化文档生成相关表"""
        try:
            logger.info("初始化文档生成相关表...")
            # DocumentGenerator的表在初始化时已经创建
            logger.info("文档生成表创建完成")
        except Exception as e:
            logger.error(f"创建文档生成表失败: {e}")
            raise

    async def _initialize_test_framework_tables(self):
        """初始化测试框架相关表"""
        try:
            logger.info("初始化测试框架相关表...")
            # TestFramework的表在初始化时已经创建
            logger.info("测试框架表创建完成")
        except Exception as e:
            logger.error(f"创建测试框架表失败: {e}")
            raise

    async def _initialize_deployment_manager_tables(self):
        """初始化部署管理相关表"""
        try:
            logger.info("初始化部署管理相关表...")
            # DeploymentManager的表在初始化时已经创建
            logger.info("部署管理表创建完成")
        except Exception as e:
            logger.error(f"创建部署管理表失败: {e}")
            raise

    async def _initialize_default_data(self):
        """初始化默认数据"""
        try:
            logger.info("初始化默认数据...")

            # 初始化系统优化默认配置
            await self._init_system_optimizer_defaults()

            # 初始化文档生成默认模板
            await self._init_doc_generator_defaults()

            # 初始化测试框架默认数据
            await self._init_test_framework_defaults()

            # 初始化部署管理默认配置
            await self._init_deployment_manager_defaults()

            logger.info("默认数据初始化完成")

        except Exception as e:
            logger.error(f"初始化默认数据失败: {e}")
            raise

    async def _init_system_optimizer_defaults(self):
        """初始化系统优化默认配置"""
        try:
            # SystemOptimizer的默认配置在初始化时已经设置
            logger.info("系统优化默认配置初始化完成")
        except Exception as e:
            logger.error(f"初始化系统优化默认配置失败: {e}")
            raise

    async def _init_doc_generator_defaults(self):
        """初始化文档生成默认模板"""
        try:
            # DocumentGenerator的默认模板在初始化时已经创建
            logger.info("文档生成默认模板初始化完成")
        except Exception as e:
            logger.error(f"初始化文档生成默认模板失败: {e}")
            raise

    async def _init_test_framework_defaults(self):
        """初始化测试框架默认数据"""
        try:
            # TestFramework的默认数据在初始化时已经创建
            logger.info("测试框架默认数据初始化完成")
        except Exception as e:
            logger.error(f"初始化测试框架默认数据失败: {e}")
            raise

    async def _init_deployment_manager_defaults(self):
        """初始化部署管理默认配置"""
        try:
            # DeploymentManager的默认配置在初始化时已经创建
            logger.info("部署管理默认配置初始化完成")
        except Exception as e:
            logger.error(f"初始化部署管理默认配置失败: {e}")
            raise

    async def check_tables_exist(self) -> dict:
        """检查所有表是否存在"""
        table_status = {}

        try:
            # 检查模型监控表
            monitor_tables = [
                "model_performance_metrics",
                "data_drift_metrics",
                "model_alerts",
                "retraining_triggers",
                "model_monitor_configs",
                "monitor_task_logs",
            ]

            # 检查系统优化表
            optimizer_tables = [
                "system_metrics",
                "performance_alerts",
                "optimization_tasks",
                "cache_strategies",
                "system_configs",
                "performance_baselines",
            ]

            # 检查文档生成表
            doc_tables = [
                "api_endpoints",
                "database_tables",
                "system_modules",
                "document_templates",
                "generated_documents",
                "document_changes",
            ]

            # 检查测试框架表
            test_tables = [
                "test_cases",
                "test_results",
                "test_suites",
                "test_reports",
                "performance_baselines",
                "test_environments",
            ]

            # 检查部署管理表
            deploy_tables = [
                "deployment_configs",
                "deployment_tasks",
                "server_info",
                "application_versions",
                "health_check_results",
                "deployment_history",
            ]

            all_tables = {
                "model_monitor": monitor_tables,
                "system_optimizer": optimizer_tables,
                "doc_generator": doc_tables,
                "test_framework": test_tables,
                "deployment_manager": deploy_tables,
            }

            with self.engine.connect() as conn:
                for module, tables in all_tables.items():
                    table_status[module] = {}
                    for table in tables:
                        try:
                            result = conn.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
                            table_status[module][table] = True
                        except Exception:
                            table_status[module][table] = False

            return table_status

        except Exception as e:
            logger.error(f"检查表状态失败: {e}")
            return {}

    async def drop_all_v3_tables(self) -> bool:
        """删除所有第三阶段表（谨慎使用）"""
        try:
            logger.warning("开始删除所有第三阶段表...")

            tables_to_drop = [
                # 模型监控表
                "monitor_task_logs",
                "model_monitor_configs",
                "retraining_triggers",
                "model_alerts",
                "data_drift_metrics",
                "model_performance_metrics",
                # 系统优化表
                "performance_baselines",
                "system_configs",
                "cache_strategies",
                "optimization_tasks",
                "performance_alerts",
                "system_metrics",
                # 文档生成表
                "document_changes",
                "generated_documents",
                "document_templates",
                "system_modules",
                "database_tables",
                "api_endpoints",
                # 测试框架表
                "test_environments",
                "test_reports",
                "test_suites",
                "test_results",
                "test_cases",
                # 部署管理表
                "deployment_history",
                "health_check_results",
                "application_versions",
                "server_info",
                "deployment_tasks",
                "deployment_configs",
            ]

            with self.engine.connect() as conn:
                for table in tables_to_drop:
                    try:
                        conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                        logger.info(f"删除表 {table} 成功")
                    except Exception as e:
                        logger.warning(f"删除表 {table} 失败: {e}")

                conn.commit()

            logger.warning("所有第三阶段表删除完成")
            return True

        except Exception as e:
            logger.error(f"删除第三阶段表失败: {e}")
            return False

    async def get_initialization_status(self) -> dict:
        """获取初始化状态"""
        try:
            table_status = await self.check_tables_exist()

            # 统计各模块表的状态
            module_status = {}
            for module, tables in table_status.items():
                total_tables = len(tables)
                existing_tables = sum(1 for exists in tables.values() if exists)
                module_status[module] = {
                    "total_tables": total_tables,
                    "existing_tables": existing_tables,
                    "completion_rate": (existing_tables / total_tables * 100) if total_tables > 0 else 0,
                    "is_complete": existing_tables == total_tables,
                    "tables": tables,
                }

            # 计算总体完成率
            total_tables = sum(status["total_tables"] for status in module_status.values())
            total_existing = sum(status["existing_tables"] for status in module_status.values())
            overall_completion = (total_existing / total_tables * 100) if total_tables > 0 else 0

            return {
                "overall_status": {
                    "total_tables": total_tables,
                    "existing_tables": total_existing,
                    "completion_rate": round(overall_completion, 2),
                    "is_complete": total_existing == total_tables,
                },
                "module_status": module_status,
                "timestamp": logger.info,
            }

        except Exception as e:
            logger.error(f"获取初始化状态失败: {e}")
            return {}


# 便捷函数
async def initialize_v3_database(engine=None) -> bool:
    """初始化第三阶段数据库"""
    initializer = DatabaseInitializerV3(engine)
    return await initializer.initialize_all_tables()


async def check_v3_database_status(engine=None) -> dict:
    """检查第三阶段数据库状态"""
    initializer = DatabaseInitializerV3(engine)
    return await initializer.get_initialization_status()


async def reset_v3_database(engine=None) -> bool:
    """重置第三阶段数据库（删除并重新创建）"""
    initializer = DatabaseInitializerV3(engine)

    # 先删除所有表
    drop_success = await initializer.drop_all_v3_tables()
    if not drop_success:
        return False

    # 重新初始化
    return await initializer.initialize_all_tables()


if __name__ == "__main__":
    import asyncio

    async def main():
        """主函数"""
        print("开始初始化第三阶段数据库...")

        # 初始化数据库
        success = await initialize_v3_database()

        if success:
            print("数据库初始化成功！")

            # 检查状态
            status = await check_v3_database_status()
            print(f"初始化状态: {status}")
        else:
            print("数据库初始化失败！")

    # 运行主函数
    asyncio.run(main())

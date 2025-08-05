import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from src.core.config import get_config
from src.monitoring import MonitoringSystem
from src.monitoring.config import MonitoringConfig

#!/usr/bin/env python3
"""
StockSchool 监控系统启动脚本

这个脚本用于独立启动监控系统，可以在不启动主应用的情况下运行监控服务。
"""


# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/monitoring_standalone.log")],
)

logger = logging.getLogger(__name__)


class MonitoringService:
    """监控服务管理器"""

    def __init__(self):
        """方法描述"""
        self.running = False

    async def start(self):
        """启动监控服务"""
        try:
            logger.info("正在启动监控服务...")

            # 加载配置
            config = get_config()
            monitoring_config = MonitoringConfig.from_config_file(
                config_file=project_root / "config" / "monitoring.yaml"
            )

            # 初始化监控系统
            self.monitoring_system = MonitoringSystem(monitoring_config)

            # 启动监控系统
            await self.monitoring_system.start()

            self.running = True
            logger.info("监控服务启动成功")

            # 保持服务运行
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"启动监控服务失败: {e}")
            raise

    async def stop(self):
        """停止监控服务"""
        try:
            logger.info("正在停止监控服务...")
            self.running = False

            if self.monitoring_system:
                await self.monitoring_system.stop()

            logger.info("监控服务已停止")

        except Exception as e:
            logger.error(f"停止监控服务失败: {e}")

    def handle_signal(self, signum, frame):
        """处理系统信号"""
        logger.info(f"收到信号 {signum}，正在关闭服务...")
        self.running = False


def main():
    """主函数"""
    # 确保日志目录存在
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    # 创建监控服务
    service = MonitoringService()

    # 注册信号处理器
    signal.signal(signal.SIGINT, service.handle_signal)
    signal.signal(signal.SIGTERM, service.handle_signal)

    try:
        # 运行监控服务
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logger.info("收到键盘中断信号")
    except Exception as e:
        logger.error(f"监控服务运行异常: {e}")
        sys.exit(1)
    finally:
        # 清理资源
        try:
            asyncio.run(service.stop())
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


if __name__ == "__main__":
    main()

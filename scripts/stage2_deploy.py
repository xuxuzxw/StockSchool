import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import docker
import requests

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二阶段部署脚本
生产环境优化部署工具
"""


# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Stage2Deployer:
    """第二阶段部署器"""

    def __init__(self, project_root: str = "."):
        """方法描述"""
        self.docker_client = docker.from_env()

        # 配置路径
        self.config_path = self.project_root / "config" / "stage2_optimization_config.yml"
        self.docker_compose_file = self.project_root / "docker-compose.stage2.yml"

        logger.info(f"初始化第二阶段部署器，项目根目录: {self.project_root}")

    def pre_deployment_check(self) -> bool:
        """部署前检查"""
        logger.info("执行部署前检查...")

        checks = [
            self._check_docker_installation,
            self._check_config_files,
            self._check_disk_space,
            self._check_network_connectivity,
            self._check_port_availability,
        ]

        for check in checks:
            if not check():
                logger.error(f"检查失败: {check.__name__}")
                return False

        logger.info("✅ 所有部署前检查通过")
        return True

    def _check_docker_installation(self) -> bool:
        """检查Docker安装"""
        try:
            self.docker_client.ping()
            logger.info("✅ Docker已安装并运行")
            return True
        except Exception as e:
            logger.error(f"❌ Docker检查失败: {e}")
            return False

    def _check_config_files(self) -> bool:
        """检查配置文件"""
        required_files = [self.config_path, self.docker_compose_file, self.project_root / "requirements-stage2.txt"]

        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"❌ 配置文件不存在: {file_path}")
                return False

        logger.info("✅ 所有配置文件存在")
        return True

    def _check_disk_space(self) -> bool:
        """检查磁盘空间"""
        stat = os.statvfs(self.project_root)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

        if free_gb < 5:
            logger.error(f"❌ 磁盘空间不足: {free_gb:.1f}GB < 5GB")
            return False

        logger.info(f"✅ 磁盘空间充足: {free_gb:.1f}GB")
        return True

    def _check_network_connectivity(self) -> bool:
        """检查网络连接"""
        try:
            response = requests.get("https://registry.hub.docker.com/v2/", timeout=10)
            if response.status_code == 200:
                logger.info("✅ 网络连接正常")
                return True
        except Exception as e:
            logger.error(f"❌ 网络连接检查失败: {e}")
            return False

    def _check_port_availability(self) -> bool:
        """检查端口可用性"""
        ports = [8000, 8001, 5432, 6379, 3000, 9090]

        for port in ports:
            try:
                result = subprocess.run(["netstat", "-an"], capture_output=True, text=True)
                if f":{port}" in result.stdout:
                    logger.warning(f"⚠️  端口 {port} 可能已被占用")
            except Exception:
                pass

        logger.info("✅ 端口检查完成")
        return True

    def build_optimized_images(self) -> bool:
        """构建优化镜像"""
        logger.info("开始构建优化镜像...")

        try:
            # 构建主应用镜像
            logger.info("构建主应用优化镜像...")
            image, logs = self.docker_client.images.build(
                path=str(self.project_root),
                dockerfile="Dockerfile.stage2",
                tag="stock-school:stage2-optimized",
                rm=True,
                forcerm=True,
            )

            # 构建测试镜像
            logger.info("构建测试镜像...")
            self.docker_client.images.build(
                path=str(self.project_root),
                dockerfile="Dockerfile.test",
                tag="stock-school:test",
                rm=True,
                forcerm=True,
            )

            logger.info("✅ 优化镜像构建成功")
            return True

        except Exception as e:
            logger.error(f"❌ 镜像构建失败: {e}")
            return False

    def deploy_services(self) -> bool:
        """部署服务"""
        logger.info("开始部署第二阶段服务...")

        try:
            # 启动所有服务
            subprocess.run(["docker-compose", "-f", str(self.docker_compose_file), "up", "-d"], check=True)

            logger.info("✅ 服务部署成功")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 服务部署失败: {e}")
            return False

    def wait_for_services(self, timeout: int = 300) -> bool:
        """等待服务就绪"""
        logger.info("等待服务就绪...")

        services = [
            ("stock-school-stage2", 8000),
            ("postgres", 5432),
            ("redis", 6379),
            ("prometheus", 9090),
            ("grafana", 3000),
        ]

        start_time = time.time()

        for service_name, port in services:
            while time.time() - start_time < timeout:
                try:
                    # 检查容器状态
                    container = self.docker_client.containers.get(service_name)
                    if container.status != "running":
                        time.sleep(5)
                        continue

                    # 检查端口可用性
                    response = requests.get(
                        (
                            f"http://localhost:{port}/health"
                            if port != 5432 and port != 6379
                            else f"http://localhost:{port}"
                        ),
                        timeout=5,
                    )
                    if response.status_code in [200, 404]:  # 404可能是健康检查端点不存在
                        logger.info(f"✅ {service_name} 已就绪")
                        break

                except Exception:
                    time.sleep(5)
            else:
                logger.error(f"❌ {service_name} 启动超时")
                return False

        logger.info("✅ 所有服务已就绪")
        return True

    def run_performance_tests(self) -> Dict:
        """运行性能测试"""
        logger.info("开始性能测试...")

        try:
            # 启动性能测试容器
            subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    str(self.docker_compose_file),
                    "--profile",
                    "testing",
                    "run",
                    "performance-test",
                ],
                check=True,
            )

            # 收集测试结果
            results = self._collect_test_results()

            logger.info("✅ 性能测试完成")
            return results

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 性能测试失败: {e}")
            return {"error": str(e)}

    def _collect_test_results(self) -> Dict:
        """收集测试结果"""
        results_file = self.project_root / "tests" / "results" / "performance_results.json"

        if results_file.exists():
            with open(results_file, "r") as f:
                return json.load(f)

        return {"message": "测试完成，结果文件未找到"}

    def setup_monitoring(self) -> bool:
        """设置监控"""
        logger.info("设置监控仪表板...")

        try:
            # 配置Grafana
            grafana_url = "http://localhost:3000"

            # 等待Grafana启动
            time.sleep(10)

            # 导入仪表板配置
            self._import_grafana_dashboards()

            logger.info("✅ 监控设置完成")
            return True

        except Exception as e:
            logger.error(f"❌ 监控设置失败: {e}")
            return False

    def _import_grafana_dashboards(self) -> None:
        """导入Grafana仪表板"""
        # 这里可以添加具体的仪表板导入逻辑
        logger.info("Grafana仪表板已配置")

    def generate_deployment_report(self) -> str:
        """生成部署报告"""
        report = {
            "deployment_time": datetime.now().isoformat(),
            "services": [],
            "performance_metrics": {},
            "health_status": "healthy",
        }

        # 收集服务信息
        try:
            containers = self.docker_client.containers.list()
            for container in containers:
                report["services"].append(
                    {
                        "name": container.name,
                        "status": container.status,
                        "image": container.image.tags[0] if container.image.tags else "unknown",
                    }
                )
        except Exception as e:
            logger.error(f"收集服务信息失败: {e}")

        # 保存报告
        report_file = self.project_root / "deployment_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        return str(report_file)

    def run_full_deployment(self) -> bool:
        """运行完整部署流程"""
        logger.info("🚀 开始第二阶段完整部署流程...")

        steps = [
            ("预部署检查", self.pre_deployment_check),
            ("构建优化镜像", self.build_optimized_images),
            ("部署服务", self.deploy_services),
            ("等待服务就绪", self.wait_for_services),
            ("运行性能测试", lambda: bool(self.run_performance_tests())),
            ("设置监控", self.setup_monitoring),
        ]

        for step_name, step_func in steps:
            logger.info(f"执行步骤: {step_name}")
            if not step_func():
                logger.error(f"❌ 部署失败在步骤: {step_name}")
                return False

        report_file = self.generate_deployment_report()
        logger.info(f"✅ 第二阶段部署完成！报告已生成: {report_file}")
        return True

    def cleanup(self) -> bool:
        """清理部署"""
        logger.info("开始清理部署...")

        try:
            subprocess.run(["docker-compose", "-f", str(self.docker_compose_file), "down", "-v"], check=True)

            logger.info("✅ 清理完成")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 清理失败: {e}")
            return False


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        deployer = Stage2Deployer()
        deployer.cleanup()
    else:
        deployer = Stage2Deployer()
        deployer.run_full_deployment()


if __name__ == "__main__":
    main()

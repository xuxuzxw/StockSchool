import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docker部署验证测试

验证StockSchool监控系统的Docker容器化部署是否正常工作

作者: StockSchool Team
创建时间: 2025-01-02
"""


try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    print("警告: aiohttp不可用，将跳过HTTP测试")
    AIOHTTP_AVAILABLE = False


class DockerDeploymentTester:
    """Docker部署测试器"""

    def __init__(self):
        """方法描述"""
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
        self.container_name = "stockschool-monitoring-test"
        self.image_name = "stockschool-monitoring:test"
        self.test_port = 8001  # 使用不同端口避免冲突

    def print_header(self, title: str):
        """打印测试标题"""
        print(f"\n{'='*80}")
        print(f"🐳 {title}")
        print(f"{'='*80}")

    def print_section(self, title: str):
        """打印测试章节"""
        print(f"\n🧪 {title}...")

    def print_result(self, test_name: str, passed: bool, details: str = ""):
        """打印测试结果"""
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {status} {test_name}")
        if details:
            print(f"    {details}")

        self.test_results["tests"][test_name] = {
            "passed": passed,
            "details": details
        }

    def test_docker_availability(self):
        """测试Docker可用性"""
        self.print_section("测试Docker环境")

        try:
            # 检查Docker是否安装
            result = subprocess.run(['docker', '--version'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.print_result("Docker安装", True, result.stdout.strip())
            else:
                self.print_result("Docker安装", False, "Docker命令失败")
                return False

            # 检查Docker是否运行
            result = subprocess.run(['docker', 'info'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.print_result("Docker服务", True, "Docker守护进程正在运行")
            else:
                self.print_result("Docker服务", False, "Docker守护进程未运行")
                return False

            return True

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.print_result("Docker环境", False, f"Docker不可用: {e}")
            return False
        except Exception as e:
            self.print_result("Docker环境", False, f"检查失败: {e}")
            return False

    def test_dockerfile_build(self):
        """测试Dockerfile构建"""
        self.print_section("测试Docker镜像构建")

        try:
            # 清理可能存在的旧镜像
            subprocess.run(['docker', 'rmi', self.image_name],
                          capture_output=True, timeout=30)

            # 构建Docker镜像
            print("  正在构建Docker镜像...")
            result = subprocess.run([
                'docker', 'build',
                '-t', self.image_name,
                '-f', 'Dockerfile',
                '.'
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                self.print_result("Docker镜像构建", True, "镜像构建成功")
                return True
            else:
                self.print_result("Docker镜像构建", False, f"构建失败: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.print_result("Docker镜像构建", False, "构建超时")
            return False
        except Exception as e:
            self.print_result("Docker镜像构建", False, f"构建异常: {e}")
            return False

    def test_container_startup(self):
        """测试容器启动"""
        self.print_section("测试容器启动")

        try:
            # 停止并删除可能存在的旧容器
            subprocess.run(['docker', 'stop', self.container_name],
                          capture_output=True, timeout=30)
            subprocess.run(['docker', 'rm', self.container_name],
                          capture_output=True, timeout=30)

            # 启动容器
            print("  正在启动Docker容器...")
            result = subprocess.run([
                'docker', 'run', '-d',
                '--name', self.container_name,
                '-p', f'{self.test_port}:8000',
                '-e', 'DEBUG=false',
                '-e', 'LOG_LEVEL=INFO',
                self.image_name
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                container_id = result.stdout.strip()
                self.print_result("容器启动", True, f"容器ID: {container_id[:12]}")

                # 等待容器启动
                print("  等待容器完全启动...")
                time.sleep(10)

                # 检查容器状态
                result = subprocess.run([
                    'docker', 'ps', '--filter', f'name={self.container_name}', '--format', 'table {{.Status}}'
                ], capture_output=True, text=True, timeout=10)

                if 'Up' in result.stdout:
                    self.print_result("容器运行状态", True, "容器正在运行")
                    return True
                else:
                    self.print_result("容器运行状态", False, "容器未正常运行")
                    return False
            else:
                self.print_result("容器启动", False, f"启动失败: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.print_result("容器启动", False, "启动超时")
            return False
        except Exception as e:
            self.print_result("容器启动", False, f"启动异常: {e}")
            return False

    async def test_container_health(self):
        """测试容器健康状态"""
        self.print_section("测试容器健康状态")

        if not AIOHTTP_AVAILABLE:
            self.print_result("容器健康检查", False, "aiohttp不可用")
            return False

        try:
            base_url = f"http://localhost:{self.test_port}"

            async with aiohttp.ClientSession() as session:
                # 测试健康检查端点
                try:
                    async with session.get(f"{base_url}/health", timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.print_result("健康检查端点", True, f"状态: {data.get('status', 'unknown')}")
                        else:
                            self.print_result("健康检查端点", False, f"HTTP {response.status}")
                            return False
                except Exception as e:
                    self.print_result("健康检查端点", False, f"请求失败: {e}")
                    return False

                # 测试系统信息端点
                try:
                    async with session.get(f"{base_url}/info", timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.print_result("系统信息端点", True, f"版本: {data.get('version', 'unknown')}")
                        else:
                            self.print_result("系统信息端点", False, f"HTTP {response.status}")
                except Exception as e:
                    self.print_result("系统信息端点", False, f"请求失败: {e}")

                # 测试API文档端点
                try:
                    async with session.get(f"{base_url}/docs", timeout=30) as response:
                        if response.status == 200:
                            self.print_result("API文档端点", True, "文档可访问")
                        else:
                            self.print_result("API文档端点", False, f"HTTP {response.status}")
                except Exception as e:
                    self.print_result("API文档端点", False, f"请求失败: {e}")

                return True

        except Exception as e:
            self.print_result("容器健康检查", False, f"测试失败: {e}")
            return False

    def test_container_logs(self):
        """测试容器日志"""
        self.print_section("测试容器日志")

        try:
            # 获取容器日志
            result = subprocess.run([
                'docker', 'logs', '--tail', '50', self.container_name
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                logs = result.stdout

                # 检查关键日志信息
                if "监控服务初始化完成" in logs or "Application startup complete" in logs:
                    self.print_result("应用启动日志", True, "发现启动成功日志")
                else:
                    self.print_result("应用启动日志", False, "未发现启动成功日志")

                # 检查错误日志
                error_lines = [line for line in logs.split('\n') if 'ERROR' in line.upper()]
                if error_lines:
                    self.print_result("错误日志检查", False, f"发现 {len(error_lines)} 个错误")
                    for error in error_lines[:3]:  # 只显示前3个错误
                        print(f"    {error}")
                else:
                    self.print_result("错误日志检查", True, "无严重错误")

                return True
            else:
                self.print_result("容器日志获取", False, f"获取失败: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.print_result("容器日志获取", False, "获取超时")
            return False
        except Exception as e:
            self.print_result("容器日志获取", False, f"获取异常: {e}")
            return False

    def test_docker_compose(self):
        """测试docker-compose部署"""
        self.print_section("测试docker-compose部署")

        if not os.path.exists('docker-compose.yml'):
            self.print_result("docker-compose文件", False, "docker-compose.yml不存在")
            return False

        try:
            # 检查docker-compose语法
            result = subprocess.run([
                'docker-compose', 'config'
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                self.print_result("docker-compose配置", True, "配置文件语法正确")

                # 尝试验证服务定义
                if 'monitoring' in result.stdout or 'app' in result.stdout:
                    self.print_result("服务定义检查", True, "发现监控服务定义")
                else:
                    self.print_result("服务定义检查", False, "未发现监控服务定义")

                return True
            else:
                self.print_result("docker-compose配置", False, f"配置错误: {result.stderr}")
                return False

        except FileNotFoundError:
            self.print_result("docker-compose工具", False, "docker-compose未安装")
            return False
        except subprocess.TimeoutExpired:
            self.print_result("docker-compose配置", False, "检查超时")
            return False
        except Exception as e:
            self.print_result("docker-compose配置", False, f"检查异常: {e}")
            return False

    def cleanup_containers(self):
        """清理测试容器"""
        self.print_section("清理测试资源")

        try:
            # 停止容器
            result = subprocess.run(['docker', 'stop', self.container_name],
                                  capture_output=True, timeout=30)
            if result.returncode == 0:
                self.print_result("停止容器", True)
            else:
                self.print_result("停止容器", False, "容器可能已停止")

            # 删除容器
            result = subprocess.run(['docker', 'rm', self.container_name],
                                  capture_output=True, timeout=30)
            if result.returncode == 0:
                self.print_result("删除容器", True)
            else:
                self.print_result("删除容器", False, "容器可能已删除")

            # 删除镜像
            result = subprocess.run(['docker', 'rmi', self.image_name],
                                  capture_output=True, timeout=30)
            if result.returncode == 0:
                self.print_result("删除镜像", True)
            else:
                self.print_result("删除镜像", False, "镜像可能已删除")

        except Exception as e:
            self.print_result("资源清理", False, f"清理异常: {e}")

    def generate_summary(self):
        """生成测试摘要"""
        total_tests = len(self.test_results["tests"])
        passed_tests = sum(1 for test in self.test_results["tests"].values() if test["passed"])
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": pass_rate,
            "status": "优秀" if pass_rate >= 90 else "良好" if pass_rate >= 70 else "需要改进"
        }

        return self.test_results["summary"]

    def print_summary(self):
        """打印测试摘要"""
        summary = self.generate_summary()

        self.print_header("Docker部署验证测试报告")
        print(f"测试时间: {self.test_results['timestamp']}")
        print(f"📋 测试结果:")

        for test_name, result in self.test_results["tests"].items():
            status = "✅ 通过" if result["passed"] else "❌ 失败"
            print(f"  {test_name}: {status}")
            if result["details"]:
                print(f"    {result['details']}")

        print(f"\n🎯 总体评估:")
        print(f"  总测试数: {summary['total_tests']}")
        print(f"  通过测试: {summary['passed_tests']}")
        print(f"  失败测试: {summary['failed_tests']}")
        print(f"  通过率: {summary['pass_rate']:.1f}%")
        print(f"  评级: {summary['status']}")

        # 保存测试结果
        with open("docker_deployment_test_results.json", "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)

        print(f"\n📄 测试结果已保存到: docker_deployment_test_results.json")

    async def run_all_tests(self):
        """运行所有测试"""
        print("🐳 开始Docker部署验证测试...")

        # 基础环境测试
        if not self.test_docker_availability():
            print("⚠️  Docker环境不可用，跳过后续测试")
            self.print_summary()
            return

        # Docker镜像构建测试
        if not self.test_dockerfile_build():
            print("⚠️  Docker镜像构建失败，跳过后续测试")
            self.cleanup_containers()
            self.print_summary()
            return

        # 容器启动测试
        if not self.test_container_startup():
            print("⚠️  容器启动失败，跳过后续测试")
            self.cleanup_containers()
            self.print_summary()
            return

        # 容器健康测试
        await self.test_container_health()

        # 容器日志测试
        self.test_container_logs()

        # docker-compose测试
        self.test_docker_compose()

        # 清理资源
        self.cleanup_containers()

        self.print_summary()


async def main():
    """主函数"""
    tester = DockerDeploymentTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest

#!/usr/bin/env python3
"""
监控数据导出和报告功能测试

测试导出服务的各项功能，包括数据导出、报告生成、邮件发送等

作者: StockSchool Team
创建时间: 2025-01-02
"""


# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 导入测试模块
try:
    from src.services.export_service import (
        DataExporter,
        EmailService,
        ExportConfig,
        ExportService,
        PDFReportGenerator,
        ReportGenerator,
        ScheduledReportService,
        create_export_service,
    )
    from src.services.monitoring_service import MonitoringService
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    pytest.skip(f"跳过测试，导入模块失败: {e}")


class MockMonitoringService:
    """模拟监控服务"""

    async def get_system_health(self) -> Dict[str, Any]:
        """模拟系统健康数据"""
        return {
            "cpu_usage": 75.5,
            "memory_usage": 68.2,
            "disk_usage": 45.8,
            "database_status": "connected",
            "redis_status": "connected",
            "celery_status": "running",
            "api_status": "healthy",
        }

    async def query_metrics_with_cache(
        self, metric_names: List[str], start_time: datetime, end_time: datetime, **kwargs
    ) -> List[Dict[str, Any]]:
        """模拟指标数据"""
        metrics_data = []

        # 生成模拟数据
        current_time = start_time
        while current_time <= end_time:
            for metric_name in metric_names:
                if metric_name == "cpu_usage":
                    value = 70 + (hash(str(current_time)) % 20)
                elif metric_name == "memory_usage":
                    value = 60 + (hash(str(current_time)) % 25)
                elif metric_name == "disk_usage":
                    value = 40 + (hash(str(current_time)) % 15)
                else:
                    value = 50 + (hash(str(current_time)) % 30)

                metrics_data.append(
                    {
                        "timestamp": current_time,
                        "metric_name": metric_name,
                        "metric_value": value,
                        "metric_type": "gauge",
                        "metric_unit": "%",
                        "source_component": "system",
                        "labels": {"host": "localhost"},
                    }
                )

            current_time += timedelta(minutes=5)

        return metrics_data

    async def get_active_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """模拟告警数据"""
        return [
            {
                "alert_id": "alert_001",
                "severity": "warning",
                "title": "CPU使用率过高",
                "description": "CPU使用率超过80%",
                "created_at": datetime.now() - timedelta(hours=2),
                "status": "active",
            },
            {
                "alert_id": "alert_002",
                "severity": "critical",
                "title": "内存使用率过高",
                "description": "内存使用率超过90%",
                "created_at": datetime.now() - timedelta(hours=1),
                "status": "active",
            },
        ]


class ExportFunctionalityTest:
    """导出功能测试类"""

    def __init__(self):
        """方法描述"""

        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp(prefix="export_test_")
        self.logger.info(f"测试临时目录: {self.temp_dir}")

        # 创建测试配置
        self.config = ExportConfig(
            export_dir=os.path.join(self.temp_dir, "exports"),
            temp_dir=os.path.join(self.temp_dir, "temp"),
            max_file_size_mb=10,
            max_records_per_export=1000,
            # 邮件配置（测试时不发送真实邮件）
            smtp_host="localhost",
            smtp_port=587,
            smtp_username="test@example.com",
            smtp_password="test_password",
            smtp_use_tls=False,
        )

        # 创建模拟监控服务
        self.mock_monitoring_service = MockMonitoringService()

        # 测试结果
        self.test_results = {
            "csv_export": False,
            "excel_export": False,
            "json_export": False,
            "pdf_report": False,
            "email_service": False,
            "scheduled_report": False,
            "data_integrity": False,
            "format_validation": False,
        }

    async def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        self.logger.info("🚀 开始导出功能测试...")

        try:
            # 测试数据导出
            await self.test_csv_export()
            await self.test_excel_export()
            await self.test_json_export()

            # 测试报告生成
            await self.test_pdf_report_generation()

            # 测试邮件服务
            await self.test_email_service()

            # 测试定时报告
            await self.test_scheduled_report()

            # 测试数据完整性
            await self.test_data_integrity()

            # 测试格式验证
            await self.test_format_validation()

        except Exception as e:
            self.logger.error(f"测试过程中发生错误: {e}")

        # 输出测试结果
        self.print_test_results()

        return self.test_results

    async def test_csv_export(self):
        """测试CSV导出功能"""
        self.logger.info("📊 测试CSV导出功能...")

        try:
            # 创建导出服务
            export_service = await create_export_service(
                config=self.config, monitoring_service=self.mock_monitoring_service
            )

            # 准备测试数据
            start_time = datetime.now() - timedelta(hours=1)
            end_time = datetime.now()

            # 导出CSV数据
            file_path = await export_service.export_monitoring_data(
                format_type="csv",
                start_time=start_time,
                end_time=end_time,
                metric_names=["cpu_usage", "memory_usage"],
                filename="test_csv_export",
            )

            # 验证文件是否存在
            if Path(file_path).exists():
                # 验证文件内容
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "cpu_usage" in content and "memory_usage" in content:
                        self.test_results["csv_export"] = True
                        self.logger.info("✅ CSV导出测试通过")
                    else:
                        self.logger.error("❌ CSV文件内容验证失败")
            else:
                self.logger.error("❌ CSV文件未生成")

        except Exception as e:
            self.logger.error(f"❌ CSV导出测试失败: {e}")

    async def test_excel_export(self):
        """测试Excel导出功能"""
        self.logger.info("📈 测试Excel导出功能...")

        try:
            # 检查Excel支持
            try:
                import openpyxl
                import pandas as pd

                excel_available = True
            except ImportError:
                excel_available = False
                self.logger.warning("⚠️ Excel依赖不可用，跳过Excel导出测试")
                return

            if excel_available:
                # 创建导出服务
                export_service = await create_export_service(
                    config=self.config, monitoring_service=self.mock_monitoring_service
                )

                # 准备测试数据
                start_time = datetime.now() - timedelta(hours=1)
                end_time = datetime.now()

                # 导出Excel数据
                file_path = await export_service.export_monitoring_data(
                    format_type="excel",
                    start_time=start_time,
                    end_time=end_time,
                    metric_names=["cpu_usage", "memory_usage"],
                    filename="test_excel_export",
                )

                # 验证文件是否存在
                if Path(file_path).exists():
                    self.test_results["excel_export"] = True
                    self.logger.info("✅ Excel导出测试通过")
                else:
                    self.logger.error("❌ Excel文件未生成")

        except Exception as e:
            self.logger.error(f"❌ Excel导出测试失败: {e}")

    async def test_json_export(self):
        """测试JSON导出功能"""
        self.logger.info("📄 测试JSON导出功能...")

        try:
            # 创建导出服务
            export_service = await create_export_service(
                config=self.config, monitoring_service=self.mock_monitoring_service
            )

            # 准备测试数据
            start_time = datetime.now() - timedelta(hours=1)
            end_time = datetime.now()

            # 导出JSON数据
            file_path = await export_service.export_monitoring_data(
                format_type="json",
                start_time=start_time,
                end_time=end_time,
                metric_names=["cpu_usage", "memory_usage"],
                filename="test_json_export",
            )

            # 验证文件是否存在
            if Path(file_path).exists():
                # 验证JSON格式
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            self.test_results["json_export"] = True
                            self.logger.info("✅ JSON导出测试通过")
                        else:
                            self.logger.error("❌ JSON数据格式验证失败")
                    except json.JSONDecodeError:
                        self.logger.error("❌ JSON文件格式无效")
            else:
                self.logger.error("❌ JSON文件未生成")

        except Exception as e:
            self.logger.error(f"❌ JSON导出测试失败: {e}")

    async def test_pdf_report_generation(self):
        """测试PDF报告生成功能"""
        self.logger.info("📋 测试PDF报告生成功能...")

        try:
            # 检查PDF支持
            try:
                from reportlab.lib import colors

                pdf_available = True
            except ImportError:
                pdf_available = False
                self.logger.warning("⚠️ PDF依赖不可用，跳过PDF报告测试")
                return

            if pdf_available:
                # 创建导出服务
                export_service = await create_export_service(
                    config=self.config, monitoring_service=self.mock_monitoring_service
                )

                # 准备测试数据
                start_time = datetime.now() - timedelta(hours=1)
                end_time = datetime.now()

                # 生成PDF报告
                file_path = await export_service.generate_report(
                    report_type="system_health",
                    start_time=start_time,
                    end_time=end_time,
                    format_type="pdf",
                    filename="test_pdf_report",
                )

                # 验证文件是否存在
                if Path(file_path).exists():
                    # 检查文件大小（PDF文件应该有一定大小）
                    file_size = Path(file_path).stat().st_size
                    if file_size > 1000:  # 至少1KB
                        self.test_results["pdf_report"] = True
                        self.logger.info("✅ PDF报告生成测试通过")
                    else:
                        self.logger.error("❌ PDF文件大小异常")
                else:
                    self.logger.error("❌ PDF文件未生成")

        except Exception as e:
            self.logger.error(f"❌ PDF报告生成测试失败: {e}")

    async def test_email_service(self):
        """测试邮件服务功能"""
        self.logger.info("📧 测试邮件服务功能...")

        try:
            # 创建邮件服务（使用模拟配置）
            email_service = EmailService(self.config)

            # 准备测试报告数据
            report_data = {
                "report_info": {
                    "title": "测试报告",
                    "generated_at": datetime.now(),
                    "period": {"start_time": datetime.now() - timedelta(hours=1), "end_time": datetime.now()},
                },
                "summary": {
                    "overall_status": "healthy",
                    "total_alerts": 2,
                    "critical_alerts": 1,
                    "avg_cpu_usage": 75.5,
                    "avg_memory_usage": 68.2,
                    "avg_disk_usage": 45.8,
                },
                "recommendations": ["建议优化CPU使用率", "建议检查内存使用情况"],
            }

            # 测试邮件正文生成
            email_body = email_service._generate_email_body(report_data)

            if "测试报告" in email_body and "healthy" in email_body:
                self.test_results["email_service"] = True
                self.logger.info("✅ 邮件服务测试通过（邮件正文生成）")
            else:
                self.logger.error("❌ 邮件正文生成失败")

            # 注意：这里不测试实际的邮件发送，因为需要真实的SMTP配置

        except Exception as e:
            self.logger.error(f"❌ 邮件服务测试失败: {e}")

    async def test_scheduled_report(self):
        """测试定时报告功能"""
        self.logger.info("⏰ 测试定时报告功能...")

        try:
            # 创建定时报告服务
            scheduled_service = ScheduledReportService(self.config, self.mock_monitoring_service)

            # 测试报告数据生成（不实际发送邮件）
            start_time = datetime.now() - timedelta(days=1)
            end_time = datetime.now()

            # 生成报告数据
            report_data = await scheduled_service.report_generator.generate_system_health_report(
                start_time, end_time, self.mock_monitoring_service
            )

            # 验证报告数据结构
            required_keys = ["report_info", "summary", "health_data", "metrics_data", "alerts_data"]
            if all(key in report_data for key in required_keys):
                self.test_results["scheduled_report"] = True
                self.logger.info("✅ 定时报告测试通过（报告数据生成）")
            else:
                self.logger.error("❌ 报告数据结构验证失败")

        except Exception as e:
            self.logger.error(f"❌ 定时报告测试失败: {e}")

    async def test_data_integrity(self):
        """测试数据完整性"""
        self.logger.info("🔍 测试数据完整性...")

        try:
            # 创建导出服务
            export_service = await create_export_service(
                config=self.config, monitoring_service=self.mock_monitoring_service
            )

            # 准备测试数据
            start_time = datetime.now() - timedelta(hours=1)
            end_time = datetime.now()
            metric_names = ["cpu_usage", "memory_usage"]

            # 获取原始数据
            original_data = await self.mock_monitoring_service.query_metrics_with_cache(
                metric_names=metric_names, start_time=start_time, end_time=end_time
            )

            # 导出JSON数据
            json_file_path = await export_service.export_monitoring_data(
                format_type="json",
                start_time=start_time,
                end_time=end_time,
                metric_names=metric_names,
                filename="integrity_test",
            )

            # 读取导出的数据
            with open(json_file_path, "r", encoding="utf-8") as f:
                exported_data = json.load(f)

            # 比较数据完整性
            if len(original_data) == len(exported_data):
                # 检查关键字段
                integrity_check = True
                for i, (orig, exp) in enumerate(zip(original_data, exported_data)):
                    if orig["metric_name"] != exp["metric_name"] or orig["metric_value"] != exp["metric_value"]:
                        integrity_check = False
                        break

                if integrity_check:
                    self.test_results["data_integrity"] = True
                    self.logger.info("✅ 数据完整性测试通过")
                else:
                    self.logger.error("❌ 数据内容不匹配")
            else:
                self.logger.error(f"❌ 数据数量不匹配: 原始{len(original_data)}, 导出{len(exported_data)}")

        except Exception as e:
            self.logger.error(f"❌ 数据完整性测试失败: {e}")

    async def test_format_validation(self):
        """测试格式验证"""
        self.logger.info("✅ 测试格式验证...")

        try:
            # 创建导出服务
            export_service = await create_export_service(
                config=self.config, monitoring_service=self.mock_monitoring_service
            )

            # 测试无效格式
            try:
                await export_service.export_monitoring_data(
                    format_type="invalid_format",
                    start_time=datetime.now() - timedelta(hours=1),
                    end_time=datetime.now(),
                    metric_names=["cpu_usage"],
                )
                self.logger.error("❌ 格式验证失败：应该拒绝无效格式")
            except ValueError as e:
                if "不支持的导出格式" in str(e):
                    self.test_results["format_validation"] = True
                    self.logger.info("✅ 格式验证测试通过")
                else:
                    self.logger.error(f"❌ 格式验证错误信息不正确: {e}")

        except Exception as e:
            self.logger.error(f"❌ 格式验证测试失败: {e}")

    def print_test_results(self):
        """打印测试结果"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("📊 导出功能测试结果汇总")
        self.logger.info("=" * 50)

        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())

        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            self.logger.info(f"{test_name.ljust(20)}: {status}")

        self.logger.info("-" * 50)
        self.logger.info(f"总测试数: {total_tests}")
        self.logger.info(f"通过数量: {passed_tests}")
        self.logger.info(f"失败数量: {total_tests - passed_tests}")
        self.logger.info(f"通过率: {(passed_tests/total_tests)*100:.1f}%")

        if passed_tests == total_tests:
            self.logger.info("🎉 所有测试通过！")
        else:
            self.logger.warning("⚠️ 部分测试失败，请检查相关功能")

    def cleanup(self):
        """清理测试文件"""
        try:
            import shutil

            shutil.rmtree(self.temp_dir)
            self.logger.info(f"清理测试目录: {self.temp_dir}")
        except Exception as e:
            self.logger.warning(f"清理测试目录失败: {e}")


async def main():
    """主函数"""
    print("🚀 开始监控数据导出和报告功能测试...")

    # 创建测试实例
    test_runner = ExportFunctionalityTest()

    try:
        # 运行所有测试
        results = await test_runner.run_all_tests()

        # 检查测试结果
        total_tests = len(results)
        passed_tests = sum(results.values())

        print(f"\n📊 测试完成！通过率: {(passed_tests/total_tests)*100:.1f}%")

        if passed_tests == total_tests:
            print("🎉 任务20：监控数据导出和报告功能 - 实现完成！")
            return True
        else:
            print("⚠️ 部分功能需要进一步完善")
            return False

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return False
    finally:
        # 清理测试文件
        test_runner.cleanup()


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

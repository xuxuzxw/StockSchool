import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import schedule
from loguru import logger

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量监控调度器

负责定期执行数据质量检查任务，包括：
- 定时数据质量检查
- 质量报告生成
- 告警处理
- 历史数据清理

作者: StockSchool Team
创建时间: 2025-01-02
"""


from .alerts import AlertManager
from .config import get_monitoring_config
from .data_quality import DataQualityMonitor, QualityReport
from .notifications import NotificationManager


class DataQualityScheduler:
    """数据质量监控调度器"""

    def __init__(self):
        """初始化调度器"""
        self.config = get_monitoring_config()
        self.monitor = DataQualityMonitor()
        self.alert_manager = AlertManager()
        self.notification_manager = NotificationManager()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False

        # 监控的表列表
        self.monitored_tables = [
            "stock_basic_info",
            "stock_daily_data",
            "stock_factors",
            "market_data",
            "financial_data",
            "news_data",
        ]

        logger.info("数据质量调度器初始化完成")

    def start(self):
        """启动调度器"""
        if self.running:
            logger.warning("调度器已在运行中")
            return

        self.running = True

        # 设置定时任务
        self._setup_schedules()

        logger.info("数据质量调度器已启动")

        # 运行调度循环
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到停止信号")
        finally:
            self.stop()

    def stop(self):
        """停止调度器"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("数据质量调度器已停止")

    def _setup_schedules(self):
        """设置定时任务"""
        # 数据质量检查任务（每5分钟执行一次）
        schedule.every(self.config.data_quality_check_interval).seconds.do(self._run_quality_check_job)

        # 质量报告生成任务（每小时执行一次）
        schedule.every().hour.do(self._run_report_generation_job)

        # 历史数据清理任务（每天凌晨2点执行）
        schedule.every().day.at("02:00").do(self._run_cleanup_job)

        # 质量趋势分析任务（每天上午8点执行）
        schedule.every().day.at("08:00").do(self._run_trend_analysis_job)

        logger.info("定时任务设置完成")

    def _run_quality_check_job(self):
        """执行数据质量检查任务"""
        logger.info("开始执行数据质量检查任务")

        try:
            # 使用线程池异步执行检查
            future = self.executor.submit(self._async_quality_check)
            future.result(timeout=300)  # 5分钟超时

        except Exception as e:
            logger.error(f"数据质量检查任务执行失败: {e}")

    def _async_quality_check(self):
        """异步执行数据质量检查"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._check_all_tables())
        finally:
            loop.close()

    async def _check_all_tables(self):
        """检查所有监控表的数据质量"""
        for table_name in self.monitored_tables:
            try:
                logger.info(f"检查表 {table_name} 的数据质量")

                # 执行质量检查
                results = await self.monitor.check_data_quality(table_name)

                if results:
                    # 保存检查结果
                    await self.monitor.save_check_results(results)

                    # 创建告警
                    alerts = await self.monitor.create_quality_alerts(results)

                    if alerts:
                        # 发送告警
                        for alert in alerts:
                            await self.alert_manager.create_alert(alert)
                            await self.notification_manager.send_alert_notification(alert)

                        logger.warning(f"表 {table_name} 发现 {len(alerts)} 个数据质量问题")
                    else:
                        logger.info(f"表 {table_name} 数据质量正常")
                else:
                    logger.warning(f"表 {table_name} 没有配置质量检查规则")

            except Exception as e:
                logger.error(f"检查表 {table_name} 时发生错误: {e}")

    def _run_report_generation_job(self):
        """执行质量报告生成任务"""
        logger.info("开始生成数据质量报告")

        try:
            future = self.executor.submit(self._async_report_generation)
            future.result(timeout=180)  # 3分钟超时

        except Exception as e:
            logger.error(f"质量报告生成任务执行失败: {e}")

    def _async_report_generation(self):
        """异步生成质量报告"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._generate_all_reports())
        finally:
            loop.close()

    async def _generate_all_reports(self):
        """生成所有表的质量报告"""
        reports = []

        for table_name in self.monitored_tables:
            try:
                logger.info(f"生成表 {table_name} 的质量报告")

                report = await self.monitor.generate_quality_report(table_name)

                if report:
                    reports.append(report)

                    # 保存报告到数据库
                    await self.monitor.save_quality_report(report)

                    logger.info(f"表 {table_name} 质量报告生成完成，总分: {report.overall_score:.2f}")

            except Exception as e:
                logger.error(f"生成表 {table_name} 质量报告时发生错误: {e}")

        # 生成汇总报告
        if reports:
            await self._generate_summary_report(reports)

    async def _generate_summary_report(self, reports: List[QualityReport]):
        """生成汇总质量报告"""
        try:
            total_score = sum(r.overall_score for r in reports) / len(reports)
            total_issues = sum(sum(r.issues_count.values()) for r in reports)

            summary = {
                "timestamp": datetime.now(),
                "total_tables": len(reports),
                "average_score": total_score,
                "total_issues": total_issues,
                "table_scores": {r.table_name: r.overall_score for r in reports},
            }

            logger.info(f"数据质量汇总报告: 平均分数 {total_score:.2f}, 总问题数 {total_issues}")

            # 如果总体质量分数过低，发送汇总告警
            if total_score < self.config.thresholds.get("data_quality_score", 80.0):
                await self._send_summary_alert(summary)

        except Exception as e:
            logger.error(f"生成汇总报告时发生错误: {e}")

    async def _send_summary_alert(self, summary: Dict[str, Any]):
        """发送汇总告警"""
        try:
            from .alerts import Alert, AlertSeverity, AlertStatus, AlertType

            alert = Alert(
                rule_id="data_quality_summary",
                title="数据质量汇总告警",
                message=f"系统整体数据质量分数过低: {summary['average_score']:.2f}, "
                f"共发现 {summary['total_issues']} 个问题",
                severity=AlertSeverity.WARNING if summary["average_score"] > 60 else AlertSeverity.CRITICAL,
                alert_type=AlertType.DATA_QUALITY,
                status=AlertStatus.ACTIVE,
                metadata=summary,
                tags=["data_quality", "summary", "system_wide"],
            )

            await self.alert_manager.create_alert(alert)
            await self.notification_manager.send_alert_notification(alert)

            logger.warning(f"已发送数据质量汇总告警: {alert.title}")

        except Exception as e:
            logger.error(f"发送汇总告警时发生错误: {e}")

    def _run_cleanup_job(self):
        """执行历史数据清理任务"""
        logger.info("开始清理历史数据")

        try:
            future = self.executor.submit(self._async_cleanup)
            future.result(timeout=300)  # 5分钟超时

        except Exception as e:
            logger.error(f"历史数据清理任务执行失败: {e}")

    def _async_cleanup(self):
        """异步清理历史数据"""
        try:
            # 清理过期的质量检查结果
            cutoff_date = datetime.now() - timedelta(days=self.config.data_quality_retention_days)

            with self.monitor.engine.connect() as conn:
                # 清理质量检查结果
                result = conn.execute("DELETE FROM data_quality_results WHERE timestamp < ?", (cutoff_date,))
                deleted_results = result.rowcount

                # 清理质量报告
                result = conn.execute("DELETE FROM data_quality_reports WHERE timestamp < ?", (cutoff_date,))
                deleted_reports = result.rowcount

                conn.commit()

            logger.info(f"历史数据清理完成: 删除 {deleted_results} 条检查结果, {deleted_reports} 条报告")

        except Exception as e:
            logger.error(f"清理历史数据时发生错误: {e}")

    def _run_trend_analysis_job(self):
        """执行质量趋势分析任务"""
        logger.info("开始执行质量趋势分析")

        try:
            future = self.executor.submit(self._async_trend_analysis)
            future.result(timeout=180)  # 3分钟超时

        except Exception as e:
            logger.error(f"质量趋势分析任务执行失败: {e}")

    def _async_trend_analysis(self):
        """异步执行趋势分析"""
        try:
            for table_name in self.monitored_tables:
                # 获取最近7天的质量历史
                history = self.monitor.get_quality_history(table_name, days=7)

                if len(history) >= 2:
                    # 分析趋势
                    latest_score = history[0]["overall_score"]
                    previous_score = history[1]["overall_score"]
                    trend = latest_score - previous_score

                    if trend < -10:  # 质量分数下降超过10分
                        logger.warning(f"表 {table_name} 数据质量呈下降趋势: {trend:.2f}")
                        # 可以在这里发送趋势告警
                    elif trend > 10:  # 质量分数提升超过10分
                        logger.info(f"表 {table_name} 数据质量呈改善趋势: {trend:.2f}")

        except Exception as e:
            logger.error(f"趋势分析时发生错误: {e}")

    def add_monitored_table(self, table_name: str):
        """添加监控表"""
        if table_name not in self.monitored_tables:
            self.monitored_tables.append(table_name)
            logger.info(f"已添加监控表: {table_name}")

    def remove_monitored_table(self, table_name: str):
        """移除监控表"""
        if table_name in self.monitored_tables:
            self.monitored_tables.remove(table_name)
            logger.info(f"已移除监控表: {table_name}")

    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        return {
            "running": self.running,
            "monitored_tables": self.monitored_tables,
            "next_run_time": schedule.next_run(),
            "scheduled_jobs": len(schedule.jobs),
            "config": {
                "check_interval": self.config.data_quality_check_interval,
                "retention_days": self.config.data_quality_retention_days,
                "batch_size": self.config.data_quality_batch_size,
            },
        }


def main():
    """主函数"""
    scheduler = DataQualityScheduler()

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("收到停止信号")
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同步监控器使用示例

演示如何使用同步监控仪表板的各项功能
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.sync_monitor import (
    SyncMonitor, SyncTaskInfo, SyncTaskStatus, get_sync_monitor
)

# 获取同步监控器实例
sync_monitor = get_sync_monitor()


def example_basic_monitoring():
    """基础监控功能示例"""
    print("=== 基础监控功能示例 ===")
    
    # 创建任务信息
    task_info = SyncTaskInfo(
        task_id="example_task_001",
        data_source="tushare",
        data_type="daily",
        target_date="2024-01-01",
        status=SyncTaskStatus.PENDING,
        priority=1
    )
    
    # 开始监控
    sync_monitor.start_task_monitoring(task_info)
    print(f"开始监控任务: {task_info.task_id}")
    
    # 模拟任务执行过程
    for progress in [20, 40, 60, 80, 100]:
        time.sleep(1)  # 模拟处理时间
        
        sync_monitor.update_task_progress(
            task_id=task_info.task_id,
            progress=progress,
            records_processed=progress * 10,
            records_failed=progress // 10
        )
        print(f"任务进度: {progress}%, 处理记录: {progress * 10}, 失败记录: {progress // 10}")
    
    # 完成任务
    sync_monitor.complete_task_monitoring(
        task_id=task_info.task_id,
        success=True,
        final_stats={
            'records_processed': 1000,
            'records_failed': 10,
            'processing_rate': 100.5
        }
    )
    print("任务监控完成")


def example_real_time_dashboard():
    """实时仪表板示例"""
    print("\n=== 实时仪表板示例 ===")
    
    # 创建多个任务进行监控
    tasks = [
        SyncTaskInfo(
            task_id=f"dashboard_task_{i:03d}",
            data_source="tushare" if i % 2 == 0 else "akshare",
            data_type="daily" if i % 3 == 0 else "sentiment",
            target_date="2024-01-01",
            status=SyncTaskStatus.PENDING,
            priority=i % 3 + 1
        )
        for i in range(5)
    ]
    
    # 开始监控所有任务
    for task in tasks:
        sync_monitor.start_task_monitoring(task)
    
    # 获取实时仪表板数据
    dashboard = sync_monitor.get_real_time_dashboard()
    
    print("实时仪表板数据:")
    print(f"  活跃任务数: {dashboard['summary']['total_active']}")
    print(f"  运行中任务: {dashboard['summary']['running_tasks']}")
    print(f"  平均进度: {dashboard['summary']['avg_progress']:.1f}%")
    print(f"  数据源分布: {dashboard['data_sources']}")
    print(f"  数据类型分布: {dashboard['data_types']}")
    print(f"  最近事件数: {len(dashboard['recent_events'])}")
    
    # 模拟部分任务完成
    for i, task in enumerate(tasks[:3]):
        sync_monitor.complete_task_monitoring(
            task_id=task.task_id,
            success=i % 2 == 0,  # 部分成功，部分失败
            error_message="模拟错误" if i % 2 != 0 else None
        )
    
    # 再次获取仪表板数据
    dashboard = sync_monitor.get_real_time_dashboard()
    print(f"\n更新后活跃任务数: {dashboard['summary']['total_active']}")


def example_event_subscription():
    """事件订阅示例"""
    print("\n=== 事件订阅示例 ===")
    
    # 定义事件处理函数
    def event_handler(event):
        print(f"收到事件: {event.event_type.value} - 任务: {event.task_id}")
        if event.event_type.value == 'task_progress':
            print(f"  进度: {event.data.get('progress', 0)}%")
        elif event.event_type.value == 'task_completed':
            print(f"  成功: {event.data.get('success', False)}")
        elif event.event_type.value == 'task_failed':
            print(f"  错误: {event.data.get('error_message', 'Unknown')}")
    
    # 订阅事件
    sync_monitor.subscribe_events(event_handler)
    
    # 创建并监控任务
    task_info = SyncTaskInfo(
        task_id="event_example_task",
        data_source="akshare",
        data_type="sentiment",
        target_date="2024-01-01",
        status=SyncTaskStatus.PENDING,
        priority=2
    )
    
    sync_monitor.start_task_monitoring(task_info)
    time.sleep(0.5)
    
    sync_monitor.update_task_progress(task_info.task_id, 50.0, 500, 5)
    time.sleep(0.5)
    
    sync_monitor.complete_task_monitoring(task_info.task_id, success=True)
    time.sleep(0.5)
    
    # 取消订阅
    sync_monitor.unsubscribe_events(event_handler)
    print("事件订阅示例完成")


def example_historical_analysis():
    """历史记录分析示例"""
    print("\n=== 历史记录分析示例 ===")
    
    try:
        # 获取同步历史记录
        history = sync_monitor.get_sync_history(
            start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            limit=10
        )
        
        print(f"历史记录查询结果:")
        print(f"  总记录数: {history['pagination']['total']}")
        print(f"  返回记录数: {len(history['records'])}")
        
        if history['records']:
            print("  最近的记录:")
            for record in history['records'][:3]:
                print(f"    任务: {record['task_id']}, 状态: {record['status']}, "
                      f"数据源: {record['data_source']}")
        
        # 获取错误日志分析
        error_analysis = sync_monitor.get_error_log_analysis(
            start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        print(f"\n错误分析结果:")
        print(f"  错误统计条目: {len(error_analysis['error_statistics'])}")
        print(f"  错误分类条目: {len(error_analysis['error_classification'])}")
        print(f"  错误趋势条目: {len(error_analysis['error_trend'])}")
        
        # 获取同步摘要报告
        summary_report = sync_monitor.get_sync_summary_report(
            start_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        print(f"\n摘要报告:")
        overall_stats = summary_report.get('overall_statistics', {})
        print(f"  总任务数: {overall_stats.get('total_tasks', 0)}")
        print(f"  成功任务数: {overall_stats.get('successful_tasks', 0)}")
        print(f"  失败任务数: {overall_stats.get('failed_tasks', 0)}")
        print(f"  成功率: {overall_stats.get('success_rate', 0):.1f}%")
        
    except Exception as e:
        print(f"历史分析示例执行出错: {e}")


def example_performance_analysis():
    """性能分析示例"""
    print("\n=== 性能分析示例 ===")
    
    try:
        # 计算性能指标
        metrics = sync_monitor.calculate_performance_metrics(
            start_date=(datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S'),
            end_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        if 'error' not in metrics:
            print("性能指标:")
            print(f"  总任务数: {metrics.get('total_tasks', 0)}")
            print(f"  成功率: {metrics.get('success_rate', 0):.1f}%")
            print(f"  失败率: {metrics.get('failure_rate', 0):.1f}%")
            print(f"  平均耗时: {metrics.get('avg_duration_seconds', 0):.2f}秒")
            print(f"  吞吐量: {metrics.get('throughput_tasks_per_hour', 0):.1f} 任务/小时")
            print(f"  数据质量率: {metrics.get('data_quality_rate', 0):.1f}%")
        else:
            print(f"性能指标计算失败: {metrics['error']}")
        
        # 分析性能趋势
        trend_analysis = sync_monitor.analyze_performance_trends(days=3, granularity='hour')
        
        if 'error' not in trend_analysis:
            print(f"\n趋势分析:")
            print(f"  分析周期: {trend_analysis['analysis_period']['days']} 天")
            print(f"  数据点数: {len(trend_analysis['trend_data'])}")
            
            trend_info = trend_analysis.get('trend_analysis', {})
            print(f"  成功率趋势: {trend_info.get('success_rate_trend', 'unknown')}")
            print(f"  耗时趋势: {trend_info.get('duration_trend', 'unknown')}")
            print(f"  任务量趋势: {trend_info.get('volume_trend', 'unknown')}")
        else:
            print(f"趋势分析失败: {trend_analysis['error']}")
        
        # 生成性能预测
        forecast = sync_monitor.generate_performance_forecast(forecast_hours=6)
        
        if 'error' not in forecast:
            print(f"\n性能预测:")
            summary = forecast.get('forecast_summary', {})
            print(f"  预测周期: {summary.get('forecast_period_hours', 0)} 小时")
            print(f"  预测平均成功率: {summary.get('avg_predicted_success_rate', 0):.1f}%")
            print(f"  预测平均耗时: {summary.get('avg_predicted_duration', 0):.2f}秒")
            print(f"  预测总任务数: {summary.get('total_predicted_tasks', 0)}")
            print(f"  预测可靠性: {summary.get('forecast_reliability', 'unknown')}")
        else:
            print(f"性能预测失败: {forecast['error']}")
        
        # 获取性能告警
        alerts = sync_monitor.get_performance_alerts()
        
        print(f"\n性能告警:")
        if alerts:
            for alert in alerts:
                print(f"  {alert['alert_type']}: {alert['message']} (严重程度: {alert['severity']})")
        else:
            print("  无性能告警")
        
    except Exception as e:
        print(f"性能分析示例执行出错: {e}")


def example_report_export():
    """报告导出示例"""
    print("\n=== 报告导出示例 ===")
    
    try:
        # 导出JSON格式报告
        json_report = sync_monitor.export_performance_report(
            start_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            format='json'
        )
        
        if json_report['success']:
            print("JSON报告导出成功:")
            print(f"  报告格式: {json_report['export_format']}")
            print(f"  包含数据: {list(json_report['data'].keys())}")
            
            # 保存报告到文件
            report_filename = f"sync_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(json_report['data'], f, indent=2, ensure_ascii=False)
            print(f"  报告已保存到: {report_filename}")
        else:
            print(f"JSON报告导出失败: {json_report['error']}")
        
        # 导出CSV格式报告
        csv_report = sync_monitor.export_performance_report(
            start_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            format='csv'
        )
        
        if csv_report['success']:
            print(f"\nCSV报告导出成功:")
            print(f"  报告格式: {csv_report['export_format']}")
            print(f"  包含表格: {list(csv_report['data'].keys())}")
        else:
            print(f"CSV报告导出失败: {csv_report['error']}")
        
    except Exception as e:
        print(f"报告导出示例执行出错: {e}")


def example_cleanup_operations():
    """清理操作示例"""
    print("\n=== 清理操作示例 ===")
    
    try:
        # 清理旧记录
        cleanup_result = sync_monitor.cleanup_old_records(retention_days=30)
        
        if cleanup_result['success']:
            print("清理操作完成:")
            print(f"  清理任务记录: {cleanup_result['cleaned_tasks']} 条")
            print(f"  清理事件记录: {cleanup_result['cleaned_events']} 条")
            print(f"  清理性能记录: {cleanup_result['cleaned_performance']} 条")
            print(f"  保留天数: {cleanup_result['retention_days']} 天")
        else:
            print(f"清理操作失败: {cleanup_result['error']}")
        
    except Exception as e:
        print(f"清理操作示例执行出错: {e}")


async def example_async_monitoring():
    """异步监控示例"""
    print("\n=== 异步监控示例 ===")
    
    async def simulate_async_task(task_id: str, duration: int):
        """模拟异步任务"""
        task_info = SyncTaskInfo(
            task_id=task_id,
            data_source="async_source",
            data_type="async_data",
            target_date=datetime.now().strftime('%Y-%m-%d'),
            status=SyncTaskStatus.PENDING,
            priority=1
        )
        
        # 开始监控
        sync_monitor.start_task_monitoring(task_info)
        
        # 模拟任务执行
        for i in range(0, 101, 20):
            await asyncio.sleep(duration / 5)  # 模拟异步处理
            sync_monitor.update_task_progress(
                task_id=task_id,
                progress=i,
                records_processed=i * 5,
                records_failed=i // 20
            )
        
        # 完成任务
        sync_monitor.complete_task_monitoring(
            task_id=task_id,
            success=True,
            final_stats={'records_processed': 500, 'records_failed': 5}
        )
        
        print(f"异步任务 {task_id} 完成")
    
    # 并发执行多个异步任务
    tasks = [
        simulate_async_task(f"async_task_{i}", 2)
        for i in range(3)
    ]
    
    await asyncio.gather(*tasks)
    print("所有异步任务完成")


def main():
    """主函数"""
    print("同步监控器功能演示")
    print("=" * 50)
    
    try:
        # 基础监控功能
        example_basic_monitoring()
        
        # 实时仪表板
        example_real_time_dashboard()
        
        # 事件订阅
        example_event_subscription()
        
        # 历史记录分析
        example_historical_analysis()
        
        # 性能分析
        example_performance_analysis()
        
        # 报告导出
        example_report_export()
        
        # 清理操作
        example_cleanup_operations()
        
        # 异步监控
        print("\n开始异步监控示例...")
        asyncio.run(example_async_monitoring())
        
        print("\n=" * 50)
        print("所有示例执行完成！")
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
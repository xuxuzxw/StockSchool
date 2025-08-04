#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控服务类

提供仪表板、性能分析等服务
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
from loguru import logger

from .models import SyncTaskInfo, SyncTaskStatus
from .constants import MonitoringConstants


class DashboardService:
    """仪表板服务"""
    
    def __init__(self, active_tasks: Dict[str, SyncTaskInfo], constants: MonitoringConstants):
        self.active_tasks = active_tasks
        self.constants = constants
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        try:
            dashboard = {
                'timestamp': datetime.now().isoformat(),
                'active_tasks': self._get_active_tasks_summary(),
                'summary': self._calculate_summary_stats(),
                'data_sources': self._group_by_data_source(),
                'data_types': self._group_by_data_type(),
                'recent_events': []  # 由事件系统提供
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"获取仪表板数据失败: {e}")
            return self._get_error_dashboard(str(e))
    
    def _get_active_tasks_summary(self) -> List[Dict[str, Any]]:
        """获取活跃任务摘要"""
        return [task.to_dict() for task in self.active_tasks.values()]
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """计算摘要统计"""
        total_active = len(self.active_tasks)
        running_tasks = 0
        pending_tasks = 0
        total_progress = 0.0
        
        for task in self.active_tasks.values():
            if task.status == SyncTaskStatus.RUNNING:
                running_tasks += 1
            elif task.status == SyncTaskStatus.PENDING:
                pending_tasks += 1
            
            total_progress += task.progress
        
        avg_progress = total_progress / total_active if total_active > 0 else 0.0
        
        return {
            'total_active': total_active,
            'running_tasks': running_tasks,
            'pending_tasks': pending_tasks,
            'avg_progress': avg_progress
        }
    
    def _group_by_data_source(self) -> Dict[str, int]:
        """按数据源分组统计"""
        sources = defaultdict(int)
        for task in self.active_tasks.values():
            sources[task.data_source] += 1
        return dict(sources)
    
    def _group_by_data_type(self) -> Dict[str, int]:
        """按数据类型分组统计"""
        types = defaultdict(int)
        for task in self.active_tasks.values():
            types[task.data_type] += 1
        return dict(types)
    
    def _get_error_dashboard(self, error_message: str) -> Dict[str, Any]:
        """获取错误仪表板"""
        return {
            'timestamp': datetime.now().isoformat(),
            'error': error_message,
            'active_tasks': [],
            'summary': {'total_active': 0, 'running_tasks': 0, 'pending_tasks': 0, 'avg_progress': 0.0},
            'data_sources': {},
            'data_types': {},
            'recent_events': []
        }


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, task_repository):
        self.task_repository = task_repository
    
    def analyze_performance(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """分析性能指标"""
        try:
            # 这里应该从数据库获取历史数据进行分析
            # 由于repository接口还没有完整实现，这里提供框架
            
            analysis = {
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                'task_statistics': self._analyze_task_statistics(),
                'performance_metrics': self._calculate_performance_metrics(),
                'error_analysis': self._analyze_errors(),
                'recommendations': self._generate_recommendations()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"性能分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_task_statistics(self) -> Dict[str, Any]:
        """分析任务统计"""
        # 实现任务统计分析逻辑
        return {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'avg_duration': 0.0,
            'success_rate': 0.0
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """计算性能指标"""
        # 实现性能指标计算逻辑
        return {
            'throughput': 0.0,  # 任务/小时
            'avg_response_time': 0.0,
            'p95_response_time': 0.0,
            'error_rate': 0.0
        }
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """分析错误模式"""
        # 实现错误分析逻辑
        return {
            'common_errors': [],
            'error_trends': [],
            'error_distribution': {}
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        # 基于分析结果生成建议
        recommendations = []
        
        # 这里可以添加各种分析逻辑
        # 例如：检查错误率、响应时间等
        
        return recommendations


class AlertService:
    """告警服务"""
    
    def __init__(self, constants: MonitoringConstants):
        self.constants = constants
        self.alert_rules = self._load_alert_rules()
    
    def _load_alert_rules(self) -> List[Dict[str, Any]]:
        """加载告警规则"""
        return [
            {
                'name': 'high_error_rate',
                'condition': lambda metrics: metrics.get('error_rate', 0) > self.constants.HIGH_ERROR_RATE_THRESHOLD,
                'message': '错误率过高',
                'severity': 'high'
            },
            {
                'name': 'slow_task',
                'condition': lambda task: task.duration and task.duration > self.constants.SLOW_TASK_THRESHOLD,
                'message': '任务执行时间过长',
                'severity': 'medium'
            }
        ]
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查告警条件"""
        alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule['condition'](metrics):
                    alerts.append({
                        'rule_name': rule['name'],
                        'message': rule['message'],
                        'severity': rule['severity'],
                        'timestamp': datetime.now().isoformat(),
                        'metrics': metrics
                    })
            except Exception as e:
                logger.error(f"告警规则 {rule['name']} 检查失败: {e}")
        
        return alerts
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """发送告警"""
        try:
            # 这里实现具体的告警发送逻辑
            # 例如：发送邮件、webhook、短信等
            logger.warning(f"告警: {alert['message']} | 严重程度: {alert['severity']}")
            return True
        except Exception as e:
            logger.error(f"发送告警失败: {e}")
            return False
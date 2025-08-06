"""
改进的验收测试编排器 - 使用现代设计模式
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Protocol
from contextlib import contextmanager
from dataclasses import dataclass

from .core.models import AcceptanceReport, TestResult, TestStatus, PhaseType
from .core.exceptions import AcceptanceTestError, ConfigurationError
from .core.phase_factory import PhaseFactory, PhaseRegistry, TestPhase
from .config.manager import ConfigManager
from .reporting.generator import ReportGenerator
from .utils.performance import PerformanceMonitor


class TestProgressObserver(Protocol):
    """测试进度观察者协议"""
    
    def on_phase_started(self, phase_name: str) -> None:
        """阶段开始时调用"""
        ...
    
    def on_phase_completed(self, phase_name: str, results: List[TestResult]) -> None:
        """阶段完成时调用"""
        ...
    
    def on_test_completed(self, report: AcceptanceReport) -> None:
        """测试完成时调用"""
        ...


@dataclass
class OrchestratorConfig:
    """编排器配置"""
    config_file: str = '.env.acceptance'
    skip_config_validation: bool = False
    terminate_on_critical_failure: bool = False
    max_parallel_phases: int = 1
    enable_performance_monitoring: bool = True


class AcceptanceTestOrchestrator:
    """改进的验收测试编排器"""
    
    def __init__(
        self, 
        config: Optional[OrchestratorConfig] = None,
        phase_factory: Optional[PhaseFactory] = None,
        config_manager: Optional[ConfigManager] = None,
        report_generator: Optional[ReportGenerator] = None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        """
        初始化编排器 - 使用依赖注入
        
        Args:
            config: 编排器配置
            phase_factory: 阶段工厂实例
            config_manager: 配置管理器实例
            report_generator: 报告生成器实例
            performance_monitor: 性能监控器实例
        """
        self.config = config or OrchestratorConfig()
        self.session_id = str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)
        
        # 依赖注入
        self.config_manager = config_manager or ConfigManager(self.config.config_file)
        self.phase_factory = phase_factory or PhaseFactory()
        self.report_generator = report_generator or ReportGenerator()
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        
        # 观察者列表
        self._observers: List[TestProgressObserver] = []
        
        # 延迟初始化的属性
        self._test_phases: Optional[List[TestPhase]] = None
        self._report: Optional[AcceptanceReport] = None
        
        # 验证配置
        if not self.config.skip_config_validation:
            self._validate_configuration()
        else:
            self.logger.warning("跳过配置验证（测试模式）")
        
        self.logger.info(f"验收测试编排器初始化完成，会话ID: {self.session_id}")
    
    def _validate_configuration(self) -> None:
        """验证配置"""
        try:
            missing_config = self.config_manager.validate_required_config()
            if missing_config:
                raise ConfigurationError(f"缺少必需配置: {missing_config}")
        except Exception as e:
            self.logger.error(f"配置验证失败: {e}")
            raise
    
    @property
    def test_phases(self) -> List[TestPhase]:
        """懒加载测试阶段"""
        if self._test_phases is None:
            self._test_phases = self._initialize_test_phases()
        return self._test_phases
    
    @property
    def report(self) -> AcceptanceReport:
        """懒加载报告"""
        if self._report is None:
            self._report = AcceptanceReport(
                test_session_id=self.session_id,
                start_time=datetime.now()
            )
        return self._report
    
    def _initialize_test_phases(self) -> List[TestPhase]:
        """
        初始化测试阶段 - 使用工厂模式
        
        Returns:
            测试阶段列表
        """
        try:
            phases = self.phase_factory.create_all_phases(
                self.config_manager._config
            )
            
            if not phases:
                raise AcceptanceTestError("没有找到任何可执行的测试阶段")
            
            self.logger.info(f"成功初始化 {len(phases)} 个测试阶段")
            return phases
            
        except Exception as e:
            self.logger.error(f"初始化测试阶段失败: {e}")
            raise AcceptanceTestError(f"初始化测试阶段失败: {e}") from e
    
    def add_observer(self, observer: TestProgressObserver) -> None:
        """添加观察者"""
        self._observers.append(observer)
    
    def remove_observer(self, observer: TestProgressObserver) -> None:
        """移除观察者"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_phase_started(self, phase_name: str) -> None:
        """通知阶段开始"""
        for observer in self._observers:
            try:
                observer.on_phase_started(phase_name)
            except Exception as e:
                self.logger.warning(f"观察者通知失败: {e}")
    
    def _notify_phase_completed(self, phase_name: str, results: List[TestResult]) -> None:
        """通知阶段完成"""
        for observer in self._observers:
            try:
                observer.on_phase_completed(phase_name, results)
            except Exception as e:
                self.logger.warning(f"观察者通知失败: {e}")
    
    def _notify_test_completed(self, report: AcceptanceReport) -> None:
        """通知测试完成"""
        for observer in self._observers:
            try:
                observer.on_test_completed(report)
            except Exception as e:
                self.logger.warning(f"观察者通知失败: {e}")
    
    @contextmanager
    def _performance_context(self, operation_name: str):
        """性能监控上下文管理器"""
        if self.config.enable_performance_monitoring:
            with self.performance_monitor.time_operation(operation_name):
                yield
        else:
            yield
    
    def run_acceptance_tests(
        self, 
        selected_phases: Optional[List[str]] = None,
        fail_fast: Optional[bool] = None
    ) -> AcceptanceReport:
        """
        执行完整的验收测试流程
        
        Args:
            selected_phases: 可选的指定阶段列表，如果为None则执行所有阶段
            fail_fast: 是否在第一个失败时停止，覆盖配置中的设置
        
        Returns:
            AcceptanceReport: 完整的验收测试报告
        """
        self.logger.info(f"开始执行验收测试，会话ID: {self.session_id}")
        
        # 确定是否快速失败
        should_fail_fast = (
            fail_fast if fail_fast is not None 
            else self.config.terminate_on_critical_failure
        )
        
        try:
            with self._performance_context("run_acceptance_tests"):
                # 筛选要执行的阶段
                phases_to_run = self._filter_phases(selected_phases)
                
                # 执行各个测试阶段
                for phase in phases_to_run:
                    phase_name = getattr(phase, 'phase_name', str(phase))
                    self.logger.info(f"开始执行阶段: {phase_name}")
                    
                    # 通知观察者
                    self._notify_phase_started(phase_name)
                    
                    try:
                        # 执行阶段测试
                        with self._performance_context(f"phase_{phase_name}"):
                            phase_results = phase.execute()
                        
                        self.report.phase_results.extend(phase_results)
                        
                        # 更新统计信息
                        self._update_statistics(phase_results)
                        
                        # 通知观察者
                        self._notify_phase_completed(phase_name, phase_results)
                        
                        # 检查是否需要提前终止
                        if should_fail_fast and self._should_terminate_early(phase_results):
                            self.logger.warning(f"阶段 {phase_name} 关键测试失败，提前终止")
                            break
                            
                    except Exception as e:
                        self.logger.error(f"阶段 {phase_name} 执行失败: {e}")
                        # 创建失败结果
                        error_result = TestResult(
                            phase=phase_name,
                            test_name="phase_execution",
                            status=TestStatus.FAILED,
                            execution_time=0.0,
                            error_message=str(e)
                        )
                        self.report.phase_results.append(error_result)
                        self.report.failed_tests += 1
                        self.report.total_tests += 1
                        
                        # 如果启用快速失败，则停止执行
                        if should_fail_fast:
                            self.logger.error("启用快速失败模式，停止执行")
                            break
                
                # 完成测试
                self._finalize_report()
                
                # 生成报告
                self._generate_reports()
                
                # 通知观察者
                self._notify_test_completed(self.report)
                
                return self.report
                
        except Exception as e:
            self.logger.error(f"验收测试执行失败: {e}")
            self.report.end_time = datetime.now()
            self.report.overall_result = False
            self.report.recommendations.append(f"测试执行异常: {str(e)}")
            raise AcceptanceTestError(f"验收测试执行失败: {e}") from e
    
    def _filter_phases(self, selected_phases: Optional[List[str]]) -> List[TestPhase]:
        """筛选要执行的测试阶段"""
        if not selected_phases:
            return self.test_phases
        
        filtered_phases = []
        available_phase_names = {
            getattr(phase, 'phase_name', str(phase)): phase 
            for phase in self.test_phases
        }
        
        for phase_name in selected_phases:
            if phase_name in available_phase_names:
                filtered_phases.append(available_phase_names[phase_name])
            else:
                self.logger.warning(f"未找到指定的测试阶段: {phase_name}")
        
        if not filtered_phases:
            raise AcceptanceTestError(f"没有找到指定的测试阶段: {selected_phases}")
        
        return filtered_phases
    
    def _update_statistics(self, phase_results: List[TestResult]) -> None:
        """更新测试统计信息"""
        for result in phase_results:
            self.report.total_tests += 1
            
            if result.status == TestStatus.PASSED:
                self.report.passed_tests += 1
            elif result.status == TestStatus.FAILED:
                self.report.failed_tests += 1
            elif result.status == TestStatus.SKIPPED:
                self.report.skipped_tests += 1
    
    def _should_terminate_early(self, phase_results: List[TestResult]) -> bool:
        """判断是否需要提前终止测试"""
        # 检查是否有关键测试失败
        critical_tests = ['docker_services', 'database_connection', 'redis_connection']
        
        for result in phase_results:
            if (result.test_name in critical_tests and 
                result.status == TestStatus.FAILED):
                return True
        
        return False
    
    def _finalize_report(self) -> None:
        """完成报告生成"""
        self.report.end_time = datetime.now()
        self.report.overall_result = (
            self.report.failed_tests == 0 and 
            self.report.total_tests > 0
        )
        
        # 添加性能指标
        if self.config.enable_performance_monitoring:
            self.report.performance_metrics = self.performance_monitor.get_summary()
        
        # 生成改进建议
        self._generate_recommendations()
        
        self.logger.info(
            f"验收测试完成: {self.report.passed_tests}/{self.report.total_tests} 通过, "
            f"整体结果: {'通过' if self.report.overall_result else '失败'}"
        )
    
    def _generate_recommendations(self) -> None:
        """生成改进建议"""
        recommendations = []
        
        # 基于失败测试生成建议
        failed_phases = set()
        for result in self.report.phase_results:
            if result.status == TestStatus.FAILED:
                failed_phases.add(result.phase)
        
        # 阶段特定建议映射
        phase_recommendations = {
            '基础设施验收': "建议检查Docker服务状态和网络连接配置",
            '数据服务验收': "建议验证Tushare API配置和数据库连接",
            '计算引擎验收': "建议检查因子计算逻辑和性能优化",
            'AI服务验收': "建议检查模型训练配置和GPU资源",
            '外接AI分析验收': "建议验证外接AI API配置和网络连接",
        }
        
        for failed_phase in failed_phases:
            if failed_phase in phase_recommendations:
                recommendations.append(phase_recommendations[failed_phase])
        
        # 基于性能指标生成建议
        if self.report.performance_metrics:
            for metric_name, metric_data in self.report.performance_metrics.items():
                if isinstance(metric_data, dict) and metric_data.get('average', 0) > 30:
                    recommendations.append(f"建议优化 {metric_name} 的执行性能")
        
        # 基于通过率生成建议
        if self.report.total_tests > 0:
            pass_rate = self.report.passed_tests / self.report.total_tests
            if pass_rate < 0.8:
                recommendations.append("整体通过率较低，建议进行全面的系统检查和优化")
            elif pass_rate < 0.95:
                recommendations.append("部分测试失败，建议针对失败项目进行专项优化")
        
        self.report.recommendations = recommendations
    
    def _generate_reports(self) -> None:
        """生成各种格式的测试报告"""
        try:
            # 生成JSON报告
            json_path = self.report_generator.generate_json_report(self.report)
            self.logger.info(f"JSON报告已生成: {json_path}")
            
            # 生成HTML报告
            html_path = self.report_generator.generate_html_report(self.report)
            self.logger.info(f"HTML报告已生成: {html_path}")
            
            # 生成Markdown报告
            md_path = self.report_generator.generate_markdown_report(self.report)
            self.logger.info(f"Markdown报告已生成: {md_path}")
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
    
    def get_session_status(self) -> Dict[str, Any]:
        """获取当前会话状态"""
        return {
            'session_id': self.session_id,
            'start_time': self.report.start_time.isoformat(),
            'end_time': self.report.end_time.isoformat() if self.report.end_time else None,
            'total_tests': self.report.total_tests,
            'passed_tests': self.report.passed_tests,
            'failed_tests': self.report.failed_tests,
            'skipped_tests': self.report.skipped_tests,
            'overall_result': self.report.overall_result,
            'current_phase': self._get_current_phase(),
            'config': {
                'config_file': self.config.config_file,
                'skip_config_validation': self.config.skip_config_validation,
                'terminate_on_critical_failure': self.config.terminate_on_critical_failure,
                'max_parallel_phases': self.config.max_parallel_phases,
                'enable_performance_monitoring': self.config.enable_performance_monitoring,
            }
        }
    
    def _get_current_phase(self) -> Optional[str]:
        """获取当前执行的阶段"""
        if self.report.end_time:
            return None  # 已完成
        
        # 简单实现：返回最后一个有结果的阶段
        if self.report.phase_results:
            return self.report.phase_results[-1].phase
        
        return "准备中"
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            # 清理各个阶段的资源
            if self._test_phases:
                for phase in self._test_phases:
                    if hasattr(phase, '_cleanup_resources'):
                        phase._cleanup_resources()
            
            # 清理工厂缓存
            self.phase_factory.clear_cache()
            
            # 清理观察者
            self._observers.clear()
            
            self.logger.info("验收测试资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")


# 简单的控制台观察者实现
class ConsoleProgressObserver:
    """控制台进度观察者"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def on_phase_started(self, phase_name: str) -> None:
        """阶段开始时调用"""
        self.logger.info(f"🚀 开始执行阶段: {phase_name}")
    
    def on_phase_completed(self, phase_name: str, results: List[TestResult]) -> None:
        """阶段完成时调用"""
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        
        if failed == 0:
            self.logger.info(f"✅ 阶段完成: {phase_name} ({passed} 通过)")
        else:
            self.logger.warning(f"⚠️ 阶段完成: {phase_name} ({passed} 通过, {failed} 失败)")
    
    def on_test_completed(self, report: AcceptanceReport) -> None:
        """测试完成时调用"""
        if report.overall_result:
            self.logger.info(f"🎉 验收测试完成: 全部通过 ({report.passed_tests}/{report.total_tests})")
        else:
            self.logger.error(f"❌ 验收测试完成: 存在失败 ({report.passed_tests}/{report.total_tests})")
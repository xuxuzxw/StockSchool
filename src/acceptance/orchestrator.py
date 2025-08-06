"""
验收测试编排器 - 基于现有框架构建
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from .core.models import AcceptanceReport, TestResult, TestStatus, PhaseType
from .core.exceptions import AcceptanceTestError, ConfigurationError
from .config.manager import ConfigManager
from .reporting.generator import ReportGenerator
from .utils.performance import PerformanceMonitor


class AcceptanceTestOrchestrator:
    """验收测试编排器 - 充分利用现有框架"""
    
    def __init__(self, config_file: str = '.env.acceptance', skip_config_validation: bool = False):
        """初始化编排器"""
        self.session_id = str(uuid.uuid4())
        self.config_manager = ConfigManager(config_file)
        self.report_generator = ReportGenerator()
        self.performance_monitor = PerformanceMonitor()
        self.logger = logging.getLogger(__name__)
        
        # 验证配置（可选跳过，用于测试）
        if not skip_config_validation:
            missing_config = self.config_manager.validate_required_config()
            if missing_config:
                raise ConfigurationError(f"缺少必需配置: {missing_config}")
        else:
            self.logger.warning("跳过配置验证（测试模式）")
        
        # 初始化测试阶段
        self.test_phases = self._initialize_test_phases()
        
        # 初始化报告
        self.report = AcceptanceReport(
            test_session_id=self.session_id,
            start_time=datetime.now()
        )
        
        self.logger.info(f"验收测试编排器初始化完成，会话ID: {self.session_id}")
    
    def _initialize_test_phases(self) -> List:
        """初始化测试阶段 - 利用现有的阶段类"""
        phases = []
        
        # 动态导入现有的测试阶段类
        try:
            # 基础设施验收阶段
            from .phases.infrastructure import InfrastructurePhase
            phases.append(InfrastructurePhase(
                phase_name="基础设施验收",
                config=self.config_manager._config
            ))
        except ImportError:
            self.logger.warning("InfrastructurePhase 未找到，将跳过基础设施测试")
        
        try:
            # 数据服务验收阶段
            from .phases.data_service import DataServicePhase
            phases.append(DataServicePhase(
                phase_name="数据服务验收",
                config=self.config_manager._config
            ))
        except ImportError:
            self.logger.warning("DataServicePhase 未找到，将跳过数据服务测试")
        
        try:
            # 计算引擎验收阶段
            from .phases.compute_engine import ComputeEnginePhase
            phases.append(ComputeEnginePhase(
                phase_name="计算引擎验收",
                config=self.config_manager._config
            ))
        except ImportError:
            self.logger.warning("ComputeEnginePhase 未找到，将跳过计算引擎测试")
        
        try:
            # AI服务验收阶段
            from .phases.ai_service import AIServicePhase
            phases.append(AIServicePhase(
                phase_name="AI服务验收",
                config=self.config_manager._config
            ))
        except ImportError:
            self.logger.warning("AIServicePhase 未找到，将跳过AI服务测试")
        
        try:
            # 外接AI分析验收阶段
            from .phases.external_ai_analysis import ExternalAIAnalysisPhase
            phases.append(ExternalAIAnalysisPhase(
                phase_name="外接AI分析验收",
                config=self.config_manager._config
            ))
        except ImportError:
            self.logger.warning("ExternalAIAnalysisPhase 未找到，将跳过外接AI分析测试")
        
        try:
            # API服务验收阶段
            from .phases.api_service import APIServicePhase
            phases.append(APIServicePhase(
                phase_name="API服务验收",
                config=self.config_manager._config
            ))
        except ImportError:
            self.logger.warning("APIServicePhase 未找到，将跳过API服务测试")
        
        try:
            # 监控系统验收阶段
            from .phases.monitoring import MonitoringPhase
            phases.append(MonitoringPhase(
                phase_name="监控系统验收",
                config=self.config_manager._config
            ))
        except ImportError:
            self.logger.warning("MonitoringPhase 未找到，将跳过监控系统测试")
        
        try:
            # 性能基准验收阶段
            from .phases.performance import PerformancePhase
            phases.append(PerformancePhase(
                phase_name="性能基准验收",
                config=self.config_manager._config
            ))
        except ImportError:
            self.logger.warning("PerformancePhase 未找到，将跳过性能基准测试")
        
        try:
            # 集成测试验收阶段
            from .phases.integration import IntegrationPhase
            phases.append(IntegrationPhase(
                phase_name="集成测试验收",
                config=self.config_manager._config
            ))
        except ImportError:
            self.logger.warning("IntegrationPhase 未找到，将跳过集成测试")
        
        try:
            # 用户验收测试阶段
            from .phases.user_acceptance import UserAcceptancePhase
            phases.append(UserAcceptancePhase(
                phase_name="用户验收测试",
                config=self.config_manager._config
            ))
        except ImportError:
            self.logger.warning("UserAcceptancePhase 未找到，将跳过用户验收测试")
        
        try:
            # 代码质量验收阶段
            from .phases.code_quality import CodeQualityPhase
            phases.append(CodeQualityPhase(
                phase_name="代码质量验收",
                config=self.config_manager._config
            ))
        except ImportError:
            self.logger.warning("CodeQualityPhase 未找到，将跳过代码质量测试")
        
        try:
            # 安全性验收阶段
            from .phases.security import SecurityPhase
            phases.append(SecurityPhase(
                phase_name="安全性验收",
                config=self.config_manager._config
            ))
        except ImportError:
            self.logger.warning("SecurityPhase 未找到，将跳过安全性测试")
        
        if not phases:
            raise AcceptanceTestError("没有找到任何可执行的测试阶段")
        
        self.logger.info(f"成功初始化 {len(phases)} 个测试阶段")
        return phases
    
    @PerformanceMonitor().time_function("run_acceptance_tests")
    def run_acceptance_tests(self, selected_phases: Optional[List[str]] = None) -> AcceptanceReport:
        """
        执行完整的验收测试流程
        
        Args:
            selected_phases: 可选的指定阶段列表，如果为None则执行所有阶段
        
        Returns:
            AcceptanceReport: 完整的验收测试报告
        """
        self.logger.info(f"开始执行验收测试，会话ID: {self.session_id}")
        
        try:
            # 筛选要执行的阶段
            phases_to_run = self._filter_phases(selected_phases)
            
            # 执行各个测试阶段
            for phase in phases_to_run:
                self.logger.info(f"开始执行阶段: {phase.phase_name}")
                
                try:
                    # 执行阶段测试
                    phase_results = phase.execute()
                    self.report.phase_results.extend(phase_results)
                    
                    # 更新统计信息
                    self._update_statistics(phase_results)
                    
                    # 检查是否需要提前终止
                    if self._should_terminate_early(phase_results):
                        self.logger.warning(f"阶段 {phase.phase_name} 关键测试失败，提前终止")
                        break
                        
                except Exception as e:
                    self.logger.error(f"阶段 {phase.phase_name} 执行失败: {e}")
                    # 创建失败结果
                    error_result = TestResult(
                        phase=phase.phase_name,
                        test_name="phase_execution",
                        status=TestStatus.FAILED,
                        execution_time=0.0,
                        error_message=str(e)
                    )
                    self.report.phase_results.append(error_result)
                    self.report.failed_tests += 1
                    self.report.total_tests += 1
            
            # 完成测试
            self._finalize_report()
            
            # 生成报告
            self._generate_reports()
            
            return self.report
            
        except Exception as e:
            self.logger.error(f"验收测试执行失败: {e}")
            self.report.end_time = datetime.now()
            self.report.overall_result = False
            self.report.recommendations.append(f"测试执行异常: {str(e)}")
            raise AcceptanceTestError(f"验收测试执行失败: {e}") from e
    
    def _filter_phases(self, selected_phases: Optional[List[str]]) -> List:
        """筛选要执行的测试阶段"""
        if not selected_phases:
            return self.test_phases
        
        filtered_phases = []
        for phase in self.test_phases:
            if phase.phase_name in selected_phases:
                filtered_phases.append(phase)
        
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
        # 如果配置了提前终止策略
        terminate_on_critical_failure = self.config_manager.get('terminate_on_critical_failure', False)
        
        if not terminate_on_critical_failure:
            return False
        
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
        
        if '基础设施验收' in failed_phases:
            recommendations.append("建议检查Docker服务状态和网络连接配置")
        
        if '数据服务验收' in failed_phases:
            recommendations.append("建议验证Tushare API配置和数据库连接")
        
        if '计算引擎验收' in failed_phases:
            recommendations.append("建议检查因子计算逻辑和性能优化")
        
        if 'AI服务验收' in failed_phases:
            recommendations.append("建议检查模型训练配置和GPU资源")
        
        if '外接AI分析验收' in failed_phases:
            recommendations.append("建议验证外接AI API配置和网络连接")
        
        # 基于性能指标生成建议
        if self.report.performance_metrics:
            for metric_name, metric_data in self.report.performance_metrics.items():
                if metric_data.get('average', 0) > 30:  # 超过30秒的操作
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
            'current_phase': self._get_current_phase()
        }
    
    def _get_current_phase(self) -> Optional[str]:
        """获取当前执行的阶段"""
        # 这里可以根据实际需要实现更复杂的状态跟踪
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
            for phase in self.test_phases:
                if hasattr(phase, '_cleanup_resources'):
                    phase._cleanup_resources()
            
            self.logger.info("验收测试资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")
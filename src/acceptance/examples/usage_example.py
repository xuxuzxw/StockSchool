"""
改进版验收测试编排器的使用示例
"""

import logging
from datetime import datetime

from ..orchestrator_improved import (
    AcceptanceTestOrchestrator, 
    OrchestratorConfig,
    ConsoleProgressObserver
)
from ..core.phase_factory import PhaseRegistry, PhaseConfig


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建编排器（使用默认配置）
    orchestrator = AcceptanceTestOrchestrator()
    
    # 添加进度观察者
    observer = ConsoleProgressObserver()
    orchestrator.add_observer(observer)
    
    try:
        # 执行所有测试
        report = orchestrator.run_acceptance_tests()
        
        # 打印结果
        print(f"测试完成: {report.passed_tests}/{report.total_tests} 通过")
        print(f"整体结果: {'✅ 通过' if report.overall_result else '❌ 失败'}")
        
    except Exception as e:
        print(f"测试执行失败: {e}")
    
    finally:
        # 清理资源
        orchestrator.cleanup()


def example_custom_config():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    # 创建自定义配置
    config = OrchestratorConfig(
        config_file='.env.acceptance',
        skip_config_validation=True,  # 跳过配置验证（测试模式）
        terminate_on_critical_failure=True,  # 关键失败时终止
        max_parallel_phases=2,  # 最大并行阶段数
        enable_performance_monitoring=True  # 启用性能监控
    )
    
    # 创建编排器
    orchestrator = AcceptanceTestOrchestrator(config=config)
    
    # 添加观察者
    observer = ConsoleProgressObserver()
    orchestrator.add_observer(observer)
    
    try:
        # 只执行指定的阶段
        selected_phases = ["基础设施验收", "数据服务验收"]
        report = orchestrator.run_acceptance_tests(
            selected_phases=selected_phases,
            fail_fast=True
        )
        
        # 获取会话状态
        status = orchestrator.get_session_status()
        print(f"会话ID: {status['session_id']}")
        print(f"配置: {status['config']}")
        
    except Exception as e:
        print(f"测试执行失败: {e}")
    
    finally:
        orchestrator.cleanup()


def example_custom_phase_registration():
    """自定义阶段注册示例"""
    print("\n=== 自定义阶段注册示例 ===")
    
    # 获取阶段注册表
    registry = PhaseRegistry()
    
    # 注册自定义阶段
    registry.register_phase("custom_test", PhaseConfig(
        module_path="src.acceptance.phases.custom",
        class_name="CustomTestPhase",
        phase_name="自定义测试阶段",
        enabled=True,
        priority=99
    ))
    
    # 创建编排器
    orchestrator = AcceptanceTestOrchestrator(
        config=OrchestratorConfig(skip_config_validation=True)
    )
    
    try:
        # 查看所有可用阶段
        phases = orchestrator.test_phases
        print(f"可用阶段数量: {len(phases)}")
        
        for phase in phases:
            phase_name = getattr(phase, 'phase_name', str(phase))
            print(f"- {phase_name}")
        
    except Exception as e:
        print(f"阶段初始化失败: {e}")
    
    finally:
        orchestrator.cleanup()


def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    from ..core.exceptions_improved import (
        ConfigurationError,
        PhaseExecutionError,
        create_configuration_error
    )
    
    try:
        # 模拟配置错误
        raise create_configuration_error(
            message="缺少必需的配置项",
            missing_configs=["TUSHARE_TOKEN", "DATABASE_URL"],
            session_id="test-session-123"
        )
        
    except ConfigurationError as e:
        print(f"配置错误: {e.message}")
        print(f"错误详情: {e.to_dict()}")
    
    try:
        # 模拟阶段执行错误
        raise PhaseExecutionError(
            message="基础设施验收阶段执行失败",
            phase_name="基础设施验收",
            failed_tests=["docker_services", "database_connection"]
        )
        
    except PhaseExecutionError as e:
        print(f"阶段执行错误: {e.message}")
        print(f"失败的测试: {e.failed_tests}")


class CustomProgressObserver:
    """自定义进度观察者示例"""
    
    def __init__(self):
        self.start_time = None
        self.phase_times = {}
    
    def on_phase_started(self, phase_name: str) -> None:
        """阶段开始时调用"""
        self.phase_times[phase_name] = datetime.now()
        print(f"📊 开始执行: {phase_name}")
    
    def on_phase_completed(self, phase_name: str, results) -> None:
        """阶段完成时调用"""
        if phase_name in self.phase_times:
            duration = datetime.now() - self.phase_times[phase_name]
            print(f"⏱️ 阶段耗时: {phase_name} - {duration.total_seconds():.2f}秒")
        
        # 统计结果
        passed = sum(1 for r in results if r.status.name == 'PASSED')
        failed = sum(1 for r in results if r.status.name == 'FAILED')
        
        print(f"📈 阶段结果: {phase_name} - 通过:{passed}, 失败:{failed}")
    
    def on_test_completed(self, report) -> None:
        """测试完成时调用"""
        total_duration = report.end_time - report.start_time
        print(f"🏁 测试完成，总耗时: {total_duration.total_seconds():.2f}秒")
        
        # 生成简单的性能报告
        if hasattr(report, 'performance_metrics') and report.performance_metrics:
            print("📊 性能指标:")
            for metric_name, metric_data in report.performance_metrics.items():
                if isinstance(metric_data, dict):
                    avg_time = metric_data.get('average', 0)
                    print(f"  - {metric_name}: {avg_time:.2f}秒")


def example_custom_observer():
    """自定义观察者示例"""
    print("\n=== 自定义观察者示例 ===")
    
    # 创建编排器
    orchestrator = AcceptanceTestOrchestrator(
        config=OrchestratorConfig(skip_config_validation=True)
    )
    
    # 添加自定义观察者
    custom_observer = CustomProgressObserver()
    orchestrator.add_observer(custom_observer)
    
    try:
        # 执行测试
        report = orchestrator.run_acceptance_tests(
            selected_phases=["基础设施验收"]
        )
        
    except Exception as e:
        print(f"测试执行失败: {e}")
    
    finally:
        orchestrator.cleanup()


def main():
    """主函数"""
    setup_logging()
    
    print("StockSchool 改进版验收测试编排器使用示例")
    print("=" * 50)
    
    # 运行各种示例
    try:
        example_basic_usage()
        example_custom_config()
        example_custom_phase_registration()
        example_error_handling()
        example_custom_observer()
        
    except Exception as e:
        print(f"示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
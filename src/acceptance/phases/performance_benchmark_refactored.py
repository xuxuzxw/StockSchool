"""
重构后的性能基准验收阶段 - 使用策略模式和配置外部化
"""
import os
import sys
import yaml
from typing import List, Dict, Any
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ..core.base_phase import BaseTestPhase
from ..core.models import TestResult, TestStatus
from ..core.exceptions import AcceptanceTestError
from ..performance.factory import PerformanceTestExecutor


class PerformanceBenchmarkPhase(BaseTestPhase):
    """重构后的性能基准验收阶段"""
    
    def __init__(self, phase_name: str, config: Dict[str, Any]):
        super().__init__(phase_name, config)
        
        try:
            # 加载性能测试配置
            self.performance_config = self._load_performance_config()
            
            # 创建性能测试执行器
            self.test_executor = PerformanceTestExecutor(self.performance_config)
            
            # 获取启用的测试策略
            self.enabled_strategies = self._get_enabled_strategies()
            
            self.logger.info(f"性能基准验收阶段初始化完成，启用策略: {self.enabled_strategies}")
            
        except Exception as e:
            self.logger.error(f"性能基准验收阶段初始化失败: {e}")
            raise AcceptanceTestError(f"性能基准验收阶段初始化失败: {e}")
    
    def _load_performance_config(self) -> Dict[str, Any]:
        """加载性能测试配置"""
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "performance_test_config.yml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('performance_test', {})
        except FileNotFoundError:
            self.logger.warning(f"性能测试配置文件未找到: {config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            self.logger.error(f"配置文件解析失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'thresholds': {
                'memory_limit_gb': 16,
                'cpu_usage_limit': 80,
                'db_query_timeout_seconds': 5
            },
            'system_baseline': {'enabled': True},
            'load_test': {'enabled': True, 'duration_seconds': 60, 'thread_count': 4},
            'stress_test': {'enabled': True},
            'stability_test': {'enabled': True}
        }
    
    def _get_enabled_strategies(self) -> List[str]:
        """获取启用的测试策略"""
        enabled = []
        
        if self.performance_config.get('system_baseline', {}).get('enabled', True):
            enabled.append('system_baseline')
        
        if self.performance_config.get('load_test', {}).get('enabled', True):
            enabled.append('load_test')
        
        # 可以继续添加其他策略的检查
        
        return enabled
    
    def _run_tests(self) -> List[TestResult]:
        """执行性能基准验收测试"""
        test_results = []
        
        # 验证前提条件
        if not self._validate_prerequisites():
            test_results.append(TestResult(
                phase=self.phase_name,
                test_name="prerequisites_validation",
                status=TestStatus.FAILED,
                execution_time=0.0,
                error_message="性能基准验收前提条件验证失败"
            ))
            return test_results
        
        # 执行启用的性能测试策略
        for strategy_name in self.enabled_strategies:
            test_results.append(
                self._execute_test(
                    f"{strategy_name}_test",
                    lambda s=strategy_name: self._execute_performance_strategy(s)
                )
            )
        
        return test_results
    
    def _execute_performance_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """执行性能测试策略"""
        self.logger.info(f"执行性能测试策略: {strategy_name}")
        
        try:
            result = self.test_executor.execute_test(strategy_name)
            
            if not result['success']:
                raise AcceptanceTestError(f"性能测试策略 {strategy_name} 执行失败: {result.get('error', '未知错误')}")
            
            return {
                "strategy_name": strategy_name,
                "test_results": result['results'],
                "validation_results": result['validation'],
                "performance_score": result['validation']['score'],
                "issues": result['validation']['issues'],
                "test_passed": result['validation']['passed']
            }
            
        except Exception as e:
            raise AcceptanceTestError(f"性能测试策略 {strategy_name} 执行异常: {e}")
    
    def _validate_prerequisites(self) -> bool:
        """验证测试前提条件"""
        try:
            # 检查必要的系统工具
            import psutil
            import numpy as np
            
            # 检查配置完整性
            required_config = ['thresholds']
            for key in required_config:
                if key not in self.performance_config:
                    self.logger.error(f"缺少必要配置: {key}")
                    return False
            
            return True
            
        except ImportError as e:
            self.logger.error(f"缺少必要的依赖包: {e}")
            return False
        except Exception as e:
            self.logger.error(f"前提条件验证失败: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """清理测试资源"""
        try:
            # 清理性能测试相关资源
            if hasattr(self, 'test_executor'):
                del self.test_executor
            
            self.logger.info("性能基准验收阶段资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")


# 使用示例
if __name__ == "__main__":
    # 测试重构后的性能基准验收阶段
    config = {
        'config_file': '.env.acceptance',
        'test_timeout': 300,
        'max_concurrent_tests': 3
    }
    
    phase = PerformanceBenchmarkPhase("性能基准验收测试", config)
    results = phase.execute()
    
    for result in results:
        print(f"测试: {result.test_name}, 状态: {result.status.value}")
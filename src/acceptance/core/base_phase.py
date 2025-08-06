"""
测试阶段基类
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable

from .models import TestResult, TestStatus
from .exceptions import handle_test_exceptions, TestExecutionError


class BaseTestPhase(ABC):
    """测试阶段基类"""
    
    def __init__(self, phase_name: str, config: Dict[str, Any]):
        self.phase_name = phase_name
        self.config = config
        self.results: List[TestResult] = []
        self.logger = logging.getLogger(f"{__name__}.{phase_name}")
    
    def execute(self) -> List[TestResult]:
        """执行测试阶段"""
        self.logger.info(f"开始执行 {self.phase_name} 阶段测试")
        start_time = time.time()
        
        try:
            self.results = self._run_tests()
            execution_time = time.time() - start_time
            
            passed_count = sum(1 for r in self.results if r.status == TestStatus.PASSED)
            total_count = len(self.results)
            
            self.logger.info(
                f"{self.phase_name} 阶段完成: {passed_count}/{total_count} 通过, "
                f"耗时 {execution_time:.2f}秒"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = TestResult(
                phase=self.phase_name,
                test_name="phase_execution",
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            self.results = [error_result]
            self.logger.error(f"{self.phase_name} 阶段执行失败: {e}")
        
        return self.results
    
    @abstractmethod
    def _run_tests(self) -> List[TestResult]:
        """子类需要实现的具体测试逻辑"""
        pass
    
    @handle_test_exceptions(logger=logging.getLogger(__name__))
    def _execute_test(
        self, 
        test_name: str, 
        test_func: Callable, 
        *args, 
        **kwargs
    ) -> TestResult:
        """执行单个测试并返回结果"""
        start_time = time.time()
        
        try:
            self.logger.debug(f"执行测试: {test_name}")
            result = test_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            return TestResult(
                phase=self.phase_name,
                test_name=test_name,
                status=TestStatus.PASSED,
                execution_time=execution_time,
                details=result if isinstance(result, dict) else None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"测试 {test_name} 失败: {e}")
            
            return TestResult(
                phase=self.phase_name,
                test_name=test_name,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """获取配置值的辅助方法"""
        return self.config.get(key, default)
    
    def _validate_prerequisites(self) -> bool:
        """验证测试前提条件"""
        return True
    
    def _cleanup_resources(self) -> None:
        """清理测试资源"""
        pass
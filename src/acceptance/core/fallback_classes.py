"""
验收测试框架的替代类定义
当核心模块导入失败时使用这些类作为后备
"""
import time
import logging
from typing import Dict, Any, Callable
from enum import Enum


class TestStatus(Enum):
    """测试状态枚举"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class TestResult:
    """测试结果数据类"""
    
    def __init__(self, phase: str, test_name: str, status: TestStatus, 
                 execution_time: float, error_message: str = None, 
                 details: Dict[str, Any] = None):
        self.phase = phase
        self.test_name = test_name
        self.status = status
        self.execution_time = execution_time
        self.error_message = error_message
        self.details = details or {}


class AcceptanceTestError(Exception):
    """验收测试异常基类"""
    pass


class BaseTestPhase:
    """测试阶段基类的后备实现"""
    
    def __init__(self, phase_name: str, config: Dict[str, Any]):
        self.phase_name = phase_name
        self.config = config
        self.logger = self._create_logger()
    
    def _create_logger(self) -> logging.Logger:
        """创建日志记录器"""
        logger = logging.getLogger(self.phase_name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _execute_test(self, test_name: str, test_func: Callable) -> TestResult:
        """执行单个测试的模板方法"""
        start_time = time.time()
        try:
            result = test_func()
            end_time = time.time()
            
            return TestResult(
                phase=self.phase_name,
                test_name=test_name,
                status=TestStatus.PASSED,
                execution_time=end_time - start_time,
                details=result
            )
        except Exception as e:
            end_time = time.time()
            self.logger.error(f"测试 {test_name} 执行失败: {e}")
            return TestResult(
                phase=self.phase_name,
                test_name=test_name,
                status=TestStatus.FAILED,
                execution_time=end_time - start_time,
                error_message=str(e)
            )
    
    def _validate_prerequisites(self) -> bool:
        """验证前提条件的默认实现"""
        return True
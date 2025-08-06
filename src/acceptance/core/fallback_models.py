"""
Fallback模型定义
当核心模块不可用时提供基本功能
"""
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


class TestStatus:
    """测试状态枚举"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class TestResult:
    """测试结果数据类"""
    phase: str
    test_name: str
    status: str
    execution_time: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class AcceptanceTestError(Exception):
    """验收测试异常"""
    pass


class LoggerManager:
    """日志管理器"""
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取或创建logger"""
        if name not in cls._loggers:
            cls._loggers[name] = cls._create_logger(name)
        return cls._loggers[name]
    
    @staticmethod
    def _create_logger(name: str) -> logging.Logger:
        """创建新的logger"""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


class TestExecutor:
    """测试执行器"""
    
    def __init__(self, phase_name: str):
        self.phase_name = phase_name
        self.logger = LoggerManager.get_logger(phase_name)
    
    def execute_test(self, test_name: str, test_func) -> TestResult:
        """执行单个测试"""
        self.logger.info(f"开始执行测试: {test_name}")
        
        start_time = time.time()
        try:
            result = test_func()
            execution_time = time.time() - start_time
            
            self.logger.info(f"测试 {test_name} 执行成功，耗时 {execution_time:.2f}秒")
            
            return TestResult(
                phase=self.phase_name,
                test_name=test_name,
                status=TestStatus.PASSED,
                execution_time=execution_time,
                details=result if isinstance(result, dict) else {"result": result}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"测试 {test_name} 执行失败: {str(e)}")
            
            return TestResult(
                phase=self.phase_name,
                test_name=test_name,
                status=TestStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )


class BaseTestPhase(ABC):
    """基础测试阶段类"""
    
    def __init__(self, phase_name: str, config: Dict[str, Any]):
        self.phase_name = phase_name
        self.config = config
        self.logger = LoggerManager.get_logger(phase_name)
        self.executor = TestExecutor(phase_name)
    
    def _execute_test(self, test_name: str, test_func) -> TestResult:
        """执行单个测试"""
        return self.executor.execute_test(test_name, test_func)
    
    def _validate_prerequisites(self) -> bool:
        """验证前提条件"""
        return True
    
    @abstractmethod
    def run_tests(self) -> Dict[str, Any]:
        """运行测试的抽象方法"""
        pass
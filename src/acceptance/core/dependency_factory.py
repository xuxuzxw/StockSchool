"""
依赖工厂 - 统一管理测试组件的创建和依赖注入
"""
from typing import Protocol, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging


class TestPhaseProtocol(Protocol):
    """测试阶段协议"""
    phase_name: str
    config: Dict[str, Any]
    logger: logging.Logger
    
    def _execute_test(self, test_name: str, test_func) -> 'TestResult':
        ...
    
    def _validate_prerequisites(self) -> bool:
        ...


class TestComponentFactory:
    """测试组件工厂"""
    
    def __init__(self):
        self._components = {}
        self._fallback_components = {}
    
    def register_component(self, name: str, component_class, fallback_class=None):
        """注册组件"""
        self._components[name] = component_class
        if fallback_class:
            self._fallback_components[name] = fallback_class
    
    def create_component(self, name: str, *args, **kwargs):
        """创建组件实例"""
        try:
            component_class = self._components.get(name)
            if component_class:
                return component_class(*args, **kwargs)
        except ImportError as e:
            logging.warning(f"无法创建组件 {name}: {e}")
            
        # 使用备用组件
        fallback_class = self._fallback_components.get(name)
        if fallback_class:
            return fallback_class(*args, **kwargs)
            
        raise ImportError(f"无法创建组件 {name}")


# 全局工厂实例
component_factory = TestComponentFactory()


class MockTestResult:
    """测试结果模拟类"""
    def __init__(self, phase: str, test_name: str, status: str, 
                 execution_time: float, error_message: str = None, details: Dict = None):
        self.phase = phase
        self.test_name = test_name
        self.status = status
        self.execution_time = execution_time
        self.error_message = error_message
        self.details = details or {}


class MockTestStatus:
    """测试状态模拟类"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class MockBaseTestPhase:
    """基础测试阶段模拟类"""
    def __init__(self, phase_name: str, config: Dict[str, Any]):
        self.phase_name = phase_name
        self.config = config
        self.logger = self._create_logger()
    
    def _create_logger(self):
        logger = logging.getLogger(self.phase_name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _execute_test(self, test_name: str, test_func):
        """执行单个测试"""
        import time
        start_time = time.time()
        try:
            result = test_func()
            end_time = time.time()
            
            return MockTestResult(
                phase=self.phase_name,
                test_name=test_name,
                status=MockTestStatus.PASSED,
                execution_time=end_time - start_time,
                details=result
            )
        except Exception as e:
            end_time = time.time()
            return MockTestResult(
                phase=self.phase_name,
                test_name=test_name,
                status=MockTestStatus.FAILED,
                execution_time=end_time - start_time,
                error_message=str(e)
            )
    
    def _validate_prerequisites(self) -> bool:
        """验证前提条件"""
        return True


# 注册组件
try:
    from ..core.base_phase import BaseTestPhase
    from ..core.models import TestResult, TestStatus
    from ..core.exceptions import AcceptanceTestError
    
    component_factory.register_component('BaseTestPhase', BaseTestPhase, MockBaseTestPhase)
    component_factory.register_component('TestResult', TestResult, MockTestResult)
    component_factory.register_component('TestStatus', TestStatus, MockTestStatus)
    
except ImportError:
    component_factory.register_component('BaseTestPhase', MockBaseTestPhase)
    component_factory.register_component('TestResult', MockTestResult)
    component_factory.register_component('TestStatus', MockTestStatus)
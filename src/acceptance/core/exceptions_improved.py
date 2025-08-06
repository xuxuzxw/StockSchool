"""
改进的异常处理 - 提供更精细的异常类型和错误信息
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ErrorContext:
    """错误上下文信息"""
    phase_name: Optional[str] = None
    test_name: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class AcceptanceTestError(Exception):
    """验收测试基础异常"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "ACCEPTANCE_TEST_ERROR"
        self.context = context or ErrorContext()
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": {
                "phase_name": self.context.phase_name,
                "test_name": self.context.test_name,
                "session_id": self.context.session_id,
                "timestamp": self.context.timestamp,
                "additional_info": self.context.additional_info
            },
            "cause": str(self.cause) if self.cause else None
        }


class ConfigurationError(AcceptanceTestError):
    """配置相关错误"""
    
    def __init__(
        self, 
        message: str, 
        missing_configs: Optional[List[str]] = None,
        invalid_configs: Optional[Dict[str, str]] = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message, 
            error_code="CONFIGURATION_ERROR",
            context=context
        )
        self.missing_configs = missing_configs or []
        self.invalid_configs = invalid_configs or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        result.update({
            "missing_configs": self.missing_configs,
            "invalid_configs": self.invalid_configs
        })
        return result


class PhaseExecutionError(AcceptanceTestError):
    """阶段执行错误"""
    
    def __init__(
        self, 
        message: str, 
        phase_name: str,
        failed_tests: Optional[List[str]] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        if not context:
            context = ErrorContext(phase_name=phase_name)
        elif not context.phase_name:
            context.phase_name = phase_name
            
        super().__init__(
            message, 
            error_code="PHASE_EXECUTION_ERROR",
            context=context,
            cause=cause
        )
        self.phase_name = phase_name
        self.failed_tests = failed_tests or []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        result.update({
            "phase_name": self.phase_name,
            "failed_tests": self.failed_tests
        })
        return result


class InfrastructureError(AcceptanceTestError):
    """基础设施相关错误"""
    
    def __init__(
        self, 
        message: str, 
        service_name: str,
        service_status: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message, 
            error_code="INFRASTRUCTURE_ERROR",
            context=context,
            cause=cause
        )
        self.service_name = service_name
        self.service_status = service_status
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        result.update({
            "service_name": self.service_name,
            "service_status": self.service_status
        })
        return result


class DataServiceError(AcceptanceTestError):
    """数据服务相关错误"""
    
    def __init__(
        self, 
        message: str, 
        data_source: str,
        operation: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message, 
            error_code="DATA_SERVICE_ERROR",
            context=context,
            cause=cause
        )
        self.data_source = data_source
        self.operation = operation
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        result.update({
            "data_source": self.data_source,
            "operation": self.operation
        })
        return result


class ComputeEngineError(AcceptanceTestError):
    """计算引擎相关错误"""
    
    def __init__(
        self, 
        message: str, 
        factor_name: Optional[str] = None,
        computation_type: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message, 
            error_code="COMPUTE_ENGINE_ERROR",
            context=context,
            cause=cause
        )
        self.factor_name = factor_name
        self.computation_type = computation_type
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        result.update({
            "factor_name": self.factor_name,
            "computation_type": self.computation_type
        })
        return result


class AIServiceError(AcceptanceTestError):
    """AI服务相关错误"""
    
    def __init__(
        self, 
        message: str, 
        model_name: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message, 
            error_code="AI_SERVICE_ERROR",
            context=context,
            cause=cause
        )
        self.model_name = model_name
        self.operation = operation
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        result.update({
            "model_name": self.model_name,
            "operation": self.operation
        })
        return result


class APIServiceError(AcceptanceTestError):
    """API服务相关错误"""
    
    def __init__(
        self, 
        message: str, 
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message, 
            error_code="API_SERVICE_ERROR",
            context=context,
            cause=cause
        )
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_body = response_body
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        result.update({
            "endpoint": self.endpoint,
            "status_code": self.status_code,
            "response_body": self.response_body
        })
        return result


class PerformanceError(AcceptanceTestError):
    """性能相关错误"""
    
    def __init__(
        self, 
        message: str, 
        metric_name: str,
        expected_value: Optional[float] = None,
        actual_value: Optional[float] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message, 
            error_code="PERFORMANCE_ERROR",
            context=context,
            cause=cause
        )
        self.metric_name = metric_name
        self.expected_value = expected_value
        self.actual_value = actual_value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        result.update({
            "metric_name": self.metric_name,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value
        })
        return result


class TimeoutError(AcceptanceTestError):
    """超时错误"""
    
    def __init__(
        self, 
        message: str, 
        timeout_seconds: float,
        operation: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message, 
            error_code="TIMEOUT_ERROR",
            context=context,
            cause=cause
        )
        self.timeout_seconds = timeout_seconds
        self.operation = operation
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        result.update({
            "timeout_seconds": self.timeout_seconds,
            "operation": self.operation
        })
        return result


# 异常工厂函数
def create_configuration_error(
    message: str,
    missing_configs: Optional[List[str]] = None,
    invalid_configs: Optional[Dict[str, str]] = None,
    session_id: Optional[str] = None
) -> ConfigurationError:
    """创建配置错误"""
    context = ErrorContext(session_id=session_id)
    return ConfigurationError(
        message=message,
        missing_configs=missing_configs,
        invalid_configs=invalid_configs,
        context=context
    )


def create_phase_execution_error(
    message: str,
    phase_name: str,
    failed_tests: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    cause: Optional[Exception] = None
) -> PhaseExecutionError:
    """创建阶段执行错误"""
    context = ErrorContext(
        phase_name=phase_name,
        session_id=session_id
    )
    return PhaseExecutionError(
        message=message,
        phase_name=phase_name,
        failed_tests=failed_tests,
        context=context,
        cause=cause
    )


def create_infrastructure_error(
    message: str,
    service_name: str,
    service_status: Optional[str] = None,
    session_id: Optional[str] = None,
    cause: Optional[Exception] = None
) -> InfrastructureError:
    """创建基础设施错误"""
    context = ErrorContext(session_id=session_id)
    return InfrastructureError(
        message=message,
        service_name=service_name,
        service_status=service_status,
        context=context,
        cause=cause
    )


# 异常处理装饰器
def handle_acceptance_test_errors(
    error_type: type = AcceptanceTestError,
    default_message: str = "验收测试执行失败"
):
    """异常处理装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type:
                # 重新抛出已知的验收测试异常
                raise
            except Exception as e:
                # 将未知异常包装为验收测试异常
                raise error_type(
                    message=f"{default_message}: {str(e)}",
                    cause=e
                ) from e
        return wrapper
    return decorator
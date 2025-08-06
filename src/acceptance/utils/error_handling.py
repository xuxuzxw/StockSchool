"""
统一的错误处理和日志记录工具
"""
import logging
import functools
import traceback
from typing import Any, Callable, Dict, Optional
from enum import Enum


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AcceptanceTestError(Exception):
    """验收测试专用异常"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.severity = severity
        self.details = details or {}
        self.timestamp = datetime.now()


class ErrorHandler:
    """统一错误处理器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_counts: Dict[str, int] = {}
    
    def handle_component_initialization_error(self, component_name: str, 
                                            error: Exception) -> None:
        """处理组件初始化错误"""
        error_key = f"init_{component_name}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        if self.error_counts[error_key] <= 3:  # 只记录前3次错误
            self.logger.warning(
                f"组件 {component_name} 初始化失败 (第{self.error_counts[error_key]}次): {error}",
                extra={
                    "component": component_name,
                    "error_type": type(error).__name__,
                    "error_count": self.error_counts[error_key]
                }
            )
        
        if self.error_counts[error_key] > 5:
            raise AcceptanceTestError(
                f"组件 {component_name} 初始化连续失败超过5次",
                severity=ErrorSeverity.HIGH,
                details={"component": component_name, "last_error": str(error)}
            )
    
    def handle_test_execution_error(self, test_name: str, error: Exception) -> Dict[str, Any]:
        """处理测试执行错误"""
        self.logger.error(
            f"测试 {test_name} 执行失败: {error}",
            extra={
                "test_name": test_name,
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc()
            }
        )
        
        return {
            "status": "failed",
            "error_message": str(error),
            "error_type": type(error).__name__,
            "test_name": test_name,
            "severity": ErrorSeverity.MEDIUM.value
        }


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"第{attempt + 1}次尝试失败，{delay}秒后重试: {e}")
                        time.sleep(delay)
                    else:
                        print(f"所有重试都失败了，最后错误: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator


def log_performance(logger: logging.Logger):
    """性能日志装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    f"方法 {func.__name__} 执行完成",
                    extra={
                        "method": func.__name__,
                        "execution_time": execution_time,
                        "status": "success"
                    }
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                logger.error(
                    f"方法 {func.__name__} 执行失败",
                    extra={
                        "method": func.__name__,
                        "execution_time": execution_time,
                        "status": "failed",
                        "error": str(e)
                    }
                )
                
                raise
        
        return wrapper
    return decorator


class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志格式"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_test_start(self, test_name: str, config: Dict[str, Any]):
        """记录测试开始"""
        self.logger.info(
            f"开始执行测试: {test_name}",
            extra={
                "event": "test_start",
                "test_name": test_name,
                "config": config
            }
        )
    
    def log_test_result(self, test_name: str, result: Dict[str, Any], 
                       execution_time: float):
        """记录测试结果"""
        status = result.get('status', 'unknown')
        
        log_method = self.logger.info if status == 'success' else self.logger.error
        
        log_method(
            f"测试 {test_name} 完成: {status}",
            extra={
                "event": "test_complete",
                "test_name": test_name,
                "status": status,
                "execution_time": execution_time,
                "result_summary": self._extract_summary(result)
            }
        )
    
    def log_component_status(self, component_name: str, status: str, 
                           details: Optional[Dict[str, Any]] = None):
        """记录组件状态"""
        self.logger.info(
            f"组件 {component_name} 状态: {status}",
            extra={
                "event": "component_status",
                "component": component_name,
                "status": status,
                "details": details or {}
            }
        )
    
    def _extract_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """提取结果摘要"""
        summary_keys = ['score', 'success_count', 'total_count', 'status']
        return {key: result.get(key) for key in summary_keys if key in result}
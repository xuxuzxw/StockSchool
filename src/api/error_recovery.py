import asyncio
import random
import time
import traceback
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

from loguru import logger

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API错误恢复机制
智能重试、指数退避、错误分类和恢复策略
"""


class ErrorType(Enum):
    """错误类型枚举"""

    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    RESOURCE_NOT_FOUND = "resource_not_found"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    DEPENDENCY_ERROR = "dependency_error"


class RetryStrategy(Enum):
    """重试策略枚举"""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"


class ErrorClassifier:
    """错误分类器"""

    @staticmethod
    def classify_error(error: Exception) -> ErrorType:
        """对错误进行分类"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # 网络错误
        if any(keyword in error_str for keyword in ["connection", "timeout", "network"]):
            return ErrorType.NETWORK_ERROR
        elif "timeout" in error_type:
            return ErrorType.TIMEOUT_ERROR

        # 数据库错误
        elif any(keyword in error_str for keyword in ["database", "sql", "connection pool"]):
            return ErrorType.DATABASE_ERROR

        # 限流错误
        elif "rate limit" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT_ERROR

        # 验证错误
        elif "validation" in error_str or "invalid" in error_str:
            return ErrorType.VALIDATION_ERROR

        # 认证错误
        elif "authentication" in error_str or "unauthorized" in error_str or "401" in error_str:
            return ErrorType.AUTHENTICATION_ERROR

        # 授权错误
        elif "authorization" in error_str or "forbidden" in error_str or "403" in error_str:
            return ErrorType.AUTHORIZATION_ERROR

        # 资源未找到
        elif "not found" in error_str or "404" in error_str:
            return ErrorType.RESOURCE_NOT_FOUND

        # 依赖错误
        elif "dependency" in error_str or "service unavailable" in error_str:
            return ErrorType.DEPENDENCY_ERROR

        # 默认内部服务器错误
        else:
            return ErrorType.INTERNAL_SERVER_ERROR


class RetryPolicy:
    """重试策略配置"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        jitter: bool = True,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.jitter = jitter
        self.exponential_base = exponential_base

    def calculate_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        if self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (self.exponential_base**attempt)
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.IMMEDIATE:
            delay = 0
        else:  # NO_RETRY
            return self.max_delay + 1  # 返回一个超过最大值的延迟

        # 添加随机抖动
        if self.jitter:
            delay *= 1 + random.uniform(-0.1, 0.1)

        return min(delay, self.max_delay)

    def should_retry(self, attempt: int, error_type: ErrorType) -> bool:
        """检查是否应该重试"""
        if attempt >= self.max_retries:
            return False

        # 根据错误类型决定是否重试
        non_retryable_errors = [
            ErrorType.VALIDATION_ERROR,
            ErrorType.AUTHENTICATION_ERROR,
            ErrorType.AUTHORIZATION_ERROR,
            ErrorType.RESOURCE_NOT_FOUND,
        ]

        return error_type not in non_retryable_errors


class ErrorRecoveryContext:
    """错误恢复上下文"""

    def __init__(self):
        """方法描述"""
        self.last_attempt_time: Dict[str, float] = {}
        self.error_history: Dict[str, List[Dict[str, Any]]] = {}
        self.circuit_breaker_status: Dict[str, bool] = {}

    def record_attempt(self, identifier: str, error: Optional[Exception] = None):
        """记录尝试"""
        current_time = time.time()

        if identifier not in self.attempt_count:
            self.attempt_count[identifier] = 0
            self.error_history[identifier] = []

        self.attempt_count[identifier] += 1
        self.last_attempt_time[identifier] = current_time

        if error:
            self.error_history[identifier].append(
                {
                    "error": str(error),
                    "type": ErrorClassifier.classify_error(error).value,
                    "timestamp": current_time,
                    "traceback": traceback.format_exc(),
                }
            )

    def get_attempt_count(self, identifier: str) -> int:
        """获取尝试次数"""
        return self.attempt_count.get(identifier, 0)

    def get_error_history(self, identifier: str) -> List[Dict[str, Any]]:
        """获取错误历史"""
        return self.error_history.get(identifier, [])

    def reset_context(self, identifier: str):
        """重置上下文"""
        if identifier in self.attempt_count:
            del self.attempt_count[identifier]
        if identifier in self.last_attempt_time:
            del self.last_attempt_time[identifier]
        if identifier in self.error_history:
            del self.error_history[identifier]

    def is_circuit_breaker_open(self, identifier: str) -> bool:
        """检查熔断器状态"""
        return self.circuit_breaker_status.get(identifier, False)

    def open_circuit_breaker(self, identifier: str):
        """打开熔断器"""
        self.circuit_breaker_status[identifier] = True

    def close_circuit_breaker(self, identifier: str):
        """关闭熔断器"""
        self.circuit_breaker_status[identifier] = False


class ErrorRecoveryManager:
    """错误恢复管理器"""

    def __init__(self):
        """方法描述"""
        self.default_policy = RetryPolicy()
        self.custom_policies: Dict[ErrorType, RetryPolicy] = {
            ErrorType.NETWORK_ERROR: RetryPolicy(max_retries=5, base_delay=2.0),
            ErrorType.DATABASE_ERROR: RetryPolicy(max_retries=3, base_delay=1.0),
            ErrorType.RATE_LIMIT_ERROR: RetryPolicy(max_retries=10, base_delay=5.0),
            ErrorType.TIMEOUT_ERROR: RetryPolicy(max_retries=3, base_delay=1.0),
            ErrorType.DEPENDENCY_ERROR: RetryPolicy(max_retries=5, base_delay=3.0),
        }

    def get_policy(self, error_type: ErrorType) -> RetryPolicy:
        """获取指定错误类型的重试策略"""
        return self.custom_policies.get(error_type, self.default_policy)

    async def execute_with_recovery(self, func: Callable, identifier: str, *args, **kwargs) -> Any:
        """执行函数并应用错误恢复"""

        # 检查熔断器
        if self.context.is_circuit_breaker_open(identifier):
            logger.warning(f"熔断器已打开，跳过执行: {identifier}")
            raise Exception("Circuit breaker is open")

        policy = None
        last_error = None

        while True:
            try:
                # 执行函数
                result = await func(*args, **kwargs)

                # 成功时重置上下文
                if self.context.get_attempt_count(identifier) > 0:
                    self.context.reset_context(identifier)
                    logger.info(f"错误恢复成功: {identifier}")

                return result

            except Exception as e:
                last_error = e
                error_type = ErrorClassifier.classify_error(e)
                policy = self.get_policy(error_type)

                # 记录尝试
                self.context.record_attempt(identifier, e)

                # 检查是否应该重试
                attempt_count = self.context.get_attempt_count(identifier)

                if not policy.should_retry(attempt_count - 1, error_type):
                    logger.error(
                        f"达到最大重试次数，放弃重试: {identifier}, "
                        f"错误类型: {error_type.value}, 尝试次数: {attempt_count}"
                    )

                    # 打开熔断器（对于某些错误类型）
                    if error_type in [ErrorType.DATABASE_ERROR, ErrorType.DEPENDENCY_ERROR]:
                        self.context.open_circuit_breaker(identifier)
                        logger.error(f"打开熔断器: {identifier}")

                    raise e

                # 计算延迟
                delay = policy.calculate_delay(attempt_count - 1)

                logger.warning(
                    f"准备重试: {identifier}, "
                    f"错误类型: {error_type.value}, "
                    f"尝试次数: {attempt_count}, "
                    f"延迟: {delay:.2f}秒"
                )

                await asyncio.sleep(delay)

    def get_recovery_stats(self, identifier: str) -> Dict[str, Any]:
        """获取恢复统计信息"""
        return {
            "attempt_count": self.context.get_attempt_count(identifier),
            "error_history": self.context.get_error_history(identifier),
            "circuit_breaker_open": self.context.is_circuit_breaker_open(identifier),
            "last_attempt_time": self.context.last_attempt_time.get(identifier),
        }

    def reset_recovery_stats(self, identifier: str):
        """重置恢复统计"""
        self.context.reset_context(identifier)
        self.context.close_circuit_breaker(identifier)


# 全局错误恢复管理器
error_recovery_manager = ErrorRecoveryManager()


def with_error_recovery(identifier_template: Optional[str] = None, custom_policy: Optional[RetryPolicy] = None):
    """
    错误恢复装饰器

    Args:
        identifier_template: 标识符模板，支持格式化字符串
        custom_policy: 自定义重试策略
    """

    def decorator(func: Callable) -> Callable:
        """方法描述"""

        async def wrapper(*args, **kwargs):
            # 生成标识符
            if identifier_template:
                try:
                    identifier = identifier_template.format(*args, **kwargs)
                except:
                    identifier = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            else:
                identifier = f"{func.__name__}_{hash(str(args) + str(kwargs))}"

            # 使用自定义策略或默认策略
            if custom_policy:
                original_policy = error_recovery_manager.default_policy
                error_recovery_manager.default_policy = custom_policy
                try:
                    return await error_recovery_manager.execute_with_recovery(func, identifier, *args, **kwargs)
                finally:
                    error_recovery_manager.default_policy = original_policy
            else:
                return await error_recovery_manager.execute_with_recovery(func, identifier, *args, **kwargs)

        return wrapper

    return decorator


# 快捷错误恢复装饰器
def network_recovery(func: Callable) -> Callable:
    """网络错误恢复装饰器"""
    policy = RetryPolicy(max_retries=5, base_delay=2.0)
    return with_error_recovery(custom_policy=policy)(func)


def database_recovery(func: Callable) -> Callable:
    """数据库错误恢复装饰器"""
    policy = RetryPolicy(max_retries=3, base_delay=1.0)
    return with_error_recovery(custom_policy=policy)(func)


def rate_limit_recovery(func: Callable) -> Callable:
    """限流错误恢复装饰器"""
    policy = RetryPolicy(max_retries=10, base_delay=5.0)
    return with_error_recovery(custom_policy=policy)(func)

import asyncio
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API限流中间件
支持基于用户角色的限流策略和智能错误恢复
"""



class RateLimitError(HTTPException):
    """限流异常"""
    def __init__(self, detail: str = "请求过于频繁，请稍后再试", retry_after: int = 60):
        """方法描述"""
            status_code=429,
            detail={
                "message": detail,
                "retry_after": retry_after,
                "error_code": "RATE_LIMIT_EXCEEDED"
            },
            headers={"Retry-After": str(retry_after)}
        )


class UserRateLimiter:
    """用户级限流器"""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10,
        window_seconds: int = 60
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit
        self.window_seconds = window_seconds
        self.user_requests: Dict[str, Dict[str, list]] = {}

    def _clean_old_requests(self, user_id: str, current_time: float):
        """清理过期的请求记录"""
        if user_id not in self.user_requests:
            self.user_requests[user_id] = {"minute": [], "hour": []}

        # 清理分钟窗口
        cutoff_minute = current_time - 60
        self.user_requests[user_id]["minute"] = [
            req_time for req_time in self.user_requests[user_id]["minute"]
            if req_time > cutoff_minute
        ]

        # 清理小时窗口
        cutoff_hour = current_time - 3600
        self.user_requests[user_id]["hour"] = [
            req_time for req_time in self.user_requests[user_id]["hour"]
            if req_time > cutoff_hour
        ]

    def allow_request(self, user_id: str) -> bool:
        """检查是否允许用户请求"""
        current_time = time.time()

        self._clean_old_requests(user_id, current_time)

        # 检查分钟限制
        if len(self.user_requests[user_id]["minute"]) >= self.requests_per_minute:
            return False

        # 检查小时限制
        if len(self.user_requests[user_id]["hour"]) >= self.requests_per_hour:
            return False

        # 记录请求
        self.user_requests[user_id]["minute"].append(current_time)
        self.user_requests[user_id]["hour"].append(current_time)

        return True

    def get_retry_after(self, user_id: str) -> int:
        """计算重试等待时间"""
        if user_id not in self.user_requests:
            return 0

        current_time = time.time()
        minute_requests = self.user_requests[user_id]["minute"]
        hour_requests = self.user_requests[user_id]["hour"]

        # 计算分钟窗口的剩余时间
        if minute_requests:
            oldest_minute = min(minute_requests)
            retry_minute = max(0, 60 - (current_time - oldest_minute))
        else:
            retry_minute = 0

        # 计算小时窗口的剩余时间
        if hour_requests:
            oldest_hour = min(hour_requests)
            retry_hour = max(0, 3600 - (current_time - oldest_hour))
        else:
            retry_hour = 0

        return max(retry_minute, retry_hour)


class APIRateLimiter:
    """API限流管理器"""

    def __init__(self):
        """方法描述"""
        self.role_limits = {
            "guest": UserRateLimiter(requests_per_minute=10, requests_per_hour=100),
            "user": UserRateLimiter(requests_per_minute=60, requests_per_hour=1000),
            "developer": UserRateLimiter(requests_per_minute=120, requests_per_hour=2000),
            "enterprise": UserRateLimiter(requests_per_minute=300, requests_per_hour=5000),
            "admin": UserRateLimiter(requests_per_minute=1000, requests_per_hour=10000)
        }

        # IP级限流（防止恶意攻击）
        self.ip_limiter = UserRateLimiter(requests_per_minute=30, requests_per_hour=500)

        # 全局API限流
        self.global_limiter = UserRateLimiter(requests_per_minute=1000, requests_per_hour=10000)

    def get_user_role(self, user_id: str) -> str:
        """获取用户角色（实际应用中应该从数据库或认证系统获取）"""
        # 这里简化处理，实际应该查询用户权限
        if user_id.startswith("admin_"):
            return "admin"
        elif user_id.startswith("dev_"):
            return "developer"
        elif user_id.startswith("ent_"):
            return "enterprise"
        elif user_id.startswith("user_"):
            return "user"
        else:
            return "guest"

    def get_user_id_from_request(self, request: Request) -> str:
        """从请求中获取用户ID"""
        # 优先从认证头获取
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # 实际应该解析JWT token
            token = auth_header.split(" ")[1]
            return f"user_{token[:8]}"  # 简化处理

        # 从IP地址获取
        client_ip = request.client.host if request.client else "unknown"
        return f"ip_{client_ip}"

    def check_rate_limit(self, user_id: str, ip_address: str) -> bool:
        """检查限流状态"""
        # 检查全局限流
        if not self.global_limiter.allow_request("global"):
            return False

        # 检查IP限流
        if not self.ip_limiter.allow_request(f"ip_{ip_address}"):
            return False

        # 检查用户级限流
        role = self.get_user_role(user_id)
        user_limiter = self.role_limits.get(role, self.role_limits["guest"])

        return user_limiter.allow_request(user_id)

    def get_retry_after(self, user_id: str, ip_address: str) -> int:
        """获取重试等待时间"""
        role = self.get_user_role(user_id)
        user_limiter = self.role_limits.get(role, self.role_limits["guest"])

        user_retry = user_limiter.get_retry_after(user_id)
        ip_retry = self.ip_limiter.get_retry_after(f"ip_{ip_address}")
        global_retry = self.global_limiter.get_retry_after("global")

        return max(user_retry, ip_retry, global_retry)


class RetryMechanism:
    """智能重试机制"""

    def __init__(self):
        """方法描述"""
        self.max_retries = 3
        self.base_delay = 1  # 基础延迟1秒
        self.max_delay = 300  # 最大延迟5分钟

    def calculate_delay(self, attempt: int) -> float:
        """计算指数退避延迟"""
        # 指数退避：2^attempt * base_delay，加入随机抖动
        import random
        delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
        return delay

    def record_attempt(self, identifier: str):
        """记录重试尝试"""
        if identifier not in self.retry_attempts:
            self.retry_attempts[identifier] = 0
        self.retry_attempts[identifier] += 1

    def should_retry(self, identifier: str) -> bool:
        """检查是否应该重试"""
        return self.retry_attempts.get(identifier, 0) < self.max_retries

    def get_retry_delay(self, identifier: str) -> float:
        """获取重试延迟"""
        attempt = self.retry_attempts.get(identifier, 0)
        return self.calculate_delay(attempt)

    def reset_attempts(self, identifier: str):
        """重置重试计数"""
        if identifier in self.retry_attempts:
            del self.retry_attempts[identifier]


# 全局限流器实例
api_rate_limiter = APIRateLimiter()
retry_mechanism = RetryMechanism()


def rate_limit_middleware(
    requests_per_minute: Optional[int] = None,
    requests_per_hour: Optional[int] = None,
    role_based: bool = True
):
    """
    API限流装饰器

    Args:
        requests_per_minute: 每分钟请求限制（覆盖角色默认）
        requests_per_hour: 每小时请求限制（覆盖角色默认）
        role_based: 是否启用基于角色的限流
    """
    def decorator(func: Callable) -> Callable:
        """方法描述"""
        async def wrapper(request: Request, *args, **kwargs):
            user_id = api_rate_limiter.get_user_id_from_request(request)
            client_ip = request.client.host if request.client else "unknown"

            # 检查限流
            if not api_rate_limiter.check_rate_limit(user_id, client_ip):
                retry_after = api_rate_limiter.get_retry_after(user_id, client_ip)

                # 记录限流事件
                logger.warning(
                    f"限流触发: 用户={user_id}, IP={client_ip}, 重试等待={retry_after}秒"
                )

                raise RateLimitError(
                    detail=f"请求过于频繁，请{retry_after}秒后再试",
                    retry_after=retry_after
                )

            try:
                # 执行原始函数
                result = await func(request, *args, **kwargs)

                # 成功时重置重试计数
                retry_mechanism.reset_attempts(f"{user_id}_{client_ip}")

                return result

            except Exception as e:
                # 错误处理
                identifier = f"{user_id}_{client_ip}"

                if retry_mechanism.should_retry(identifier):
                    retry_mechanism.record_attempt(identifier)
                    delay = retry_mechanism.get_retry_delay(identifier)

                    logger.warning(
                        f"API调用失败，准备重试: 用户={user_id}, "
                        f"延迟={delay:.2f}秒, 尝试次数={retry_mechanism.retry_attempts[identifier]}"
                    )

                    await asyncio.sleep(delay)
                    return await wrapper(request, *args, **kwargs)
                else:
                    # 达到最大重试次数
                    retry_mechanism.reset_attempts(identifier)
                    logger.error(f"API调用失败，已达到最大重试次数: 用户={user_id}")
                    raise

        return wrapper
    return decorator


# 快捷限流装饰器
def rate_limit_guest(func: Callable) -> Callable:
    """访客用户限流装饰器"""
    return rate_limit_middleware(
        requests_per_minute=10,
        requests_per_hour=100,
        role_based=False
    )(func)


def rate_limit_user(func: Callable) -> Callable:
    """普通用户限流装饰器"""
    return rate_limit_middleware(
        requests_per_minute=60,
        requests_per_hour=1000,
        role_based=True
    )(func)


def rate_limit_developer(func: Callable) -> Callable:
    """开发者用户限流装饰器"""
    return rate_limit_middleware(
        requests_per_minute=120,
        requests_per_hour=2000,
        role_based=True
    )(func)


def rate_limit_enterprise(func: Callable) -> Callable:
    """企业用户限流装饰器"""
    return rate_limit_middleware(
        requests_per_minute=300,
        requests_per_hour=5000,
        role_based=True
    )(func)
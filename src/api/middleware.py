import time
import uuid
from typing import Any, Dict, Optional

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.api.error_recovery import ErrorClassifier, ErrorType, error_recovery_manager
from src.api.rate_limit_config import rate_limit_manager
from src.api.rate_limiter import RateLimitError, api_rate_limiter

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API中间件
集成限流、错误恢复、监控和日志功能
"""


class RateLimitMiddleware(BaseHTTPMiddleware):
    """限流中间件"""

    def __init__(self, app):
        """方法描述"""
        self.rate_limiter = api_rate_limiter

    async def dispatch(self, request: Request, call_next):
        """处理请求限流"""

        # 跳过健康检查和文档请求
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # 获取用户标识
        user_id = self.rate_limiter.get_user_id_from_request(request)
        client_ip = request.client.host if request.client else "unknown"

        # 检查白名单
        config = rate_limit_manager.get_config()
        if config.is_whitelisted(client_ip, "ip") or config.is_whitelisted(user_id, "user"):
            return await call_next(request)

        # 检查限流
        if not self.rate_limiter.check_rate_limit(user_id, client_ip):
            retry_after = self.rate_limiter.get_retry_after(user_id, client_ip)

            logger.warning(
                f"限流触发: 路径={request.url.path}, " f"用户={user_id}, IP={client_ip}, 重试等待={retry_after}秒"
            )

            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": f"请求过于频繁，请{retry_after}秒后再试",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": retry_after,
                    "data": None,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # 继续处理请求
        return await call_next(request)


class ErrorRecoveryMiddleware(BaseHTTPMiddleware):
    """错误恢复中间件"""

    def __init__(self, app):
        """方法描述"""

    async def dispatch(self, request: Request, call_next):
        """处理错误恢复"""

        try:
            return await call_next(request)

        except Exception as e:
            error_type = ErrorClassifier.classify_error(e)

            # 记录错误
            logger.error(
                f"API错误: 路径={request.url.path}, "
                f"方法={request.method}, 错误类型={error_type.value}, "
                f"错误={str(e)}"
            )

            # 根据错误类型返回适当的响应
            if error_type == ErrorType.RATE_LIMIT_ERROR:
                return JSONResponse(
                    status_code=429,
                    content={
                        "success": False,
                        "message": "请求过于频繁",
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "data": None,
                    },
                )
            elif error_type == ErrorType.AUTHENTICATION_ERROR:
                return JSONResponse(
                    status_code=401,
                    content={
                        "success": False,
                        "message": "认证失败",
                        "error_code": "AUTHENTICATION_FAILED",
                        "data": None,
                    },
                )
            elif error_type == ErrorType.AUTHORIZATION_ERROR:
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "message": "权限不足",
                        "error_code": "AUTHORIZATION_FAILED",
                        "data": None,
                    },
                )
            elif error_type == ErrorType.VALIDATION_ERROR:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "请求参数错误",
                        "error_code": "VALIDATION_ERROR",
                        "data": None,
                    },
                )
            elif error_type == ErrorType.RESOURCE_NOT_FOUND:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "message": "资源未找到",
                        "error_code": "RESOURCE_NOT_FOUND",
                        "data": None,
                    },
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "message": "服务器内部错误",
                        "error_code": "INTERNAL_SERVER_ERROR",
                        "data": None,
                    },
                )


class LoggingMiddleware(BaseHTTPMiddleware):
    """日志中间件"""

    def __init__(self, app):
        """方法描述"""

    async def dispatch(self, request: Request, call_next):
        """记录请求和响应日志"""

        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # 记录请求
        start_time = time.time()
        logger.info(
            f"API请求开始: ID={request_id}, "
            f"方法={request.method}, 路径={request.url.path}, "
            f"IP={request.client.host if request.client else 'unknown'}"
        )

        # 处理请求
        response = await call_next(request)

        # 记录响应
        duration = time.time() - start_time
        logger.info(f"API请求完成: ID={request_id}, " f"状态码={response.status_code}, 耗时={duration:.3f}秒")

        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}"

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """指标收集中间件"""

    def __init__(self, app):
        """方法描述"""
        self.request_count = 0
        self.error_count = 0
        self.request_times: Dict[str, list] = {}

    async def dispatch(self, request: Request, call_next):
        """收集请求指标"""

        start_time = time.time()
        path = request.url.path
        method = request.method

        # 跳过健康检查和文档请求
        if path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        try:
            response = await call_next(request)

            duration = time.time() - start_time

            # 记录指标
            key = f"{method}:{path}"
            if key not in self.request_times:
                self.request_times[key] = []

            self.request_times[key].append(
                {"duration": duration, "status_code": response.status_code, "timestamp": time.time()}
            )

            # 保持最近1000条记录
            if len(self.request_times[key]) > 1000:
                self.request_times[key] = self.request_times[key][-1000:]

            self.request_count += 1

            if response.status_code >= 400:
                self.error_count += 1

            return response

        except Exception as e:
            self.error_count += 1
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标数据"""
        metrics = {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "endpoint_stats": {},
        }

        for key, times in self.request_times.items():
            if times:
                durations = [t["duration"] for t in times]
                status_codes = [t["status_code"] for t in times]

                metrics["endpoint_stats"][key] = {
                    "count": len(times),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "status_codes": dict((str(code), status_codes.count(code)) for code in set(status_codes)),
                }

        return metrics


# 全局中间件实例
metrics_middleware = MetricsMiddleware(None)  # 将在初始化时设置


class CORSMiddlewareEnhanced(BaseHTTPMiddleware):
    """增强的CORS中间件"""

    def __init__(self, app):
        """方法描述"""

    async def dispatch(self, request: Request, call_next):
        """处理CORS预检请求"""

        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response

        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"

        return response

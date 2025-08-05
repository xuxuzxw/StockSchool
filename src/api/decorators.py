import logging
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict

from fastapi import HTTPException

from src.api.factor_api import APIResponse

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API装饰器
统一异常处理、日志记录和响应格式化
"""


logger = logging.getLogger(__name__)


def api_exception_handler(operation_name: str):
    """
    API异常处理装饰器

    Args:
        operation_name: 操作名称，用于日志记录
    """

    def decorator(func: Callable) -> Callable:
        """方法描述"""

        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # 重新抛出HTTP异常
                raise
            except Exception as e:
                logger.error(f"{operation_name}失败: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"{operation_name}失败: {str(e)}")

        return wrapper

    return decorator


def api_response_builder(success_message: str = "操作成功"):
    """
    API响应构建装饰器

    Args:
        success_message: 成功消息模板
    """

    def decorator(func: Callable) -> Callable:
        """方法描述"""

        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            if isinstance(result, APIResponse):
                return result

            # 如果返回的是数据，包装成APIResponse
            if isinstance(result, dict):
                return APIResponse(success=True, data=result, message=success_message, timestamp=datetime.now())

            return result

        return wrapper

    return decorator


class ResponseBuilder:
    """响应构建器类"""

    @staticmethod
    def success(data: Any = None, message: str = "操作成功", **kwargs) -> APIResponse:
        """构建成功响应"""
        return APIResponse(success=True, data=data, message=message, timestamp=datetime.now(), **kwargs)

    @staticmethod
    def error(message: str = "操作失败", error_code: str = "UNKNOWN_ERROR", **kwargs) -> APIResponse:
        """构建错误响应"""
        return APIResponse(
            success=False,
            data={"error_code": error_code, "error_message": message},
            message=message,
            timestamp=datetime.now(),
            **kwargs,
        )


def log_api_call(operation_name: str):
    """
    API调用日志装饰器

    Args:
        operation_name: 操作名称
    """

    def decorator(func: Callable) -> Callable:
        """方法描述"""

        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.info(f"开始执行 {operation_name}")

            try:
                result = await func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"{operation_name} 执行成功，耗时: {duration:.2f}秒")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"{operation_name} 执行失败，耗时: {duration:.2f}秒，错误: {e}")
                raise

        return wrapper

    return decorator

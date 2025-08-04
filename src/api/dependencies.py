#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API依赖注入管理
统一管理API层的依赖项，提供单例和生命周期管理
"""

from typing import Dict, Any, Optional
from functools import lru_cache
import logging

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.utils.db import get_db_engine
from src.services.factor_service import FactorService, FactorCalculationService
from src.compute.factor_standardizer import FactorStandardizer
from src.compute.factor_effectiveness_analyzer import FactorEffectivenessAnalyzer
from src.compute.task_scheduler import TaskScheduler

logger = logging.getLogger(__name__)

# 安全认证
security = HTTPBearer()


class AuthService:
    """认证服务类"""
    
    def __init__(self):
        # TODO: 从配置文件加载认证配置
        self.secret_key = "your-secret-key"
        self.algorithm = "HS256"
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        验证访问令牌
        
        Args:
            token: 访问令牌
            
        Returns:
            用户信息字典
            
        Raises:
            HTTPException: 当令牌无效时
        """
        # TODO: 实现实际的JWT token验证逻辑
        if not token or token == "invalid":
            raise HTTPException(status_code=401, detail="无效的访问令牌")
        
        # 模拟用户信息，实际应该从token解析
        return {
            "user_id": "test_user",
            "username": "test_user",
            "permissions": ["read", "write"]
        }


# 使用lru_cache实现单例模式
@lru_cache()
def get_auth_service() -> AuthService:
    """获取认证服务单例"""
    return AuthService()


@lru_cache()
def get_factor_service() -> FactorService:
    """获取因子服务单例"""
    return FactorService()


@lru_cache()
def get_calculation_service() -> FactorCalculationService:
    """获取计算服务单例"""
    return FactorCalculationService()


@lru_cache()
def get_standardizer() -> FactorStandardizer:
    """获取标准化器单例"""
    engine = get_db_engine()
    return FactorStandardizer(engine)


@lru_cache()
def get_effectiveness_analyzer() -> FactorEffectivenessAnalyzer:
    """获取有效性分析器单例"""
    engine = get_db_engine()
    return FactorEffectivenessAnalyzer(engine)


@lru_cache()
def get_task_scheduler() -> TaskScheduler:
    """获取任务调度器单例"""
    engine = get_db_engine()
    return TaskScheduler(engine)


# 依赖注入函数
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict[str, Any]:
    """验证访问令牌的依赖注入函数"""
    return auth_service.verify_token(credentials.credentials)


def get_current_user(
    user_info: Dict[str, Any] = Depends(verify_token)
) -> Dict[str, Any]:
    """获取当前用户信息"""
    return user_info


# 权限检查装饰器
def require_permission(permission: str):
    """
    权限检查装饰器
    
    Args:
        permission: 所需权限
    """
    def dependency(user: Dict[str, Any] = Depends(get_current_user)):
        user_permissions = user.get("permissions", [])
        if permission not in user_permissions:
            raise HTTPException(
                status_code=403, 
                detail=f"缺少权限: {permission}"
            )
        return user
    
    return dependency


# 数据库健康检查
async def check_database_health():
    """检查数据库连接健康状态"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"数据库健康检查失败: {e}")
        return False


# 服务健康检查依赖
async def require_healthy_database():
    """要求数据库健康的依赖"""
    if not await check_database_health():
        raise HTTPException(
            status_code=503, 
            detail="数据库服务不可用"
        )
    return True
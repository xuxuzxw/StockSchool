#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子管理API接口
提供因子元数据管理、配置管理、监控管理等功能
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Dict, Any, Optional
from datetime import date, datetime, timedelta
from pydantic import BaseModel, Field, validator
import pandas as pd
import logging

from src.utils.config_loader import config
from src.utils.db import get_db_engine
from src.compute.task_scheduler import TaskScheduler, TaskConfig, TaskPriority, TaskType
from src.compute.calculation_monitor import FactorCalculationMonitor
from src.compute.factor_cache import FactorCache

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1/factor-management", tags=["factor-management"])

# 安全认证
security = HTTPBearer()

# 数据模型定义
class FactorDefinition(BaseModel):
    """因子定义"""
    factor_name: str = Field(..., description="因子名称")
    factor_type: str = Field(..., description="因子类型")
    category: str = Field(..., description="因子分类")
    description: str = Field(..., description="因子描述")
    formula: Optional[str] = Field(None, description="计算公式")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="计算参数")
    data_requirements: List[str] = Field(default_factory=list, description="数据依赖")
    update_frequency: str = Field("daily", description="更新频率")
    is_active: bool = Field(True, description="是否启用")

class TaskScheduleConfig(BaseModel):
    """任务调度配置"""
    task_name: str = Field(..., description="任务名称")
    task_type: str = Field(..., description="任务类型")
    schedule_time: str = Field(..., description="调度时间")
    priority: str = Field("normal", description="任务优先级")
    dependencies: List[str] = Field(default_factory=list, description="依赖任务")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="任务参数")
    is_enabled: bool = Field(True, description="是否启用")
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ['low', 'normal', 'high', 'urgent']:
            raise ValueError('优先级必须是: low, normal, high, urgent')
        return v

class MonitoringRule(BaseModel):
    """监控规则"""
    rule_name: str = Field(..., description="规则名称")
    rule_type: str = Field(..., description="规则类型")
    condition: str = Field(..., description="触发条件")
    threshold: float = Field(..., description="阈值")
    alert_level: str = Field("warning", description="告警级别")
    notification_channels: List[str] = Field(default_factory=list, description="通知渠道")
    is_enabled: bool = Field(True, description="是否启用")
    
    @validator('alert_level')
    def validate_alert_level(cls, v):
        if v not in ['info', 'warning', 'error', 'critical']:
            raise ValueError('告警级别必须是: info, warning, error, critical')
        return v

class CacheConfig(BaseModel):
    """缓存配置"""
    cache_type: str = Field(..., description="缓存类型")
    ttl_seconds: int = Field(3600, description="缓存过期时间(秒)")
    max_size: int = Field(1000, description="最大缓存条目数")
    compression_enabled: bool = Field(True, description="是否启用压缩")
    eviction_policy: str = Field("lru", description="淘汰策略")

# 响应模型
class APIResponse(BaseModel):
    """API统一响应格式"""
    success: bool = Field(..., description="请求是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    message: str = Field("", description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")

# 依赖注入
async def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证管理员访问令牌"""
    token = credentials.credentials
    # TODO: 实现实际的管理员token验证逻辑
    if not token or token == "invalid":
        raise HTTPException(status_code=401, detail="无效的访问令牌")
    # 检查管理员权限
    if not token.startswith("admin_"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return {"user_id": "admin_user", "permissions": ["admin"]}

def get_task_scheduler():
    """获取任务调度器"""
    engine = get_db_engine()
    return TaskScheduler(engine)

def get_calculation_monitor():
    """获取计算监控器"""
    engine = get_db_engine()
    return FactorCalculationMonitor(engine)

def get_factor_cache():
    """获取因子缓存"""
    return FactorCache()

# 因子定义管理API
@router.get("/factors", response_model=APIResponse, summary="获取因子定义列表")
async def get_factor_definitions(
    factor_type: Optional[str] = Query(None, description="因子类型"),
    category: Optional[str] = Query(None, description="因子分类"),
    is_active: Optional[bool] = Query(None, description="是否启用"),
    user: dict = Depends(verify_admin_token)
):
    """获取因子定义列表"""
    try:
        engine = get_db_engine()
        
        # 构建查询条件
        conditions = []
        params = {}
        
        if factor_type:
            conditions.append("factor_type = :factor_type")
            params['factor_type'] = factor_type
            
        if category:
            conditions.append("category = :category")
            params['category'] = category
            
        if is_active is not None:
            conditions.append("is_active = :is_active")
            params['is_active'] = is_active
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
        SELECT factor_name, factor_type, category, description, formula,
               parameters, data_requirements, update_frequency, is_active,
               created_at, updated_at
        FROM factor_definitions
        {where_clause}
        ORDER BY factor_type, category, factor_name
        """
        
        with engine.connect() as conn:
            result = conn.execute(query, params)
            factors = []
            
            for row in result:
                factors.append({
                    'factor_name': row.factor_name,
                    'factor_type': row.factor_type,
                    'category': row.category,
                    'description': row.description,
                    'formula': row.formula,
                    'parameters': row.parameters,
                    'data_requirements': row.data_requirements,
                    'update_frequency': row.update_frequency,
                    'is_active': row.is_active,
                    'created_at': row.created_at.isoformat() if row.created_at else None,
                    'updated_at': row.updated_at.isoformat() if row.updated_at else None
                })
        
        return APIResponse(
            success=True,
            data={
                "factors": factors,
                "total_count": len(factors)
            },
            message=f"成功获取{len(factors)}个因子定义"
        )
        
    except Exception as e:
        logger.error(f"获取因子定义失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取因子定义失败: {str(e)}")

@router.post("/factors", response_model=APIResponse, summary="创建因子定义")
async def create_factor_definition(
    factor: FactorDefinition,
    user: dict = Depends(verify_admin_token)
):
    """创建新的因子定义"""
    try:
        engine = get_db_engine()
        
        # 检查因子名称是否已存在
        check_query = "SELECT COUNT(*) as count FROM factor_definitions WHERE factor_name = :factor_name"
        
        with engine.connect() as conn:
            result = conn.execute(check_query, {'factor_name': factor.factor_name})
            if result.fetchone().count > 0:
                raise HTTPException(status_code=400, detail="因子名称已存在")
            
            # 插入新因子定义
            insert_query = """
            INSERT INTO factor_definitions (
                factor_name, factor_type, category, description, formula,
                parameters, data_requirements, update_frequency, is_active,
                created_at, updated_at
            ) VALUES (
                :factor_name, :factor_type, :category, :description, :formula,
                :parameters, :data_requirements, :update_frequency, :is_active,
                :created_at, :updated_at
            )
            """
            
            conn.execute(insert_query, {
                'factor_name': factor.factor_name,
                'factor_type': factor.factor_type,
                'category': factor.category,
                'description': factor.description,
                'formula': factor.formula,
                'parameters': factor.parameters,
                'data_requirements': factor.data_requirements,
                'update_frequency': factor.update_frequency,
                'is_active': factor.is_active,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            })
        
        return APIResponse(
            success=True,
            data={"factor_name": factor.factor_name},
            message="因子定义创建成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建因子定义失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建因子定义失败: {str(e)}")

@router.put("/factors/{factor_name}", response_model=APIResponse, summary="更新因子定义")
async def update_factor_definition(
    factor_name: str,
    factor: FactorDefinition,
    user: dict = Depends(verify_admin_token)
):
    """更新因子定义"""
    try:
        engine = get_db_engine()
        
        update_query = """
        UPDATE factor_definitions SET
            factor_type = :factor_type,
            category = :category,
            description = :description,
            formula = :formula,
            parameters = :parameters,
            data_requirements = :data_requirements,
            update_frequency = :update_frequency,
            is_active = :is_active,
            updated_at = :updated_at
        WHERE factor_name = :factor_name
        """
        
        with engine.connect() as conn:
            result = conn.execute(update_query, {
                'factor_name': factor_name,
                'factor_type': factor.factor_type,
                'category': factor.category,
                'description': factor.description,
                'formula': factor.formula,
                'parameters': factor.parameters,
                'data_requirements': factor.data_requirements,
                'update_frequency': factor.update_frequency,
                'is_active': factor.is_active,
                'updated_at': datetime.now()
            })
            
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="因子定义不存在")
        
        return APIResponse(
            success=True,
            data={"factor_name": factor_name},
            message="因子定义更新成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新因子定义失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新因子定义失败: {str(e)}")

@router.delete("/factors/{factor_name}", response_model=APIResponse, summary="删除因子定义")
async def delete_factor_definition(
    factor_name: str,
    user: dict = Depends(verify_admin_token)
):
    """删除因子定义"""
    try:
        engine = get_db_engine()
        
        delete_query = "DELETE FROM factor_definitions WHERE factor_name = :factor_name"
        
        with engine.connect() as conn:
            result = conn.execute(delete_query, {'factor_name': factor_name})
            
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="因子定义不存在")
        
        return APIResponse(
            success=True,
            data={"factor_name": factor_name},
            message="因子定义删除成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除因子定义失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除因子定义失败: {str(e)}")

# 任务调度管理API
@router.get("/schedules", response_model=APIResponse, summary="获取调度任务列表")
async def get_scheduled_tasks(
    task_type: Optional[str] = Query(None, description="任务类型"),
    is_enabled: Optional[bool] = Query(None, description="是否启用"),
    user: dict = Depends(verify_admin_token)
):
    """获取调度任务列表"""
    try:
        scheduler = get_task_scheduler()
        tasks = scheduler.get_scheduled_tasks(task_type=task_type, is_enabled=is_enabled)
        
        return APIResponse(
            success=True,
            data={
                "tasks": tasks,
                "total_count": len(tasks)
            },
            message=f"成功获取{len(tasks)}个调度任务"
        )
        
    except Exception as e:
        logger.error(f"获取调度任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取调度任务失败: {str(e)}")

@router.post("/schedules", response_model=APIResponse, summary="创建调度任务")
async def create_scheduled_task(
    task_config: TaskScheduleConfig,
    user: dict = Depends(verify_admin_token)
):
    """创建新的调度任务"""
    try:
        scheduler = get_task_scheduler()
        
        # 转换为TaskConfig对象
        config = TaskConfig(
            task_id=f"scheduled_{task_config.task_name}",
            task_name=task_config.task_name,
            task_type=TaskType(task_config.task_type),
            priority=TaskPriority(task_config.priority.upper()),
            dependencies=task_config.dependencies,
            parameters=task_config.parameters,
            schedule_time=task_config.schedule_time,
            is_enabled=task_config.is_enabled
        )
        
        task_id = scheduler.add_scheduled_task(config)
        
        return APIResponse(
            success=True,
            data={"task_id": task_id},
            message="调度任务创建成功"
        )
        
    except Exception as e:
        logger.error(f"创建调度任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建调度任务失败: {str(e)}")

@router.put("/schedules/{task_id}/enable", response_model=APIResponse, summary="启用调度任务")
async def enable_scheduled_task(
    task_id: str,
    user: dict = Depends(verify_admin_token)
):
    """启用调度任务"""
    try:
        scheduler = get_task_scheduler()
        scheduler.enable_task(task_id)
        
        return APIResponse(
            success=True,
            data={"task_id": task_id},
            message="调度任务已启用"
        )
        
    except Exception as e:
        logger.error(f"启用调度任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"启用调度任务失败: {str(e)}")

@router.put("/schedules/{task_id}/disable", response_model=APIResponse, summary="禁用调度任务")
async def disable_scheduled_task(
    task_id: str,
    user: dict = Depends(verify_admin_token)
):
    """禁用调度任务"""
    try:
        scheduler = get_task_scheduler()
        scheduler.disable_task(task_id)
        
        return APIResponse(
            success=True,
            data={"task_id": task_id},
            message="调度任务已禁用"
        )
        
    except Exception as e:
        logger.error(f"禁用调度任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"禁用调度任务失败: {str(e)}")

# 监控管理API
@router.get("/monitoring/status", response_model=APIResponse, summary="获取监控状态")
async def get_monitoring_status(
    user: dict = Depends(verify_admin_token)
):
    """获取系统监控状态"""
    try:
        monitor = get_calculation_monitor()
        status = monitor.get_system_status()
        
        return APIResponse(
            success=True,
            data=status,
            message="监控状态获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取监控状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取监控状态失败: {str(e)}")

@router.get("/monitoring/metrics", response_model=APIResponse, summary="获取性能指标")
async def get_performance_metrics(
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    user: dict = Depends(verify_admin_token)
):
    """获取性能指标数据"""
    try:
        monitor = get_calculation_monitor()
        
        if not start_time:
            start_time = datetime.now() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.now()
        
        metrics = monitor.get_performance_metrics(start_time, end_time)
        
        return APIResponse(
            success=True,
            data={
                "metrics": metrics,
                "time_range": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }
            },
            message="性能指标获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")

@router.get("/monitoring/alerts", response_model=APIResponse, summary="获取告警信息")
async def get_alerts(
    alert_level: Optional[str] = Query(None, description="告警级别"),
    is_resolved: Optional[bool] = Query(None, description="是否已解决"),
    limit: int = Query(100, description="返回数量限制"),
    user: dict = Depends(verify_admin_token)
):
    """获取告警信息"""
    try:
        monitor = get_calculation_monitor()
        alerts = monitor.get_alerts(
            alert_level=alert_level,
            is_resolved=is_resolved,
            limit=limit
        )
        
        return APIResponse(
            success=True,
            data={
                "alerts": alerts,
                "total_count": len(alerts)
            },
            message=f"成功获取{len(alerts)}条告警信息"
        )
        
    except Exception as e:
        logger.error(f"获取告警信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取告警信息失败: {str(e)}")

# 缓存管理API
@router.get("/cache/status", response_model=APIResponse, summary="获取缓存状态")
async def get_cache_status(
    user: dict = Depends(verify_admin_token)
):
    """获取缓存状态"""
    try:
        cache = get_factor_cache()
        status = cache.get_cache_status()
        
        return APIResponse(
            success=True,
            data=status,
            message="缓存状态获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取缓存状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取缓存状态失败: {str(e)}")

@router.post("/cache/clear", response_model=APIResponse, summary="清空缓存")
async def clear_cache(
    cache_type: Optional[str] = Query(None, description="缓存类型"),
    user: dict = Depends(verify_admin_token)
):
    """清空缓存"""
    try:
        cache = get_factor_cache()
        
        if cache_type:
            cleared_count = cache.clear_cache_by_type(cache_type)
        else:
            cleared_count = cache.clear_all_cache()
        
        return APIResponse(
            success=True,
            data={"cleared_count": cleared_count},
            message=f"成功清空{cleared_count}个缓存条目"
        )
        
    except Exception as e:
        logger.error(f"清空缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空缓存失败: {str(e)}")

@router.post("/cache/warm-up", response_model=APIResponse, summary="缓存预热")
async def warm_up_cache(
    factor_names: Optional[List[str]] = Query(None, description="因子名称列表"),
    date_range: int = Query(30, description="预热天数"),
    user: dict = Depends(verify_admin_token)
):
    """缓存预热"""
    try:
        cache = get_factor_cache()
        
        end_date = date.today()
        start_date = end_date - timedelta(days=date_range)
        
        warmed_count = cache.warm_up_cache(
            factor_names=factor_names,
            start_date=start_date,
            end_date=end_date
        )
        
        return APIResponse(
            success=True,
            data={
                "warmed_count": warmed_count,
                "date_range": f"{start_date} - {end_date}"
            },
            message=f"成功预热{warmed_count}个缓存条目"
        )
        
    except Exception as e:
        logger.error(f"缓存预热失败: {e}")
        raise HTTPException(status_code=500, detail=f"缓存预热失败: {str(e)}")

# 系统配置API
@router.get("/config", response_model=APIResponse, summary="获取系统配置")
async def get_system_config(
    user: dict = Depends(verify_admin_token)
):
    """获取系统配置"""
    try:
        # 获取当前配置
        current_config = {
            "database": {
                "pool_size": config.get('database.pool_size', 20),
                "max_overflow": config.get('database.max_overflow', 30)
            },
            "computation": {
                "max_workers": config.get('computation.max_workers', 4),
                "batch_size": config.get('computation.batch_size', 1000)
            },
            "cache": {
                "redis_ttl": config.get('cache.redis_ttl', 3600),
                "memory_limit": config.get('cache.memory_limit', 1000)
            },
            "monitoring": {
                "alert_enabled": config.get('monitoring.alert_enabled', True),
                "metrics_retention_days": config.get('monitoring.metrics_retention_days', 30)
            }
        }
        
        return APIResponse(
            success=True,
            data=current_config,
            message="系统配置获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取系统配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统配置失败: {str(e)}")

@router.put("/config", response_model=APIResponse, summary="更新系统配置")
async def update_system_config(
    config_data: Dict[str, Any],
    user: dict = Depends(verify_admin_token)
):
    """更新系统配置"""
    try:
        # TODO: 实现配置更新逻辑
        # 这里应该验证配置的有效性并更新到配置文件或数据库
        
        return APIResponse(
            success=True,
            data={"updated_keys": list(config_data.keys())},
            message="系统配置更新成功"
        )
        
    except Exception as e:
        logger.error(f"更新系统配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新系统配置失败: {str(e)}")
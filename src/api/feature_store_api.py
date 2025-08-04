#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征商店API接口
提供因子特征商店的版本管理、元数据管理和查询功能
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import date, datetime
from pydantic import BaseModel, Field, validator
import pandas as pd
import logging

from src.utils.db import get_db_engine
from src.features.factor_feature_store import FactorFeatureStore, FactorMetadata, FactorVersion
from src.api.auth import require_factor_read, require_factor_write, require_factor_admin

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1/feature-store", tags=["feature-store"])

# 请求模型
class FactorMetadataRequest(BaseModel):
    """因子元数据请求"""
    factor_name: str = Field(..., description="因子名称")
    factor_type: str = Field(..., description="因子类型")
    category: str = Field(..., description="因子分类")
    description: str = Field(..., description="因子描述")
    formula: str = Field(..., description="计算公式")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="计算参数")
    data_requirements: List[str] = Field(default_factory=list, description="数据依赖")
    update_frequency: str = Field("daily", description="更新频率")
    data_schema: Dict[str, str] = Field(default_factory=dict, description="数据结构")
    tags: List[str] = Field(default_factory=list, description="标签")

class FactorVersionRequest(BaseModel):
    """因子版本请求"""
    factor_name: str = Field(..., description="因子名称")
    algorithm_code: str = Field(..., description="算法代码")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="算法参数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="版本元数据")

class FactorDataRequest(BaseModel):
    """因子数据请求"""
    factor_name: str = Field(..., description="因子名称")
    version_id: Optional[str] = Field(None, description="版本ID")
    ts_codes: Optional[List[str]] = Field(None, description="股票代码列表")
    start_date: Optional[date] = Field(None, description="开始日期")
    end_date: Optional[date] = Field(None, description="结束日期")

class FactorSearchRequest(BaseModel):
    """因子搜索请求"""
    query: str = Field(..., description="搜索关键词")
    factor_type: Optional[str] = Field(None, description="因子类型")
    category: Optional[str] = Field(None, description="因子分类")
    tags: Optional[List[str]] = Field(None, description="标签列表")

# 响应模型
class APIResponse(BaseModel):
    """API统一响应格式"""
    success: bool = Field(..., description="请求是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    message: str = Field("", description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")

# 依赖注入
def get_factor_feature_store():
    """获取因子特征商店"""
    engine = get_db_engine()
    return FactorFeatureStore(engine)

# API端点实现
@router.post("/factors/register", response_model=APIResponse, summary="注册因子元数据")
async def register_factor_metadata(
    request: FactorMetadataRequest,
    user=Depends(require_factor_admin()),
    feature_store=Depends(get_factor_feature_store)
):
    """注册因子元数据"""
    try:
        metadata = FactorMetadata(
            factor_name=request.factor_name,
            factor_type=request.factor_type,
            category=request.category,
            description=request.description,
            formula=request.formula,
            parameters=request.parameters,
            data_requirements=request.data_requirements,
            update_frequency=request.update_frequency,
            data_schema=request.data_schema,
            tags=request.tags
        )
        
        factor_name = feature_store.register_factor(metadata)
        
        return APIResponse(
            success=True,
            data={"factor_name": factor_name},
            message="因子元数据注册成功"
        )
        
    except Exception as e:
        logger.error(f"注册因子元数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"注册因子元数据失败: {str(e)}")

@router.get("/factors/metadata", response_model=APIResponse, summary="获取因子元数据")
async def get_factor_metadata(
    factor_name: Optional[str] = Query(None, description="因子名称"),
    user=Depends(require_factor_read()),
    feature_store=Depends(get_factor_feature_store)
):
    """获取因子元数据"""
    try:
        if factor_name:
            metadata = feature_store.get_factor_metadata(factor_name)
            if not metadata:
                raise HTTPException(status_code=404, detail="因子不存在")
            
            return APIResponse(
                success=True,
                data=metadata.to_dict(),
                message="因子元数据获取成功"
            )
        else:
            metadata_list = feature_store.get_factor_metadata()
            
            return APIResponse(
                success=True,
                data={
                    "factors": [meta.to_dict() for meta in metadata_list],
                    "total_count": len(metadata_list)
                },
                message=f"成功获取{len(metadata_list)}个因子元数据"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取因子元数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取因子元数据失败: {str(e)}")

@router.post("/factors/versions", response_model=APIResponse, summary="创建因子版本")
async def create_factor_version(
    request: FactorVersionRequest,
    user=Depends(require_factor_write()),
    feature_store=Depends(get_factor_feature_store)
):
    """创建因子版本"""
    try:
        version_id = feature_store.create_factor_version(
            factor_name=request.factor_name,
            algorithm_code=request.algorithm_code,
            parameters=request.parameters,
            metadata=request.metadata
        )
        
        return APIResponse(
            success=True,
            data={"version_id": version_id},
            message="因子版本创建成功"
        )
        
    except Exception as e:
        logger.error(f"创建因子版本失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建因子版本失败: {str(e)}")

@router.get("/factors/{factor_name}/versions", response_model=APIResponse, summary="获取因子版本列表")
async def get_factor_versions(
    factor_name: str,
    user=Depends(require_factor_read()),
    feature_store=Depends(get_factor_feature_store)
):
    """获取因子版本列表"""
    try:
        versions = feature_store.get_factor_versions(factor_name)
        
        return APIResponse(
            success=True,
            data={
                "versions": [version.to_dict() for version in versions],
                "total_count": len(versions)
            },
            message=f"成功获取{len(versions)}个版本"
        )
        
    except Exception as e:
        logger.error(f"获取因子版本失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取因子版本失败: {str(e)}")

@router.put("/versions/{version_id}/activate", response_model=APIResponse, summary="激活因子版本")
async def activate_factor_version(
    version_id: str,
    user=Depends(require_factor_write()),
    feature_store=Depends(get_factor_feature_store)
):
    """激活因子版本"""
    try:
        success = feature_store.activate_version(version_id)
        
        if success:
            return APIResponse(
                success=True,
                data={"version_id": version_id},
                message="因子版本激活成功"
            )
        else:
            raise HTTPException(status_code=400, detail="因子版本激活失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"激活因子版本失败: {e}")
        raise HTTPException(status_code=500, detail=f"激活因子版本失败: {str(e)}")

@router.get("/factors/data", response_model=APIResponse, summary="获取因子数据")
async def get_factor_data(
    request: FactorDataRequest = Depends(),
    user=Depends(require_factor_read()),
    feature_store=Depends(get_factor_feature_store)
):
    """获取因子数据"""
    try:
        data = feature_store.get_factor_data(
            factor_name=request.factor_name,
            version_id=request.version_id,
            ts_codes=request.ts_codes,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # 转换DataFrame为字典格式
        if not data.empty:
            data_dict = data.to_dict('records')
        else:
            data_dict = []
        
        return APIResponse(
            success=True,
            data={
                "factor_data": data_dict,
                "total_count": len(data_dict),
                "columns": list(data.columns) if not data.empty else []
            },
            message=f"成功获取{len(data_dict)}条因子数据"
        )
        
    except Exception as e:
        logger.error(f"获取因子数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取因子数据失败: {str(e)}")

@router.get("/factors/{factor_name}/lineage", response_model=APIResponse, summary="获取因子血缘")
async def get_factor_lineage(
    factor_name: str,
    version_id: Optional[str] = Query(None, description="版本ID"),
    user=Depends(require_factor_read()),
    feature_store=Depends(get_factor_feature_store)
):
    """获取因子数据血缘"""
    try:
        lineage = feature_store.get_factor_lineage(factor_name, version_id)
        
        return APIResponse(
            success=True,
            data=lineage,
            message="因子血缘获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取因子血缘失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取因子血缘失败: {str(e)}")

@router.get("/factors/{factor_name}/quality", response_model=APIResponse, summary="获取因子质量指标")
async def get_factor_quality_metrics(
    factor_name: str,
    version_id: Optional[str] = Query(None, description="版本ID"),
    start_date: Optional[date] = Query(None, description="开始日期"),
    end_date: Optional[date] = Query(None, description="结束日期"),
    user=Depends(require_factor_read()),
    feature_store=Depends(get_factor_feature_store)
):
    """获取因子质量指标"""
    try:
        metrics = feature_store.get_quality_metrics(
            factor_name=factor_name,
            version_id=version_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # 转换DataFrame为字典格式
        if not metrics.empty:
            metrics_dict = metrics.to_dict('records')
        else:
            metrics_dict = []
        
        return APIResponse(
            success=True,
            data={
                "quality_metrics": metrics_dict,
                "total_count": len(metrics_dict)
            },
            message=f"成功获取{len(metrics_dict)}条质量指标"
        )
        
    except Exception as e:
        logger.error(f"获取因子质量指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取因子质量指标失败: {str(e)}")

@router.post("/factors/search", response_model=APIResponse, summary="搜索因子")
async def search_factors(
    request: FactorSearchRequest,
    user=Depends(require_factor_read()),
    feature_store=Depends(get_factor_feature_store)
):
    """搜索因子"""
    try:
        factors = feature_store.search_factors(
            query=request.query,
            factor_type=request.factor_type,
            category=request.category,
            tags=request.tags
        )
        
        return APIResponse(
            success=True,
            data={
                "factors": [factor.to_dict() for factor in factors],
                "total_count": len(factors)
            },
            message=f"搜索到{len(factors)}个因子"
        )
        
    except Exception as e:
        logger.error(f"搜索因子失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索因子失败: {str(e)}")

@router.get("/statistics", response_model=APIResponse, summary="获取存储统计")
async def get_storage_statistics(
    user=Depends(require_factor_read()),
    feature_store=Depends(get_factor_feature_store)
):
    """获取特征商店存储统计"""
    try:
        stats = feature_store.get_storage_statistics()
        
        return APIResponse(
            success=True,
            data=stats,
            message="存储统计获取成功"
        )
        
    except Exception as e:
        logger.error(f"获取存储统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取存储统计失败: {str(e)}")

@router.get("/health", response_model=APIResponse, summary="健康检查")
async def health_check():
    """特征商店健康检查"""
    try:
        feature_store = get_factor_feature_store()
        
        # 简单的健康检查
        stats = feature_store.get_storage_statistics()
        
        return APIResponse(
            success=True,
            data={
                "status": "healthy",
                "factor_count": stats.get('factor_count', 0),
                "version_count": stats.get('version_count', 0),
                "timestamp": datetime.now().isoformat()
            },
            message="特征商店运行正常"
        )
        
    except Exception as e:
        logger.error(f"特征商店健康检查失败: {e}")
        return APIResponse(
            success=False,
            data={
                "status": "unhealthy",
                "error": str(e)
            },
            message="特征商店异常"
        )
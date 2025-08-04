#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API分页支持
提供统一的分页查询功能
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from math import ceil


class PaginationParams(BaseModel):
    """分页参数"""
    page: int = Field(1, ge=1, description="页码，从1开始")
    page_size: int = Field(20, ge=1, le=1000, description="每页大小，最大1000")
    
    @validator('page_size')
    def validate_page_size(cls, v):
        if v > 1000:
            raise ValueError('每页最多1000条记录')
        return v


class PaginationResult(BaseModel):
    """分页结果"""
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页大小")
    total: int = Field(..., description="总记录数")
    total_pages: int = Field(..., description="总页数")
    has_next: bool = Field(..., description="是否有下一页")
    has_prev: bool = Field(..., description="是否有上一页")


class PaginatedResponse(BaseModel):
    """分页响应"""
    items: List[Any] = Field(..., description="数据项列表")
    pagination: PaginationResult = Field(..., description="分页信息")


def paginate_data(
    data: List[Any],
    page: int,
    page_size: int
) -> PaginatedResponse:
    """
    对数据进行分页处理
    
    Args:
        data: 要分页的数据列表
        page: 页码
        page_size: 每页大小
        
    Returns:
        分页后的响应数据
    """
    total = len(data)
    total_pages = ceil(total / page_size) if total > 0 else 0
    
    # 计算起始和结束索引
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    # 获取当前页数据
    items = data[start_idx:end_idx]
    
    # 构建分页信息
    pagination = PaginationResult(
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )
    
    return PaginatedResponse(
        items=items,
        pagination=pagination
    )


def create_pagination_links(
    base_url: str,
    pagination: PaginationResult,
    **query_params
) -> Dict[str, Optional[str]]:
    """
    创建分页链接
    
    Args:
        base_url: 基础URL
        pagination: 分页信息
        **query_params: 查询参数
        
    Returns:
        包含分页链接的字典
    """
    def build_url(page: int) -> str:
        params = {**query_params, 'page': page, 'page_size': pagination.page_size}
        param_str = '&'.join(f"{k}={v}" for k, v in params.items() if v is not None)
        return f"{base_url}?{param_str}"
    
    links = {
        'self': build_url(pagination.page),
        'first': build_url(1) if pagination.total_pages > 0 else None,
        'last': build_url(pagination.total_pages) if pagination.total_pages > 0 else None,
        'next': build_url(pagination.page + 1) if pagination.has_next else None,
        'prev': build_url(pagination.page - 1) if pagination.has_prev else None
    }
    
    return links
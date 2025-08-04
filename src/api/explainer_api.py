#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型解释器API服务

提供RESTful API接口，支持模型解释功能的在线服务
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

from src.strategy.model_explainer import ModelExplainer, ModelExplainerError
from src.config.unified_config import config
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/explainer", tags=["model_explainer"])

class ExplainRequest(BaseModel):
    """预测解释请求模型"""
    model_path: str
    data: List[List[float]]
    feature_names: List[str]
    method: str = "shap"
    sample_idx: int = 0
    target_names: Optional[List[str]] = None

class BatchExplainRequest(BaseModel):
    """批量预测解释请求模型"""
    model_path: str
    data_list: List[List[List[float]]]
    feature_names: List[str]
    method: str = "shap"
    target_names: Optional[List[str]] = None

class FeatureImportanceRequest(BaseModel):
    """特征重要性请求模型"""
    model_path: str
    data: List[List[float]]
    feature_names: List[str]
    target: Optional[List[float]] = None
    method: str = "shap"
    target_names: Optional[List[str]] = None

class ExplainResponse(BaseModel):
    """预测解释响应模型"""
    status: str
    explanation: Dict[str, Any]
    timestamp: str

class BatchExplainResponse(BaseModel):
    """批量预测解释响应模型"""
    status: str
    explanations: List[Dict[str, Any]]
    timestamp: str

class FeatureImportanceResponse(BaseModel):
    """特征重要性响应模型"""
    status: str
    importance: List[Dict[str, Any]]
    timestamp: str

def load_model(model_path: str):
    """加载模型"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"模型加载成功: {model_path}")
        return model
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: ExplainRequest):
    """
    单个预测解释API
    
    Args:
        request: 解释请求参数
        
    Returns:
        解释结果
    """
    start_time = datetime.now()
    try:
        logger.info(f"开始处理预测解释请求，方法: {request.method}")
        
        # 加载模型
        model = load_model(request.model_path)
        
        # 转换数据
        X = pd.DataFrame(request.data, columns=request.feature_names)
        
        # 创建解释器
        explainer = ModelExplainer(model, request.feature_names)
        
        # 执行解释
        explanation = explainer.explain_prediction(X, request.sample_idx)
        
        # 添加目标名称（如果提供）
        if request.target_names:
            explanation['target_names'] = request.target_names
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"预测解释完成，耗时: {processing_time:.2f}秒")
        
        return ExplainResponse(
            status="success",
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
        
    except ModelExplainerError as e:
        logger.error(f"模型解释器错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"预测解释失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch_explain", response_model=BatchExplainResponse)
async def batch_explain(request: BatchExplainRequest):
    """
    批量预测解释API
    
    Args:
        request: 批量解释请求参数
        
    Returns:
        批量解释结果
    """
    start_time = datetime.now()
    try:
        logger.info(f"开始处理批量预测解释请求，样本数: {len(request.data_list)}")
        
        # 加载模型
        model = load_model(request.model_path)
        
        # 创建解释器
        explainer = ModelExplainer(model, request.feature_names)
        
        # 批量解释
        results = []
        for i, data in enumerate(request.data_list):
            # 确保data是正确的格式
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                # 如果data是二维列表，取第一行
                sample_data = data[0] if len(data) > 0 else data
            else:
                sample_data = data
            X = pd.DataFrame([sample_data], columns=request.feature_names)
            explanation = explainer.explain_prediction(X, 0)
            
            # 添加目标名称（如果提供）
            if request.target_names:
                explanation['target_names'] = request.target_names
                
            explanation['batch_index'] = i
            results.append(explanation)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"批量预测解释完成，耗时: {processing_time:.2f}秒")
        
        return BatchExplainResponse(
            status="success",
            explanations=results,
            timestamp=datetime.now().isoformat()
        )
        
    except ModelExplainerError as e:
        logger.error(f"模型解释器错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"批量预测解释失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feature_importance", response_model=FeatureImportanceResponse)
async def feature_importance(request: FeatureImportanceRequest):
    """
    特征重要性计算API
    
    Args:
        request: 特征重要性请求参数
        
    Returns:
        特征重要性结果
    """
    start_time = datetime.now()
    try:
        logger.info(f"开始计算特征重要性，方法: {request.method}")
        
        # 加载模型
        model = load_model(request.model_path)
        
        # 转换数据
        X = pd.DataFrame(request.data, columns=request.feature_names)
        y = pd.Series(request.target) if request.target else None
        
        # 创建解释器
        explainer = ModelExplainer(model, request.feature_names)
        
        # 计算特征重要性
        importance = explainer.calculate_feature_importance(X, y, method=request.method)
        
        # 转换为字典列表
        importance_records = importance.to_dict('records')
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"特征重要性计算完成，耗时: {processing_time:.2f}秒")
        
        return FeatureImportanceResponse(
            status="success",
            importance=importance_records,
            timestamp=datetime.now().isoformat()
        )
        
    except ModelExplainerError as e:
        logger.error(f"模型解释器错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"特征重要性计算失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model_info/{model_path}")
async def get_model_info(model_path: str):
    """
    获取模型信息
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        模型信息
    """
    try:
        # 加载模型
        model = load_model(model_path)
        
        # 获取模型信息
        model_info = {
            "model_type": type(model).__name__,
            "model_module": type(model).__module__,
            "has_predict": hasattr(model, 'predict'),
            "has_feature_importances": hasattr(model, 'feature_importances_'),
            "has_coef": hasattr(model, 'coef_')
        }
        
        return {"status": "success", "model_info": model_info}
        
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model_summary")
async def model_summary(request: ExplainRequest):
    """
    生成模型摘要
    
    Args:
        request: 解释请求参数
        
    Returns:
        模型摘要
    """
    try:
        # 加载模型
        model = load_model(request.model_path)
        
        # 转换数据
        X = pd.DataFrame(request.data, columns=request.feature_names)
        y = pd.Series([])  # 空的目标变量用于摘要
        
        # 创建解释器
        explainer = ModelExplainer(model, request.feature_names)
        
        # 生成模型摘要
        summary = explainer.generate_model_summary(X, y)
        
        return {"status": "success", "summary": summary}
        
    except Exception as e:
        logger.error(f"生成模型摘要失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

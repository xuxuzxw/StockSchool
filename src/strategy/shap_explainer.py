#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP解释器模块

专门处理基于SHAP的模型解释，提供高性能的特征重要性计算和预测解释功能。
"""

import numpy as np
import pandas as pd
import shap
import torch
import gc
from typing import Any, Dict, List, Optional, Union
from sklearn.base import BaseEstimator
import logging
from src.utils.gpu_utils import get_device, get_batch_size
from src.config.unified_config import config
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

class SHAPExplainer:
    """SHAP解释器类"""
    
    def __init__(self, model: BaseEstimator, feature_names: List[str]):
        """
        初始化SHAP解释器
        
        Args:
            model: 机器学习模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.device = get_device()
        self.batch_size = get_batch_size()
        
        # 初始化SHAP解释器
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """初始化SHAP解释器"""
        try:
            model_name = type(self.model).__name__.lower()
            
            if 'tree' in model_name or 'forest' in model_name or 'gbm' in model_name:
                # 树模型使用TreeExplainer
                self.explainer = shap.TreeExplainer(
                    self.model, 
                    model_output='raw_values',
                    feature_perturbation='tree_path_dependent'
                )
            elif 'linear' in model_name or 'logistic' in model_name:
                # 线性模型使用LinearExplainer
                self.explainer = shap.LinearExplainer(self.model, feature_perturbation='interventional')
            else:
                # 通用解释器
                self.explainer = shap.Explainer(self.model)
            
            logger.info("SHAP解释器初始化成功")
            
        except Exception as e:
            logger.error(f"SHAP解释器初始化失败: {e}")
            raise
    
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series = None, 
                                   background_samples: int = 100) -> pd.DataFrame:
        """
        计算特征重要性
        
        Args:
            X: 特征数据
            y: 目标变量（可选）
            background_samples: 背景样本数
            
        Returns:
            特征重要性DataFrame
        """
        try:
            # 选择背景数据
            if len(X) > background_samples:
                background_data = X.sample(n=background_samples, random_state=42)
            else:
                background_data = X
            
            # 计算SHAP值
            shap_values = self.explainer.shap_values(background_data)
            
            # 处理多输出情况
            if isinstance(shap_values, list):
                # 多输出模型，取第一个输出
                shap_values = shap_values[0]
            
            # 计算特征重要性（绝对值平均）
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # 创建结果DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            
            logger.info(f"SHAP特征重要性计算完成，共{len(importance_df)}个特征")
            return importance_df
            
        except Exception as e:
            logger.error(f"SHAP特征重要性计算失败: {e}")
            raise
    
    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        """
        解释单个预测
        
        Args:
            X: 输入数据
            sample_idx: 样本索引
            
        Returns:
            预测解释字典
        """
        try:
            # 获取单个样本
            sample = X.iloc[sample_idx:sample_idx+1] if len(X) > sample_idx else X
            
            # 计算SHAP值
            shap_values = self.explainer.shap_values(sample)
            
            # 处理多输出情况
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # 获取基础值和预测值
            base_values = self.explainer.expected_value
            if isinstance(base_values, list):
                base_values = base_values[0]
            
            prediction = self.model.predict(sample)[0]
            
            # 创建解释结果
            explanation = {
                'sample_idx': sample_idx,
                'feature_names': self.feature_names,
                'feature_values': sample.values[0].tolist(),
                'shap_values': shap_values[0].tolist(),
                'base_value': float(base_values),
                'prediction': float(prediction),
                'model_type': 'tree' if hasattr(self.model, 'tree_') else 'other'
            }
            
            logger.info(f"预测解释完成，样本索引: {sample_idx}")
            return explanation
            
        except Exception as e:
            logger.error(f"预测解释失败: {e}")
            raise
    
    def batch_explain(self, X: pd.DataFrame, batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        批量解释预测
        
        Args:
            X: 输入数据
            batch_size: 批量大小
            
        Returns:
            预测解释列表
        """
        batch_size = batch_size or self.batch_size
        explanations = []
        
        try:
            for i in range(0, len(X), batch_size):
                batch = X.iloc[i:i+batch_size]
                batch_explanations = []
                
                for j in range(len(batch)):
                    explanation = self.explain_prediction(batch, j)
                    explanation['batch_index'] = i + j
                    batch_explanations.append(explanation)
                
                explanations.extend(batch_explanations)
                
                # 清理内存
                if i % (batch_size * 10) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            logger.info(f"批量预测解释完成，共{len(explanations)}个样本")
            return explanations
            
        except Exception as e:
            logger.error(f"批量预测解释失败: {e}")
            raise

def create_shap_explainer(model: BaseEstimator, feature_names: List[str]) -> SHAPExplainer:
    """
    创建SHAP解释器的便捷函数
    
    Args:
        model: 机器学习模型
        feature_names: 特征名称列表
        
    Returns:
        SHAPExplainer实例
    """
    return SHAPExplainer(model, feature_names)

if __name__ == '__main__':
    # 测试代码
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    
    print("测试SHAP解释器...")
    
    # 创建测试数据
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    feature_names = [f'feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    # 创建模型
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_df, y_series)
    
    # 创建解释器
    explainer = SHAPExplainer(model, feature_names)
    
    # 测试特征重要性
    importance = explainer.calculate_feature_importance(X_df)
    print(f"特征重要性计算完成，前5个特征:")
    print(importance.head())
    
    # 测试单个预测解释
    explanation = explainer.explain_prediction(X_df, 0)
    print(f"\n单个预测解释完成:")
    print(f"预测值: {explanation['prediction']:.4f}")
    print(f"基础值: {explanation['base_value']:.4f}")
    
    # 测试批量解释
    batch_explanations = explainer.batch_explain(X_df.head(10))
    print(f"\n批量解释完成，共{len(batch_explanations)}个样本")
    
    print("\nSHAP解释器测试完成!")

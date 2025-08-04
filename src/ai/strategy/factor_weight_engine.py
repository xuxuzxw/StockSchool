"""因子权重引擎

实现因子重要性计算、权重动态调整、约束和稳定性控制。
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import shap
from sklearn.preprocessing import StandardScaler

from ...utils.db import get_db_manager

logger = logging.getLogger(__name__)

@dataclass
class FactorWeight:
    """因子权重信息"""
    factor_name: str
    weight: float
    importance_score: float
    shap_value: float
    stability_score: float
    last_updated: datetime
    model_name: str
    model_version: str

class FactorWeightEngine:
    """因子权重引擎
    
    主要功能：
    - 基于SHAP值计算因子重要性
    - 动态调整因子权重
    - 权重约束和稳定性控制
    - 权重历史记录和分析
    """
    
    def __init__(self):
        self.engine = get_db_manager().engine
        self._ensure_tables_exist()
        
        # 权重约束参数
        self.max_weight = 0.3  # 单个因子最大权重
        self.min_weight = 0.01  # 单个因子最小权重
        self.stability_threshold = 0.1  # 稳定性阈值
        self.decay_factor = 0.95  # 历史权重衰减因子
    
    def _ensure_tables_exist(self):
        """确保因子权重相关表存在"""
        create_tables_sql = """
        -- 因子权重表
        CREATE TABLE IF NOT EXISTS factor_weights (
            id SERIAL PRIMARY KEY,
            factor_name VARCHAR(100) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            weight DECIMAL(10, 6) NOT NULL,
            importance_score DECIMAL(10, 6),
            shap_value DECIMAL(10, 6),
            stability_score DECIMAL(10, 6),
            calculation_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB,
            UNIQUE(factor_name, model_name, model_version, calculation_date)
        );
        
        -- 因子权重历史表
        CREATE TABLE IF NOT EXISTS factor_weight_history (
            id SERIAL PRIMARY KEY,
            factor_name VARCHAR(100) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            old_weight DECIMAL(10, 6),
            new_weight DECIMAL(10, 6),
            change_reason TEXT,
            changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            changed_by VARCHAR(50)
        );
        
        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_factor_weights_model ON factor_weights(model_name, model_version);
        CREATE INDEX IF NOT EXISTS idx_factor_weights_date ON factor_weights(calculation_date);
        CREATE INDEX IF NOT EXISTS idx_factor_weights_factor ON factor_weights(factor_name);
        CREATE INDEX IF NOT EXISTS idx_factor_weight_history_factor ON factor_weight_history(factor_name, model_name);
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
            logger.info("因子权重表创建成功")
        except Exception as e:
            logger.error(f"创建因子权重表失败: {e}")
            raise
    
    def calculate_factor_importance(self, 
                                  model_data: Dict[str, Any],
                                  X_test: pd.DataFrame,
                                  sample_size: int = 1000) -> Dict[str, float]:
        """计算因子重要性
        
        Args:
            model_data: 模型数据字典
            X_test: 测试数据
            sample_size: SHAP计算样本大小
            
        Returns:
            因子重要性字典
        """
        try:
            model = model_data['model']
            feature_columns = model_data['feature_columns']
            
            # 确保特征列顺序一致
            X_test_aligned = X_test[feature_columns]
            
            # 采样以提高计算效率
            if len(X_test_aligned) > sample_size:
                X_sample = X_test_aligned.sample(n=sample_size, random_state=42)
            else:
                X_sample = X_test_aligned
            
            # 计算SHAP值
            if hasattr(model, 'predict'):
                # 对于树模型使用TreeExplainer
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    # 对于其他模型使用KernelExplainer
                    explainer = shap.KernelExplainer(model.predict, X_sample.iloc[:100])
                    shap_values = explainer.shap_values(X_sample.iloc[:100])
            else:
                logger.warning("模型不支持SHAP分析，使用特征重要性")
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(feature_columns, model.feature_importances_))
                    return importance_dict
                else:
                    # 返回均等权重
                    return {col: 1.0/len(feature_columns) for col in feature_columns}
            
            # 计算平均绝对SHAP值作为重要性
            if isinstance(shap_values, list):
                # 多分类情况，取第一个类别
                shap_values = shap_values[0]
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance_dict = dict(zip(feature_columns, mean_abs_shap))
            
            # 归一化重要性分数
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
            
            logger.info(f"计算因子重要性完成，共{len(importance_dict)}个因子")
            return importance_dict
            
        except Exception as e:
            logger.error(f"计算因子重要性失败: {e}")
            # 返回均等权重作为备选
            feature_columns = model_data.get('feature_columns', [])
            if feature_columns:
                return {col: 1.0/len(feature_columns) for col in feature_columns}
            return {}
    
    def calculate_factor_weights(self,
                               model_name: str,
                               model_version: str,
                               importance_scores: Dict[str, float],
                               historical_weights: Optional[Dict[str, float]] = None,
                               constraints: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """计算因子权重
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            importance_scores: 重要性分数
            historical_weights: 历史权重
            constraints: 权重约束
            
        Returns:
            因子权重字典
        """
        try:
            # 应用约束参数
            if constraints:
                max_weight = constraints.get('max_weight', self.max_weight)
                min_weight = constraints.get('min_weight', self.min_weight)
                stability_threshold = constraints.get('stability_threshold', self.stability_threshold)
                decay_factor = constraints.get('decay_factor', self.decay_factor)
            else:
                max_weight = self.max_weight
                min_weight = self.min_weight
                stability_threshold = self.stability_threshold
                decay_factor = self.decay_factor
            
            # 初始权重基于重要性分数
            raw_weights = importance_scores.copy()
            
            # 如果有历史权重，进行平滑处理
            if historical_weights:
                for factor in raw_weights:
                    if factor in historical_weights:
                        # 加权平均：新权重 * (1-decay) + 历史权重 * decay
                        raw_weights[factor] = (
                            raw_weights[factor] * (1 - decay_factor) + 
                            historical_weights[factor] * decay_factor
                        )
            
            # 应用权重约束
            constrained_weights = {}
            for factor, weight in raw_weights.items():
                # 应用最大最小权重约束
                constrained_weight = max(min_weight, min(max_weight, weight))
                constrained_weights[factor] = constrained_weight
            
            # 重新归一化
            total_weight = sum(constrained_weights.values())
            if total_weight > 0:
                normalized_weights = {k: v/total_weight for k, v in constrained_weights.items()}
            else:
                # 如果总权重为0，使用均等权重
                n_factors = len(constrained_weights)
                normalized_weights = {k: 1.0/n_factors for k in constrained_weights}
            
            logger.info(f"计算因子权重完成: {model_name} v{model_version}")
            return normalized_weights
            
        except Exception as e:
            logger.error(f"计算因子权重失败: {e}")
            # 返回均等权重
            n_factors = len(importance_scores)
            return {k: 1.0/n_factors for k in importance_scores}
    
    def calculate_weight_stability(self,
                                 current_weights: Dict[str, float],
                                 historical_weights: Dict[str, float]) -> Dict[str, float]:
        """计算权重稳定性
        
        Args:
            current_weights: 当前权重
            historical_weights: 历史权重
            
        Returns:
            稳定性分数字典
        """
        stability_scores = {}
        
        for factor in current_weights:
            if factor in historical_weights:
                # 计算权重变化幅度
                weight_change = abs(current_weights[factor] - historical_weights[factor])
                # 稳定性分数 = 1 - 变化幅度（越小越稳定）
                stability_score = max(0, 1 - weight_change / max(historical_weights[factor], 0.01))
                stability_scores[factor] = stability_score
            else:
                # 新因子稳定性设为0.5
                stability_scores[factor] = 0.5
        
        return stability_scores
    
    def save_factor_weights(self,
                          model_name: str,
                          model_version: str,
                          weights: Dict[str, float],
                          importance_scores: Dict[str, float],
                          shap_values: Optional[Dict[str, float]] = None,
                          stability_scores: Optional[Dict[str, float]] = None,
                          calculation_date: Optional[str] = None) -> bool:
        """保存因子权重
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            weights: 权重字典
            importance_scores: 重要性分数
            shap_values: SHAP值
            stability_scores: 稳定性分数
            calculation_date: 计算日期
            
        Returns:
            是否保存成功
        """
        if calculation_date is None:
            calculation_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # 删除当天的旧记录
            delete_sql = """
            DELETE FROM factor_weights 
            WHERE model_name = :model_name 
            AND model_version = :model_version 
            AND calculation_date = :calculation_date
            """
            
            # 插入新记录
            insert_sql = """
            INSERT INTO factor_weights (
                factor_name, model_name, model_version, weight, 
                importance_score, shap_value, stability_score, 
                calculation_date, metadata
            ) VALUES (
                :factor_name, :model_name, :model_version, :weight,
                :importance_score, :shap_value, :stability_score,
                :calculation_date, :metadata
            )
            """
            
            with self.engine.connect() as conn:
                # 删除旧记录
                conn.execute(text(delete_sql), {
                    'model_name': model_name,
                    'model_version': model_version,
                    'calculation_date': calculation_date
                })
                
                # 插入新记录
                for factor_name, weight in weights.items():
                    metadata = {
                        'calculation_method': 'shap_based',
                        'constraints_applied': True
                    }
                    
                    conn.execute(text(insert_sql), {
                        'factor_name': factor_name,
                        'model_name': model_name,
                        'model_version': model_version,
                        'weight': weight,
                        'importance_score': importance_scores.get(factor_name, 0),
                        'shap_value': shap_values.get(factor_name, 0) if shap_values else 0,
                        'stability_score': stability_scores.get(factor_name, 0) if stability_scores else 0,
                        'calculation_date': calculation_date,
                        'metadata': json.dumps(metadata)
                    })
                
                conn.commit()
            
            logger.info(f"保存因子权重成功: {model_name} v{model_version} ({len(weights)}个因子)")
            return True
            
        except Exception as e:
            logger.error(f"保存因子权重失败: {e}")
            return False
    
    def get_factor_weights(self,
                         model_name: str,
                         model_version: str,
                         calculation_date: Optional[str] = None) -> Dict[str, FactorWeight]:
        """获取因子权重
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            calculation_date: 计算日期，如果为None则获取最新的
            
        Returns:
            因子权重字典
        """
        if calculation_date:
            query_sql = """
            SELECT factor_name, weight, importance_score, shap_value, 
                   stability_score, calculation_date
            FROM factor_weights 
            WHERE model_name = :model_name 
            AND model_version = :model_version 
            AND calculation_date = :calculation_date
            ORDER BY factor_name
            """
            params = {
                'model_name': model_name,
                'model_version': model_version,
                'calculation_date': calculation_date
            }
        else:
            query_sql = """
            SELECT factor_name, weight, importance_score, shap_value, 
                   stability_score, calculation_date
            FROM factor_weights 
            WHERE model_name = :model_name 
            AND model_version = :model_version 
            AND calculation_date = (
                SELECT MAX(calculation_date) FROM factor_weights 
                WHERE model_name = :model_name AND model_version = :model_version
            )
            ORDER BY factor_name
            """
            params = {
                'model_name': model_name,
                'model_version': model_version
            }
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                rows = result.fetchall()
            
            factor_weights = {}
            for row in rows:
                factor_weights[row[0]] = FactorWeight(
                    factor_name=row[0],
                    weight=float(row[1]),
                    importance_score=float(row[2] or 0),
                    shap_value=float(row[3] or 0),
                    stability_score=float(row[4] or 0),
                    last_updated=row[5],
                    model_name=model_name,
                    model_version=model_version
                )
            
            return factor_weights
            
        except Exception as e:
            logger.error(f"获取因子权重失败: {e}")
            return {}
    
    def update_factor_weights(self,
                            model_name: str,
                            model_version: str,
                            model_data: Dict[str, Any],
                            X_test: pd.DataFrame,
                            constraints: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """更新因子权重
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            model_data: 模型数据
            X_test: 测试数据
            constraints: 权重约束
            
        Returns:
            更新后的权重字典
        """
        try:
            # 计算因子重要性
            importance_scores = self.calculate_factor_importance(model_data, X_test)
            
            if not importance_scores:
                logger.warning("无法计算因子重要性")
                return {}
            
            # 获取历史权重
            historical_weights_data = self.get_factor_weights(model_name, model_version)
            historical_weights = {k: v.weight for k, v in historical_weights_data.items()}
            
            # 计算新权重
            new_weights = self.calculate_factor_weights(
                model_name=model_name,
                model_version=model_version,
                importance_scores=importance_scores,
                historical_weights=historical_weights,
                constraints=constraints
            )
            
            # 计算稳定性分数
            stability_scores = self.calculate_weight_stability(new_weights, historical_weights)
            
            # 保存权重
            success = self.save_factor_weights(
                model_name=model_name,
                model_version=model_version,
                weights=new_weights,
                importance_scores=importance_scores,
                stability_scores=stability_scores
            )
            
            if success:
                # 记录权重变化历史
                self._record_weight_changes(model_name, historical_weights, new_weights, "自动更新")
                logger.info(f"因子权重更新成功: {model_name} v{model_version}")
                return new_weights
            else:
                logger.error("保存因子权重失败")
                return {}
                
        except Exception as e:
            logger.error(f"更新因子权重失败: {e}")
            return {}
    
    def _record_weight_changes(self,
                             model_name: str,
                             old_weights: Dict[str, float],
                             new_weights: Dict[str, float],
                             change_reason: str,
                             changed_by: str = "system"):
        """记录权重变化历史"""
        try:
            insert_sql = """
            INSERT INTO factor_weight_history (
                factor_name, model_name, old_weight, new_weight, 
                change_reason, changed_by
            ) VALUES (
                :factor_name, :model_name, :old_weight, :new_weight,
                :change_reason, :changed_by
            )
            """
            
            with self.engine.connect() as conn:
                for factor_name in new_weights:
                    old_weight = old_weights.get(factor_name, 0)
                    new_weight = new_weights[factor_name]
                    
                    # 只记录有显著变化的权重
                    if abs(new_weight - old_weight) > 0.001:
                        conn.execute(text(insert_sql), {
                            'factor_name': factor_name,
                            'model_name': model_name,
                            'old_weight': old_weight,
                            'new_weight': new_weight,
                            'change_reason': change_reason,
                            'changed_by': changed_by
                        })
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"记录权重变化历史失败: {e}")
    
    def get_weight_history(self,
                         model_name: str,
                         factor_name: Optional[str] = None,
                         days: int = 30) -> pd.DataFrame:
        """获取权重变化历史
        
        Args:
            model_name: 模型名称
            factor_name: 因子名称，如果为None则获取所有因子
            days: 历史天数
            
        Returns:
            权重历史DataFrame
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        if factor_name:
            query_sql = """
            SELECT factor_name, old_weight, new_weight, change_reason, 
                   changed_at, changed_by
            FROM factor_weight_history 
            WHERE model_name = :model_name 
            AND factor_name = :factor_name
            AND changed_at >= :start_date
            ORDER BY changed_at DESC
            """
            params = {
                'model_name': model_name,
                'factor_name': factor_name,
                'start_date': start_date
            }
        else:
            query_sql = """
            SELECT factor_name, old_weight, new_weight, change_reason, 
                   changed_at, changed_by
            FROM factor_weight_history 
            WHERE model_name = :model_name 
            AND changed_at >= :start_date
            ORDER BY changed_at DESC, factor_name
            """
            params = {
                'model_name': model_name,
                'start_date': start_date
            }
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                rows = result.fetchall()
            
            if rows:
                df = pd.DataFrame(rows, columns=[
                    'factor_name', 'old_weight', 'new_weight', 
                    'change_reason', 'changed_at', 'changed_by'
                ])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取权重历史失败: {e}")
            return pd.DataFrame()
    
    def analyze_weight_stability(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """分析权重稳定性
        
        Args:
            model_name: 模型名称
            days: 分析天数
            
        Returns:
            稳定性分析结果
        """
        try:
            # 获取权重历史
            history_df = self.get_weight_history(model_name, days=days)
            
            if history_df.empty:
                return {'status': 'no_data', 'message': '没有权重历史数据'}
            
            # 计算每个因子的权重变化统计
            stability_analysis = {}
            
            for factor in history_df['factor_name'].unique():
                factor_history = history_df[history_df['factor_name'] == factor]
                
                if len(factor_history) > 1:
                    weight_changes = factor_history['new_weight'] - factor_history['old_weight']
                    
                    stability_analysis[factor] = {
                        'change_count': len(factor_history),
                        'avg_change': float(weight_changes.mean()),
                        'std_change': float(weight_changes.std()),
                        'max_change': float(weight_changes.abs().max()),
                        'stability_score': float(1 / (1 + weight_changes.abs().mean()))
                    }
                else:
                    stability_analysis[factor] = {
                        'change_count': len(factor_history),
                        'avg_change': 0,
                        'std_change': 0,
                        'max_change': 0,
                        'stability_score': 1.0
                    }
            
            # 计算整体稳定性
            overall_stability = np.mean([v['stability_score'] for v in stability_analysis.values()])
            
            return {
                'status': 'success',
                'overall_stability': float(overall_stability),
                'factor_analysis': stability_analysis,
                'analysis_period_days': days,
                'total_factors': len(stability_analysis)
            }
            
        except Exception as e:
            logger.error(f"分析权重稳定性失败: {e}")
            return {'status': 'error', 'message': str(e)}
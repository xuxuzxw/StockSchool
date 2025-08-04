"""AI模型管理器

扩展现有的ModelTrainingPipeline，增加模型版本管理、A/B测试等功能。
"""

import os
import json
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

from ..training_pipeline import ModelTrainingPipeline
from ...utils.db import get_db_manager

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """模型版本信息"""
    id: int
    name: str
    model_type: str
    version: str
    file_path: str
    status: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    training_start_date: str
    training_end_date: str
    created_at: datetime
    created_by: str
    description: str

class AIModelManager(ModelTrainingPipeline):
    """AI模型管理器
    
    扩展ModelTrainingPipeline，增加以下功能：
    - 模型版本管理
    - 模型性能对比
    - A/B测试支持
    - 自动重训练
    - 模型部署管理
    """
    
    def __init__(self, model_storage_path: str = "models"):
        super().__init__()
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(exist_ok=True)
        self.engine = get_db_manager().engine
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """确保AI模型相关表存在"""
        create_tables_sql = """
        -- AI模型表
        CREATE TABLE IF NOT EXISTS ai_models (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            version VARCHAR(20) NOT NULL,
            file_path TEXT NOT NULL,
            status VARCHAR(20) DEFAULT 'active',
            metrics JSONB,
            hyperparameters JSONB,
            feature_columns TEXT[],
            training_start_date DATE,
            training_end_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by VARCHAR(50),
            description TEXT,
            UNIQUE(name, version)
        );
        
        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_ai_models_status ON ai_models(status);
        CREATE INDEX IF NOT EXISTS idx_ai_models_type_version ON ai_models(model_type, version);
        CREATE INDEX IF NOT EXISTS idx_ai_models_name ON ai_models(name);
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
            logger.info("AI模型表创建成功")
        except Exception as e:
            logger.error(f"创建AI模型表失败: {e}")
            raise
    
    def create_model_version(self, 
                           name: str,
                           model_type: str,
                           description: str = "",
                           created_by: str = "system") -> str:
        """创建新的模型版本
        
        Args:
            name: 模型名称
            model_type: 模型类型
            description: 模型描述
            created_by: 创建者
            
        Returns:
            模型版本号
        """
        # 生成版本号
        version = self._generate_version(name)
        
        # 创建模型记录
        insert_sql = """
        INSERT INTO ai_models (name, model_type, version, file_path, status, created_by, description)
        VALUES (:name, :model_type, :version, :file_path, 'created', :created_by, :description)
        RETURNING id
        """
        
        file_path = str(self.model_storage_path / name / version)
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(insert_sql), {
                    'name': name,
                    'model_type': model_type,
                    'version': version,
                    'file_path': file_path,
                    'created_by': created_by,
                    'description': description
                })
                conn.commit()
                model_id = result.fetchone()[0]
                
            logger.info(f"创建模型版本: {name} v{version} (ID: {model_id})")
            return version
            
        except Exception as e:
            logger.error(f"创建模型版本失败: {e}")
            raise
    
    def _generate_version(self, model_name: str) -> str:
        """生成模型版本号
        
        Args:
            model_name: 模型名称
            
        Returns:
            版本号 (格式: v1.0.0)
        """
        # 查询最新版本
        query_sql = """
        SELECT version FROM ai_models 
        WHERE name = :name 
        ORDER BY created_at DESC 
        LIMIT 1
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'name': model_name})
                latest_version = result.fetchone()
                
            if latest_version is None:
                return "v1.0.0"
            
            # 解析版本号并递增
            version_str = latest_version[0]
            if version_str.startswith('v'):
                version_parts = version_str[1:].split('.')
                major, minor, patch = map(int, version_parts)
                return f"v{major}.{minor}.{patch + 1}"
            else:
                return "v1.0.0"
                
        except Exception as e:
            logger.error(f"生成版本号失败: {e}")
            return f"v1.0.{int(datetime.now().timestamp())}"
    
    def train_and_save_model(self,
                           model_name: str,
                           factor_names: List[str],
                           start_date: str,
                           end_date: str,
                           model_configs: List[Dict],
                           target_period: int = 5,
                           stock_list: Optional[List[str]] = None,
                           description: str = "",
                           created_by: str = "system") -> Dict[str, Any]:
        """训练并保存模型
        
        Args:
            model_name: 模型名称
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            model_configs: 模型配置列表
            target_period: 目标周期
            stock_list: 股票列表
            description: 模型描述
            created_by: 创建者
            
        Returns:
            训练结果
        """
        logger.info(f"开始训练模型: {model_name}")
        
        # 创建模型版本
        version = self.create_model_version(
            name=model_name,
            model_type=model_configs[0]['type'] if model_configs else 'unknown',
            description=description,
            created_by=created_by
        )
        
        try:
            # 运行训练流水线
            result = self.run_training_pipeline(
                factor_names=factor_names,
                start_date=start_date,
                end_date=end_date,
                model_configs=model_configs,
                model_name=f"{model_name}_{version}",
                target_period=target_period,
                stock_list=stock_list,
                save_models=True
            )
            
            if result and result.get('best_model'):
                # 更新模型记录
                self._update_model_record(
                    model_name=model_name,
                    version=version,
                    training_result=result,
                    start_date=start_date,
                    end_date=end_date
                )
                
                logger.info(f"模型训练完成: {model_name} v{version}")
                return {
                    'model_name': model_name,
                    'version': version,
                    'status': 'success',
                    'result': result
                }
            else:
                # 标记为失败
                self._mark_model_failed(model_name, version, "训练失败")
                return {
                    'model_name': model_name,
                    'version': version,
                    'status': 'failed',
                    'error': '训练失败'
                }
                
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            self._mark_model_failed(model_name, version, str(e))
            return {
                'model_name': model_name,
                'version': version,
                'status': 'failed',
                'error': str(e)
            }
    
    def _update_model_record(self,
                           model_name: str,
                           version: str,
                           training_result: Dict[str, Any],
                           start_date: str,
                           end_date: str):
        """更新模型记录"""
        best_model = training_result['best_model']
        
        update_sql = """
        UPDATE ai_models SET
            status = 'trained',
            metrics = :metrics,
            hyperparameters = :hyperparameters,
            feature_columns = :feature_columns,
            training_start_date = :start_date,
            training_end_date = :end_date,
            updated_at = CURRENT_TIMESTAMP
        WHERE name = :name AND version = :version
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(update_sql), {
                    'name': model_name,
                    'version': version,
                    'metrics': json.dumps({
                        'train_metrics': best_model['train_metrics'],
                        'test_metrics': best_model['test_metrics'],
                        'cv_scores': best_model['cv_scores']
                    }),
                    'hyperparameters': json.dumps(best_model['model_params']),
                    'feature_columns': best_model['feature_columns'],
                    'start_date': start_date,
                    'end_date': end_date
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"更新模型记录失败: {e}")
            raise
    
    def _mark_model_failed(self, model_name: str, version: str, error_message: str):
        """标记模型为失败状态"""
        update_sql = """
        UPDATE ai_models SET
            status = 'failed',
            description = COALESCE(description, '') || ' Error: ' || :error,
            updated_at = CURRENT_TIMESTAMP
        WHERE name = :name AND version = :version
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(update_sql), {
                    'name': model_name,
                    'version': version,
                    'error': error_message
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"标记模型失败状态失败: {e}")
    
    def get_model_versions(self, model_name: str) -> List[ModelVersion]:
        """获取模型的所有版本
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型版本列表
        """
        query_sql = """
        SELECT id, name, model_type, version, file_path, status, 
               metrics, hyperparameters, feature_columns,
               training_start_date, training_end_date, created_at, 
               created_by, description
        FROM ai_models 
        WHERE name = :name 
        ORDER BY created_at DESC
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'name': model_name})
                rows = result.fetchall()
                
            versions = []
            for row in rows:
                versions.append(ModelVersion(
                    id=row[0],
                    name=row[1],
                    model_type=row[2],
                    version=row[3],
                    file_path=row[4],
                    status=row[5],
                    metrics=row[6] or {},
                    hyperparameters=row[7] or {},
                    feature_columns=row[8] or [],
                    training_start_date=str(row[9]) if row[9] else "",
                    training_end_date=str(row[10]) if row[10] else "",
                    created_at=row[11],
                    created_by=row[12] or "",
                    description=row[13] or ""
                ))
                
            return versions
            
        except Exception as e:
            logger.error(f"获取模型版本失败: {e}")
            return []
    
    def get_best_model(self, model_name: str, metric: str = 'r2') -> Optional[ModelVersion]:
        """获取最佳模型版本
        
        Args:
            model_name: 模型名称
            metric: 评估指标
            
        Returns:
            最佳模型版本
        """
        versions = self.get_model_versions(model_name)
        
        if not versions:
            return None
        
        # 过滤已训练的模型
        trained_versions = [v for v in versions if v.status == 'trained']
        
        if not trained_versions:
            return None
        
        # 根据指标选择最佳模型
        best_version = None
        best_score = float('-inf')
        
        for version in trained_versions:
            if version.metrics and 'test_metrics' in version.metrics:
                score = version.metrics['test_metrics'].get(metric, float('-inf'))
                if score > best_score:
                    best_score = score
                    best_version = version
        
        return best_version
    
    def deploy_model(self, model_name: str, version: str) -> bool:
        """部署模型
        
        Args:
            model_name: 模型名称
            version: 版本号
            
        Returns:
            是否部署成功
        """
        try:
            # 将当前活跃模型设为非活跃
            deactivate_sql = """
            UPDATE ai_models SET status = 'inactive' 
            WHERE name = :name AND status = 'active'
            """
            
            # 激活指定版本
            activate_sql = """
            UPDATE ai_models SET status = 'active', updated_at = CURRENT_TIMESTAMP
            WHERE name = :name AND version = :version AND status = 'trained'
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(deactivate_sql), {'name': model_name})
                result = conn.execute(text(activate_sql), {
                    'name': model_name,
                    'version': version
                })
                conn.commit()
                
                if result.rowcount > 0:
                    logger.info(f"模型部署成功: {model_name} v{version}")
                    return True
                else:
                    logger.warning(f"模型部署失败: {model_name} v{version} 不存在或状态不正确")
                    return False
                    
        except Exception as e:
            logger.error(f"模型部署失败: {e}")
            return False
    
    def get_active_model(self, model_name: str) -> Optional[ModelVersion]:
        """获取当前活跃的模型版本
        
        Args:
            model_name: 模型名称
            
        Returns:
            活跃的模型版本
        """
        query_sql = """
        SELECT id, name, model_type, version, file_path, status, 
               metrics, hyperparameters, feature_columns,
               training_start_date, training_end_date, created_at, 
               created_by, description
        FROM ai_models 
        WHERE name = :name AND status = 'active'
        LIMIT 1
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'name': model_name})
                row = result.fetchone()
                
            if row:
                return ModelVersion(
                    id=row[0],
                    name=row[1],
                    model_type=row[2],
                    version=row[3],
                    file_path=row[4],
                    status=row[5],
                    metrics=row[6] or {},
                    hyperparameters=row[7] or {},
                    feature_columns=row[8] or [],
                    training_start_date=str(row[9]) if row[9] else "",
                    training_end_date=str(row[10]) if row[10] else "",
                    created_at=row[11],
                    created_by=row[12] or "",
                    description=row[13] or ""
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取活跃模型失败: {e}")
            return None
    
    def compare_models(self, model_name: str, versions: List[str]) -> pd.DataFrame:
        """比较模型版本性能
        
        Args:
            model_name: 模型名称
            versions: 版本列表
            
        Returns:
            比较结果DataFrame
        """
        comparison_data = []
        
        for version in versions:
            model_versions = self.get_model_versions(model_name)
            model_version = next((v for v in model_versions if v.version == version), None)
            
            if model_version and model_version.metrics:
                test_metrics = model_version.metrics.get('test_metrics', {})
                train_metrics = model_version.metrics.get('train_metrics', {})
                
                comparison_data.append({
                    'version': version,
                    'status': model_version.status,
                    'model_type': model_version.model_type,
                    'test_r2': test_metrics.get('r2', 0),
                    'test_rmse': test_metrics.get('rmse', 0),
                    'test_mae': test_metrics.get('mae', 0),
                    'train_r2': train_metrics.get('r2', 0),
                    'train_rmse': train_metrics.get('rmse', 0),
                    'train_mae': train_metrics.get('mae', 0),
                    'created_at': model_version.created_at
                })
        
        return pd.DataFrame(comparison_data)
    
    def delete_model_version(self, model_name: str, version: str) -> bool:
        """删除模型版本
        
        Args:
            model_name: 模型名称
            version: 版本号
            
        Returns:
            是否删除成功
        """
        try:
            # 检查是否为活跃模型
            active_model = self.get_active_model(model_name)
            if active_model and active_model.version == version:
                logger.warning(f"无法删除活跃模型: {model_name} v{version}")
                return False
            
            # 删除数据库记录
            delete_sql = """
            DELETE FROM ai_models 
            WHERE name = :name AND version = :version
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(delete_sql), {
                    'name': model_name,
                    'version': version
                })
                conn.commit()
                
                if result.rowcount > 0:
                    # 删除模型文件
                    model_path = self.model_storage_path / model_name / version
                    if model_path.exists():
                        import shutil
                        shutil.rmtree(model_path)
                    
                    logger.info(f"模型版本删除成功: {model_name} v{version}")
                    return True
                else:
                    logger.warning(f"模型版本不存在: {model_name} v{version}")
                    return False
                    
        except Exception as e:
            logger.error(f"删除模型版本失败: {e}")
            return False
    
    def load_model_for_prediction(self, model_name: str, version: str = None) -> Optional[Dict[str, Any]]:
        """加载模型用于预测
        
        Args:
            model_name: 模型名称
            version: 版本号，如果为None则使用活跃版本
            
        Returns:
            模型字典
        """
        if version is None:
            model_version = self.get_active_model(model_name)
        else:
            model_versions = self.get_model_versions(model_name)
            model_version = next((v for v in model_versions if v.version == version), None)
        
        if not model_version:
            logger.error(f"模型不存在: {model_name} v{version or 'active'}")
            return None
        
        try:
            # 加载模型文件
            model_path = Path(model_version.file_path)
            if not model_path.exists():
                logger.error(f"模型文件不存在: {model_path}")
                return None
            
            model_data = self.load_model(str(model_path))
            model_data['version_info'] = model_version
            
            return model_data
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None
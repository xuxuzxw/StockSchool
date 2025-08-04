"""模型监控器

提供AI模型的实时监控功能，包括性能监控、数据漂移检测、自动重训练等。
"""

import logging
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

from ...utils.db import get_db_manager
from .model_manager import AIModelManager
from .factor_weight_engine import FactorWeightEngine
from .prediction_service import PredictionService

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """模型性能指标"""
    model_name: str
    model_version: str
    date: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float]
    prediction_count: int
    avg_prediction_time: float
    error_rate: float
    confidence_distribution: Dict[str, float]
    feature_importance_drift: Optional[float]
    created_at: Optional[datetime] = None

@dataclass
class DataDriftMetrics:
    """数据漂移指标"""
    model_name: str
    feature_name: str
    date: datetime
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    drift_score: float
    drift_type: str  # 'mean_shift', 'variance_change', 'distribution_change'
    is_significant: bool
    p_value: float
    threshold: float
    created_at: Optional[datetime] = None

@dataclass
class ModelAlert:
    """模型告警"""
    alert_id: str
    model_name: str
    alert_type: str  # 'performance_degradation', 'data_drift', 'prediction_anomaly'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    details: Dict[str, Any]
    threshold_value: float
    actual_value: float
    status: str  # 'active', 'acknowledged', 'resolved'
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

@dataclass
class RetrainingTrigger:
    """重训练触发器"""
    trigger_id: str
    model_name: str
    trigger_type: str  # 'performance_threshold', 'data_drift', 'scheduled', 'manual'
    trigger_condition: Dict[str, Any]
    is_triggered: bool
    trigger_time: Optional[datetime]
    retraining_status: str  # 'pending', 'running', 'completed', 'failed'
    retraining_job_id: Optional[str]
    created_at: Optional[datetime] = None

class ModelMonitor:
    """模型监控器
    
    主要功能：
    - 模型性能监控
    - 数据漂移检测
    - 预测异常检测
    - 自动告警机制
    - 重训练触发
    - 监控报告生成
    """
    
    def __init__(self):
        self.engine = get_db_manager().engine
        self.model_manager = AIModelManager()
        self.factor_engine = FactorWeightEngine()
        self.prediction_service = PredictionService()
        
        # 监控配置
        self.performance_thresholds = {
            'accuracy': 0.6,
            'precision': 0.6,
            'recall': 0.6,
            'f1_score': 0.6,
            'error_rate': 0.3
        }
        
        self.drift_thresholds = {
            'mean_shift': 2.0,  # 标准差倍数
            'variance_change': 0.5,  # 方差变化比例
            'distribution_change': 0.05  # KS检验p值
        }
        
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """确保监控相关表存在"""
        create_tables_sql = """
        -- 模型性能监控表
        CREATE TABLE IF NOT EXISTS model_performance_metrics (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            accuracy DECIMAL(6, 4),
            precision_score DECIMAL(6, 4),
            recall_score DECIMAL(6, 4),
            f1_score DECIMAL(6, 4),
            auc_score DECIMAL(6, 4),
            prediction_count INTEGER,
            avg_prediction_time DECIMAL(8, 4),
            error_rate DECIMAL(6, 4),
            confidence_distribution JSONB,
            feature_importance_drift DECIMAL(6, 4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 数据漂移监控表
        CREATE TABLE IF NOT EXISTS data_drift_metrics (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            feature_name VARCHAR(100) NOT NULL,
            date DATE NOT NULL,
            reference_mean DECIMAL(12, 6),
            current_mean DECIMAL(12, 6),
            reference_std DECIMAL(12, 6),
            current_std DECIMAL(12, 6),
            drift_score DECIMAL(8, 4),
            drift_type VARCHAR(50),
            is_significant BOOLEAN,
            p_value DECIMAL(10, 8),
            threshold_value DECIMAL(8, 4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 模型告警表
        CREATE TABLE IF NOT EXISTS model_alerts (
            id SERIAL PRIMARY KEY,
            alert_id VARCHAR(100) UNIQUE NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            alert_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            message TEXT NOT NULL,
            details JSONB,
            threshold_value DECIMAL(10, 4),
            actual_value DECIMAL(10, 4),
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP
        );
        
        -- 重训练触发器表
        CREATE TABLE IF NOT EXISTS retraining_triggers (
            id SERIAL PRIMARY KEY,
            trigger_id VARCHAR(100) UNIQUE NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            trigger_type VARCHAR(50) NOT NULL,
            trigger_condition JSONB NOT NULL,
            is_triggered BOOLEAN DEFAULT false,
            trigger_time TIMESTAMP,
            retraining_status VARCHAR(20) DEFAULT 'pending',
            retraining_job_id VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 模型监控配置表
        CREATE TABLE IF NOT EXISTS model_monitor_configs (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) UNIQUE NOT NULL,
            performance_thresholds JSONB,
            drift_thresholds JSONB,
            monitoring_frequency VARCHAR(20) DEFAULT 'daily',
            alert_channels JSONB,
            auto_retrain_enabled BOOLEAN DEFAULT false,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 监控任务日志表
        CREATE TABLE IF NOT EXISTS monitor_task_logs (
            id SERIAL PRIMARY KEY,
            task_type VARCHAR(50) NOT NULL,
            model_name VARCHAR(100),
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            status VARCHAR(20) NOT NULL,
            metrics_collected INTEGER DEFAULT 0,
            alerts_generated INTEGER DEFAULT 0,
            errors JSONB,
            execution_time DECIMAL(8, 4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_model_performance_model_date ON model_performance_metrics(model_name, date);
        CREATE INDEX IF NOT EXISTS idx_data_drift_model_feature ON data_drift_metrics(model_name, feature_name, date);
        CREATE INDEX IF NOT EXISTS idx_model_alerts_model_status ON model_alerts(model_name, status, created_at);
        CREATE INDEX IF NOT EXISTS idx_retraining_triggers_model ON retraining_triggers(model_name, is_triggered);
        CREATE INDEX IF NOT EXISTS idx_monitor_task_logs_type_time ON monitor_task_logs(task_type, start_time);
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
            logger.info("模型监控表创建成功")
        except Exception as e:
            logger.error(f"创建模型监控表失败: {e}")
            raise
    
    def monitor_model_performance(self, model_name: str, model_version: str = None) -> Optional[ModelPerformanceMetrics]:
        """监控模型性能
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            
        Returns:
            性能指标对象
        """
        try:
            start_time = datetime.now()
            
            # 获取最新预测结果
            predictions = self._get_recent_predictions(model_name, model_version)
            if not predictions:
                logger.warning(f"没有找到模型{model_name}的预测结果")
                return None
            
            # 获取真实标签（这里需要根据实际业务逻辑实现）
            true_labels = self._get_true_labels(predictions)
            if not true_labels:
                logger.warning("无法获取真实标签")
                return None
            
            # 计算性能指标
            pred_labels = [p['prediction'] > 0.5 for p in predictions]  # 假设二分类
            
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
            recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
            
            # 计算错误率
            error_count = sum(1 for p, t in zip(pred_labels, true_labels) if p != t)
            error_rate = error_count / len(predictions)
            
            # 计算平均预测时间
            avg_prediction_time = np.mean([p.get('prediction_time', 0) for p in predictions])
            
            # 计算置信度分布
            confidences = [p.get('confidence', 0.5) for p in predictions]
            confidence_distribution = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'q25': float(np.percentile(confidences, 25)),
                'q50': float(np.percentile(confidences, 50)),
                'q75': float(np.percentile(confidences, 75))
            }
            
            # 计算特征重要性漂移（如果有历史数据）
            feature_importance_drift = self._calculate_feature_importance_drift(model_name)
            
            # 创建性能指标对象
            metrics = ModelPerformanceMetrics(
                model_name=model_name,
                model_version=model_version or 'latest',
                date=datetime.now().date(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_score=None,  # 需要额外计算
                prediction_count=len(predictions),
                avg_prediction_time=avg_prediction_time,
                error_rate=error_rate,
                confidence_distribution=confidence_distribution,
                feature_importance_drift=feature_importance_drift,
                created_at=datetime.now()
            )
            
            # 保存性能指标
            self._save_performance_metrics(metrics)
            
            # 检查性能阈值
            self._check_performance_thresholds(metrics)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._log_monitor_task('performance_monitoring', model_name, start_time, 
                                 datetime.now(), 'completed', 1, execution_time=execution_time)
            
            logger.info(f"模型{model_name}性能监控完成")
            return metrics
            
        except Exception as e:
            logger.error(f"监控模型性能失败: {e}")
            self._log_monitor_task('performance_monitoring', model_name, start_time,
                                 datetime.now(), 'failed', 0, errors={'error': str(e)})
            return None
    
    def detect_data_drift(self, model_name: str, feature_data: pd.DataFrame) -> List[DataDriftMetrics]:
        """检测数据漂移
        
        Args:
            model_name: 模型名称
            feature_data: 当前特征数据
            
        Returns:
            漂移指标列表
        """
        try:
            start_time = datetime.now()
            drift_metrics = []
            
            # 获取参考数据（训练时的数据分布）
            reference_data = self._get_reference_data(model_name)
            if reference_data is None:
                logger.warning(f"无法获取模型{model_name}的参考数据")
                return []
            
            # 对每个特征检测漂移
            for feature_name in feature_data.columns:
                if feature_name not in reference_data.columns:
                    continue
                
                current_values = feature_data[feature_name].dropna()
                reference_values = reference_data[feature_name].dropna()
                
                if len(current_values) == 0 or len(reference_values) == 0:
                    continue
                
                # 计算统计量
                ref_mean = reference_values.mean()
                ref_std = reference_values.std()
                curr_mean = current_values.mean()
                curr_std = current_values.std()
                
                # 检测均值漂移
                mean_drift_score = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
                mean_drift = mean_drift_score > self.drift_thresholds['mean_shift']
                
                # 检测方差漂移
                var_drift_score = abs(curr_std - ref_std) / (ref_std + 1e-8)
                var_drift = var_drift_score > self.drift_thresholds['variance_change']
                
                # KS检验检测分布漂移
                ks_stat, ks_p_value = stats.ks_2samp(reference_values, current_values)
                dist_drift = ks_p_value < self.drift_thresholds['distribution_change']
                
                # 确定主要漂移类型
                if mean_drift and mean_drift_score >= var_drift_score:
                    drift_type = 'mean_shift'
                    drift_score = mean_drift_score
                    is_significant = mean_drift
                    p_value = None
                    threshold = self.drift_thresholds['mean_shift']
                elif var_drift:
                    drift_type = 'variance_change'
                    drift_score = var_drift_score
                    is_significant = var_drift
                    p_value = None
                    threshold = self.drift_thresholds['variance_change']
                else:
                    drift_type = 'distribution_change'
                    drift_score = ks_stat
                    is_significant = dist_drift
                    p_value = ks_p_value
                    threshold = self.drift_thresholds['distribution_change']
                
                # 创建漂移指标
                drift_metric = DataDriftMetrics(
                    model_name=model_name,
                    feature_name=feature_name,
                    date=datetime.now().date(),
                    reference_mean=ref_mean,
                    current_mean=curr_mean,
                    reference_std=ref_std,
                    current_std=curr_std,
                    drift_score=drift_score,
                    drift_type=drift_type,
                    is_significant=is_significant,
                    p_value=p_value or 0.0,
                    threshold=threshold,
                    created_at=datetime.now()
                )
                
                drift_metrics.append(drift_metric)
                
                # 保存漂移指标
                self._save_drift_metrics(drift_metric)
                
                # 如果检测到显著漂移，生成告警
                if is_significant:
                    self._create_drift_alert(drift_metric)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._log_monitor_task('data_drift_detection', model_name, start_time,
                                 datetime.now(), 'completed', len(drift_metrics), 
                                 execution_time=execution_time)
            
            logger.info(f"模型{model_name}数据漂移检测完成，检测到{len([d for d in drift_metrics if d.is_significant])}个显著漂移")
            return drift_metrics
            
        except Exception as e:
            logger.error(f"检测数据漂移失败: {e}")
            self._log_monitor_task('data_drift_detection', model_name, start_time,
                                 datetime.now(), 'failed', 0, errors={'error': str(e)})
            return []
    
    def check_retraining_triggers(self, model_name: str) -> List[RetrainingTrigger]:
        """检查重训练触发条件
        
        Args:
            model_name: 模型名称
            
        Returns:
            触发的重训练条件列表
        """
        try:
            triggered_conditions = []
            
            # 获取最新性能指标
            latest_metrics = self._get_latest_performance_metrics(model_name)
            if not latest_metrics:
                return []
            
            # 检查性能阈值触发
            performance_trigger = self._check_performance_trigger(latest_metrics)
            if performance_trigger:
                triggered_conditions.append(performance_trigger)
            
            # 检查数据漂移触发
            drift_trigger = self._check_drift_trigger(model_name)
            if drift_trigger:
                triggered_conditions.append(drift_trigger)
            
            # 检查定时触发
            scheduled_trigger = self._check_scheduled_trigger(model_name)
            if scheduled_trigger:
                triggered_conditions.append(scheduled_trigger)
            
            # 保存触发条件
            for trigger in triggered_conditions:
                self._save_retraining_trigger(trigger)
                
                # 如果启用自动重训练，启动重训练任务
                if self._is_auto_retrain_enabled(model_name):
                    self._start_retraining_job(trigger)
            
            return triggered_conditions
            
        except Exception as e:
            logger.error(f"检查重训练触发条件失败: {e}")
            return []
    
    def generate_monitoring_report(self, model_name: str, days: int = 7) -> Dict[str, Any]:
        """生成监控报告
        
        Args:
            model_name: 模型名称
            days: 报告天数
            
        Returns:
            监控报告
        """
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # 获取性能指标
            performance_data = self._get_performance_data(model_name, start_date, end_date)
            
            # 获取漂移数据
            drift_data = self._get_drift_data(model_name, start_date, end_date)
            
            # 获取告警数据
            alert_data = self._get_alert_data(model_name, start_date, end_date)
            
            # 生成图表
            charts = self._generate_monitoring_charts(model_name, performance_data, drift_data)
            
            # 计算汇总统计
            summary_stats = self._calculate_summary_stats(performance_data, drift_data, alert_data)
            
            report = {
                'model_name': model_name,
                'report_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': days
                },
                'summary': summary_stats,
                'performance_metrics': performance_data,
                'drift_metrics': drift_data,
                'alerts': alert_data,
                'charts': charts,
                'recommendations': self._generate_recommendations(summary_stats),
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成监控报告失败: {e}")
            return {}
    
    def _get_recent_predictions(self, model_name: str, model_version: str = None, days: int = 1) -> List[Dict]:
        """获取最近的预测结果"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            query_sql = """
            SELECT *
            FROM prediction_results
            WHERE model_name = :model_name
            AND created_at >= :start_date
            AND created_at <= :end_date
            """
            
            params = {
                'model_name': model_name,
                'start_date': start_date,
                'end_date': end_date
            }
            
            if model_version:
                query_sql += " AND model_version = :model_version"
                params['model_version'] = model_version
            
            query_sql += " ORDER BY created_at DESC LIMIT 1000"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                rows = result.fetchall()
            
            predictions = []
            for row in rows:
                predictions.append({
                    'stock_code': row[1],
                    'prediction': row[3],
                    'confidence': row[4],
                    'prediction_time': row[6],
                    'created_at': row[7]
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"获取预测结果失败: {e}")
            return []
    
    def _get_true_labels(self, predictions: List[Dict]) -> List[bool]:
        """获取真实标签（需要根据实际业务逻辑实现）"""
        try:
            # 这里是示例实现，实际需要根据业务逻辑获取真实标签
            # 比如根据股票后续涨跌情况确定预测是否正确
            
            true_labels = []
            for pred in predictions:
                # 示例：随机生成真实标签（实际应该从数据库获取）
                # 这里需要根据股票代码和预测时间获取后续的真实涨跌情况
                true_label = np.random.choice([True, False])  # 临时实现
                true_labels.append(true_label)
            
            return true_labels
            
        except Exception as e:
            logger.error(f"获取真实标签失败: {e}")
            return []
    
    def _calculate_feature_importance_drift(self, model_name: str) -> Optional[float]:
        """计算特征重要性漂移"""
        try:
            # 获取当前模型的特征重要性
            current_importance = self.factor_engine.get_factor_weights(model_name)
            if not current_importance:
                return None
            
            # 获取历史特征重要性
            history = self.factor_engine.get_factor_weight_history(model_name, days=30)
            if not history:
                return None
            
            # 计算重要性变化
            if len(history) > 0:
                latest_history = history[0]
                current_weights = {fw.factor_name: fw.weight for fw in current_importance}
                historical_weights = json.loads(latest_history['weights'])
                
                # 计算余弦相似度
                common_factors = set(current_weights.keys()) & set(historical_weights.keys())
                if len(common_factors) > 0:
                    curr_vec = np.array([current_weights[f] for f in common_factors])
                    hist_vec = np.array([historical_weights[f] for f in common_factors])
                    
                    cosine_sim = np.dot(curr_vec, hist_vec) / (np.linalg.norm(curr_vec) * np.linalg.norm(hist_vec))
                    drift_score = 1 - cosine_sim
                    
                    return float(drift_score)
            
            return None
            
        except Exception as e:
            logger.error(f"计算特征重要性漂移失败: {e}")
            return None
    
    def _get_reference_data(self, model_name: str) -> Optional[pd.DataFrame]:
        """获取参考数据（训练时的数据分布）"""
        try:
            # 这里需要根据实际情况获取训练时的数据
            # 可以从模型训练记录中获取，或者从专门的参考数据表中获取
            
            query_sql = """
            SELECT training_data_stats
            FROM ai_models
            WHERE model_name = :model_name
            AND status = 'active'
            ORDER BY created_at DESC
            LIMIT 1
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'model_name': model_name})
                row = result.fetchone()
            
            if row and row[0]:
                # 假设训练数据统计以JSON格式存储
                stats_data = json.loads(row[0])
                # 这里需要根据实际数据格式转换为DataFrame
                # 示例实现
                return pd.DataFrame(stats_data)
            
            return None
            
        except Exception as e:
            logger.error(f"获取参考数据失败: {e}")
            return None
    
    def _save_performance_metrics(self, metrics: ModelPerformanceMetrics):
        """保存性能指标"""
        try:
            insert_sql = """
            INSERT INTO model_performance_metrics (
                model_name, model_version, date, accuracy, precision_score,
                recall_score, f1_score, auc_score, prediction_count,
                avg_prediction_time, error_rate, confidence_distribution,
                feature_importance_drift
            ) VALUES (
                :model_name, :model_version, :date, :accuracy, :precision_score,
                :recall_score, :f1_score, :auc_score, :prediction_count,
                :avg_prediction_time, :error_rate, :confidence_distribution,
                :feature_importance_drift
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'model_name': metrics.model_name,
                    'model_version': metrics.model_version,
                    'date': metrics.date,
                    'accuracy': metrics.accuracy,
                    'precision_score': metrics.precision,
                    'recall_score': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'auc_score': metrics.auc_score,
                    'prediction_count': metrics.prediction_count,
                    'avg_prediction_time': metrics.avg_prediction_time,
                    'error_rate': metrics.error_rate,
                    'confidence_distribution': json.dumps(metrics.confidence_distribution),
                    'feature_importance_drift': metrics.feature_importance_drift
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"保存性能指标失败: {e}")
            raise
    
    def _save_drift_metrics(self, drift_metric: DataDriftMetrics):
        """保存漂移指标"""
        try:
            insert_sql = """
            INSERT INTO data_drift_metrics (
                model_name, feature_name, date, reference_mean, current_mean,
                reference_std, current_std, drift_score, drift_type,
                is_significant, p_value, threshold_value
            ) VALUES (
                :model_name, :feature_name, :date, :reference_mean, :current_mean,
                :reference_std, :current_std, :drift_score, :drift_type,
                :is_significant, :p_value, :threshold_value
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'model_name': drift_metric.model_name,
                    'feature_name': drift_metric.feature_name,
                    'date': drift_metric.date,
                    'reference_mean': drift_metric.reference_mean,
                    'current_mean': drift_metric.current_mean,
                    'reference_std': drift_metric.reference_std,
                    'current_std': drift_metric.current_std,
                    'drift_score': drift_metric.drift_score,
                    'drift_type': drift_metric.drift_type,
                    'is_significant': drift_metric.is_significant,
                    'p_value': drift_metric.p_value,
                    'threshold_value': drift_metric.threshold
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"保存漂移指标失败: {e}")
            raise
    
    def _check_performance_thresholds(self, metrics: ModelPerformanceMetrics):
        """检查性能阈值"""
        try:
            alerts = []
            
            # 检查各项性能指标
            if metrics.accuracy < self.performance_thresholds['accuracy']:
                alerts.append(self._create_performance_alert(
                    metrics.model_name, 'accuracy', 
                    self.performance_thresholds['accuracy'], metrics.accuracy
                ))
            
            if metrics.precision < self.performance_thresholds['precision']:
                alerts.append(self._create_performance_alert(
                    metrics.model_name, 'precision',
                    self.performance_thresholds['precision'], metrics.precision
                ))
            
            if metrics.recall < self.performance_thresholds['recall']:
                alerts.append(self._create_performance_alert(
                    metrics.model_name, 'recall',
                    self.performance_thresholds['recall'], metrics.recall
                ))
            
            if metrics.f1_score < self.performance_thresholds['f1_score']:
                alerts.append(self._create_performance_alert(
                    metrics.model_name, 'f1_score',
                    self.performance_thresholds['f1_score'], metrics.f1_score
                ))
            
            if metrics.error_rate > self.performance_thresholds['error_rate']:
                alerts.append(self._create_performance_alert(
                    metrics.model_name, 'error_rate',
                    self.performance_thresholds['error_rate'], metrics.error_rate
                ))
            
            # 保存告警
            for alert in alerts:
                if alert:
                    self._save_alert(alert)
            
        except Exception as e:
            logger.error(f"检查性能阈值失败: {e}")
    
    def _create_performance_alert(self, model_name: str, metric_name: str, 
                                threshold: float, actual_value: float) -> Optional[ModelAlert]:
        """创建性能告警"""
        try:
            severity = 'high' if actual_value < threshold * 0.8 else 'medium'
            
            alert = ModelAlert(
                alert_id=f"PERF_{model_name}_{metric_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                model_name=model_name,
                alert_type='performance_degradation',
                severity=severity,
                message=f"模型{model_name}的{metric_name}指标({actual_value:.4f})低于阈值({threshold:.4f})",
                details={
                    'metric_name': metric_name,
                    'threshold': threshold,
                    'actual_value': actual_value,
                    'degradation_ratio': (threshold - actual_value) / threshold
                },
                threshold_value=threshold,
                actual_value=actual_value,
                status='active',
                created_at=datetime.now()
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"创建性能告警失败: {e}")
            return None
    
    def _create_drift_alert(self, drift_metric: DataDriftMetrics):
        """创建漂移告警"""
        try:
            severity = 'high' if drift_metric.drift_score > drift_metric.threshold * 2 else 'medium'
            
            alert = ModelAlert(
                alert_id=f"DRIFT_{drift_metric.model_name}_{drift_metric.feature_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                model_name=drift_metric.model_name,
                alert_type='data_drift',
                severity=severity,
                message=f"模型{drift_metric.model_name}的特征{drift_metric.feature_name}检测到{drift_metric.drift_type}漂移",
                details={
                    'feature_name': drift_metric.feature_name,
                    'drift_type': drift_metric.drift_type,
                    'drift_score': drift_metric.drift_score,
                    'threshold': drift_metric.threshold,
                    'p_value': drift_metric.p_value
                },
                threshold_value=drift_metric.threshold,
                actual_value=drift_metric.drift_score,
                status='active',
                created_at=datetime.now()
            )
            
            self._save_alert(alert)
            
        except Exception as e:
            logger.error(f"创建漂移告警失败: {e}")
    
    def _save_alert(self, alert: ModelAlert):
        """保存告警"""
        try:
            insert_sql = """
            INSERT INTO model_alerts (
                alert_id, model_name, alert_type, severity, message,
                details, threshold_value, actual_value, status
            ) VALUES (
                :alert_id, :model_name, :alert_type, :severity, :message,
                :details, :threshold_value, :actual_value, :status
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'alert_id': alert.alert_id,
                    'model_name': alert.model_name,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'details': json.dumps(alert.details),
                    'threshold_value': alert.threshold_value,
                    'actual_value': alert.actual_value,
                    'status': alert.status
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"保存告警失败: {e}")
            raise
    
    def _log_monitor_task(self, task_type: str, model_name: str, start_time: datetime,
                         end_time: datetime, status: str, metrics_collected: int,
                         alerts_generated: int = 0, errors: Dict = None, execution_time: float = None):
        """记录监控任务日志"""
        try:
            insert_sql = """
            INSERT INTO monitor_task_logs (
                task_type, model_name, start_time, end_time, status,
                metrics_collected, alerts_generated, errors, execution_time
            ) VALUES (
                :task_type, :model_name, :start_time, :end_time, :status,
                :metrics_collected, :alerts_generated, :errors, :execution_time
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'task_type': task_type,
                    'model_name': model_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'status': status,
                    'metrics_collected': metrics_collected,
                    'alerts_generated': alerts_generated,
                    'errors': json.dumps(errors) if errors else None,
                    'execution_time': execution_time
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"记录监控任务日志失败: {e}")
    
    def get_active_alerts(self, model_name: str = None, severity: str = None) -> List[ModelAlert]:
        """获取活跃告警
        
        Args:
            model_name: 模型名称
            severity: 告警级别
            
        Returns:
            告警列表
        """
        try:
            where_clause = "WHERE status = 'active'"
            params = {}
            
            if model_name:
                where_clause += " AND model_name = :model_name"
                params['model_name'] = model_name
            
            if severity:
                where_clause += " AND severity = :severity"
                params['severity'] = severity
            
            query_sql = f"""
            SELECT *
            FROM model_alerts
            {where_clause}
            ORDER BY created_at DESC
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                rows = result.fetchall()
            
            alerts = []
            for row in rows:
                alert = ModelAlert(
                    alert_id=row[1],
                    model_name=row[2],
                    alert_type=row[3],
                    severity=row[4],
                    message=row[5],
                    details=json.loads(row[6]) if row[6] else {},
                    threshold_value=float(row[7]) if row[7] else 0,
                    actual_value=float(row[8]) if row[8] else 0,
                    status=row[9],
                    created_at=row[10],
                    resolved_at=row[11]
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"获取活跃告警失败: {e}")
            return []
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警
        
        Args:
            alert_id: 告警ID
            
        Returns:
            是否成功
        """
        try:
            update_sql = """
            UPDATE model_alerts
            SET status = 'resolved', resolved_at = CURRENT_TIMESTAMP
            WHERE alert_id = :alert_id
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(update_sql), {'alert_id': alert_id})
                conn.commit()
                
                return result.rowcount > 0
            
        except Exception as e:
            logger.error(f"解决告警失败: {e}")
            return False
    
    def _generate_monitoring_charts(self, model_name: str, performance_data: List[Dict],
                                  drift_data: List[Dict]) -> Dict[str, str]:
        """生成监控图表"""
        try:
            charts = {}
            
            # 性能趋势图
            if performance_data:
                charts['performance_trend'] = self._create_performance_trend_chart(performance_data)
            
            # 漂移热力图
            if drift_data:
                charts['drift_heatmap'] = self._create_drift_heatmap(drift_data)
            
            return charts
            
        except Exception as e:
            logger.error(f"生成监控图表失败: {e}")
            return {}
    
    def _create_performance_trend_chart(self, performance_data: List[Dict]) -> str:
        """创建性能趋势图"""
        try:
            df = pd.DataFrame(performance_data)
            df['date'] = pd.to_datetime(df['date'])
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('模型性能趋势', fontsize=16)
            
            # 准确率趋势
            axes[0, 0].plot(df['date'], df['accuracy'], marker='o')
            axes[0, 0].set_title('准确率趋势')
            axes[0, 0].set_ylabel('准确率')
            axes[0, 0].grid(True)
            
            # F1分数趋势
            axes[0, 1].plot(df['date'], df['f1_score'], marker='o', color='orange')
            axes[0, 1].set_title('F1分数趋势')
            axes[0, 1].set_ylabel('F1分数')
            axes[0, 1].grid(True)
            
            # 错误率趋势
            axes[1, 0].plot(df['date'], df['error_rate'], marker='o', color='red')
            axes[1, 0].set_title('错误率趋势')
            axes[1, 0].set_ylabel('错误率')
            axes[1, 0].grid(True)
            
            # 预测时间趋势
            axes[1, 1].plot(df['date'], df['avg_prediction_time'], marker='o', color='green')
            axes[1, 1].set_title('平均预测时间趋势')
            axes[1, 1].set_ylabel('预测时间(秒)')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 转换为base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
            
        except Exception as e:
            logger.error(f"创建性能趋势图失败: {e}")
            return ""
    
    def _create_drift_heatmap(self, drift_data: List[Dict]) -> str:
        """创建漂移热力图"""
        try:
            df = pd.DataFrame(drift_data)
            
            # 创建透视表
            pivot_table = df.pivot_table(
                values='drift_score',
                index='feature_name',
                columns='date',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.3f')
            plt.title('特征漂移热力图')
            plt.xlabel('日期')
            plt.ylabel('特征名称')
            plt.tight_layout()
            
            # 转换为base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
            
        except Exception as e:
            logger.error(f"创建漂移热力图失败: {e}")
            return ""
    
    def _calculate_summary_stats(self, performance_data: List[Dict],
                               drift_data: List[Dict], alert_data: List[Dict]) -> Dict[str, Any]:
        """计算汇总统计"""
        try:
            summary = {
                'performance_summary': {},
                'drift_summary': {},
                'alert_summary': {}
            }
            
            # 性能汇总
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                summary['performance_summary'] = {
                    'avg_accuracy': float(perf_df['accuracy'].mean()),
                    'avg_f1_score': float(perf_df['f1_score'].mean()),
                    'avg_error_rate': float(perf_df['error_rate'].mean()),
                    'total_predictions': int(perf_df['prediction_count'].sum()),
                    'performance_trend': 'improving' if perf_df['accuracy'].iloc[-1] > perf_df['accuracy'].iloc[0] else 'declining'
                }
            
            # 漂移汇总
            if drift_data:
                drift_df = pd.DataFrame(drift_data)
                summary['drift_summary'] = {
                    'total_features_monitored': len(drift_df['feature_name'].unique()),
                    'features_with_drift': len(drift_df[drift_df['is_significant']]['feature_name'].unique()),
                    'avg_drift_score': float(drift_df['drift_score'].mean()),
                    'max_drift_score': float(drift_df['drift_score'].max()),
                    'drift_types': drift_df['drift_type'].value_counts().to_dict()
                }
            
            # 告警汇总
            if alert_data:
                alert_df = pd.DataFrame(alert_data)
                summary['alert_summary'] = {
                    'total_alerts': len(alert_data),
                    'active_alerts': len(alert_df[alert_df['status'] == 'active']),
                    'alert_by_severity': alert_df['severity'].value_counts().to_dict(),
                    'alert_by_type': alert_df['alert_type'].value_counts().to_dict()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"计算汇总统计失败: {e}")
            return {}
    
    def _generate_recommendations(self, summary_stats: Dict[str, Any]) -> List[str]:
        """生成建议"""
        try:
            recommendations = []
            
            # 基于性能统计生成建议
            perf_summary = summary_stats.get('performance_summary', {})
            if perf_summary.get('avg_accuracy', 1.0) < 0.7:
                recommendations.append("模型准确率较低，建议检查训练数据质量或调整模型参数")
            
            if perf_summary.get('avg_error_rate', 0.0) > 0.3:
                recommendations.append("模型错误率较高，建议重新训练模型或增加训练数据")
            
            # 基于漂移统计生成建议
            drift_summary = summary_stats.get('drift_summary', {})
            if drift_summary.get('features_with_drift', 0) > 0:
                recommendations.append(f"检测到{drift_summary['features_with_drift']}个特征存在漂移，建议重新训练模型")
            
            # 基于告警统计生成建议
            alert_summary = summary_stats.get('alert_summary', {})
            if alert_summary.get('active_alerts', 0) > 5:
                recommendations.append("活跃告警数量较多，建议及时处理并优化模型")
            
            if not recommendations:
                recommendations.append("模型运行状态良好，继续保持监控")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            return ["无法生成建议"]
    
    def _get_performance_data(self, model_name: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """获取性能数据"""
        try:
            query_sql = """
            SELECT
                date,
                accuracy,
                precision_score AS precision,
                recall_score AS recall,
                f1_score,
                error_rate,
                prediction_count,
                avg_prediction_time
            FROM model_performance_metrics
            WHERE model_name = :model_name
            AND date BETWEEN :start_date AND :end_date
            ORDER BY date
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {
                    'model_name': model_name,
                    'start_date': start_date,
                    'end_date': end_date
                })
                rows = result.fetchall()
            
            data = []
            for row in rows:
                data.append({
                    'date': row[3].isoformat(),
                    'accuracy': float(row[4]) if row[4] else 0,
                    'precision': float(row[5]) if row[5] else 0,
                    'recall': float(row[6]) if row[6] else 0,
                    'f1_score': float(row[7]) if row[7] else 0,
                    'error_rate': float(row[11]) if row[11] else 0,
                    'prediction_count': row[9],
                    'avg_prediction_time': float(row[10]) if row[10] else 0
                })
            
            return data
            
        except Exception as e:
            logger.error(f"获取性能数据失败: {e}")
            return []
    
    def _get_drift_data(self, model_name: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """获取漂移数据"""
        try:
            query_sql = """
            SELECT
                feature_name,
                date,
                drift_score,
                drift_type,
                is_significant,
                p_value
            FROM data_drift_metrics
            WHERE model_name = :model_name
            AND date BETWEEN :start_date AND :end_date
            ORDER BY date, feature_name
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {
                    'model_name': model_name,
                    'start_date': start_date,
                    'end_date': end_date
                })
                rows = result.fetchall()
            
            data = []
            for row in rows:
                data.append({
                    'feature_name': row[2],
                    'date': row[3].isoformat(),
                    'drift_score': float(row[8]),
                    'drift_type': row[9],
                    'is_significant': row[10],
                    'p_value': float(row[11]) if row[11] else 0
                })
            
            return data
            
        except Exception as e:
            logger.error(f"获取漂移数据失败: {e}")
            return []
    
    def _get_alert_data(self, model_name: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """获取告警数据"""
        try:
            query_sql = """
            SELECT
                alert_id,
                model_name,
                alert_type,
                severity,
                message,
                status,
                created_at
            FROM model_alerts
            WHERE model_name = :model_name
            AND DATE(created_at) BETWEEN :start_date AND :end_date
            ORDER BY created_at DESC
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {
                    'model_name': model_name,
                    'start_date': start_date,
                    'end_date': end_date
                })
                rows = result.fetchall()
            
            data = []
            for row in rows:
                data.append({
                    'alert_id': row[1],
                    'alert_type': row[3],
                    'severity': row[4],
                    'message': row[5],
                    'status': row[9],
                    'created_at': row[10].isoformat() if row[10] else None
                })
            
            return data
            
        except Exception as e:
            logger.error(f"获取告警数据失败: {e}")
            return []
    
    def _check_performance_trigger(self, metrics: ModelPerformanceMetrics) -> Optional[RetrainingTrigger]:
        """检查性能触发条件"""
        try:
            # 检查是否有性能指标低于阈值
            performance_issues = []
            
            if metrics.accuracy < self.performance_thresholds['accuracy']:
                performance_issues.append(f"accuracy: {metrics.accuracy:.4f} < {self.performance_thresholds['accuracy']}")
            
            if metrics.f1_score < self.performance_thresholds['f1_score']:
                performance_issues.append(f"f1_score: {metrics.f1_score:.4f} < {self.performance_thresholds['f1_score']}")
            
            if metrics.error_rate > self.performance_thresholds['error_rate']:
                performance_issues.append(f"error_rate: {metrics.error_rate:.4f} > {self.performance_thresholds['error_rate']}")
            
            if performance_issues:
                trigger = RetrainingTrigger(
                    trigger_id=f"PERF_TRIGGER_{metrics.model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    model_name=metrics.model_name,
                    trigger_type='performance_threshold',
                    trigger_condition={
                        'issues': performance_issues,
                        'metrics': asdict(metrics)
                    },
                    is_triggered=True,
                    trigger_time=datetime.now(),
                    retraining_status='pending',
                    created_at=datetime.now()
                )
                return trigger
            
            return None
            
        except Exception as e:
            logger.error(f"检查性能触发条件失败: {e}")
            return None
    
    def _check_drift_trigger(self, model_name: str) -> Optional[RetrainingTrigger]:
        """检查漂移触发条件"""
        try:
            # 获取最近的漂移数据
            recent_drifts = self._get_recent_significant_drifts(model_name, days=1)
            
            if len(recent_drifts) >= 3:  # 如果有3个或以上特征发生显著漂移
                trigger = RetrainingTrigger(
                    trigger_id=f"DRIFT_TRIGGER_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    model_name=model_name,
                    trigger_type='data_drift',
                    trigger_condition={
                        'significant_drifts': len(recent_drifts),
                        'drift_features': [d['feature_name'] for d in recent_drifts]
                    },
                    is_triggered=True,
                    trigger_time=datetime.now(),
                    retraining_status='pending',
                    created_at=datetime.now()
                )
                return trigger
            
            return None
            
        except Exception as e:
            logger.error(f"检查漂移触发条件失败: {e}")
            return None
    
    def _check_scheduled_trigger(self, model_name: str) -> Optional[RetrainingTrigger]:
        """检查定时触发条件"""
        try:
            # 获取模型最后训练时间
            last_training_time = self._get_last_training_time(model_name)
            if not last_training_time:
                return None
            
            # 检查是否超过重训练周期（例如30天）
            days_since_training = (datetime.now() - last_training_time).days
            if days_since_training >= 30:
                trigger = RetrainingTrigger(
                    trigger_id=f"SCHEDULED_TRIGGER_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    model_name=model_name,
                    trigger_type='scheduled',
                    trigger_condition={
                        'days_since_training': days_since_training,
                        'last_training_time': last_training_time.isoformat(),
                        'retraining_cycle_days': 30
                    },
                    is_triggered=True,
                    trigger_time=datetime.now(),
                    retraining_status='pending',
                    created_at=datetime.now()
                )
                return trigger
            
            return None
            
        except Exception as e:
            logger.error(f"检查定时触发条件失败: {e}")
            return None
    
    def _save_retraining_trigger(self, trigger: RetrainingTrigger):
        """保存重训练触发器"""
        try:
            insert_sql = """
            INSERT INTO retraining_triggers (
                trigger_id, model_name, trigger_type, trigger_condition,
                is_triggered, trigger_time, retraining_status
            ) VALUES (
                :trigger_id, :model_name, :trigger_type, :trigger_condition,
                :is_triggered, :trigger_time, :retraining_status
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'trigger_id': trigger.trigger_id,
                    'model_name': trigger.model_name,
                    'trigger_type': trigger.trigger_type,
                    'trigger_condition': json.dumps(trigger.trigger_condition),
                    'is_triggered': trigger.is_triggered,
                    'trigger_time': trigger.trigger_time,
                    'retraining_status': trigger.retraining_status
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"保存重训练触发器失败: {e}")
            raise
    
    def _is_auto_retrain_enabled(self, model_name: str) -> bool:
        """检查是否启用自动重训练"""
        try:
            query_sql = """
            SELECT auto_retrain_enabled
            FROM model_monitor_configs
            WHERE model_name = :model_name
            AND is_active = true
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'model_name': model_name})
                row = result.fetchone()
            
            return row[0] if row else False
            
        except Exception as e:
            logger.error(f"检查自动重训练配置失败: {e}")
            return False
    
    def _start_retraining_job(self, trigger: RetrainingTrigger):
        """启动重训练任务"""
        try:
            # 这里应该启动实际的重训练任务
            # 可以通过消息队列、任务调度器等方式实现
            job_id = f"RETRAIN_JOB_{trigger.model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 更新触发器状态
            update_sql = """
            UPDATE retraining_triggers
            SET retraining_status = 'running', retraining_job_id = :job_id
            WHERE trigger_id = :trigger_id
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(update_sql), {
                    'job_id': job_id,
                    'trigger_id': trigger.trigger_id
                })
                conn.commit()
            
            logger.info(f"启动重训练任务: {job_id}")
            
        except Exception as e:
            logger.error(f"启动重训练任务失败: {e}")
    
    def _get_latest_performance_metrics(self, model_name: str) -> Optional[ModelPerformanceMetrics]:
        """获取最新性能指标"""
        try:
            query_sql = """
            SELECT *
            FROM model_performance_metrics
            WHERE model_name = :model_name
            ORDER BY date DESC, created_at DESC
            LIMIT 1
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'model_name': model_name})
                row = result.fetchone()
            
            if row:
                return ModelPerformanceMetrics(
                    model_name=row[1],
                    model_version=row[2],
                    date=row[3],
                    accuracy=float(row[4]) if row[4] else 0,
                    precision=float(row[5]) if row[5] else 0,
                    recall=float(row[6]) if row[6] else 0,
                    f1_score=float(row[7]) if row[7] else 0,
                    auc_score=float(row[8]) if row[8] else None,
                    prediction_count=row[9],
                    avg_prediction_time=float(row[10]) if row[10] else 0,
                    error_rate=float(row[11]) if row[11] else 0,
                    confidence_distribution=json.loads(row[12]) if row[12] else {},
                    feature_importance_drift=float(row[13]) if row[13] else None,
                    created_at=row[14]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取最新性能指标失败: {e}")
            return None
    
    def _get_recent_significant_drifts(self, model_name: str, days: int = 1) -> List[Dict]:
        """获取最近的显著漂移"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            query_sql = """
            SELECT feature_name, drift_score, drift_type
            FROM data_drift_metrics
            WHERE model_name = :model_name
            AND date >= :start_date
            AND date <= :end_date
            AND is_significant = true
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {
                    'model_name': model_name,
                    'start_date': start_date,
                    'end_date': end_date
                })
                rows = result.fetchall()
            
            drifts = []
            for row in rows:
                drifts.append({
                    'feature_name': row[0],
                    'drift_score': float(row[1]),
                    'drift_type': row[2]
                })
            
            return drifts
            
        except Exception as e:
            logger.error(f"获取最近显著漂移失败: {e}")
            return []
    
    def _get_last_training_time(self, model_name: str) -> Optional[datetime]:
        """获取最后训练时间"""
        try:
            query_sql = """
            SELECT created_at
            FROM ai_models
            WHERE model_name = :model_name
            ORDER BY created_at DESC
            LIMIT 1
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'model_name': model_name})
                row = result.fetchone()
            
            return row[0] if row else None
            
        except Exception as e:
            logger.error(f"获取最后训练时间失败: {e}")
            return None
    
    def get_monitoring_dashboard_data(self, model_name: str = None) -> Dict[str, Any]:
        """获取监控仪表板数据
        
        Args:
            model_name: 模型名称，如果为None则获取所有模型的数据
            
        Returns:
            仪表板数据
        """
        try:
            dashboard_data = {
                'overview': {},
                'models': [],
                'alerts': [],
                'performance_trends': {},
                'drift_summary': {},
                'system_health': {}
            }
            
            # 获取概览数据
            dashboard_data['overview'] = self._get_overview_data(model_name)
            
            # 获取模型列表
            dashboard_data['models'] = self._get_models_summary(model_name)
            
            # 获取活跃告警
            dashboard_data['alerts'] = [asdict(alert) for alert in self.get_active_alerts(model_name)]
            
            # 获取性能趋势
            dashboard_data['performance_trends'] = self._get_performance_trends(model_name)
            
            # 获取漂移汇总
            dashboard_data['drift_summary'] = self._get_drift_summary(model_name)
            
            # 获取系统健康状态
            dashboard_data['system_health'] = self._get_system_health()
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"获取监控仪表板数据失败: {e}")
            return {}
    
    def _get_overview_data(self, model_name: str = None) -> Dict[str, Any]:
        """获取概览数据"""
        try:
            where_clause = ""
            params = {}
            
            if model_name:
                where_clause = "WHERE model_name = :model_name"
                params['model_name'] = model_name
            
            # 获取模型数量
            models_query = f"""
            SELECT COUNT(DISTINCT model_name)
            FROM model_performance_metrics
            {where_clause}
            """
            
            # 获取活跃告警数量
            alerts_query = f"""
            SELECT COUNT(*)
            FROM model_alerts
            WHERE status = 'active'
            {"AND model_name = :model_name" if model_name else ""}
            """
            
            # 获取今日预测数量
            predictions_query = f"""
            SELECT COALESCE(SUM(prediction_count), 0)
            FROM model_performance_metrics
            WHERE date = CURRENT_DATE
            {"AND model_name = :model_name" if model_name else ""}
            """
            
            with self.engine.connect() as conn:
                models_count = conn.execute(text(models_query), params).scalar()
                alerts_count = conn.execute(text(alerts_query), params).scalar()
                predictions_count = conn.execute(text(predictions_query), params).scalar()
            
            return {
                'total_models': models_count or 0,
                'active_alerts': alerts_count or 0,
                'daily_predictions': predictions_count or 0,
                'monitoring_status': 'healthy' if alerts_count == 0 else 'warning' if alerts_count < 5 else 'critical'
            }
            
        except Exception as e:
            logger.error(f"获取概览数据失败: {e}")
            return {}
    
    def _get_models_summary(self, model_name: str = None) -> List[Dict[str, Any]]:
        """获取模型汇总"""
        try:
            where_clause = ""
            params = {}
            
            if model_name:
                where_clause = "WHERE model_name = :model_name"
                params['model_name'] = model_name
            
            query_sql = f"""
            SELECT 
                model_name,
                COUNT(*) as metrics_count,
                AVG(accuracy) as avg_accuracy,
                AVG(f1_score) as avg_f1_score,
                AVG(error_rate) as avg_error_rate,
                MAX(date) as last_monitored
            FROM model_performance_metrics
            {where_clause}
            GROUP BY model_name
            ORDER BY last_monitored DESC
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                rows = result.fetchall()
            
            models = []
            for row in rows:
                # 获取该模型的活跃告警数量
                alert_query = """
                SELECT COUNT(*)
                FROM model_alerts
                WHERE model_name = :model_name AND status = 'active'
                """
                
                alert_count = conn.execute(text(alert_query), {'model_name': row[0]}).scalar()
                
                models.append({
                    'model_name': row[0],
                    'metrics_count': row[1],
                    'avg_accuracy': float(row[2]) if row[2] else 0,
                    'avg_f1_score': float(row[3]) if row[3] else 0,
                    'avg_error_rate': float(row[4]) if row[4] else 0,
                    'last_monitored': row[5].isoformat() if row[5] else None,
                    'active_alerts': alert_count or 0,
                    'health_status': 'healthy' if alert_count == 0 else 'warning' if alert_count < 3 else 'critical'
                })
            
            return models
            
        except Exception as e:
            logger.error(f"获取模型汇总失败: {e}")
            return []
    
    def _get_performance_trends(self, model_name: str = None) -> Dict[str, Any]:
        """获取性能趋势"""
        try:
            where_clause = ""
            params = {}
            
            if model_name:
                where_clause = "WHERE model_name = :model_name"
                params['model_name'] = model_name
            
            # 获取最近7天的性能数据
            query_sql = f"""
            SELECT 
                date,
                AVG(accuracy) as avg_accuracy,
                AVG(f1_score) as avg_f1_score,
                AVG(error_rate) as avg_error_rate
            FROM model_performance_metrics
            {where_clause}
            AND date >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY date
            ORDER BY date
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                rows = result.fetchall()
            
            trends = {
                'dates': [],
                'accuracy': [],
                'f1_score': [],
                'error_rate': []
            }
            
            for row in rows:
                trends['dates'].append(row[0].isoformat())
                trends['accuracy'].append(float(row[1]) if row[1] else 0)
                trends['f1_score'].append(float(row[2]) if row[2] else 0)
                trends['error_rate'].append(float(row[3]) if row[3] else 0)
            
            return trends
            
        except Exception as e:
            logger.error(f"获取性能趋势失败: {e}")
            return {}
    
    def _get_drift_summary(self, model_name: str = None) -> Dict[str, Any]:
        """获取漂移汇总"""
        try:
            where_clause = ""
            params = {}
            
            if model_name:
                where_clause = "WHERE model_name = :model_name"
                params['model_name'] = model_name
            
            # 获取最近7天的漂移数据
            query_sql = f"""
            SELECT 
                COUNT(*) as total_checks,
                COUNT(CASE WHEN is_significant THEN 1 END) as significant_drifts,
                AVG(drift_score) as avg_drift_score,
                MAX(drift_score) as max_drift_score
            FROM data_drift_metrics
            {where_clause}
            AND date >= CURRENT_DATE - INTERVAL '7 days'
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                row = result.fetchone()
            
            if row:
                return {
                    'total_checks': row[0] or 0,
                    'significant_drifts': row[1] or 0,
                    'avg_drift_score': float(row[2]) if row[2] else 0,
                    'max_drift_score': float(row[3]) if row[3] else 0,
                    'drift_rate': (row[1] or 0) / (row[0] or 1) * 100
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"获取漂移汇总失败: {e}")
            return {}
    
    def _get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        try:
            # 获取最近的监控任务执行情况
            query_sql = """
            SELECT 
                task_type,
                COUNT(*) as total_tasks,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_tasks,
                AVG(execution_time) as avg_execution_time
            FROM monitor_task_logs
            WHERE start_time >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
            GROUP BY task_type
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql))
                rows = result.fetchall()
            
            task_health = {}
            overall_success_rate = 0
            total_tasks = 0
            
            for row in rows:
                task_type = row[0]
                total = row[1]
                successful = row[2]
                avg_time = float(row[3]) if row[3] else 0
                
                success_rate = (successful / total) * 100 if total > 0 else 0
                
                task_health[task_type] = {
                    'total_tasks': total,
                    'successful_tasks': successful,
                    'success_rate': success_rate,
                    'avg_execution_time': avg_time
                }
                
                overall_success_rate += successful
                total_tasks += total
            
            overall_success_rate = (overall_success_rate / total_tasks) * 100 if total_tasks > 0 else 100
            
            return {
                'overall_success_rate': overall_success_rate,
                'total_tasks_24h': total_tasks,
                'task_health': task_health,
                'system_status': 'healthy' if overall_success_rate >= 95 else 'warning' if overall_success_rate >= 80 else 'critical'
            }
            
        except Exception as e:
            logger.error(f"获取系统健康状态失败: {e}")
            return {}
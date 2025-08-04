"""AI预测服务

提供实时预测API，支持缓存、批量处理和模型热更新。
"""

import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import asyncio

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import redis
from sklearn.preprocessing import StandardScaler

from ...utils.db import get_db_manager
from .model_manager import AIModelManager
from .factor_weight_engine import FactorWeightEngine
from .stock_scoring_engine import StockScoringEngine

logger = logging.getLogger(__name__)

@dataclass
class PredictionRequest:
    """预测请求"""
    stock_codes: List[str]
    model_name: str
    model_version: Optional[str] = None
    prediction_date: Optional[str] = None
    return_scores: bool = True
    return_factors: bool = False
    cache_ttl: int = 300  # 缓存时间（秒）

@dataclass
class PredictionResult:
    """预测结果"""
    stock_code: str
    prediction_value: float
    confidence_score: float
    score: Optional[float] = None
    rank: Optional[int] = None
    factor_contributions: Optional[Dict[str, float]] = None
    prediction_date: Optional[str] = None
    model_info: Optional[Dict[str, str]] = None

@dataclass
class BatchPredictionResult:
    """批量预测结果"""
    request_id: str
    predictions: List[PredictionResult]
    model_info: Dict[str, str]
    prediction_date: str
    processing_time: float
    cache_hit_ratio: float
    status: str
    error_message: Optional[str] = None

class PredictionService:
    """AI预测服务
    
    主要功能：
    - 实时股票预测
    - 批量预测处理
    - 结果缓存管理
    - 模型热更新
    - 预测性能监控
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, redis_db: int = 0):
        self.engine = get_db_manager().engine
        self.model_manager = AIModelManager()
        self.factor_weight_engine = FactorWeightEngine()
        self.stock_scoring_engine = StockScoringEngine()
        
        # Redis缓存
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                db=redis_db,
                decode_responses=True
            )
            self.redis_client.ping()
            self.cache_enabled = True
            logger.info("Redis缓存连接成功")
        except Exception as e:
            logger.warning(f"Redis连接失败，禁用缓存: {e}")
            self.redis_client = None
            self.cache_enabled = False
        
        # 模型缓存
        self.model_cache = {}
        self.cache_ttl = 3600  # 模型缓存1小时
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """确保预测服务相关表存在"""
        create_tables_sql = """
        -- 预测请求日志表
        CREATE TABLE IF NOT EXISTS prediction_requests (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(100) UNIQUE NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20),
            stock_codes TEXT[],
            prediction_date DATE,
            request_params JSONB,
            processing_time DECIMAL(10, 4),
            cache_hit_ratio DECIMAL(5, 4),
            status VARCHAR(20),
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 预测结果表
        CREATE TABLE IF NOT EXISTS prediction_results (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(100) NOT NULL,
            stock_code VARCHAR(20) NOT NULL,
            prediction_value DECIMAL(15, 6),
            confidence_score DECIMAL(5, 4),
            score DECIMAL(10, 4),
            rank_value INTEGER,
            factor_contributions JSONB,
            prediction_date DATE,
            model_name VARCHAR(100),
            model_version VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 预测性能监控表
        CREATE TABLE IF NOT EXISTS prediction_performance (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20),
            date DATE NOT NULL,
            total_requests INTEGER DEFAULT 0,
            avg_processing_time DECIMAL(10, 4),
            cache_hit_ratio DECIMAL(5, 4),
            error_rate DECIMAL(5, 4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model_name, model_version, date)
        );
        
        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_prediction_requests_model ON prediction_requests(model_name, model_version);
        CREATE INDEX IF NOT EXISTS idx_prediction_requests_date ON prediction_requests(prediction_date);
        CREATE INDEX IF NOT EXISTS idx_prediction_results_request ON prediction_results(request_id);
        CREATE INDEX IF NOT EXISTS idx_prediction_results_stock ON prediction_results(stock_code, prediction_date);
        CREATE INDEX IF NOT EXISTS idx_prediction_performance_model ON prediction_performance(model_name, date);
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
            logger.info("预测服务表创建成功")
        except Exception as e:
            logger.error(f"创建预测服务表失败: {e}")
            raise
    
    def _get_cache_key(self, stock_code: str, model_name: str, model_version: str, prediction_date: str) -> str:
        """生成缓存键"""
        return f"prediction:{model_name}:{model_version}:{stock_code}:{prediction_date}"
    
    def _get_model_cache_key(self, model_name: str, model_version: str) -> str:
        """生成模型缓存键"""
        return f"model:{model_name}:{model_version}"
    
    def _load_model(self, model_name: str, model_version: str = None) -> Optional[Dict[str, Any]]:
        """加载模型（带缓存）"""
        try:
            # 如果没有指定版本，使用活跃版本
            if model_version is None:
                active_model = self.model_manager.get_active_model(model_name)
                if not active_model:
                    logger.error(f"没有找到活跃模型: {model_name}")
                    return None
                model_version = active_model.version
            
            cache_key = self._get_model_cache_key(model_name, model_version)
            
            # 检查内存缓存
            if cache_key in self.model_cache:
                cache_time, model_data = self.model_cache[cache_key]
                if datetime.now().timestamp() - cache_time < self.cache_ttl:
                    return model_data
                else:
                    # 缓存过期，删除
                    del self.model_cache[cache_key]
            
            # 从磁盘加载模型
            model_data = self.model_manager.load_model_for_prediction(model_name, model_version)
            
            if model_data:
                # 缓存模型
                self.model_cache[cache_key] = (datetime.now().timestamp(), model_data)
                logger.info(f"模型加载成功: {model_name} v{model_version}")
                return model_data
            else:
                logger.error(f"模型加载失败: {model_name} v{model_version}")
                return None
                
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None
    
    def _get_factor_data(self, stock_codes: List[str], prediction_date: str) -> pd.DataFrame:
        """获取因子数据"""
        try:
            # 构建查询SQL
            stock_list_str = ','.join([f"'{code}'" for code in stock_codes])
            
            query_sql = f"""
            SELECT f.*, s.stock_name, s.industry_code, s.industry_name
            FROM factor_data f
            LEFT JOIN stock_info s ON f.stock_code = s.stock_code
            WHERE f.stock_code IN ({stock_list_str})
            AND f.date = %s
            AND s.is_active = true
            ORDER BY f.stock_code
            """
            
            df = pd.read_sql(query_sql, self.engine, params=[prediction_date])
            
            if df.empty:
                logger.warning(f"没有找到因子数据: {prediction_date}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"获取因子数据失败: {e}")
            return pd.DataFrame()
    
    def _predict_single_stock(self, 
                            stock_code: str,
                            model_data: Dict[str, Any],
                            factor_data: pd.DataFrame,
                            prediction_date: str) -> Optional[PredictionResult]:
        """预测单只股票"""
        try:
            # 检查缓存
            if self.cache_enabled:
                cache_key = self._get_cache_key(
                    stock_code, 
                    model_data['version_info'].model_name,
                    model_data['version_info'].version,
                    prediction_date
                )
                
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    result_dict = json.loads(cached_result)
                    return PredictionResult(**result_dict)
            
            # 获取股票的因子数据
            stock_data = factor_data[factor_data['stock_code'] == stock_code]
            if stock_data.empty:
                logger.warning(f"没有找到股票因子数据: {stock_code}")
                return None
            
            # 准备特征数据
            feature_columns = model_data['feature_columns']
            X = stock_data[feature_columns].values
            
            # 检查缺失值
            if pd.isna(X).any():
                # 使用训练时的均值填充
                for i, col in enumerate(feature_columns):
                    if pd.isna(X[0, i]):
                        X[0, i] = 0  # 或使用训练时的均值
            
            # 标准化（如果模型有标准化器）
            if 'scaler' in model_data and model_data['scaler'] is not None:
                X = model_data['scaler'].transform(X)
            
            # 预测
            model = model_data['model']
            prediction = model.predict(X)[0]
            
            # 计算置信度（如果模型支持）
            confidence = 0.8  # 默认置信度
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)[0]
                    confidence = float(np.max(proba))
                except:
                    pass
            elif hasattr(model, 'decision_function'):
                try:
                    decision = model.decision_function(X)[0]
                    confidence = float(1 / (1 + np.exp(-abs(decision))))  # sigmoid
                except:
                    pass
            
            # 计算因子贡献（如果需要）
            factor_contributions = None
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    factor_contributions = dict(zip(feature_columns, importances.tolist()))
            except:
                pass
            
            result = PredictionResult(
                stock_code=stock_code,
                prediction_value=float(prediction),
                confidence_score=confidence,
                factor_contributions=factor_contributions,
                prediction_date=prediction_date,
                model_info={
                    'model_name': model_data['version_info'].model_name,
                    'model_version': model_data['version_info'].version,
                    'model_type': model_data['version_info'].model_type
                }
            )
            
            # 缓存结果
            if self.cache_enabled:
                try:
                    self.redis_client.setex(
                        cache_key,
                        300,  # 5分钟缓存
                        json.dumps(asdict(result), default=str)
                    )
                except Exception as e:
                    logger.warning(f"缓存预测结果失败: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"预测股票失败 {stock_code}: {e}")
            return None
    
    def predict(self, request: PredictionRequest) -> BatchPredictionResult:
        """执行预测
        
        Args:
            request: 预测请求
            
        Returns:
            批量预测结果
        """
        start_time = datetime.now()
        request_id = f"pred_{start_time.strftime('%Y%m%d_%H%M%S')}_{hash(str(request.stock_codes)) % 10000}"
        
        try:
            # 参数验证
            if not request.stock_codes:
                return BatchPredictionResult(
                    request_id=request_id,
                    predictions=[],
                    model_info={},
                    prediction_date=request.prediction_date or datetime.now().strftime('%Y-%m-%d'),
                    processing_time=0,
                    cache_hit_ratio=0,
                    status='error',
                    error_message='股票代码列表为空'
                )
            
            # 设置预测日期
            if request.prediction_date is None:
                prediction_date = datetime.now().strftime('%Y-%m-%d')
            else:
                prediction_date = request.prediction_date
            
            # 加载模型
            model_data = self._load_model(request.model_name, request.model_version)
            if not model_data:
                return BatchPredictionResult(
                    request_id=request_id,
                    predictions=[],
                    model_info={},
                    prediction_date=prediction_date,
                    processing_time=0,
                    cache_hit_ratio=0,
                    status='error',
                    error_message=f'模型加载失败: {request.model_name}'
                )
            
            model_info = {
                'model_name': model_data['version_info'].model_name,
                'model_version': model_data['version_info'].version,
                'model_type': model_data['version_info'].model_type
            }
            
            # 获取因子数据
            factor_data = self._get_factor_data(request.stock_codes, prediction_date)
            if factor_data.empty:
                return BatchPredictionResult(
                    request_id=request_id,
                    predictions=[],
                    model_info=model_info,
                    prediction_date=prediction_date,
                    processing_time=0,
                    cache_hit_ratio=0,
                    status='error',
                    error_message=f'没有找到因子数据: {prediction_date}'
                )
            
            # 执行预测
            predictions = []
            cache_hits = 0
            
            for stock_code in request.stock_codes:
                # 检查缓存
                cache_hit = False
                if self.cache_enabled:
                    cache_key = self._get_cache_key(
                        stock_code, model_info['model_name'], 
                        model_info['model_version'], prediction_date
                    )
                    cached_result = self.redis_client.get(cache_key)
                    if cached_result:
                        try:
                            result_dict = json.loads(cached_result)
                            prediction_result = PredictionResult(**result_dict)
                            predictions.append(prediction_result)
                            cache_hits += 1
                            cache_hit = True
                        except Exception as e:
                            logger.warning(f"解析缓存结果失败: {e}")
                
                if not cache_hit:
                    # 执行预测
                    prediction_result = self._predict_single_stock(
                        stock_code, model_data, factor_data, prediction_date
                    )
                    if prediction_result:
                        predictions.append(prediction_result)
            
            # 计算评分和排名（如果需要）
            if request.return_scores and predictions:
                predictions = self._add_scores_and_rankings(
                    predictions, model_info['model_name'], 
                    model_info['model_version'], prediction_date
                )
            
            # 计算处理时间和缓存命中率
            processing_time = (datetime.now() - start_time).total_seconds()
            cache_hit_ratio = cache_hits / len(request.stock_codes) if request.stock_codes else 0
            
            result = BatchPredictionResult(
                request_id=request_id,
                predictions=predictions,
                model_info=model_info,
                prediction_date=prediction_date,
                processing_time=processing_time,
                cache_hit_ratio=cache_hit_ratio,
                status='success'
            )
            
            # 记录请求日志
            self._log_prediction_request(request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"预测执行失败: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return BatchPredictionResult(
                request_id=request_id,
                predictions=[],
                model_info={},
                prediction_date=request.prediction_date or datetime.now().strftime('%Y-%m-%d'),
                processing_time=processing_time,
                cache_hit_ratio=0,
                status='error',
                error_message=str(e)
            )
    
    def _add_scores_and_rankings(self, 
                               predictions: List[PredictionResult],
                               model_name: str,
                               model_version: str,
                               prediction_date: str) -> List[PredictionResult]:
        """添加评分和排名"""
        try:
            # 获取股票评分
            stock_codes = [p.stock_code for p in predictions]
            stock_scores = self.stock_scoring_engine.get_stock_scores(
                model_name, model_version, prediction_date
            )
            
            # 创建评分字典
            score_dict = {s.stock_code: s for s in stock_scores}
            
            # 更新预测结果
            for prediction in predictions:
                if prediction.stock_code in score_dict:
                    score_info = score_dict[prediction.stock_code]
                    prediction.score = score_info.score
                    prediction.rank = score_info.rank
            
            return predictions
            
        except Exception as e:
            logger.error(f"添加评分和排名失败: {e}")
            return predictions
    
    def _log_prediction_request(self, request: PredictionRequest, result: BatchPredictionResult):
        """记录预测请求日志"""
        try:
            # 记录请求日志
            insert_request_sql = """
            INSERT INTO prediction_requests (
                request_id, model_name, model_version, stock_codes,
                prediction_date, request_params, processing_time,
                cache_hit_ratio, status, error_message
            ) VALUES (
                :request_id, :model_name, :model_version, :stock_codes,
                :prediction_date, :request_params, :processing_time,
                :cache_hit_ratio, :status, :error_message
            )
            """
            
            # 记录预测结果
            insert_result_sql = """
            INSERT INTO prediction_results (
                request_id, stock_code, prediction_value, confidence_score,
                score, rank_value, factor_contributions, prediction_date,
                model_name, model_version
            ) VALUES (
                :request_id, :stock_code, :prediction_value, :confidence_score,
                :score, :rank_value, :factor_contributions, :prediction_date,
                :model_name, :model_version
            )
            """
            
            with self.engine.connect() as conn:
                # 记录请求
                conn.execute(text(insert_request_sql), {
                    'request_id': result.request_id,
                    'model_name': result.model_info.get('model_name', ''),
                    'model_version': result.model_info.get('model_version', ''),
                    'stock_codes': request.stock_codes,
                    'prediction_date': result.prediction_date,
                    'request_params': json.dumps(asdict(request), default=str),
                    'processing_time': result.processing_time,
                    'cache_hit_ratio': result.cache_hit_ratio,
                    'status': result.status,
                    'error_message': result.error_message
                })
                
                # 记录结果
                for prediction in result.predictions:
                    conn.execute(text(insert_result_sql), {
                        'request_id': result.request_id,
                        'stock_code': prediction.stock_code,
                        'prediction_value': prediction.prediction_value,
                        'confidence_score': prediction.confidence_score,
                        'score': prediction.score,
                        'rank_value': prediction.rank,
                        'factor_contributions': json.dumps(prediction.factor_contributions or {}),
                        'prediction_date': result.prediction_date,
                        'model_name': result.model_info.get('model_name', ''),
                        'model_version': result.model_info.get('model_version', '')
                    })
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"记录预测请求日志失败: {e}")
    
    def clear_cache(self, pattern: str = "prediction:*") -> int:
        """清理缓存
        
        Args:
            pattern: 缓存键模式
            
        Returns:
            清理的缓存数量
        """
        if not self.cache_enabled:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"清理缓存: {deleted}个键")
                return deleted
            return 0
            
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
            return 0
    
    def get_prediction_history(self, 
                             stock_code: str,
                             model_name: str,
                             days: int = 30) -> pd.DataFrame:
        """获取预测历史
        
        Args:
            stock_code: 股票代码
            model_name: 模型名称
            days: 历史天数
            
        Returns:
            预测历史DataFrame
        """
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            query_sql = """
            SELECT prediction_date, prediction_value, confidence_score,
                   score, rank_value, model_version
            FROM prediction_results
            WHERE stock_code = :stock_code
            AND model_name = :model_name
            AND prediction_date >= :start_date
            ORDER BY prediction_date DESC
            """
            
            df = pd.read_sql(query_sql, self.engine, params={
                'stock_code': stock_code,
                'model_name': model_name,
                'start_date': start_date
            })
            
            return df
            
        except Exception as e:
            logger.error(f"获取预测历史失败: {e}")
            return pd.DataFrame()
    
    def get_performance_stats(self, 
                            model_name: str,
                            days: int = 7) -> Dict[str, Any]:
        """获取预测性能统计
        
        Args:
            model_name: 模型名称
            days: 统计天数
            
        Returns:
            性能统计字典
        """
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            query_sql = """
            SELECT 
                COUNT(*) as total_requests,
                AVG(processing_time) as avg_processing_time,
                AVG(cache_hit_ratio) as avg_cache_hit_ratio,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as error_rate
            FROM prediction_requests
            WHERE model_name = :model_name
            AND DATE(created_at) >= :start_date
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {
                    'model_name': model_name,
                    'start_date': start_date
                })
                row = result.fetchone()
            
            if row:
                return {
                    'total_requests': int(row[0]),
                    'avg_processing_time': float(row[1] or 0),
                    'avg_cache_hit_ratio': float(row[2] or 0),
                    'error_rate': float(row[3] or 0),
                    'period_days': days
                }
            else:
                return {
                    'total_requests': 0,
                    'avg_processing_time': 0,
                    'avg_cache_hit_ratio': 0,
                    'error_rate': 0,
                    'period_days': days
                }
                
        except Exception as e:
            logger.error(f"获取性能统计失败: {e}")
            return {}
    
    def reload_model(self, model_name: str, model_version: str = None) -> bool:
        """重新加载模型（热更新）
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            
        Returns:
            是否重新加载成功
        """
        try:
            # 清除模型缓存
            if model_version is None:
                # 清除该模型的所有版本缓存
                keys_to_remove = [k for k in self.model_cache.keys() if k.startswith(f"model:{model_name}:")]
            else:
                # 清除指定版本缓存
                cache_key = self._get_model_cache_key(model_name, model_version)
                keys_to_remove = [cache_key] if cache_key in self.model_cache else []
            
            for key in keys_to_remove:
                del self.model_cache[key]
            
            # 清除Redis缓存
            if self.cache_enabled:
                if model_version is None:
                    pattern = f"prediction:{model_name}:*"
                else:
                    pattern = f"prediction:{model_name}:{model_version}:*"
                self.clear_cache(pattern)
            
            logger.info(f"模型缓存清除成功: {model_name} v{model_version or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"重新加载模型失败: {e}")
            return False
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI模型训练流水线

实现模型训练、验证、保存等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sqlalchemy import create_engine, text
import logging
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path
import json
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from src.utils.config_loader import config
from src.utils.db import get_db_engine
from src.compute.processing import FactorProcessor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """AI模型训练流水线
    
    提供完整的模型训练、验证、保存功能
    """
    
    def __init__(self):
        """初始化训练流水线"""
        self.engine = get_db_engine()
        self.factor_processor = FactorProcessor()
        self.training_config = config.get('model_training', {})
        self.models_dir = Path(config.get('paths.models_dir', 'models'))
        self.models_dir.mkdir(exist_ok=True)
        
        # 支持的模型类型
        self.model_classes = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'xgboost': xgb.XGBRegressor,
            'lightgbm': lgb.LGBMRegressor
        }
        
        logger.info("AI模型训练流水线初始化完成")
    
    def prepare_training_data(self, 
                            factor_names: List[str],
                            start_date: str,
                            end_date: str,
                            target_period: int = 5,
                            min_samples: int = 100,
                            stock_list: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """准备训练数据
        
        Args:
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            target_period: 目标收益率计算周期（天）
            min_samples: 最小样本数
            stock_list: 股票列表
            
        Returns:
            特征数据和目标数据
        """
        logger.info(f"准备训练数据: {start_date} 到 {end_date}")
        
        # 1. 加载因子数据
        factor_df = self.factor_processor.load_factor_data(
            factor_names, start_date, end_date, stock_list
        )
        
        if factor_df.empty:
            logger.error("未找到因子数据")
            return pd.DataFrame(), pd.DataFrame()
        
        # 2. 加载价格数据计算目标收益率
        target_df = self._calculate_target_returns(
            factor_df['ts_code'].unique().tolist(),
            start_date,
            end_date,
            target_period
        )
        
        # 3. 合并数据
        training_data = factor_df.merge(
            target_df, 
            on=['ts_code', 'trade_date'], 
            how='inner'
        )
        
        # 4. 数据清洗
        training_data = self._clean_training_data(training_data, factor_names)
        
        if len(training_data) < min_samples:
            logger.warning(f"训练样本数量不足: {len(training_data)} < {min_samples}")
            return pd.DataFrame(), pd.DataFrame()
        
        # 5. 分离特征和目标
        feature_columns = factor_names
        features = training_data[['ts_code', 'trade_date'] + feature_columns]
        targets = training_data[['ts_code', 'trade_date', 'target_return']]
        
        logger.info(f"训练数据准备完成: {len(training_data)} 个样本, {len(feature_columns)} 个特征")
        return features, targets
    
    def _calculate_target_returns(self, 
                                stock_list: List[str],
                                start_date: str,
                                end_date: str,
                                target_period: int) -> pd.DataFrame:
        """计算目标收益率
        
        Args:
            stock_list: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            target_period: 目标周期
            
        Returns:
            目标收益率数据
        """
        logger.info(f"计算 {target_period} 日目标收益率")
        
        # 扩展结束日期以获取足够的未来数据
        extended_end_date = (pd.to_datetime(end_date) + timedelta(days=target_period + 10)).strftime('%Y-%m-%d')
        
        query = """
            SELECT ts_code, trade_date, close, 
                   LEAD(close, :target_period) OVER (PARTITION BY ts_code ORDER BY trade_date) as future_close
            FROM stock_daily
            WHERE ts_code = ANY(:stock_list)
            AND trade_date BETWEEN :start_date AND :extended_end_date
            ORDER BY ts_code, trade_date
        """
        
        params = {
            'stock_list': stock_list,
            'start_date': start_date,
            'extended_end_date': extended_end_date,
            'target_period': target_period
        }
        
        try:
            with self.engine.connect() as conn:
                price_df = pd.read_sql_query(text(query), conn, params=params)
            
            # 计算收益率
            price_df['target_return'] = (price_df['future_close'] - price_df['close']) / price_df['close']
            
            # 过滤有效数据
            target_df = price_df[['ts_code', 'trade_date', 'target_return']].dropna()
            
            # 只保留原始日期范围内的数据
            target_df = target_df[
                (target_df['trade_date'] >= start_date) & 
                (target_df['trade_date'] <= end_date)
            ]
            
            logger.info(f"目标收益率计算完成: {len(target_df)} 个样本")
            return target_df
            
        except Exception as e:
            logger.error(f"计算目标收益率失败: {str(e)}")
            return pd.DataFrame()
    
    def _clean_training_data(self, 
                           training_data: pd.DataFrame,
                           factor_names: List[str]) -> pd.DataFrame:
        """清洗训练数据
        
        Args:
            training_data: 原始训练数据
            factor_names: 因子名称列表
            
        Returns:
            清洗后的训练数据
        """
        logger.info("清洗训练数据")
        
        df = training_data.copy()
        original_count = len(df)
        
        # 1. 删除目标变量为空的样本
        df = df.dropna(subset=['target_return'])
        
        # 2. 删除目标变量异常值（超过±50%的收益率）
        df = df[(df['target_return'] >= -0.5) & (df['target_return'] <= 0.5)]
        
        # 3. 删除因子值全为空的样本
        df = df.dropna(subset=factor_names, how='all')
        
        # 4. 删除因子值异常的样本（超过10个标准差）
        for factor in factor_names:
            if factor in df.columns:
                factor_values = df[factor].dropna()
                if len(factor_values) > 0:
                    mean = factor_values.mean()
                    std = factor_values.std()
                    if std > 0:
                        lower_bound = mean - 10 * std
                        upper_bound = mean + 10 * std
                        df = df[(df[factor] >= lower_bound) | (df[factor] <= upper_bound) | df[factor].isna()]
        
        cleaned_count = len(df)
        logger.info(f"数据清洗完成: {original_count} -> {cleaned_count} ({cleaned_count/original_count:.2%} 保留)")
        
        return df
    
    def train_model(self, 
                   features: pd.DataFrame,
                   targets: pd.DataFrame,
                   model_type: str = 'xgboost',
                   model_params: Optional[Dict] = None,
                   cv_folds: int = 5,
                   test_size: float = 0.2) -> Dict[str, Any]:
        """训练模型
        
        Args:
            features: 特征数据
            targets: 目标数据
            model_type: 模型类型
            model_params: 模型参数
            cv_folds: 交叉验证折数
            test_size: 测试集比例
            
        Returns:
            训练结果字典
        """
        logger.info(f"开始训练 {model_type} 模型")
        
        if model_type not in self.model_classes:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 合并特征和目标数据
        training_data = features.merge(targets, on=['ts_code', 'trade_date'], how='inner')
        
        # 准备特征矩阵和目标向量
        feature_columns = [col for col in features.columns if col not in ['ts_code', 'trade_date']]
        X = training_data[feature_columns].fillna(0)  # 简单填充缺失值
        y = training_data['target_return']
        
        # 时间序列分割
        training_data_sorted = training_data.sort_values('trade_date')
        split_idx = int(len(training_data_sorted) * (1 - test_size))
        
        train_data = training_data_sorted.iloc[:split_idx]
        test_data = training_data_sorted.iloc[split_idx:]
        
        X_train = train_data[feature_columns].fillna(0)
        y_train = train_data['target_return']
        X_test = test_data[feature_columns].fillna(0)
        y_test = test_data['target_return']
        
        # 特征标准化（对线性模型）
        scaler = None
        if model_type in ['linear', 'ridge', 'lasso']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # 获取模型参数
        if model_params is None:
            model_params = self._get_default_model_params(model_type)
        
        # 创建模型
        model = self.model_classes[model_type](**model_params)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 计算评估指标
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        
        # 交叉验证
        cv_scores = None
        if cv_folds > 1:
            try:
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                cv_scores = cross_val_score(
                    self.model_classes[model_type](**model_params),
                    X_train, y_train, cv=tscv, scoring='neg_mean_squared_error'
                )
                cv_scores = -cv_scores  # 转换为正值
            except Exception as e:
                logger.warning(f"交叉验证失败: {str(e)}")
        
        # 特征重要性
        feature_importance = self._get_feature_importance(model, feature_columns)
        
        # 构建结果
        result = {
            'model': model,
            'scaler': scaler,
            'model_type': model_type,
            'model_params': model_params,
            'feature_columns': feature_columns,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores.tolist() if cv_scores is not None else None,
            'feature_importance': feature_importance,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'training_date_range': {
                'start': train_data['trade_date'].min(),
                'end': train_data['trade_date'].max()
            },
            'test_date_range': {
                'start': test_data['trade_date'].min(),
                'end': test_data['trade_date'].max()
            }
        }
        
        logger.info(f"模型训练完成 - 测试集 R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
        return result
    
    def _get_default_model_params(self, model_type: str) -> Dict:
        """获取默认模型参数
        
        Args:
            model_type: 模型类型
            
        Returns:
            默认参数字典
        """
        default_params = {
            'linear': {},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 0.1},
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbosity': 0
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbosity': -1
            }
        }
        
        return default_params.get(model_type, {})
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            评估指标字典
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _get_feature_importance(self, model, feature_columns: List[str]) -> Dict[str, float]:
        """获取特征重要性
        
        Args:
            model: 训练好的模型
            feature_columns: 特征列名
            
        Returns:
            特征重要性字典
        """
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # 树模型的特征重要性
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_columns, importances))
            elif hasattr(model, 'coef_'):
                # 线性模型的系数
                coefs = np.abs(model.coef_)
                importance_dict = dict(zip(feature_columns, coefs))
        except Exception as e:
            logger.warning(f"获取特征重要性失败: {str(e)}")
        
        return importance_dict
    
    def save_model(self, 
                  training_result: Dict[str, Any],
                  model_name: str,
                  version: str = None) -> str:
        """保存模型
        
        Args:
            training_result: 训练结果
            model_name: 模型名称
            version: 模型版本
            
        Returns:
            保存路径
        """
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_dir = self.models_dir / f"{model_name}_v{version}"
        model_dir.mkdir(exist_ok=True)
        
        # 保存模型文件
        model_path = model_dir / 'model.joblib'
        joblib.dump(training_result['model'], model_path)
        
        # 保存标准化器（如果有）
        if training_result['scaler'] is not None:
            scaler_path = model_dir / 'scaler.joblib'
            joblib.dump(training_result['scaler'], scaler_path)
        
        # 保存模型元数据
        metadata = {
            'model_type': training_result['model_type'],
            'model_params': training_result['model_params'],
            'feature_columns': training_result['feature_columns'],
            'train_metrics': training_result['train_metrics'],
            'test_metrics': training_result['test_metrics'],
            'cv_scores': training_result['cv_scores'],
            'feature_importance': training_result['feature_importance'],
            'training_samples': training_result['training_samples'],
            'test_samples': training_result['test_samples'],
            'training_date_range': training_result['training_date_range'],
            'test_date_range': training_result['test_date_range'],
            'created_at': datetime.now().isoformat(),
            'version': version
        }
        
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型保存完成: {model_dir}")
        return str(model_dir)
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            模型字典
        """
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 加载模型
        model_file = model_dir / 'model.joblib'
        model = joblib.load(model_file)
        
        # 加载标准化器（如果有）
        scaler_file = model_dir / 'scaler.joblib'
        scaler = None
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)
        
        # 加载元数据
        metadata_file = model_dir / 'metadata.json'
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        result = {
            'model': model,
            'scaler': scaler,
            **metadata
        }
        
        logger.info(f"模型加载完成: {model_path}")
        return result
    
    def run_training_pipeline(self, 
                            factor_names: List[str],
                            start_date: str,
                            end_date: str,
                            model_configs: List[Dict],
                            model_name: str,
                            target_period: int = 5,
                            stock_list: Optional[List[str]] = None,
                            save_models: bool = True) -> Dict[str, Any]:
        """运行完整的训练流水线
        
        Args:
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            model_configs: 模型配置列表
            model_name: 模型名称
            target_period: 目标周期
            stock_list: 股票列表
            save_models: 是否保存模型
            
        Returns:
            训练结果字典
        """
        logger.info(f"开始运行训练流水线: {model_name}")
        
        # 1. 准备训练数据
        features, targets = self.prepare_training_data(
            factor_names, start_date, end_date, target_period, stock_list=stock_list
        )
        
        if features.empty or targets.empty:
            logger.error("训练数据准备失败")
            return {}
        
        # 2. 训练多个模型
        training_results = {}
        
        for config in model_configs:
            model_type = config['type']
            model_params = config.get('params', {})
            
            try:
                logger.info(f"训练模型: {model_type}")
                
                result = self.train_model(
                    features=features,
                    targets=targets,
                    model_type=model_type,
                    model_params=model_params
                )
                
                training_results[model_type] = result
                
                # 保存模型
                if save_models:
                    model_path = self.save_model(
                        result, 
                        f"{model_name}_{model_type}"
                    )
                    result['saved_path'] = model_path
                
            except Exception as e:
                logger.error(f"模型 {model_type} 训练失败: {str(e)}")
                continue
        
        # 3. 选择最佳模型
        best_model = self._select_best_model(training_results)
        
        pipeline_result = {
            'model_name': model_name,
            'factor_names': factor_names,
            'date_range': {'start': start_date, 'end': end_date},
            'target_period': target_period,
            'training_results': training_results,
            'best_model': best_model,
            'total_samples': len(features),
            'completed_at': datetime.now().isoformat()
        }
        
        logger.info(f"训练流水线完成，最佳模型: {best_model['model_type'] if best_model else 'None'}")
        return pipeline_result
    
    def _select_best_model(self, training_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """选择最佳模型
        
        Args:
            training_results: 训练结果字典
            
        Returns:
            最佳模型结果
        """
        if not training_results:
            return None
        
        best_model = None
        best_score = float('-inf')
        
        for model_type, result in training_results.items():
            # 使用测试集R²作为评估指标
            score = result['test_metrics']['r2']
            
            if score > best_score:
                best_score = score
                best_model = result
                best_model['model_type'] = model_type
        
        return best_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI模型训练流水线')
    parser.add_argument('--factors', nargs='+', required=True, help='因子名称列表')
    parser.add_argument('--start-date', required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--model-name', required=True, help='模型名称')
    parser.add_argument('--models', nargs='+', default=['xgboost'], 
                       choices=['linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'],
                       help='模型类型列表')
    parser.add_argument('--target-period', type=int, default=5, help='目标收益率周期（天）')
    parser.add_argument('--stocks', nargs='*', help='股票代码列表')
    parser.add_argument('--no-save', action='store_true', help='不保存模型')
    
    args = parser.parse_args()
    
    # 创建训练流水线
    pipeline = ModelTrainingPipeline()
    
    # 构建模型配置
    model_configs = [{'type': model_type} for model_type in args.models]
    
    # 运行训练流水线
    result = pipeline.run_training_pipeline(
        factor_names=args.factors,
        start_date=args.start_date,
        end_date=args.end_date,
        model_configs=model_configs,
        model_name=args.model_name,
        target_period=args.target_period,
        stock_list=args.stocks,
        save_models=not args.no_save
    )
    
    # 输出结果
    if result:
        print(f"\n训练完成: {args.model_name}")
        print(f"总样本数: {result['total_samples']}")
        print(f"训练模型数: {len(result['training_results'])}")
        
        if result['best_model']:
            best = result['best_model']
            print(f"\n最佳模型: {best['model_type']}")
            print(f"测试集 R²: {best['test_metrics']['r2']:.4f}")
            print(f"测试集 RMSE: {best['test_metrics']['rmse']:.4f}")
            
            # 显示特征重要性前10
            if best['feature_importance']:
                importance_sorted = sorted(
                    best['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                print("\n特征重要性 (Top 10):")
                for feature, importance in importance_sorted:
                    print(f"  {feature}: {importance:.4f}")
    else:
        print("训练失败")
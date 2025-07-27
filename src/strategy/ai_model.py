#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI模型训练和预测模块

该模块实现了机器学习模型的训练和预测功能，支持多种算法。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import joblib
import os
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from ..utils.config_loader import ConfigLoader
from ..utils.db import DatabaseManager
from ..monitoring.logger import get_logger

logger = get_logger(__name__)

class AIModelTrainer:
    """
    AI模型训练器 - 负责训练机器学习模型
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化AI模型训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = ConfigLoader(config_path).config if config_path else {}
        self.db_manager = DatabaseManager()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'future_return'
        
        # 模型配置
        self.model_configs = {
            'lightgbm': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            },
            'xgboost': {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        }
    
    def prepare_training_data(self, start_date: str, end_date: str, 
                            stock_pool: List[str] = None) -> pd.DataFrame:
        """
        准备训练数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_pool: 股票池
        
        Returns:
            训练数据DataFrame
        """
        try:
            logger.info(f"准备训练数据: {start_date} 到 {end_date}")
            
            # 构建SQL查询
            stock_filter = ""
            if stock_pool:
                stock_list = "','".join(stock_pool)
                stock_filter = f"AND f.ts_code IN ('{stock_list}')"
            
            query = f"""
            SELECT 
                f.ts_code,
                f.trade_date,
                f.rsi_14,
                f.macd_signal,
                f.bollinger_upper,
                f.bollinger_lower,
                f.volume_ratio,
                f.price_change_rate,
                f.ma_5,
                f.ma_20,
                f.atr_14,
                -- 计算未来收益率作为标签
                LEAD(d.close, 1) OVER (PARTITION BY f.ts_code ORDER BY f.trade_date) / d.close - 1 as future_return
            FROM factor_values f
            JOIN stock_daily d ON f.ts_code = d.ts_code AND f.trade_date = d.trade_date
            WHERE f.trade_date >= '{start_date}' 
                AND f.trade_date <= '{end_date}'
                {stock_filter}
            ORDER BY f.ts_code, f.trade_date
            """
            
            df = pd.read_sql(query, self.db_manager.get_connection())
            
            # 删除缺失值
            df = df.dropna()
            
            # 设置特征列
            self.feature_columns = [
                'rsi_14', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
                'volume_ratio', 'price_change_rate', 'ma_5', 'ma_20', 'atr_14'
            ]
            
            logger.info(f"训练数据准备完成，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"准备训练数据失败: {e}")
            raise
    
    def train_model(self, data: pd.DataFrame, model_type: str = 'lightgbm',
                   test_size: float = 0.2, validation_split: bool = True) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            data: 训练数据
            model_type: 模型类型
            test_size: 测试集比例
            validation_split: 是否使用验证集
        
        Returns:
            训练结果
        """
        try:
            logger.info(f"开始训练 {model_type} 模型")
            
            # 准备特征和标签
            X = data[self.feature_columns]
            y = data[self.target_column]
            
            # 数据标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # 训练模型
            if model_type == 'lightgbm':
                model = self._train_lightgbm(X_train, y_train, X_test, y_test, validation_split)
            elif model_type == 'xgboost':
                model = self._train_xgboost(X_train, y_train, X_test, y_test, validation_split)
            elif model_type == 'random_forest':
                model = self._train_random_forest(X_train, y_train)
            elif model_type == 'linear':
                model = self._train_linear(X_train, y_train)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 评估模型
            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            
            # 保存模型和标准化器
            self.models[model_type] = model
            self.scalers[model_type] = scaler
            
            result = {
                'model_type': model_type,
                'metrics': metrics,
                'feature_importance': self._get_feature_importance(model, model_type),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logger.info(f"{model_type} 模型训练完成，R²: {metrics['r2_score']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"训练模型失败: {e}")
            raise
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test, validation_split):
        """训练LightGBM模型"""
        params = self.model_configs['lightgbm']
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data) if validation_split else None
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data] if valid_data else None,
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10)] if validation_split else None,
            verbose_eval=False
        )
        
        return model
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test, validation_split):
        """训练XGBoost模型"""
        params = self.model_configs['xgboost']
        
        model = xgb.XGBRegressor(**params)
        
        if validation_split:
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def _train_random_forest(self, X_train, y_train):
        """训练随机森林模型"""
        params = self.model_configs['random_forest']
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        return model
    
    def _train_linear(self, X_train, y_train):
        """训练线性回归模型"""
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """计算评估指标"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
    
    def _get_feature_importance(self, model, model_type) -> Dict[str, float]:
        """获取特征重要性"""
        try:
            if model_type in ['lightgbm', 'xgboost', 'random_forest']:
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'feature_importance'):
                    importance = model.feature_importance()
                else:
                    return {}
                
                return dict(zip(self.feature_columns, importance))
            else:
                return {}
        except Exception as e:
            logger.warning(f"获取特征重要性失败: {e}")
            return {}
    
    def save_model(self, model_type: str, model_path: str = None) -> str:
        """
        保存模型
        
        Args:
            model_type: 模型类型
            model_path: 保存路径
        
        Returns:
            保存路径
        """
        try:
            if model_type not in self.models:
                raise ValueError(f"模型 {model_type} 不存在")
            
            if not model_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = f"models/{model_type}_{timestamp}.joblib"
            
            # 确保目录存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 保存模型和标准化器
            model_data = {
                'model': self.models[model_type],
                'scaler': self.scalers[model_type],
                'feature_columns': self.feature_columns,
                'model_type': model_type,
                'timestamp': datetime.now()
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"模型已保存到: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise

class AIModelPredictor:
    """
    AI模型预测器 - 负责使用训练好的模型进行预测
    """
    
    def __init__(self, model_path: str = None):
        """
        初始化AI模型预测器
        
        Args:
            model_path: 模型文件路径
        """
        self.db_manager = DatabaseManager()
        self.model_data = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            self.model_data = joblib.load(model_path)
            logger.info(f"模型已加载: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def predict(self, stock_codes: List[str], trade_date: str) -> Dict[str, float]:
        """
        预测股票收益率
        
        Args:
            stock_codes: 股票代码列表
            trade_date: 交易日期
        
        Returns:
            预测结果字典 {股票代码: 预测收益率}
        """
        try:
            if not self.model_data:
                raise ValueError("模型未加载")
            
            logger.info(f"预测 {len(stock_codes)} 只股票在 {trade_date} 的收益率")
            
            # 获取特征数据
            features_df = self._get_features(stock_codes, trade_date)
            
            if features_df.empty:
                logger.warning("没有可用的特征数据")
                return {}
            
            # 数据预处理
            X = features_df[self.model_data['feature_columns']]
            X_scaled = self.model_data['scaler'].transform(X)
            
            # 预测
            predictions = self.model_data['model'].predict(X_scaled)
            
            # 构建结果
            result = dict(zip(features_df['ts_code'], predictions))
            
            logger.info(f"预测完成，共 {len(result)} 只股票")
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise
    
    def _get_features(self, stock_codes: List[str], trade_date: str) -> pd.DataFrame:
        """
        获取特征数据
        
        Args:
            stock_codes: 股票代码列表
            trade_date: 交易日期
        
        Returns:
            特征数据DataFrame
        """
        try:
            stock_list = "','".join(stock_codes)
            
            query = f"""
            SELECT 
                ts_code,
                trade_date,
                rsi_14,
                macd_signal,
                bollinger_upper,
                bollinger_lower,
                volume_ratio,
                price_change_rate,
                ma_5,
                ma_20,
                atr_14
            FROM factor_values
            WHERE ts_code IN ('{stock_list}')
                AND trade_date = '{trade_date}'
            """
            
            df = pd.read_sql(query, self.db_manager.get_connection())
            return df.dropna()
            
        except Exception as e:
            logger.error(f"获取特征数据失败: {e}")
            raise
    
    def batch_predict(self, stock_codes: List[str], start_date: str, 
                     end_date: str) -> pd.DataFrame:
        """
        批量预测
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            预测结果DataFrame
        """
        try:
            logger.info(f"批量预测: {start_date} 到 {end_date}")
            
            # 获取交易日历
            trade_dates = self._get_trade_dates(start_date, end_date)
            
            results = []
            for trade_date in trade_dates:
                predictions = self.predict(stock_codes, trade_date)
                for ts_code, prediction in predictions.items():
                    results.append({
                        'ts_code': ts_code,
                        'trade_date': trade_date,
                        'predicted_return': prediction,
                        'prediction_time': datetime.now()
                    })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"批量预测失败: {e}")
            raise
    
    def _get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            交易日期列表
        """
        try:
            query = f"""
            SELECT cal_date
            FROM trade_calendar
            WHERE cal_date >= '{start_date}'
                AND cal_date <= '{end_date}'
                AND is_open = 1
            ORDER BY cal_date
            """
            
            df = pd.read_sql(query, self.db_manager.get_connection())
            return df['cal_date'].tolist()
            
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []

if __name__ == '__main__':
    # 测试代码
    print("测试AI模型模块...")
    
    # 测试训练器
    trainer = AIModelTrainer()
    
    # 生成模拟数据进行测试
    np.random.seed(42)
    n_samples = 1000
    
    mock_data = pd.DataFrame({
        'ts_code': ['000001.SZ'] * n_samples,
        'trade_date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'rsi_14': np.random.uniform(20, 80, n_samples),
        'macd_signal': np.random.normal(0, 0.1, n_samples),
        'bollinger_upper': np.random.uniform(10, 20, n_samples),
        'bollinger_lower': np.random.uniform(8, 15, n_samples),
        'volume_ratio': np.random.uniform(0.5, 2.0, n_samples),
        'price_change_rate': np.random.normal(0, 0.02, n_samples),
        'ma_5': np.random.uniform(10, 20, n_samples),
        'ma_20': np.random.uniform(10, 20, n_samples),
        'atr_14': np.random.uniform(0.5, 2.0, n_samples),
        'future_return': np.random.normal(0.001, 0.02, n_samples)
    })
    
    print("\n训练LightGBM模型...")
    result = trainer.train_model(mock_data, 'lightgbm')
    print(f"训练结果: R² = {result['metrics']['r2_score']:.4f}")
    
    print("\nAI模型模块测试完成!")
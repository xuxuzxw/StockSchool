import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一模型训练服务

提供标准化的AI模型训练接口，支持多种模型类型和训练配置。
"""


logger = logging.getLogger(__name__)


class TrainingConfig:
    """训练配置类"""

    def __init__(self, model_type: str, **kwargs):
        """方法描述"""
        self.model_params = kwargs.get("model_params", {})
        self.cv_folds = kwargs.get("cv_folds", 5)
        self.test_size = kwargs.get("test_size", 0.2)
        self.random_state = kwargs.get("random_state", 42)
        self.scaling = kwargs.get("scaling", True)


class TrainingResult:
    """训练结果类"""

    def __init__(self, model: Any, metrics: Dict[str, Any], metadata: Dict[str, Any]):
        """方法描述"""
        self.metrics = metrics
        self.metadata = metadata
        self.training_time = metadata.get("training_time", 0)
        self.feature_importance = metrics.get("feature_importance", {})
        self.model_type = metadata.get("model_type", "")


class ModelTrainingService:
    """统一模型训练服务

    提供标准化的AI模型训练接口，支持：
    - 多种模型类型（线性、树模型、集成学习）
    - 标准化评估指标
    - 交叉验证
    - 特征重要性分析
    - 模型持久化
    """

    def __init__(self, models_dir: str = "models"):
        """方法描述"""
        self.models_dir.mkdir(exist_ok=True)

        # 支持的模型类型
        self.model_classes = {
            "linear": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "random_forest": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor,
            "xgboost": xgb.XGBRegressor,
            "lightgbm": lgb.LGBMRegressor,
        }

        # 默认模型参数
        self.default_params = {
            "linear": {},
            "ridge": {"alpha": 1.0, "random_state": 42},
            "lasso": {"alpha": 0.1, "random_state": 42},
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
            "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42},
            "xgboost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42, "verbosity": 0},
            "lightgbm": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42,
                "verbosity": -1,
            },
        }

    def train_model(
        self, features: pd.DataFrame, targets: pd.DataFrame, model_type: str, config: Optional[TrainingConfig] = None
    ) -> TrainingResult:
        """训练模型

        Args:
            features: 特征数据 (包含ts_code, trade_date列)
            targets: 目标数据 (包含target_return列)
            model_type: 模型类型
            config: 训练配置

        Returns:
            TrainingResult: 训练结果
        """
        if config is None:
            config = TrainingConfig(model_type)

        if model_type not in self.model_classes:
            raise ValueError(f"不支持的模型类型: {model_type}")

        logger.info(f"开始训练 {model_type} 模型")
        start_time = datetime.now()

        try:
            # 准备数据
            X, y, feature_columns = self._prepare_data(features, targets)

            # 数据分割
            X_train, X_test, y_train, y_test = self._split_data(X, y, config.test_size)

            # 特征标准化（对线性模型）
            scaler = None
            if config.scaling and model_type in ["linear", "ridge", "lasso"]:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # 获取模型参数
            model_params = self._get_model_params(model_type, config.model_params)

            # 创建和训练模型
            model = self.model_classes[model_type](**model_params)
            model.fit(X_train, y_train)

            # 评估模型
            metrics = self._evaluate_model(model, X_train, X_test, y_train, y_test, feature_columns)

            # 计算训练时间
            training_time = (datetime.now() - start_time).total_seconds()

            # 构建结果
            metadata = {
                "model_type": model_type,
                "training_time": training_time,
                "feature_columns": feature_columns,
                "model_params": model_params,
                "scaler": scaler,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
            }

            result = TrainingResult(model, metrics, metadata)

            logger.info(f"模型训练完成 - 测试集R²: {metrics['test']['r2']:.4f}, 耗时: {training_time:.2f}秒")
            return result

        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise

    def save_model(self, result: TrainingResult, model_name: str, version: str = None) -> str:
        """保存模型和元数据

        Args:
            result: 训练结果
            model_name: 模型名称
            version: 版本号（可选）

        Returns:
            模型保存路径
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d%H%M%S")

        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True, parents=True)

        model_path = model_dir / f"{version}.pkl"
        metadata_path = model_dir / f"{version}_metadata.json"

        # 保存模型
        save_data = {
            "model": result.model,
            "scaler": result.metadata.get("scaler"),
            "feature_columns": result.metadata["feature_columns"],
        }
        joblib.dump(save_data, model_path)

        # 保存元数据
        metadata = {
            "model_name": model_name,
            "version": version,
            "model_type": result.model_type,
            "training_time": result.training_time,
            "metrics": result.metrics,
            "feature_columns": result.metadata["feature_columns"],
            "model_params": result.metadata["model_params"],
            "training_samples": result.metadata["training_samples"],
            "test_samples": result.metadata["test_samples"],
            "saved_at": datetime.now().isoformat(),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"模型已保存: {model_path}")
        return str(model_path)

    def load_model(self, model_name: str, version: str) -> Tuple[Any, Dict[str, Any]]:
        """加载模型和元数据

        Args:
            model_name: 模型名称
            version: 版本号

        Returns:
            (模型, 元数据)
        """
        model_path = self.models_dir / model_name / f"{version}.pkl"
        metadata_path = self.models_dir / model_name / f"{version}_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型
        save_data = joblib.load(model_path)
        model = save_data["model"]

        # 加载元数据
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return model, metadata

    def _prepare_data(self, features: pd.DataFrame, targets: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备训练数据"""
        # 合并数据
        data = features.merge(targets, on=["ts_code", "trade_date"], how="inner")

        # 获取特征列
        feature_columns = [col for col in data.columns if col not in ["ts_code", "trade_date", "target_return"]]

        # 准备特征和目标
        X = data[feature_columns].fillna(0).values
        y = data["target_return"].values

        return X, y, feature_columns

    def _split_data(
        self, X: np.ndarray, y: np.ndarray, test_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """时间序列数据分割"""
        split_idx = int(len(X) * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def _get_model_params(self, model_type: str, custom_params: Optional[Dict] = None) -> Dict[str, Any]:
        """获取合并后的模型参数"""
        params = self.default_params.get(model_type, {}).copy()
        if custom_params:
            params.update(custom_params)
        return params

    def _evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_columns: List[str],
    ) -> Dict[str, Any]:
        """评估模型性能"""
        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 计算指标
        train_metrics = {
            "mse": mean_squared_error(y_train, y_train_pred),
            "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "mae": mean_absolute_error(y_train, y_train_pred),
            "r2": r2_score(y_train, y_train_pred),
        }

        test_metrics = {
            "mse": mean_squared_error(y_test, y_test_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "mae": mean_absolute_error(y_test, y_test_pred),
            "r2": r2_score(y_test, y_test_pred),
        }

        # 特征重要性
        feature_importance = self._get_feature_importance(model, feature_columns)

        # 交叉验证
        cv_scores = self._cross_validation(model, X_train, y_train)

        return {
            "train": train_metrics,
            "test": test_metrics,
            "cv_scores": cv_scores,
            "feature_importance": feature_importance,
        }

    def _get_feature_importance(self, model: Any, feature_columns: List[str]) -> Dict[str, float]:
        """获取特征重要性"""
        importance_dict = {}

        try:
            if hasattr(model, "feature_importances_"):
                # 树模型的特征重要性
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_columns, importances))
            elif hasattr(model, "coef_"):
                # 线性模型的系数
                coefs = np.abs(model.coef_)
                importance_dict = dict(zip(feature_columns, coefs))
        except Exception as e:
            logger.warning(f"获取特征重要性失败: {str(e)}")

        return importance_dict

    def _cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Optional[List[float]]:
        """交叉验证"""
        try:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
            return (-scores).tolist()  # 转换为正值
        except Exception as e:
            logger.warning(f"交叉验证失败: {str(e)}")
            return None


# 为向后兼容创建别名
TrainingService = ModelTrainingService

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error

from src.monitoring.logger import get_logger

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
置换解释器模块

实现基于置换重要性的模型解释方法，提供模型无关的特征重要性计算。
"""


logger = get_logger(__name__)


class PermutationExplainer:
    """置换解释器类"""

    def __init__(self, model: BaseEstimator, feature_names: List[str]):
        """
        初始化置换解释器

        Args:
            model: 机器学习模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        self.is_fitted = False

        # 检查模型是否已训练
        if hasattr(model, "predict"):
            self.is_fitted = True
        else:
            logger.warning("模型似乎未训练，某些功能可能不可用")

    def calculate_feature_importance(
        self, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 42, scoring: str = "auto"
    ) -> pd.DataFrame:
        """
        计算置换特征重要性

        Args:
            X: 特征数据
            y: 目标变量
            n_repeats: 置换重复次数
            random_state: 随机种子
            scoring: 评分方法 ('auto', 'neg_mean_squared_error', 'accuracy', 'f1')

        Returns:
            特征重要性DataFrame
        """
        try:
            # 自动确定评分方法
            if scoring == "auto":
                scoring = self._determine_scoring_method(y)

            # 计算置换重要性
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring
            )

            # 创建结果DataFrame
            importance_df = (
                pd.DataFrame(
                    {
                        "feature": self.feature_names,
                        "importance": perm_importance.importances_mean,
                        "importance_std": perm_importance.importances_std,
                        "importance_min": perm_importance.importances.min(axis=1),
                        "importance_max": perm_importance.importances.max(axis=1),
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

            logger.info(f"置换特征重要性计算完成，共{len(importance_df)}个特征")
            return importance_df

        except Exception as e:
            logger.error(f"置换特征重要性计算失败: {e}")
            raise

    def _determine_scoring_method(self, y: pd.Series) -> str:
        """
        自动确定评分方法

        Args:
            y: 目标变量

        Returns:
            评分方法字符串
        """
        # 检查目标变量类型
        if y.dtype in ["int64", "int32", "object"]:
            # 分类问题
            unique_values = len(y.unique())
            if unique_values == 2:
                return "accuracy"  # 二分类
            else:
                return "f1_macro"  # 多分类
        else:
            # 回归问题
            return "neg_mean_squared_error"

    def calculate_partial_dependence(
        self, X: pd.DataFrame, feature_idx: Union[int, str], grid_resolution: int = 50
    ) -> Dict[str, Any]:
        """
        计算部分依赖关系

        Args:
            X: 特征数据
            feature_idx: 特征索引或名称
            grid_resolution: 网格分辨率

        Returns:
            部分依赖关系字典
        """
        try:
            # 获取特征索引
            if isinstance(feature_idx, str):
                feature_idx = self.feature_names.index(feature_idx)

            # 创建特征网格
            feature_values = X.iloc[:, feature_idx]
            feature_grid = np.linspace(feature_values.min(), feature_values.max(), grid_resolution)

            # 计算部分依赖
            pdp_values = []
            X_copy = X.copy()

            for value in feature_grid:
                X_copy.iloc[:, feature_idx] = value
                predictions = self.model.predict(X_copy)
                pdp_values.append(predictions.mean())

            result = {
                "feature_name": self.feature_names[feature_idx],
                "feature_values": feature_grid.tolist(),
                "pdp_values": pdp_values,
                "feature_original_range": [float(feature_values.min()), float(feature_values.max())],
            }

            logger.info(f"部分依赖关系计算完成: {self.feature_names[feature_idx]}")
            return result

        except Exception as e:
            logger.error(f"部分依赖关系计算失败: {e}")
            raise

    def calculate_interaction_strength(
        self, X: pd.DataFrame, y: pd.Series, feature_pairs: List[Tuple[int, int]] = None, n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        计算特征交互强度

        Args:
            X: 特征数据
            y: 目标变量
            feature_pairs: 特征对列表（可选）
            n_samples: 采样数量

        Returns:
            特征交互强度DataFrame
        """
        try:
            # 如果没有指定特征对，计算前几个重要特征的交互
            if feature_pairs is None:
                # 先计算特征重要性
                importance = self.calculate_feature_importance(X, y, n_repeats=5)
                top_features = importance.head(5)["feature"].tolist()

                # 生成特征对
                feature_pairs = []
                for i, feat1 in enumerate(top_features):
                    for feat2 in top_features[i + 1 :]:
                        idx1 = self.feature_names.index(feat1)
                        idx2 = self.feature_names.index(feat2)
                        feature_pairs.append((idx1, idx2))

            # 计算交互强度
            interactions = []

            for idx1, idx2 in feature_pairs:
                interaction_strength = self._compute_pairwise_interaction(X, y, idx1, idx2, n_samples)

                interactions.append(
                    {
                        "feature1": self.feature_names[idx1],
                        "feature2": self.feature_names[idx2],
                        "interaction_strength": interaction_strength,
                        "feature1_idx": idx1,
                        "feature2_idx": idx2,
                    }
                )

            interaction_df = (
                pd.DataFrame(interactions).sort_values("interaction_strength", ascending=False).reset_index(drop=True)
            )

            logger.info(f"特征交互强度计算完成，共{len(interaction_df)}个特征对")
            return interaction_df

        except Exception as e:
            logger.error(f"特征交互强度计算失败: {e}")
            raise

    def _compute_pairwise_interaction(
        self, X: pd.DataFrame, y: pd.Series, idx1: int, idx2: int, n_samples: int
    ) -> float:
        """
        计算成对特征交互强度

        Args:
            X: 特征数据
            y: 目标变量
            idx1: 第一个特征索引
            idx2: 第二个特征索引
            n_samples: 采样数量

        Returns:
            交互强度值
        """
        # 采样数据
        if len(X) > n_samples:
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
        else:
            X_sample = X
            y_sample = y

        # 计算基准预测
        baseline_pred = self.model.predict(X_sample)

        # 计算单独置换的效果
        X_perm1 = X_sample.copy()
        X_perm1.iloc[:, idx1] = np.random.permutation(X_perm1.iloc[:, idx1])
        pred1 = self.model.predict(X_perm1)

        X_perm2 = X_sample.copy()
        X_perm2.iloc[:, idx2] = np.random.permutation(X_perm2.iloc[:, idx2])
        pred2 = self.model.predict(X_perm2)

        # 计算联合置换的效果
        X_perm_both = X_sample.copy()
        X_perm_both.iloc[:, idx1] = np.random.permutation(X_perm_both.iloc[:, idx1])
        X_perm_both.iloc[:, idx2] = np.random.permutation(X_perm_both.iloc[:, idx2])
        pred_both = self.model.predict(X_perm_both)

        # 计算交互效应（H-statistic）
        individual_effects = np.abs(baseline_pred - pred1) + np.abs(baseline_pred - pred2)
        joint_effect = np.abs(baseline_pred - pred_both)

        # 交互强度 = 联合效应与个体效应的比例
        interaction_strength = np.mean(joint_effect) / (np.mean(individual_effects) + 1e-8)

        return float(interaction_strength)

    def get_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        获取模型性能指标

        Args:
            X: 特征数据
            y: 目标变量

        Returns:
            性能指标字典
        """
        try:
            predictions = self.model.predict(X)

            # 判断是分类还是回归
            if y.dtype in ["int64", "int32", "object"] or len(y.unique()) < 20:
                # 分类问题
                accuracy = accuracy_score(y, predictions)
                performance = {"accuracy": accuracy, "problem_type": "classification"}
            else:
                # 回归问题
                mse = mean_squared_error(y, predictions)
                rmse = np.sqrt(mse)
                performance = {"mse": mse, "rmse": rmse, "problem_type": "regression"}

            logger.info(f"模型性能评估完成: {performance}")
            return performance

        except Exception as e:
            logger.error(f"模型性能评估失败: {e}")
            raise


def create_permutation_explainer(model: BaseEstimator, feature_names: List[str]) -> PermutationExplainer:
    """
    创建置换解释器的便捷函数

    Args:
        model: 机器学习模型
        feature_names: 特征名称列表

    Returns:
        PermutationExplainer实例
    """
    return PermutationExplainer(model, feature_names)


if __name__ == "__main__":
    # 测试代码
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split

    print("测试置换解释器...")

    # 测试回归模型
    print("\n=== 回归模型测试 ===")
    X_reg, y_reg = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    feature_names = [f"feature_{i}" for i in range(10)]
    X_reg_df = pd.DataFrame(X_reg, columns=feature_names)
    y_reg_series = pd.Series(y_reg)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg_df, y_reg_series, test_size=0.2, random_state=42
    )

    reg_model = RandomForestRegressor(n_estimators=50, random_state=42)
    reg_model.fit(X_train_reg, y_train_reg)

    reg_explainer = PermutationExplainer(reg_model, feature_names)

    # 测试特征重要性
    reg_importance = reg_explainer.calculate_feature_importance(X_test_reg, y_test_reg, n_repeats=5)
    print(f"回归模型特征重要性计算完成，前5个特征:")
    print(reg_importance.head())

    # 测试模型性能
    reg_performance = reg_explainer.get_model_performance(X_test_reg, y_test_reg)
    print(f"回归模型性能: {reg_performance}")

    # 测试分类模型
    print("\n=== 分类模型测试 ===")
    X_clf, y_clf = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    X_clf_df = pd.DataFrame(X_clf, columns=feature_names)
    y_clf_series = pd.Series(y_clf)

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf_df, y_clf_series, test_size=0.2, random_state=42
    )

    clf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_model.fit(X_train_clf, y_train_clf)

    clf_explainer = PermutationExplainer(clf_model, feature_names)

    # 测试特征重要性
    clf_importance = clf_explainer.calculate_feature_importance(X_test_clf, y_test_clf, n_repeats=5)
    print(f"分类模型特征重要性计算完成，前5个特征:")
    print(clf_importance.head())

    # 测试模型性能
    clf_performance = clf_explainer.get_model_performance(X_test_clf, y_test_clf)
    print(f"分类模型性能: {clf_performance}")

    print("\n置换解释器测试完成!")

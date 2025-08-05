from functools import wraps
from pathlib import Path

from src.config.unified_config import config
from src.utils.gpu_utils import (
    SHAP值计算,
    Any,
    Dict,
    List,
    Optional,
    StockSchool,
    Team,
    Union,
    1.,
    2.,
    3.,
    4.,
    5.,
    6.,
    2025-07-31,
    """,
    check_memory_sufficient,
    fallback_to_cpu,
    from,
    get_batch_size,
    get_device,
    get_gpu_info,
    get_logger,
    handle_oom,
    import,
    joblib,
    logging,
)
from src.utils.gpu_utils import matplotlib.pyplot as plt  # !/usr/bin/env python3; -*- coding: utf-8 -*-
from src.utils.gpu_utils import numpy as np
from src.utils.gpu_utils import pandas as pd
from src.utils.gpu_utils import seaborn as sns
from src.utils.gpu_utils import (
    shap,
    src.utils.logger,
    time,
    torch,
    typing,
    作者:,
    创建时间:,
    单样本解释,
    可视化功能,
    排列重要性,
    模型解释器模块,
    特征交互分析,
    该模块提供多种模型解释方法，包括：,
    默认特征重要性,
)

logger = get_logger(__name__)

class ModelExplainerError(Exception):
    """模型解释器自定义异常"""
    pass

class ModelExplainer:
    """
    模型解释器类，提供多种特征重要性计算方法和可视化功能
    """

    def __init__(self, model, feature_names: List[str]):
        """
        初始化模型解释器

        Args:
            model: 机器学习模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.device = get_device()
        self.cache_dir = Path(config.get('feature_params.cache_dir', './cache/explanations'))
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.enable_cache = config.get('feature_params.cache_explanations', True)
        self.max_cache_size = config.get('feature_params.max_cache_size', 1000)

        # Windows环境特定配置
        self.windows_cuda_workaround = config.get('feature_params.windows_cuda_workaround', True)

        logger.info(f"ModelExplainer初始化完成，设备: {self.device}")

    def _get_model_type(self) -> str:
        """获取模型类型"""
        model_name = type(self.model).__name__.lower()

        if any(x in model_name for x in ['xgb', 'xgboost', 'lgb', 'lightgbm', 'catboost', 'tree', 'forest']):
            return 'tree'
        elif any(x in model_name for x in ['linear', 'logistic', 'ridge', 'lasso']):
            return 'linear'
        elif any(x in model_name for x in ['neural', 'mlp', 'cnn', 'rnn']):
            return 'neural_network'
        else:
            return 'unknown'

    def _initialize_shap_explainer(self, X: pd.DataFrame):
        """
        初始化SHAP解释器（包含显存溢出处理）

        Args:
            X: 特征数据
        """
        try:
            model_type = self._get_model_type()
            logger.info(f"检测到模型类型: {model_type}")

            if self.device.type == 'cuda':
                # 根据模型类型选择合适的解释器
                if model_type == 'tree':
                    # 树模型使用TreeExplainer
                    background_samples = config.get('feature_params.shap_background_samples', 100)
                    background_data = X[:background_samples] if len(X) > background_samples else X

                    # Windows环境下的CUDA工作区处理
                    if self.windows_cuda_workaround:
                        logger.info("应用Windows CUDA工作区处理")
                        # 降低背景样本数量以减少内存使用
                        background_data = background_data[:min(50, len(background_data))]

                    self.explainer = shap.TreeExplainer(
                        self.model,
                        background_data,
                        model_output='raw',
                        feature_perturbation='interventional'
                    )
                elif model_type == 'linear':
                    # 线性模型使用LinearExplainer
                    self.explainer = shap.LinearExplainer(self.model, X)
                else:
                    # 通用解释器
                    background_samples = config.get('feature_params.shap_background_samples', 100)
                    background_data = X[:background_samples] if len(X) > background_samples else X
                    self.explainer = shap.Explainer(self.model, background_data)
            else:
                # CPU模式通用解释器
                background_samples = config.get('feature_params.shap_background_samples', 100)
                background_data = X[:background_samples] if len(X) > background_samples else X
                self.explainer = shap.Explainer(self.model, background_data)

            logger.info("SHAP解释器初始化成功")

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"CUDA显存不足: {e}")
            # 尝试降级到CPU
            if config.get('feature_params.fallback_to_cpu', True):
                logger.info("降级到CPU模式")
                fallback_to_cpu()
                self.device = get_device()
                self._initialize_shap_explainer(X)
            else:
                raise ModelExplainerError(f"SHAP解释器初始化失败: {e}")
        except Exception as e:
            logger.error(f"初始化SHAP解释器失败: {e}")
            raise ModelExplainerError(f"SHAP解释器初始化失败: {e}")

    def _get_cache_key(self, method: str, data_hash: str = None) -> str:
        """生成缓存键"""
        model_hash = str(hash(str(self.model)))
        return f"{model_hash}_{method}_{data_hash or 'default'}"

    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存加载结果"""
        if not self.enable_cache:
            return None

        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                result = joblib.load(cache_file)
                logger.info(f"从缓存加载结果: {cache_key}")
                return result
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")

        return None

    def _save_to_cache(self, cache_key: str, result: Any):
        """保存结果到缓存"""
        if not self.enable_cache:
            return

        try:
            # 检查缓存大小
            cache_files = list(self.cache_dir.glob("*.pkl"))
            if len(cache_files) >= self.max_cache_size:
                # 删除最旧的缓存文件
                oldest_file = min(cache_files, key=lambda x: x.stat().st_mtime)
                oldest_file.unlink()
                logger.info(f"删除旧缓存文件: {oldest_file}")

            cache_file = self.cache_dir / f"{cache_key}.pkl"
            joblib.dump(result, cache_file)
            logger.info(f"结果已缓存: {cache_key}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series = None,
                                   method: str = 'default') -> pd.DataFrame:
        """
        计算特征重要性（支持分批处理）

        Args:
            X: 特征数据
            y: 目标变量（用于排列重要性）
            method: 计算方法 ('default', 'permutation', 'shap')

        Returns:
            特征重要性DataFrame
        """
        try:
            # 生成缓存键
            data_hash = str(hash(str(X.shape) + str(X.columns.tolist())))
            cache_key = self._get_cache_key(f"importance_{method}", data_hash)

            # 检查缓存
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            logger.info(f"计算特征重要性，方法: {method}")

            if method == 'default':
                result = self._get_default_importance(X)
            elif method == 'permutation':
                if y is None:
                    raise ValueError("排列重要性需要目标变量y")
                result = self._get_permutation_importance(X, y)
            elif method == 'shap':
                result = self._get_shap_importance(X)
            else:
                raise ValueError(f"不支持的计算方法: {method}")

            # 保存到缓存
            self._save_to_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"特征重要性计算失败: {e}")
            raise ModelExplainerError(f"特征重要性计算失败: {e}")

    def _get_default_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """获取默认特征重要性"""
        try:
            # 尝试从模型获取特征重要性
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_)
            else:
                # 如果模型不支持默认重要性，使用方差方法
                importances = np.var(X, axis=0)

            # 创建结果DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            logger.info("默认特征重要性计算完成")
            return importance_df

        except Exception as e:
            logger.error(f"默认特征重要性计算失败: {e}")
            # 返回基于方差的重要性
            importances = np.var(X, axis=0)
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

    def _get_permutation_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """获取排列重要性"""
        try:
            from sklearn.inspection import permutation_importance

            # 检查内存是否足够
            required_memory = len(X) * len(self.feature_names) * 8 / (1024**2)  # 估算内存需求(MB)
            if not check_memory_sufficient(required_memory * 2):  # 预留缓冲
                logger.warning("内存不足，降低批量大小")
                # 减少样本数量
                sample_size = min(1000, len(X))
                X_sample = X.sample(n=sample_size, random_state=42)
                y_sample = y.sample(n=sample_size, random_state=42)
            else:
                X_sample, y_sample = X, y

            # 计算排列重要性
            perm_importance = permutation_importance(
                self.model, X_sample, y_sample,
                n_repeats=10,
                random_state=42,
                n_jobs=1  # 避免多进程内存问题
            )

            # 创建结果DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)

            logger.info("排列重要性计算完成")
            return importance_df

        except Exception as e:
            logger.error(f"排列重要性计算失败: {e}")
            raise ModelExplainerError(f"排列重要性计算失败: {e}")

    def _get_shap_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """获取SHAP重要性"""
        try:
            # 初始化SHAP解释器
            if self.explainer is None:
                self._initialize_shap_explainer(X)

            # 获取批量大小
            batch_size = get_batch_size(len(X))
            max_objects = config.get('feature_params.shap_max_objects', 10000)

            # 限制数据大小以避免内存问题
            if len(X) > max_objects:
                logger.info(f"数据量过大({len(X)})，采样到{max_objects}")
                X_sample = X.sample(n=max_objects, random_state=42)
            else:
                X_sample = X

            # 分批处理大型数据集
            if len(X_sample) > batch_size:
                logger.info(f"分批处理SHAP计算，批量大小: {batch_size}")

                # 获取背景数据用于SHAP计算
                background_samples = min(100, len(X_sample))
                background_data = X_sample[:background_samples]

                # 重新初始化解释器（使用较小的背景数据）
                if self.windows_cuda_workaround:
                    background_data = background_data[:min(50, len(background_data))]

                self._initialize_shap_explainer(background_data)

                # 计算SHAP值（只对样本子集）
                sample_size = min(1000, len(X_sample))
                X_subset = X_sample.sample(n=sample_size, random_state=42)
                shap_values = self.explainer.shap_values(X_subset)
            else:
                # 直接计算SHAP值
                shap_values = self.explainer.shap_values(X_sample)

            # 处理SHAP值
            if isinstance(shap_values, list):
                # 多输出情况，取第一个输出
                shap_values = shap_values[0]

            # 计算特征重要性（SHAP值的绝对值均值）
            if len(shap_values.shape) == 1:
                # 一维情况
                importances = np.abs(shap_values)
            else:
                # 多维情况
                importances = np.mean(np.abs(shap_values), axis=0)

            # 创建结果DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)

            logger.info("SHAP重要性计算完成")
            return importance_df

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"SHAP计算显存不足: {e}")
            # 尝试减小批量大小并重试
            current_batch_size = config.get('feature_params.shap_batch_size', 500)
            new_batch_size = max(50, int(current_batch_size * 0.5))
            config.set('feature_params.shap_batch_size', new_batch_size)
            logger.info(f"批量大小调整为: {new_batch_size}")

            # 重新尝试计算
            return self._get_shap_importance(X)

        except Exception as e:
            logger.error(f"SHAP重要性计算失败: {e}")
            raise ModelExplainerError(f"SHAP重要性计算失败: {e}")

    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        """
        解释单个样本的预测

        Args:
            X: 特征数据
            sample_idx: 样本索引

        Returns:
            预测解释字典
        """
        try:
            # 生成缓存键
            data_hash = str(hash(str(X.iloc[sample_idx].values)))
            cache_key = self._get_cache_key("prediction_explanation", data_hash)

            # 检查缓存
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            logger.info(f"解释预测，样本索引: {sample_idx}")

            # 初始化SHAP解释器
            if self.explainer is None:
                self._initialize_shap_explainer(X)

            # 获取单个样本
            single_sample = X.iloc[[sample_idx]]

            # 计算SHAP值
            shap_values = self.explainer.shap_values(single_sample)

            # 处理SHAP值
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

            # 获取基础值和预测值
            base_value = getattr(self.explainer, 'expected_value', 0)
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict(single_sample)[0]
            else:
                prediction = self.model(single_sample.values)[0]

            # 创建解释结果
            feature_values = single_sample.iloc[0].values
            explanation = {
                'sample_idx': sample_idx,
                'feature_names': self.feature_names,
                'feature_values': feature_values,
                'shap_values': shap_values,
                'base_value': base_value,
                'prediction': prediction,
                'model_type': self._get_model_type()
            }

            # 保存到缓存
            self._save_to_cache(cache_key, explanation)

            return explanation

        except Exception as e:
            logger.error(f"预测解释失败: {e}")
            raise ModelExplainerError(f"预测解释失败: {e}")

    def analyze_feature_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        分析特征交互作用

        Args:
            X: 特征数据

        Returns:
            特征交互分析DataFrame
        """
        try:
            logger.info("分析特征交互作用")

            # 简单的相关性分析作为特征交互的近似
            correlation_matrix = X.corr()

            # 找出高相关性的特征对
            interactions = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:  # 高相关性阈值
                        interactions.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value,
                            'interaction_strength': abs(corr_value)
                        })

            # 创建结果DataFrame
            interaction_df = pd.DataFrame(interactions).sort_values(
                'interaction_strength', ascending=False
            )

            logger.info(f"特征交互分析完成，发现{len(interaction_df)}个高相关性特征对")
            return interaction_df

        except Exception as e:
            logger.error(f"特征交互分析失败: {e}")
            return pd.DataFrame()

    def generate_model_summary(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Any]:
        """
        生成模型摘要

        Args:
            X: 特征数据
            y: 目标变量（可选）

        Returns:
            模型摘要字典
        """
        try:
            logger.info("生成模型摘要")

            summary = {
                'model_type': self._get_model_type(),
                'feature_count': len(self.feature_names),
                'feature_names': self.feature_names,
                'device': str(self.device),
                'cache_enabled': self.enable_cache,
                'samples_count': len(X) if X is not None else 0
            }

            # 如果提供了目标变量，计算一些基本统计
            if y is not None:
                summary['target_stats'] = {
                    'mean': float(y.mean()),
                    'std': float(y.std()),
                    'min': float(y.min()),
                    'max': float(y.max())
                }

            # GPU信息（如果使用GPU）
            if self.device.type == 'cuda':
                gpu_info = get_gpu_info()
                summary['gpu_info'] = gpu_info

            logger.info("模型摘要生成完成")
            return summary

        except Exception as e:
            logger.error(f"模型摘要生成失败: {e}")
            return {}

    def plot_feature_importance(self, X: pd.DataFrame, y: pd.Series = None,
                              method: str = 'default', top_n: int = 20,
                              figsize: tuple = (12, 8)) -> plt.Figure:
        """
        绘制特征重要性图

        Args:
            X: 特征数据
            y: 目标变量
            method: 计算方法
            top_n: 显示前N个特征
            figsize: 图形大小

        Returns:
            matplotlib图形对象
        """
        try:
            logger.info(f"绘制特征重要性图，方法: {method}")

            # 计算特征重要性
            importance_df = self.calculate_feature_importance(X, y, method)

            # 选择前N个特征
            top_features = importance_df.head(top_n)

            # 创建图形
            fig, ax = plt.subplots(figsize=figsize)

            # 绘制条形图
            bars = ax.barh(range(len(top_features)), top_features['importance'],
                          color='skyblue', alpha=0.7)

            # 设置标签
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('重要性')
            ax.set_title(f'特征重要性 ({method.upper()}方法)')
            ax.invert_yaxis()  # 重要性从高到低

            # 添加数值标签
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                ax.text(bar.get_width(), i, f'{importance:.4f}',
                       va='center', ha='left', fontsize=8)

            plt.tight_layout()
            logger.info("特征重要性图绘制完成")
            return fig

        except Exception as e:
            logger.error(f"特征重要性图绘制失败: {e}")
            raise ModelExplainerError(f"特征重要性图绘制失败: {e}")

    def plot_shap_summary(self, X: pd.DataFrame, plot_type: str = 'dot',
                         figsize: tuple = (12, 8)) -> plt.Figure:
        """
        绘制SHAP摘要图

        Args:
            X: 特征数据
            plot_type: 图形类型 ('dot', 'bar', 'violin')
            figsize: 图形大小

        Returns:
            matplotlib图形对象
        """
        try:
            logger.info(f"绘制SHAP摘要图，类型: {plot_type}")

            # 初始化SHAP解释器
            if self.explainer is None:
                self._initialize_shap_explainer(X)

            # 限制数据大小
            max_samples = min(1000, len(X))
            X_sample = X.sample(n=max_samples, random_state=42) if len(X) > max_samples else X

            # 计算SHAP值
            shap_values = self.explainer.shap_values(X_sample)

            # 处理SHAP值
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # 创建图形
            fig, ax = plt.subplots(figsize=figsize)

            # 绘制不同类型的图
            if plot_type == 'bar':
                shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
            elif plot_type == 'violin':
                shap.summary_plot(shap_values, X_sample, plot_type='violin', show=False)
            else:  # dot plot
                shap.summary_plot(shap_values, X_sample, show=False)

            plt.tight_layout()
            logger.info("SHAP摘要图绘制完成")
            return fig

        except Exception as e:
            logger.error(f"SHAP摘要图绘制失败: {e}")
            raise ModelExplainerError(f"SHAP摘要图绘制失败: {e}")

    def plot_prediction_explanation(self, X: pd.DataFrame, sample_idx: int = 0,
                                  figsize: tuple = (12, 8)) -> plt.Figure:
        """
        绘制预测解释图

        Args:
            X: 特征数据
            sample_idx: 样本索引
            figsize: 图形大小

        Returns:
            matplotlib图形对象
        """
        try:
            logger.info(f"绘制预测解释图，样本索引: {sample_idx}")

            # 获取预测解释
            explanation = self.explain_prediction(X, sample_idx)

            # 创建图形
            fig, ax = plt.subplots(figsize=figsize)

            # 准备数据
            features = explanation['feature_names']
            shap_values = explanation['shap_values']
            feature_values = explanation['feature_values']

            # 创建瀑布图数据
            y_pos = np.arange(len(features))

            # 绘制条形图
            colors = ['red' if val < 0 else 'blue' for val in shap_values]
            bars = ax.barh(y_pos, shap_values, color=colors, alpha=0.7)

            # 设置标签
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{feat}\n({val:.2f})" for feat, val in zip(features, feature_values)])
            ax.set_xlabel('SHAP值')
            ax.set_title(f'预测解释 - 样本 {sample_idx}')
            ax.invert_yaxis()

            # 添加基准线
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            # 添加数值标签
            for i, (bar, shap_val) in enumerate(zip(bars, shap_values)):
                ax.text(bar.get_width(), i, f'{shap_val:.4f}',
                       va='center', ha='left' if shap_val >= 0 else 'right', fontsize=8)

            plt.tight_layout()
            logger.info("预测解释图绘制完成")
            return fig

        except Exception as e:
            logger.error(f"预测解释图绘制失败: {e}")
            raise ModelExplainerError(f"预测解释图绘制失败: {e}")

# 便捷函数
def create_model_explainer(model, feature_names: List[str]) -> ModelExplainer:
    """
    创建模型解释器的便捷函数

    Args:
        model: 机器学习模型
        feature_names: 特征名称列表

    Returns:
        ModelExplainer实例
    """
    return ModelExplainer(model, feature_names)

def explain_model_predictions(model, X: pd.DataFrame, feature_names: List[str],
                            method: str = 'default', sample_idx: int = 0) -> Dict[str, Any]:
    """
    解释模型预测的便捷函数

    Args:
        model: 机器学习模型
        X: 特征数据
        feature_names: 特征名称列表
        method: 解释方法
        sample_idx: 样本索引

    Returns:
        解释结果字典
    """
    explainer = ModelExplainer(model, feature_names)
    results = {}

    # 计算特征重要性
    results['feature_importance'] = explainer.calculate_feature_importance(X, method=method)

    # 解释单个预测
    results['prediction_explanation'] = explainer.explain_prediction(X, sample_idx)

    # 生成模型摘要
    results['model_summary'] = explainer.generate_model_summary(X)

    return results

# 性能监控装饰器
def performance_monitor(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        """方法描述"""
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            logger.info(f"{func.__name__} 执行时间: {end_time - start_time:.2f}秒")
            logger.info(f"{func.__name__} 内存变化: {end_memory - start_memory:.2f}MB")

            return result
        except Exception as e:
            end_time = time.time()
            logger.error(f"{func.__name__} 执行失败，耗时: {end_time - start_time:.2f}秒")
            raise

    return wrapper

# Windows环境特定的错误处理
def windows_error_handler(func):
    """Windows环境错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        """方法描述"""
            return func(*args, **kwargs)
        except Exception as e:
            # Windows特定的错误处理
            error_msg = str(e).lower()
            if 'cuda' in error_msg or 'driver' in error_msg:
                logger.warning(f"Windows CUDA错误: {e}")
                logger.info("建议检查CUDA驱动版本和PyTorch兼容性")
                logger.info("可以尝试设置 feature_params.windows_cuda_workaround = true")

            raise

    return wrapper

if __name__ == '__main__':
    # 测试代码
    print("测试ModelExplainer模块...")

    # 创建简单的测试模型
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor

        # 创建测试数据
        np.random.seed(42)
        X_test = pd.DataFrame(np.random.rand(100, 5),
                             columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        y_test = pd.Series(np.random.rand(100))

        # 训练简单模型
        test_model = RandomForestRegressor(n_estimators=10, random_state=42)
        test_model.fit(X_test, y_test)

        # 创建解释器
        explainer = ModelExplainer(test_model, X_test.columns.tolist())

        # 测试功能
        print("测试默认特征重要性...")
        importance = explainer.calculate_feature_importance(X_test, method='default')
        print(f"重要性结果形状: {importance.shape}")

        print("测试预测解释...")
        explanation = explainer.explain_prediction(X_test, sample_idx=0)
        print(f"解释结果键: {list(explanation.keys())}")

        print("测试模型摘要...")
        summary = explainer.generate_model_summary(X_test, y_test)
        print(f"摘要键: {list(summary.keys())}")

        print("ModelExplainer模块测试完成!")

    except Exception as e:
        print(f"测试失败: {e}")
        logger.error(f"ModelExplainer测试失败: {e}")

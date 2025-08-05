import base64
import io
import json
import logging
import pickle
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sqlalchemy import create_engine, text

from src.utils.db import get_db_manager

"""模型解释器

提供AI模型决策解释功能，帮助用户理解模型预测的原因。
"""

warnings.filterwarnings("ignore")


# 机器学习解释性工具
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP未安装，部分解释功能将不可用")

try:
    import lime
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME未安装，部分解释功能将不可用")

try:
    from sklearn.inspection import permutation_importance

    PERMUTATION_AVAILABLE = True
except ImportError:
    PERMUTATION_AVAILABLE = False
    logging.warning("sklearn版本过低，排列重要性功能不可用")

from .model_manager import AIModelManager

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class ExplanationResult:
    """解释结果"""

    stock_code: str
    model_name: str
    model_version: str
    explanation_type: str  # 'shap', 'lime', 'permutation'
    feature_importances: Dict[str, float]
    prediction_value: float
    base_value: Optional[float] = None
    explanation_data: Optional[Dict[str, Any]] = None
    visualization_data: Optional[str] = None  # base64编码的图片
    created_at: Optional[datetime] = None


@dataclass
class GlobalExplanation:
    """全局解释结果"""

    model_name: str
    model_version: str
    explanation_type: str
    feature_importances: Dict[str, float]
    feature_interactions: Optional[Dict[str, Dict[str, float]]] = None
    summary_statistics: Optional[Dict[str, Any]] = None
    visualization_data: Optional[str] = None
    created_at: Optional[datetime] = None


class ModelExplainer:
    """模型解释器

    主要功能：
    - SHAP值计算和可视化
    - LIME局部解释
    - 排列重要性分析
    - 特征交互分析
    - 全局和局部解释
    - 解释结果存储和查询
    """

    def __init__(self):
        """方法描述"""
        self.model_manager = AIModelManager()

        # 解释器缓存
        self.explainer_cache = {}
        self.cache_ttl = 3600  # 1小时缓存

        self._ensure_tables_exist()

    def _ensure_tables_exist(self):
        """确保解释相关表存在"""
        create_tables_sql = """
        -- 模型解释结果表
        CREATE TABLE IF NOT EXISTS model_explanations (
            id SERIAL PRIMARY KEY,
            stock_code VARCHAR(20),
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            explanation_type VARCHAR(20) NOT NULL,
            prediction_value DECIMAL(15, 6),
            base_value DECIMAL(15, 6),
            feature_importances JSONB,
            explanation_data JSONB,
            visualization_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 全局解释结果表
        CREATE TABLE IF NOT EXISTS global_explanations (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            explanation_type VARCHAR(20) NOT NULL,
            feature_importances JSONB,
            feature_interactions JSONB,
            summary_statistics JSONB,
            visualization_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model_name, model_version, explanation_type)
        );

        -- 解释任务表
        CREATE TABLE IF NOT EXISTS explanation_tasks (
            id SERIAL PRIMARY KEY,
            task_id VARCHAR(100) UNIQUE NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20),
            explanation_type VARCHAR(20) NOT NULL,
            task_type VARCHAR(20) NOT NULL, -- 'local', 'global'
            stock_codes TEXT[],
            parameters JSONB,
            status VARCHAR(20) DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            result_count INTEGER DEFAULT 0,
            error_message TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_model_explanations_model ON model_explanations(model_name, model_version);
        CREATE INDEX IF NOT EXISTS idx_model_explanations_stock ON model_explanations(stock_code, created_at);
        CREATE INDEX IF NOT EXISTS idx_global_explanations_model ON global_explanations(model_name, model_version);
        CREATE INDEX IF NOT EXISTS idx_explanation_tasks_status ON explanation_tasks(status, created_at);
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
            logger.info("模型解释表创建成功")
        except Exception as e:
            logger.error(f"创建模型解释表失败: {e}")
            raise

    def _get_explainer_cache_key(self, model_name: str, model_version: str, explanation_type: str) -> str:
        """生成解释器缓存键"""
        return f"explainer:{model_name}:{model_version}:{explanation_type}"

    def _load_model_data(self, model_name: str, model_version: str = None) -> Optional[Dict[str, Any]]:
        """加载模型数据"""
        try:
            return self.model_manager.load_model_for_prediction(model_name, model_version)
        except Exception as e:
            logger.error(f"加载模型数据失败: {e}")
            return None

    def _get_training_data_sample(
        self, model_name: str, model_version: str, sample_size: int = 1000
    ) -> Optional[pd.DataFrame]:
        """获取训练数据样本用于解释器初始化"""
        try:
            # 从数据库获取最近的因子数据作为训练数据样本
            query_sql = """
            SELECT *
            FROM factor_data
            WHERE date >= (SELECT MAX(date) - INTERVAL '30 days' FROM factor_data)
            AND stock_code IN (
                SELECT stock_code
                FROM stock_info
                WHERE is_active = true
                ORDER BY RANDOM()
                LIMIT :sample_size
            )
            ORDER BY date DESC, stock_code
            LIMIT :sample_size
            """

            df = pd.read_sql(query_sql, self.engine, params={"sample_size": sample_size})

            if df.empty:
                logger.warning("没有找到训练数据样本")
                return None

            return df

        except Exception as e:
            logger.error(f"获取训练数据样本失败: {e}")
            return None

    def _create_shap_explainer(self, model_data: Dict[str, Any], background_data: pd.DataFrame) -> Optional[Any]:
        """创建SHAP解释器"""
        if not SHAP_AVAILABLE:
            logger.error("SHAP未安装")
            return None

        try:
            model = model_data["model"]
            feature_columns = model_data["feature_columns"]

            # 准备背景数据
            X_background = background_data[feature_columns].fillna(0)

            # 标准化（如果模型有标准化器）
            if "scaler" in model_data and model_data["scaler"] is not None:
                X_background = model_data["scaler"].transform(X_background)

            # 根据模型类型选择合适的解释器
            model_type = model_data["version_info"].model_type.lower()

            if (
                "tree" in model_type
                or "forest" in model_type
                or "gbm" in model_type
                or "xgb" in model_type
                or "lgb" in model_type
            ):
                # 树模型使用TreeExplainer
                explainer = shap.TreeExplainer(model)
            else:
                # 其他模型使用KernelExplainer
                explainer = shap.KernelExplainer(model.predict, X_background.sample(min(100, len(X_background))))

            return explainer

        except Exception as e:
            logger.error(f"创建SHAP解释器失败: {e}")
            return None

    def _create_lime_explainer(self, model_data: Dict[str, Any], background_data: pd.DataFrame) -> Optional[Any]:
        """创建LIME解释器"""
        if not LIME_AVAILABLE:
            logger.error("LIME未安装")
            return None

        try:
            feature_columns = model_data["feature_columns"]
            X_background = background_data[feature_columns].fillna(0)

            # 标准化（如果模型有标准化器）
            if "scaler" in model_data and model_data["scaler"] is not None:
                X_background = model_data["scaler"].transform(X_background)

            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_background.values,
                feature_names=feature_columns,
                mode="regression",
                training_labels=None,
                verbose=False,
            )

            return explainer

        except Exception as e:
            logger.error(f"创建LIME解释器失败: {e}")
            return None

    def _get_or_create_explainer(self, model_name: str, model_version: str, explanation_type: str) -> Optional[Any]:
        """获取或创建解释器（带缓存）"""
        cache_key = self._get_explainer_cache_key(model_name, model_version, explanation_type)

        # 检查缓存
        if cache_key in self.explainer_cache:
            cache_time, explainer = self.explainer_cache[cache_key]
            if datetime.now().timestamp() - cache_time < self.cache_ttl:
                return explainer
            else:
                del self.explainer_cache[cache_key]

        # 加载模型数据
        model_data = self._load_model_data(model_name, model_version)
        if not model_data:
            return None

        # 获取背景数据
        background_data = self._get_training_data_sample(model_name, model_version)
        if background_data is None:
            return None

        # 创建解释器
        explainer = None
        if explanation_type == "shap":
            explainer = self._create_shap_explainer(model_data, background_data)
        elif explanation_type == "lime":
            explainer = self._create_lime_explainer(model_data, background_data)

        if explainer:
            # 缓存解释器
            self.explainer_cache[cache_key] = (datetime.now().timestamp(), explainer)
            logger.info(f"解释器创建成功: {explanation_type} for {model_name} v{model_version}")

        return explainer

    def explain_prediction_shap(
        self, stock_code: str, model_name: str, model_version: str = None, prediction_date: str = None
    ) -> Optional[ExplanationResult]:
        """使用SHAP解释单个预测

        Args:
            stock_code: 股票代码
            model_name: 模型名称
            model_version: 模型版本
            prediction_date: 预测日期

        Returns:
            解释结果
        """
        try:
            # 获取SHAP解释器
            explainer = self._get_or_create_explainer(model_name, model_version, "shap")
            if not explainer:
                return None

            # 加载模型数据
            model_data = self._load_model_data(model_name, model_version)
            if not model_data:
                return None

            # 获取股票因子数据
            if prediction_date is None:
                prediction_date = datetime.now().strftime("%Y-%m-%d")

            query_sql = """
            SELECT *
            FROM factor_data
            WHERE stock_code = :stock_code
            AND date = :prediction_date
            """

            stock_data = pd.read_sql(
                query_sql, self.engine, params={"stock_code": stock_code, "prediction_date": prediction_date}
            )

            if stock_data.empty:
                logger.warning(f"没有找到股票因子数据: {stock_code} {prediction_date}")
                return None

            # 准备特征数据
            feature_columns = model_data["feature_columns"]
            X = stock_data[feature_columns].fillna(0).values

            # 标准化
            if "scaler" in model_data and model_data["scaler"] is not None:
                X = model_data["scaler"].transform(X)

            # 计算SHAP值
            shap_values = explainer.shap_values(X)

            # 处理SHAP值（可能是多维的）
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # 取第一个类别的SHAP值

            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # 取第一个样本的SHAP值

            # 获取基准值
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
            else:
                base_value = float(base_value)

            # 计算预测值
            model = model_data["model"]
            prediction_value = float(model.predict(X)[0])

            # 创建特征重要性字典
            feature_importances = dict(zip(feature_columns, shap_values.tolist()))

            # 生成可视化
            visualization_data = self._create_shap_visualization(shap_values, feature_columns, base_value, stock_code)

            # 创建解释结果
            result = ExplanationResult(
                stock_code=stock_code,
                model_name=model_name,
                model_version=model_version or model_data["version_info"].version,
                explanation_type="shap",
                feature_importances=feature_importances,
                prediction_value=prediction_value,
                base_value=base_value,
                explanation_data={
                    "shap_values": shap_values.tolist(),
                    "feature_values": X[0].tolist(),
                    "prediction_date": prediction_date,
                },
                visualization_data=visualization_data,
                created_at=datetime.now(),
            )

            # 保存解释结果
            self._save_explanation_result(result)

            return result

        except Exception as e:
            logger.error(f"SHAP解释失败: {e}")
            return None

    def explain_prediction_lime(
        self,
        stock_code: str,
        model_name: str,
        model_version: str = None,
        prediction_date: str = None,
        num_features: int = 10,
    ) -> Optional[ExplanationResult]:
        """使用LIME解释单个预测

        Args:
            stock_code: 股票代码
            model_name: 模型名称
            model_version: 模型版本
            prediction_date: 预测日期
            num_features: 显示的特征数量

        Returns:
            解释结果
        """
        try:
            # 获取LIME解释器
            explainer = self._get_or_create_explainer(model_name, model_version, "lime")
            if not explainer:
                return None

            # 加载模型数据
            model_data = self._load_model_data(model_name, model_version)
            if not model_data:
                return None

            # 获取股票因子数据
            if prediction_date is None:
                prediction_date = datetime.now().strftime("%Y-%m-%d")

            query_sql = """
            SELECT *
            FROM factor_data
            WHERE stock_code = :stock_code
            AND date = :prediction_date
            """

            stock_data = pd.read_sql(
                query_sql, self.engine, params={"stock_code": stock_code, "prediction_date": prediction_date}
            )

            if stock_data.empty:
                logger.warning(f"没有找到股票因子数据: {stock_code} {prediction_date}")
                return None

            # 准备特征数据
            feature_columns = model_data["feature_columns"]
            X = stock_data[feature_columns].fillna(0).values

            # 标准化
            if "scaler" in model_data and model_data["scaler"] is not None:
                X = model_data["scaler"].transform(X)

            # 创建预测函数
            model = model_data["model"]

            def predict_fn(x):
                """方法描述"""

            # 生成LIME解释
            explanation = explainer.explain_instance(X[0], predict_fn, num_features=num_features)

            # 获取特征重要性
            feature_importances = dict(explanation.as_list())

            # 计算预测值
            prediction_value = float(model.predict(X)[0])

            # 生成可视化
            visualization_data = self._create_lime_visualization(explanation, stock_code)

            # 创建解释结果
            result = ExplanationResult(
                stock_code=stock_code,
                model_name=model_name,
                model_version=model_version or model_data["version_info"].version,
                explanation_type="lime",
                feature_importances=feature_importances,
                prediction_value=prediction_value,
                explanation_data={
                    "lime_explanation": explanation.as_list(),
                    "feature_values": X[0].tolist(),
                    "prediction_date": prediction_date,
                    "num_features": num_features,
                },
                visualization_data=visualization_data,
                created_at=datetime.now(),
            )

            # 保存解释结果
            self._save_explanation_result(result)

            return result

        except Exception as e:
            logger.error(f"LIME解释失败: {e}")
            return None

    def explain_prediction_permutation(
        self, stock_code: str, model_name: str, model_version: str = None, prediction_date: str = None
    ) -> Optional[ExplanationResult]:
        """使用排列重要性解释预测

        Args:
            stock_code: 股票代码
            model_name: 模型名称
            model_version: 模型版本
            prediction_date: 预测日期

        Returns:
            解释结果
        """
        if not PERMUTATION_AVAILABLE:
            logger.error("排列重要性功能不可用")
            return None

        try:
            # 加载模型数据
            model_data = self._load_model_data(model_name, model_version)
            if not model_data:
                return None

            # 获取测试数据
            background_data = self._get_training_data_sample(model_name, model_version, 500)
            if background_data is None:
                return None

            # 准备特征数据
            feature_columns = model_data["feature_columns"]
            X_test = background_data[feature_columns].fillna(0)

            # 标准化
            if "scaler" in model_data and model_data["scaler"] is not None:
                X_test = model_data["scaler"].transform(X_test)

            # 创建虚拟目标变量（用于计算重要性）
            model = model_data["model"]
            y_test = model.predict(X_test)

            # 计算排列重要性
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

            # 创建特征重要性字典
            feature_importances = dict(zip(feature_columns, perm_importance.importances_mean.tolist()))

            # 获取股票预测值
            if prediction_date is None:
                prediction_date = datetime.now().strftime("%Y-%m-%d")

            query_sql = """
            SELECT *
            FROM factor_data
            WHERE stock_code = :stock_code
            AND date = :prediction_date
            """

            stock_data = pd.read_sql(
                query_sql, self.engine, params={"stock_code": stock_code, "prediction_date": prediction_date}
            )

            prediction_value = 0.0
            if not stock_data.empty:
                X_stock = stock_data[feature_columns].fillna(0).values
                if "scaler" in model_data and model_data["scaler"] is not None:
                    X_stock = model_data["scaler"].transform(X_stock)
                prediction_value = float(model.predict(X_stock)[0])

            # 生成可视化
            visualization_data = self._create_permutation_visualization(feature_importances, stock_code)

            # 创建解释结果
            result = ExplanationResult(
                stock_code=stock_code,
                model_name=model_name,
                model_version=model_version or model_data["version_info"].version,
                explanation_type="permutation",
                feature_importances=feature_importances,
                prediction_value=prediction_value,
                explanation_data={
                    "importances_mean": perm_importance.importances_mean.tolist(),
                    "importances_std": perm_importance.importances_std.tolist(),
                    "prediction_date": prediction_date,
                },
                visualization_data=visualization_data,
                created_at=datetime.now(),
            )

            # 保存解释结果
            self._save_explanation_result(result)

            return result

        except Exception as e:
            logger.error(f"排列重要性解释失败: {e}")
            return None

    def create_global_explanation(
        self, model_name: str, model_version: str = None, explanation_type: str = "shap", sample_size: int = 1000
    ) -> Optional[GlobalExplanation]:
        """创建全局模型解释

        Args:
            model_name: 模型名称
            model_version: 模型版本
            explanation_type: 解释类型
            sample_size: 样本大小

        Returns:
            全局解释结果
        """
        try:
            # 加载模型数据
            model_data = self._load_model_data(model_name, model_version)
            if not model_data:
                return None

            if model_version is None:
                model_version = model_data["version_info"].version

            # 获取样本数据
            sample_data = self._get_training_data_sample(model_name, model_version, sample_size)
            if sample_data is None:
                return None

            feature_columns = model_data["feature_columns"]
            X_sample = sample_data[feature_columns].fillna(0)

            # 标准化
            if "scaler" in model_data and model_data["scaler"] is not None:
                X_sample = model_data["scaler"].transform(X_sample)

            feature_importances = {}
            feature_interactions = None
            summary_statistics = {}

            if explanation_type == "shap" and SHAP_AVAILABLE:
                # 使用SHAP进行全局解释
                explainer = self._get_or_create_explainer(model_name, model_version, "shap")
                if explainer:
                    shap_values = explainer.shap_values(X_sample)

                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]

                    # 计算平均绝对SHAP值作为特征重要性
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                    feature_importances = dict(zip(feature_columns, mean_abs_shap.tolist()))

                    # 计算统计信息
                    summary_statistics = {
                        "mean_prediction": float(np.mean(explainer.expected_value)),
                        "feature_count": len(feature_columns),
                        "sample_size": len(X_sample),
                        "top_features": sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:10],
                    }

            elif explanation_type == "permutation" and PERMUTATION_AVAILABLE:
                # 使用排列重要性进行全局解释
                model = model_data["model"]
                y_sample = model.predict(X_sample)

                perm_importance = permutation_importance(
                    model, X_sample, y_sample, n_repeats=10, random_state=42, n_jobs=-1
                )

                feature_importances = dict(zip(feature_columns, perm_importance.importances_mean.tolist()))

                summary_statistics = {
                    "feature_count": len(feature_columns),
                    "sample_size": len(X_sample),
                    "importance_std": dict(zip(feature_columns, perm_importance.importances_std.tolist())),
                    "top_features": sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:10],
                }

            # 生成可视化
            visualization_data = self._create_global_visualization(feature_importances, model_name, explanation_type)

            # 创建全局解释结果
            result = GlobalExplanation(
                model_name=model_name,
                model_version=model_version,
                explanation_type=explanation_type,
                feature_importances=feature_importances,
                feature_interactions=feature_interactions,
                summary_statistics=summary_statistics,
                visualization_data=visualization_data,
                created_at=datetime.now(),
            )

            # 保存全局解释结果
            self._save_global_explanation(result)

            return result

        except Exception as e:
            logger.error(f"创建全局解释失败: {e}")
            return None

    def _create_shap_visualization(
        self, shap_values: np.ndarray, feature_names: List[str], base_value: float, stock_code: str
    ) -> Optional[str]:
        """创建SHAP可视化"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # 排序特征重要性
            abs_shap = np.abs(shap_values)
            sorted_indices = np.argsort(abs_shap)[-10:]  # 取前10个重要特征

            sorted_shap = shap_values[sorted_indices]
            sorted_features = [feature_names[i] for i in sorted_indices]

            # 创建水平条形图
            colors = ["red" if x < 0 else "blue" for x in sorted_shap]
            bars = ax.barh(range(len(sorted_shap)), sorted_shap, color=colors, alpha=0.7)

            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel("SHAP值")
            ax.set_title(f"{stock_code} - SHAP特征重要性\n基准值: {base_value:.4f}")
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

            # 添加数值标签
            for i, (bar, value) in enumerate(zip(bars, sorted_shap)):
                ax.text(
                    value + (0.01 if value >= 0 else -0.01),
                    i,
                    f"{value:.4f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                )

            plt.tight_layout()

            # 转换为base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            logger.error(f"创建SHAP可视化失败: {e}")
            return None

    def _create_lime_visualization(self, explanation: Any, stock_code: str) -> Optional[str]:
        """创建LIME可视化"""
        try:
            # 获取特征重要性
            feature_importance = explanation.as_list()

            if not feature_importance:
                return None

            # 排序并取前10个
            feature_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)[:10]

            features = [item[0] for item in feature_importance]
            values = [item[1] for item in feature_importance]

            fig, ax = plt.subplots(figsize=(10, 6))

            # 创建水平条形图
            colors = ["red" if x < 0 else "blue" for x in values]
            bars = ax.barh(range(len(values)), values, color=colors, alpha=0.7)

            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel("LIME重要性")
            ax.set_title(f"{stock_code} - LIME特征重要性")
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

            # 添加数值标签
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(
                    value + (0.01 if value >= 0 else -0.01),
                    i,
                    f"{value:.4f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                )

            plt.tight_layout()

            # 转换为base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            logger.error(f"创建LIME可视化失败: {e}")
            return None

    def _create_permutation_visualization(
        self, feature_importances: Dict[str, float], stock_code: str
    ) -> Optional[str]:
        """创建排列重要性可视化"""
        try:
            # 排序并取前10个
            sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

            features = [item[0] for item in sorted_features]
            values = [item[1] for item in sorted_features]

            fig, ax = plt.subplots(figsize=(10, 6))

            # 创建水平条形图
            colors = ["red" if x < 0 else "blue" for x in values]
            bars = ax.barh(range(len(values)), values, color=colors, alpha=0.7)

            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel("排列重要性")
            ax.set_title(f"{stock_code} - 排列重要性分析")
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

            # 添加数值标签
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(
                    value + (0.01 if value >= 0 else -0.01),
                    i,
                    f"{value:.4f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                )

            plt.tight_layout()

            # 转换为base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            logger.error(f"创建排列重要性可视化失败: {e}")
            return None

    def _create_global_visualization(
        self, feature_importances: Dict[str, float], model_name: str, explanation_type: str
    ) -> Optional[str]:
        """创建全局解释可视化"""
        try:
            # 排序并取前15个
            sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:15]

            features = [item[0] for item in sorted_features]
            values = [item[1] for item in sorted_features]

            fig, ax = plt.subplots(figsize=(12, 8))

            # 创建水平条形图
            colors = ["red" if x < 0 else "blue" for x in values]
            bars = ax.barh(range(len(values)), values, color=colors, alpha=0.7)

            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel(f"{explanation_type.upper()}重要性")
            ax.set_title(f"{model_name} - 全局特征重要性 ({explanation_type.upper()})")
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

            # 添加数值标签
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(
                    value + (0.01 if value >= 0 else -0.01),
                    i,
                    f"{value:.4f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                )

            plt.tight_layout()

            # 转换为base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            logger.error(f"创建全局可视化失败: {e}")
            return None

    def _save_explanation_result(self, result: ExplanationResult):
        """保存解释结果"""
        try:
            insert_sql = """
            INSERT INTO model_explanations (
                stock_code, model_name, model_version, explanation_type,
                prediction_value, base_value, feature_importances,
                explanation_data, visualization_data
            ) VALUES (
                :stock_code, :model_name, :model_version, :explanation_type,
                :prediction_value, :base_value, :feature_importances,
                :explanation_data, :visualization_data
            )
            """

            with self.engine.connect() as conn:
                conn.execute(
                    text(insert_sql),
                    {
                        "stock_code": result.stock_code,
                        "model_name": result.model_name,
                        "model_version": result.model_version,
                        "explanation_type": result.explanation_type,
                        "prediction_value": result.prediction_value,
                        "base_value": result.base_value,
                        "feature_importances": json.dumps(result.feature_importances),
                        "explanation_data": json.dumps(result.explanation_data, default=str),
                        "visualization_data": result.visualization_data,
                    },
                )
                conn.commit()

        except Exception as e:
            logger.error(f"保存解释结果失败: {e}")

    def _save_global_explanation(self, result: GlobalExplanation):
        """保存全局解释结果"""
        try:
            insert_sql = """
            INSERT INTO global_explanations (
                model_name, model_version, explanation_type,
                feature_importances, feature_interactions,
                summary_statistics, visualization_data
            ) VALUES (
                :model_name, :model_version, :explanation_type,
                :feature_importances, :feature_interactions,
                :summary_statistics, :visualization_data
            )
            ON CONFLICT (model_name, model_version, explanation_type)
            DO UPDATE SET
                feature_importances = EXCLUDED.feature_importances,
                feature_interactions = EXCLUDED.feature_interactions,
                summary_statistics = EXCLUDED.summary_statistics,
                visualization_data = EXCLUDED.visualization_data,
                created_at = CURRENT_TIMESTAMP
            """

            with self.engine.connect() as conn:
                conn.execute(
                    text(insert_sql),
                    {
                        "model_name": result.model_name,
                        "model_version": result.model_version,
                        "explanation_type": result.explanation_type,
                        "feature_importances": json.dumps(result.feature_importances),
                        "feature_interactions": json.dumps(result.feature_interactions or {}),
                        "summary_statistics": json.dumps(result.summary_statistics, default=str),
                        "visualization_data": result.visualization_data,
                    },
                )
                conn.commit()

        except Exception as e:
            logger.error(f"保存全局解释结果失败: {e}")

    def get_explanation_history(
        self, stock_code: str, model_name: str, explanation_type: str = None, days: int = 30
    ) -> pd.DataFrame:
        """获取解释历史

        Args:
            stock_code: 股票代码
            model_name: 模型名称
            explanation_type: 解释类型
            days: 历史天数

        Returns:
            解释历史DataFrame
        """
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            where_clause = "WHERE stock_code = :stock_code AND model_name = :model_name AND created_at >= :start_date"
            params = {"stock_code": stock_code, "model_name": model_name, "start_date": start_date}

            if explanation_type:
                where_clause += " AND explanation_type = :explanation_type"
                params["explanation_type"] = explanation_type

            query_sql = f"""
            SELECT id, model_version, explanation_type, prediction_value,
                   base_value, feature_importances, created_at
            FROM model_explanations
            {where_clause}
            ORDER BY created_at DESC
            """

            df = pd.read_sql(query_sql, self.engine, params=params)
            return df

        except Exception as e:
            logger.error(f"获取解释历史失败: {e}")
            return pd.DataFrame()

    def get_global_explanation(
        self, model_name: str, model_version: str = None, explanation_type: str = "shap"
    ) -> Optional[GlobalExplanation]:
        """获取全局解释

        Args:
            model_name: 模型名称
            model_version: 模型版本
            explanation_type: 解释类型

        Returns:
            全局解释结果
        """
        try:
            where_clause = "WHERE model_name = :model_name AND explanation_type = :explanation_type"
            params = {"model_name": model_name, "explanation_type": explanation_type}

            if model_version:
                where_clause += " AND model_version = :model_version"
                params["model_version"] = model_version

            query_sql = f"""
            SELECT *
            FROM global_explanations
            {where_clause}
            ORDER BY created_at DESC
            LIMIT 1
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                row = result.fetchone()

            if row:
                return GlobalExplanation(
                    model_name=row[1],
                    model_version=row[2],
                    explanation_type=row[3],
                    feature_importances=json.loads(row[4]),
                    feature_interactions=json.loads(row[5]) if row[5] else None,
                    summary_statistics=json.loads(row[6]) if row[6] else None,
                    visualization_data=row[7],
                    created_at=row[8],
                )

            return None

        except Exception as e:
            logger.error(f"获取全局解释失败: {e}")
            return None

    def clear_explanation_cache(self) -> int:
        """清理解释器缓存

        Returns:
            清理的缓存数量
        """
        try:
            count = len(self.explainer_cache)
            self.explainer_cache.clear()
            logger.info(f"解释器缓存清理完成: {count}个")
            return count

        except Exception as e:
            logger.error(f"清理解释器缓存失败: {e}")
            return 0

    def get_explanation_summary(self, model_name: str, days: int = 7) -> Dict[str, Any]:
        """获取解释统计摘要

        Args:
            model_name: 模型名称
            days: 统计天数

        Returns:
            统计摘要
        """
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            query_sql = """
            SELECT
                explanation_type,
                COUNT(*) as explanation_count,
                COUNT(DISTINCT stock_code) as unique_stocks,
                AVG(prediction_value) as avg_prediction
            FROM model_explanations
            WHERE model_name = :model_name
            AND created_at >= :start_date
            GROUP BY explanation_type
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {"model_name": model_name, "start_date": start_date})
                rows = result.fetchall()

            summary = {
                "model_name": model_name,
                "period_days": days,
                "explanation_types": {},
                "total_explanations": 0,
                "total_unique_stocks": 0,
            }

            unique_stocks = set()
            for row in rows:
                explanation_type = row[0]
                count = int(row[1])
                stocks = int(row[2])
                avg_pred = float(row[3] or 0)

                summary["explanation_types"][explanation_type] = {
                    "count": count,
                    "unique_stocks": stocks,
                    "avg_prediction": avg_pred,
                }

                summary["total_explanations"] += count
                unique_stocks.add(stocks)

            summary["total_unique_stocks"] = len(unique_stocks)

            return summary

        except Exception as e:
            logger.error(f"获取解释统计摘要失败: {e}")
            return {}

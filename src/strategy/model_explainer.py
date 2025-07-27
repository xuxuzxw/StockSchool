import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import shap
import warnings
warnings.filterwarnings('ignore')
import sys
import os
from loguru import logger
from src.utils.config_loader import config

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class ModelExplainer:
    """
    模型解释器 - 解释机器学习模型的预测结果
    """
    
    def __init__(self, model, feature_names: List[str] = None):
        """
        初始化模型解释器
        
        Args:
            model: 训练好的机器学习模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series = None, 
                                   method: str = 'default') -> pd.DataFrame:
        """
        计算特征重要性
        
        Args:
            X: 特征数据
            y: 目标变量（用于排列重要性）
            method: 计算方法 ('default', 'permutation', 'shap')
        
        Returns:
            特征重要性DataFrame
        """
        if method == 'default':
            return self._get_default_importance(X)
        elif method == 'permutation':
            return self._get_permutation_importance(X, y)
        elif method == 'shap':
            return self._get_shap_importance(X)
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def _get_default_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        获取模型默认的特征重要性
        
        Args:
            X: 特征数据
        
        Returns:
            特征重要性DataFrame
        """
        feature_names = self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        if hasattr(self.model, 'feature_importances_'):
            # 树模型等
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # 线性模型
            importance = np.abs(self.model.coef_).flatten()
        else:
            logger.warning("模型不支持默认特征重要性，返回空结果")
            importance = np.zeros(len(feature_names))
        
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def _get_permutation_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        计算排列重要性
        
        Args:
            X: 特征数据
            y: 目标变量
        
        Returns:
            特征重要性DataFrame
        """
        if y is None:
            raise ValueError("排列重要性需要提供目标变量y")
        
        feature_names = self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        try:
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=10, random_state=42
            )
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        except Exception as e:
            logger.error(f"计算排列重要性失败: {e}")
            return pd.DataFrame(columns=['feature', 'importance', 'std'])
    
    def _get_shap_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        计算SHAP重要性
        
        Args:
            X: 特征数据
        
        Returns:
            特征重要性DataFrame
        """
        try:
            # 初始化SHAP解释器
            if self.explainer is None:
                self._initialize_shap_explainer(X)
            
            # 计算SHAP值
            if self.shap_values is None:
                self.shap_values = self.explainer.shap_values(X)
            
            # 如果是多分类，取第一类的SHAP值
            shap_vals = self.shap_values
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            
            # 计算特征重要性（SHAP值绝对值的均值）
            feature_names = self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            importance = np.abs(shap_vals).mean(axis=0)
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        except Exception as e:
            logger.error(f"计算SHAP重要性失败: {e}")
            return pd.DataFrame(columns=['feature', 'importance'])
    
    def _initialize_shap_explainer(self, X: pd.DataFrame):
        """
        初始化SHAP解释器
        
        Args:
            X: 特征数据
        """
        try:
            # 根据模型类型选择合适的解释器
            model_name = type(self.model).__name__.lower()
            
            if 'tree' in model_name or 'forest' in model_name or 'gbm' in model_name or 'xgb' in model_name or 'lgb' in model_name:
                # 树模型
                self.explainer = shap.TreeExplainer(self.model)
            elif 'linear' in model_name or 'logistic' in model_name:
                # 线性模型
                self.explainer = shap.LinearExplainer(self.model, X)
            else:
                # 通用解释器
                background_samples = config.get('feature_params.shap_background_samples', 100)
                self.explainer = shap.Explainer(self.model, X[:background_samples])  # 使用前N个样本作为背景
        
        except Exception as e:
            logger.warning(f"初始化SHAP解释器失败: {e}，使用通用解释器")
            try:
                background_samples = config.get('feature_params.shap_background_samples', 100)
                self.explainer = shap.Explainer(self.model.predict, X[:background_samples])
            except:
                logger.error("无法初始化SHAP解释器")
                self.explainer = None
    
    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        """
        解释单个预测结果
        
        Args:
            X: 特征数据
            sample_idx: 样本索引
        
        Returns:
            解释结果字典
        """
        if sample_idx >= len(X):
            raise ValueError(f"样本索引 {sample_idx} 超出范围")
        
        sample = X.iloc[sample_idx:sample_idx+1]
        prediction = self.model.predict(sample)[0]
        
        explanation = {
            'prediction': prediction,
            'sample_index': sample_idx,
            'feature_values': sample.iloc[0].to_dict()
        }
        
        # 添加SHAP解释
        try:
            if self.explainer is None:
                self._initialize_shap_explainer(X)
            
            if self.explainer is not None:
                shap_values = self.explainer.shap_values(sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                feature_names = self.feature_names or [f'feature_{i}' for i in range(len(shap_values[0]))]
                
                explanation['shap_values'] = dict(zip(feature_names, shap_values[0]))
                explanation['base_value'] = self.explainer.expected_value
                
                if isinstance(explanation['base_value'], np.ndarray):
                    explanation['base_value'] = explanation['base_value'][0]
        
        except Exception as e:
            logger.warning(f"SHAP解释失败: {e}")
        
        return explanation
    
    def analyze_feature_interactions(self, X: pd.DataFrame, max_features: int = 10) -> pd.DataFrame:
        """
        分析特征交互作用
        
        Args:
            X: 特征数据
            max_features: 最大特征数量
        
        Returns:
            特征交互作用DataFrame
        """
        try:
            if self.explainer is None:
                self._initialize_shap_explainer(X)
            
            if self.explainer is None:
                return pd.DataFrame()
            
            # 计算SHAP交互值
            interaction_samples = config.get('feature_params.interaction_samples', 100)
            shap_interaction_values = self.explainer.shap_interaction_values(X[:min(interaction_samples, len(X))])
            
            # 计算特征交互强度
            feature_names = self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            n_features = min(len(feature_names), max_features)
            
            interactions = []
            for i in range(n_features):
                for j in range(i+1, n_features):
                    interaction_strength = np.abs(shap_interaction_values[:, i, j]).mean()
                    interactions.append({
                        'feature_1': feature_names[i],
                        'feature_2': feature_names[j],
                        'interaction_strength': interaction_strength
                    })
            
            interaction_df = pd.DataFrame(interactions).sort_values(
                'interaction_strength', ascending=False
            )
            
            return interaction_df
        
        except Exception as e:
            logger.error(f"分析特征交互作用失败: {e}")
            return pd.DataFrame()
    
    def generate_model_summary(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Any]:
        """
        生成模型总结报告
        
        Args:
            X: 特征数据
            y: 目标变量
        
        Returns:
            模型总结字典
        """
        summary = {
            'model_type': type(self.model).__name__,
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        
        # 模型性能
        if y is not None:
            try:
                predictions = self.model.predict(X)
                summary['r2_score'] = r2_score(y, predictions)
                summary['mse'] = mean_squared_error(y, predictions)
                summary['rmse'] = np.sqrt(summary['mse'])
                
                # 交叉验证分数
                cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
                summary['cv_r2_mean'] = cv_scores.mean()
                summary['cv_r2_std'] = cv_scores.std()
            
            except Exception as e:
                logger.warning(f"计算模型性能失败: {e}")
        
        # 特征重要性
        try:
            importance_df = self.calculate_feature_importance(X, y, method='default')
            summary['top_features'] = importance_df.head(10).to_dict('records')
        except Exception as e:
            logger.warning(f"计算特征重要性失败: {e}")
        
        return summary
    
    def plot_feature_importance(self, X: pd.DataFrame, y: pd.Series = None, 
                               method: str = 'default', top_n: int = None, 
                               figsize: Tuple[int, int] = (10, 8)):
        """
        绘制特征重要性图
        
        Args:
            X: 特征数据
            y: 目标变量
            method: 计算方法
            top_n: 显示前N个特征
            figsize: 图表大小
        """
        if top_n is None:
            top_n = config.get('strategy_params.max_display_items', 20)
            
        importance_df = self.calculate_feature_importance(X, y, method)
        
        if importance_df.empty:
            print("无法计算特征重要性")
            return
        
        plt.figure(figsize=figsize)
        
        # 取前top_n个特征
        plot_data = importance_df.head(top_n)
        
        # 创建水平条形图
        bars = plt.barh(range(len(plot_data)), plot_data['importance'])
        plt.yticks(range(len(plot_data)), plot_data['feature'])
        plt.xlabel('重要性')
        plt.title(f'特征重要性 ({method}方法)')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.gca().invert_yaxis()  # 最重要的特征在顶部
        plt.tight_layout()
        plt.show()
    
    def plot_shap_summary(self, X: pd.DataFrame, plot_type: str = 'bar', 
                         max_display: int = None, figsize: Tuple[int, int] = (10, 8)):
        """
        绘制SHAP总结图
        
        Args:
            X: 特征数据
            plot_type: 图表类型 ('bar', 'beeswarm', 'violin')
            max_display: 最大显示特征数
            figsize: 图表大小
        """
        if max_display is None:
            max_display = config.get('strategy_params.max_display_items', 20)
            
        try:
            if self.explainer is None:
                self._initialize_shap_explainer(X)
            
            if self.explainer is None:
                print("无法初始化SHAP解释器")
                return
            
            # 计算SHAP值
            sample_size = min(config.get('strategy_params.shap_sample_size', 1000), len(X))  # 限制样本数量以提高性能
            X_sample = X.sample(n=sample_size, random_state=42)
            shap_values = self.explainer.shap_values(X_sample)
            
            plt.figure(figsize=figsize)
            
            if plot_type == 'bar':
                shap.summary_plot(shap_values, X_sample, plot_type='bar', 
                                max_display=max_display, show=False)
            elif plot_type == 'beeswarm':
                shap.summary_plot(shap_values, X_sample, 
                                max_display=max_display, show=False)
            elif plot_type == 'violin':
                shap.summary_plot(shap_values, X_sample, plot_type='violin',
                                max_display=max_display, show=False)
            
            plt.title(f'SHAP特征重要性总结 ({plot_type})')
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            logger.error(f"绘制SHAP总结图失败: {e}")
            print(f"绘制SHAP图表失败: {e}")
    
    def plot_prediction_explanation(self, X: pd.DataFrame, sample_idx: int = 0, 
                                  figsize: Tuple[int, int] = (12, 6)):
        """
        绘制单个预测的解释图
        
        Args:
            X: 特征数据
            sample_idx: 样本索引
            figsize: 图表大小
        """
        explanation = self.explain_prediction(X, sample_idx)
        
        if 'shap_values' not in explanation:
            print("无法获取SHAP解释")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：SHAP值瀑布图
        shap_values = list(explanation['shap_values'].values())
        feature_names = list(explanation['shap_values'].keys())
        
        # 按绝对值排序
        max_features_display = config.get('strategy_params.max_features_display', 15)
        sorted_indices = np.argsort(np.abs(shap_values))[::-1][:max_features_display]  # 显示前N个
        sorted_shap = [shap_values[i] for i in sorted_indices]
        sorted_features = [feature_names[i] for i in sorted_indices]
        
        colors = ['red' if x < 0 else 'blue' for x in sorted_shap]
        bars = ax1.barh(range(len(sorted_shap)), sorted_shap, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(sorted_shap)))
        ax1.set_yticklabels(sorted_features)
        ax1.set_xlabel('SHAP值')
        ax1.set_title(f'样本 {sample_idx} 的特征贡献')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + (0.01 if width >= 0 else -0.01), 
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center')
        
        ax1.invert_yaxis()
        
        # 右图：预测组成
        base_value = explanation.get('base_value', 0)
        prediction = explanation['prediction']
        shap_sum = sum(shap_values)
        
        components = ['基准值', 'SHAP贡献', '预测值']
        values = [base_value, base_value + shap_sum, prediction]
        
        ax2.bar(components, values, color=['gray', 'orange', 'green'], alpha=0.7)
        ax2.set_ylabel('值')
        ax2.set_title('预测值组成')
        
        # 添加数值标签
        for i, v in enumerate(values):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细信息
        print(f"\n样本 {sample_idx} 预测解释:")
        print(f"预测值: {prediction:.4f}")
        print(f"基准值: {base_value:.4f}")
        print(f"SHAP贡献总和: {shap_sum:.4f}")
        print("\n主要特征贡献:")
        top_features_count = config.get('strategy_params.top_features_count', 10)
        for feature, shap_val in zip(sorted_features[:top_features_count], sorted_shap[:top_features_count]):
            print(f"  {feature}: {shap_val:.4f}")

if __name__ == '__main__':
    # 测试代码
    print("测试模型解释模块...")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 创建目标变量（线性关系 + 噪声）
    true_coef = np.random.randn(n_features)
    y = X.dot(true_coef) + 0.1 * np.random.randn(n_samples)
    
    # 训练简单的线性模型
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    
    # 创建解释器
    explainer = ModelExplainer(model, X.columns.tolist())
    
    # 测试特征重要性
    print("\n计算特征重要性...")
    importance_df = explainer.calculate_feature_importance(X, y, method='default')
    print(importance_df.head())
    
    # 测试单个预测解释
    print("\n解释单个预测...")
    explanation = explainer.explain_prediction(X, sample_idx=0)
    print(f"预测值: {explanation['prediction']:.4f}")
    
    # 测试模型总结
    print("\n生成模型总结...")
    summary = explainer.generate_model_summary(X, y)
    print(f"模型类型: {summary['model_type']}")
    print(f"R²分数: {summary.get('r2_score', 'N/A'):.4f}")
    print(f"RMSE: {summary.get('rmse', 'N/A'):.4f}")
    
    print("\n模型解释模块测试完成!")
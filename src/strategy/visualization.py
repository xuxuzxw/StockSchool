#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型解释可视化模块

提供丰富的可视化功能，包括特征重要性图、SHAP值图、预测解释图等。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
from typing import Any, Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.base import BaseEstimator
import logging
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelVisualizer:
    """模型可视化器类"""
    
    def __init__(self, model: BaseEstimator, feature_names: List[str]):
        """
        初始化模型可视化器
        
        Args:
            model: 机器学习模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = None
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               top_n: int = 20,
                               figsize: Tuple[int, int] = (12, 8),
                               title: str = "特征重要性") -> plt.Figure:
        """
        绘制特征重要性图
        
        Args:
            importance_df: 特征重要性DataFrame
            top_n: 显示前N个特征
            figsize: 图形大小
            title: 图形标题
            
        Returns:
            matplotlib Figure对象
        """
        try:
            # 选择前N个重要特征
            top_features = importance_df.head(top_n)
            
            # 创建图形
            fig, ax = plt.subplots(figsize=figsize)
            
            # 绘制水平条形图
            y_pos = np.arange(len(top_features))
            bars = ax.barh(y_pos, top_features['importance'], 
                          xerr=top_features.get('importance_std', None),
                          height=0.8, alpha=0.8, color='skyblue')
            
            # 设置标签和标题
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('重要性')
            ax.set_title(title)
            ax.invert_yaxis()  # 重要性从高到低
            
            # 添加数值标签
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                ax.text(bar.get_width() + bar.get_width() * 0.01, 
                       bar.get_y() + bar.get_height()/2, 
                       f'{importance:.3f}', 
                       ha='left', va='center', fontsize=9)
            
            # 网格和布局
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            logger.info(f"特征重要性图绘制完成，显示前{len(top_features)}个特征")
            return fig
            
        except Exception as e:
            logger.error(f"特征重要性图绘制失败: {e}")
            raise
    
    def plot_shap_summary(self, shap_values: np.ndarray, 
                         X: pd.DataFrame,
                         max_display: int = 20) -> plt.Figure:
        """
        绘制SHAP摘要图
        
        Args:
            shap_values: SHAP值数组
            X: 特征数据
            max_display: 最大显示特征数
            
        Returns:
            matplotlib Figure对象
        """
        try:
            # 使用SHAP库绘制摘要图
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 创建SHAP Summary Plot
            shap.summary_plot(shap_values, X, 
                            feature_names=self.feature_names,
                            max_display=max_display,
                            show=False,
                            plot_type="dot")
            
            plt.tight_layout()
            logger.info("SHAP摘要图绘制完成")
            return fig
            
        except Exception as e:
            logger.error(f"SHAP摘要图绘制失败: {e}")
            raise
    
    def plot_prediction_explanation(self, explanation: Dict[str, Any],
                                   figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        绘制预测解释图
        
        Args:
            explanation: 预测解释字典
            figsize: 图形大小
            
        Returns:
            matplotlib Figure对象
        """
        try:
            # 提取数据
            feature_names = explanation['feature_names']
            feature_values = explanation['feature_values']
            shap_values = explanation['shap_values']
            base_value = explanation['base_value']
            prediction = explanation['prediction']
            
            # 创建DataFrame
            expl_df = pd.DataFrame({
                'feature': feature_names,
                'feature_value': feature_values,
                'shap_value': shap_values
            }).sort_values('shap_value', key=abs, ascending=False).head(15)
            
            # 创建图形
            fig, ax = plt.subplots(figsize=figsize)
            
            # 颜色映射（正负值不同颜色）
            colors = ['red' if x > 0 else 'blue' for x in expl_df['shap_value']]
            
            # 绘制条形图
            y_pos = np.arange(len(expl_df))
            bars = ax.barh(y_pos, expl_df['shap_value'], color=colors, alpha=0.7)
            
            # 设置标签
            ax.set_yticks(y_pos)
            ax.set_yticklabels(expl_df['feature'])
            ax.set_xlabel('SHAP值')
            ax.set_title(f'预测解释 (预测值: {prediction:.3f}, 基础值: {base_value:.3f})')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.invert_yaxis()
            
            # 添加特征值标签
            for i, (bar, feat_val) in enumerate(zip(bars, expl_df['feature_value'])):
                ax.text(bar.get_width() + (0.01 if bar.get_width() >= 0 else -0.01), 
                       bar.get_y() + bar.get_height()/2,
                       f'{feat_val:.2f}',
                       ha='left' if bar.get_width() >= 0 else 'right', 
                       va='center', fontsize=8)
            
            plt.tight_layout()
            logger.info("预测解释图绘制完成")
            return fig
            
        except Exception as e:
            logger.error(f"预测解释图绘制失败: {e}")
            raise
    
    def plot_feature_interactions(self, interaction_df: pd.DataFrame,
                                 top_n: int = 15,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制特征交互强度图
        
        Args:
            interaction_df: 特征交互DataFrame
            top_n: 显示前N个交互
            figsize: 图形大小
            
        Returns:
            matplotlib Figure对象
        """
        try:
            # 选择前N个交互
            top_interactions = interaction_df.head(top_n)
            
            # 创建交互特征名称
            interaction_labels = [
                f"{row['feature1']} × {row['feature2']}" 
                for _, row in top_interactions.iterrows()
            ]
            
            # 创建图形
            fig, ax = plt.subplots(figsize=figsize)
            
            # 绘制条形图
            y_pos = np.arange(len(top_interactions))
            bars = ax.barh(y_pos, top_interactions['interaction_strength'], 
                          height=0.8, alpha=0.8, color='orange')
            
            # 设置标签
            ax.set_yticks(y_pos)
            ax.set_yticklabels(interaction_labels)
            ax.set_xlabel('交互强度')
            ax.set_title('特征交互强度')
            ax.invert_yaxis()
            
            # 添加数值标签
            for i, (bar, strength) in enumerate(zip(bars, top_interactions['interaction_strength'])):
                ax.text(bar.get_width() + bar.get_width() * 0.01,
                       bar.get_y() + bar.get_height()/2,
                       f'{strength:.3f}',
                       ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            logger.info("特征交互强度图绘制完成")
            return fig
            
        except Exception as e:
            logger.error(f"特征交互强度图绘制失败: {e}")
            raise
    
    def create_interactive_feature_importance(self, importance_df: pd.DataFrame,
                                            top_n: int = 20) -> go.Figure:
        """
        创建交互式特征重要性图
        
        Args:
            importance_df: 特征重要性DataFrame
            top_n: 显示前N个特征
            
        Returns:
            Plotly Figure对象
        """
        try:
            # 选择前N个重要特征
            top_features = importance_df.head(top_n)
            
            # 创建交互式条形图
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=top_features['importance'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f'{imp:.3f}' for imp in top_features['importance']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='交互式特征重要性',
                xaxis_title='重要性',
                yaxis_title='特征',
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            logger.info("交互式特征重要性图创建完成")
            return fig
            
        except Exception as e:
            logger.error(f"交互式特征重要性图创建失败: {e}")
            raise
    
    def create_interactive_prediction_explanation(self, explanation: Dict[str, Any]) -> go.Figure:
        """
        创建交互式预测解释图
        
        Args:
            explanation: 预测解释字典
            
        Returns:
            Plotly Figure对象
        """
        try:
            # 提取数据
            feature_names = explanation['feature_names']
            feature_values = explanation['feature_values']
            shap_values = explanation['shap_values']
            base_value = explanation['base_value']
            prediction = explanation['prediction']
            
            # 创建DataFrame并排序
            expl_df = pd.DataFrame({
                'feature': feature_names,
                'feature_value': feature_values,
                'shap_value': shap_values
            }).sort_values('shap_value', key=abs, ascending=False).head(15)
            
            # 颜色映射
            colors = ['red' if x > 0 else 'blue' for x in expl_df['shap_value']]
            
            # 创建交互式条形图
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=expl_df['shap_value'],
                y=expl_df['feature'],
                orientation='h',
                marker=dict(color=colors),
                text=[f'{val:.2f}<br>({shap:.3f})' for val, shap in zip(expl_df['feature_value'], expl_df['shap_value'])],
                textposition='outside'
            ))
            
            # 添加基准线
            fig.add_vline(x=base_value, line_dash="dash", line_color="black", 
                         annotation_text=f"基准值: {base_value:.3f}")
            
            fig.update_layout(
                title=f'交互式预测解释<br>预测值: {prediction:.3f}',
                xaxis_title='SHAP值',
                yaxis_title='特征',
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            logger.info("交互式预测解释图创建完成")
            return fig
            
        except Exception as e:
            logger.error(f"交互式预测解释图创建失败: {e}")
            raise
    
    def plot_model_comparison(self, models_data: List[Dict[str, Any]],
                             metric_name: str = 'accuracy') -> plt.Figure:
        """
        绘制模型比较图
        
        Args:
            models_data: 模型数据列表，每个字典包含'model_name'和指标数据
            metric_name: 比较的指标名称
            
        Returns:
            matplotlib Figure对象
        """
        try:
            # 提取模型名称和指标值
            model_names = [data['model_name'] for data in models_data]
            metric_values = [data[metric_name] for data in models_data]
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制条形图
            bars = ax.bar(model_names, metric_values, color='lightcoral', alpha=0.8)
            
            # 添加数值标签
            for bar, value in zip(bars, metric_values):
                ax.text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + 0.01,
                       f'{value:.3f}',
                       ha='center', va='bottom')
            
            # 设置标签和标题
            ax.set_xlabel('模型')
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f'模型性能比较 - {metric_name.capitalize()}')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            logger.info("模型比较图绘制完成")
            return fig
            
        except Exception as e:
            logger.error(f"模型比较图绘制失败: {e}")
            raise
    
    def save_plot(self, fig: Union[plt.Figure, go.Figure], 
                  filename: str, 
                  format: str = 'png',
                  dpi: int = 300) -> str:
        """
        保存图形
        
        Args:
            fig: 图形对象
            filename: 文件名
            format: 文件格式
            dpi: 分辨率
            
        Returns:
            保存的文件路径
        """
        try:
            if isinstance(fig, plt.Figure):
                # matplotlib图形
                full_filename = f"{filename}.{format}"
                fig.savefig(full_filename, dpi=dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            else:
                # plotly图形
                if format.lower() == 'html':
                    full_filename = f"{filename}.html"
                    pyo.plot(fig, filename=full_filename, auto_open=False)
                else:
                    full_filename = f"{filename}.{format}"
                    fig.write_image(full_filename)
            
            logger.info(f"图形保存完成: {full_filename}")
            return full_filename
            
        except Exception as e:
            logger.error(f"图形保存失败: {e}")
            raise

def create_model_visualizer(model: BaseEstimator, feature_names: List[str]) -> ModelVisualizer:
    """
    创建模型可视化器的便捷函数
    
    Args:
        model: 机器学习模型
        feature_names: 特征名称列表
        
    Returns:
        ModelVisualizer实例
    """
    return ModelVisualizer(model, feature_names)

# 便捷绘图函数
def quick_feature_importance_plot(importance_df: pd.DataFrame, 
                                 top_n: int = 20,
                                 title: str = "特征重要性") -> plt.Figure:
    """
    快速绘制特征重要性图
    
    Args:
        importance_df: 特征重要性DataFrame
        top_n: 显示前N个特征
        title: 图形标题
        
    Returns:
        matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_features = importance_df.head(top_n)
    y_pos = np.arange(len(top_features))
    
    bars = ax.barh(y_pos, top_features['importance'], height=0.8, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('重要性')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def quick_shap_waterfall(explanation: Dict[str, Any]) -> go.Figure:
    """
    快速创建SHAP瀑布图
    
    Args:
        explanation: 预测解释字典
        
    Returns:
        Plotly Figure对象
    """
    shap_values = explanation['shap_values']
    feature_names = explanation['feature_names']
    base_value = explanation['base_value']
    prediction = explanation['prediction']
    
    # 创建瀑布图数据
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["absolute"] + ["relative"] * len(shap_values),
        y=["Base Value"] + feature_names[:len(shap_values)],
        x=[base_value] + shap_values,
        connector={"mode": "between", "line": {"width": 4, "color": "rgb(0, 0, 0)", "dash": "dot"}},
        totals={"marker": {"color": "deepskyblue", "line": {"color": "blue", "width": 3}}}
    ))
    
    fig.update_layout(
        title=f"SHAP瀑布图<br>预测值: {prediction:.3f}",
        height=600
    )
    
    return fig

if __name__ == '__main__':
    # 测试代码
    print("测试模型可视化器...")
    
    # 创建测试数据
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    import pandas as pd
    import numpy as np
    
    # 生成测试数据
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    feature_names = [f'feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 创建模型
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_df, y)
    
    # 创建可视化器
    visualizer = ModelVisualizer(model, feature_names)
    
    # 创建模拟特征重要性数据
    importance_data = {
        'feature': feature_names,
        'importance': np.random.rand(10) * 0.5 + 0.1,
        'importance_std': np.random.rand(10) * 0.1
    }
    importance_df = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
    
    print("1. 测试特征重要性图...")
    fig1 = visualizer.plot_feature_importance(importance_df, top_n=10)
    plt.show()
    plt.close()
    
    print("2. 测试交互式特征重要性图...")
    fig2 = visualizer.create_interactive_feature_importance(importance_df, top_n=10)
    fig2.show()
    
    # 创建模拟解释数据
    explanation = {
        'feature_names': feature_names[:5],
        'feature_values': [0.1, 0.2, 0.3, 0.4, 0.5],
        'shap_values': [0.1, -0.2, 0.15, -0.1, 0.05],
        'base_value': 0.5,
        'prediction': 0.6
    }
    
    print("3. 测试预测解释图...")
    fig3 = visualizer.plot_prediction_explanation(explanation)
    plt.show()
    plt.close()
    
    print("4. 测试交互式预测解释图...")
    fig4 = visualizer.create_interactive_prediction_explanation(explanation)
    fig4.show()
    
    print("\n模型可视化器测试完成!")

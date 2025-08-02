#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型解释器用户界面

使用Streamlit创建简单的Web界面，用于测试和演示模型解释功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import List, Dict, Any
import joblib

from src.strategy.model_explainer import ModelExplainer
from src.utils.config_loader import config

# 页面配置
st.set_page_config(
    page_title="StockSchool 模型解释器",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """主界面"""
    st.title("📊 StockSchool 模型解释器")
    st.markdown("""
    这是一个用于解释机器学习模型预测结果的工具。
    支持多种解释方法，包括SHAP值、特征重要性等。
    """)
    
    # 侧边栏
    st.sidebar.header("⚙️ 配置")
    
    # 选择功能
    feature = st.sidebar.radio(
        "选择功能",
        ["模型信息", "特征重要性", "预测解释", "批量解释", "模型对比"]
    )
    
    if feature == "模型信息":
        model_info_page()
    elif feature == "特征重要性":
        feature_importance_page()
    elif feature == "预测解释":
        prediction_explanation_page()
    elif feature == "批量解释":
        batch_explanation_page()
    elif feature == "模型对比":
        model_comparison_page()

def model_info_page():
    """模型信息页面"""
    st.header("🔍 模型信息")
    
    # 上传模型文件
    model_file = st.file_uploader("上传模型文件", type=['pkl', 'joblib'])
    
    if model_file is not None:
        try:
            # 保存临时文件
            temp_path = f"temp_model_{model_file.name}"
            with open(temp_path, "wb") as f:
                f.write(model_file.getbuffer())
            
            # 加载模型
            model = joblib.load(temp_path)
            
            # 显示模型信息
            st.subheader("模型基本信息")
            model_info = {
                "模型类型": type(model).__name__,
                "模型模块": type(model).__module__,
                "支持预测": hasattr(model, 'predict'),
                "支持特征重要性": hasattr(model, 'feature_importances_'),
                "支持系数": hasattr(model, 'coef_')
            }
            
            col1, col2 = st.columns(2)
            with col1:
                st.json(model_info)
            
            with col2:
                st.info("💡 提示：确保上传的模型文件格式正确")
            
            # 清理临时文件
            os.remove(temp_path)
            
        except Exception as e:
            st.error(f"加载模型失败: {e}")

def feature_importance_page():
    """特征重要性页面"""
    st.header("📈 特征重要性分析")
    
    # 上传模型文件
    model_file = st.file_uploader("上传模型文件", type=['pkl', 'joblib'], key="fi_model")
    
    if model_file is not None:
        try:
            # 上传数据文件
            data_file = st.file_uploader("上传数据文件 (CSV)", type=['csv'], key="fi_data")
            
            if data_file is not None:
                # 读取数据
                df = pd.read_csv(data_file)
                st.subheader("数据预览")
                st.dataframe(df.head())
                
                # 选择特征列
                feature_columns = st.multiselect(
                    "选择特征列", 
                    df.columns.tolist(),
                    default=df.columns.tolist()[:5] if len(df.columns) >= 5 else df.columns.tolist()
                )
                
                if feature_columns:
                    # 选择目标列（可选）
                    target_column = st.selectbox(
                        "选择目标列（可选）", 
                        ["无"] + [col for col in df.columns if col not in feature_columns]
                    )
                    
                    # 选择解释方法
                    method = st.selectbox(
                        "选择解释方法", 
                        ["default", "permutation", "shap"],
                        format_func=lambda x: {
                            "default": "默认重要性",
                            "permutation": "排列重要性",
                            "shap": "SHAP值"
                        }[x]
                    )
                    
                    if st.button("计算特征重要性", type="primary"):
                        with st.spinner("正在计算特征重要性..."):
                            try:
                                # 保存并加载模型
                                temp_model_path = f"temp_model_{model_file.name}"
                                with open(temp_model_path, "wb") as f:
                                    f.write(model_file.getbuffer())
                                
                                model = joblib.load(temp_model_path)
                                
                                # 准备数据
                                X = df[feature_columns]
                                y = df[target_column] if target_column != "无" else None
                                
                                # 创建解释器并计算重要性
                                explainer = ModelExplainer(model, feature_columns)
                                importance = explainer.calculate_feature_importance(X, y, method=method)
                                
                                # 显示结果
                                st.subheader("特征重要性结果")
                                st.dataframe(importance)
                                
                                # 可视化
                                fig, ax = plt.subplots(figsize=(10, 6))
                                top_features = importance.head(10)
                                bars = ax.barh(range(len(top_features)), top_features['importance'])
                                ax.set_yticks(range(len(top_features)))
                                ax.set_yticklabels(top_features['feature'])
                                ax.set_xlabel('重要性')
                                ax.set_title(f'特征重要性 ({method.upper()}方法)')
                                ax.invert_yaxis()
                                
                                # 添加数值标签
                                for i, (bar, importance_val) in enumerate(zip(bars, top_features['importance'])):
                                    ax.text(bar.get_width(), i, f'{importance_val:.4f}', 
                                           va='center', ha='left', fontsize=8)
                                
                                st.pyplot(fig)
                                
                                # 清理临时文件
                                os.remove(temp_model_path)
                                
                            except Exception as e:
                                st.error(f"计算特征重要性失败: {e}")
                
        except Exception as e:
            st.error(f"处理文件失败: {e}")

def prediction_explanation_page():
    """预测解释页面"""
    st.header("🎯 预测结果解释")
    
    # 上传模型文件
    model_file = st.file_uploader("上传模型文件", type=['pkl', 'joblib'], key="pe_model")
    
    if model_file is not None:
        try:
            # 上传数据文件
            data_file = st.file_uploader("上传数据文件 (CSV)", type=['csv'], key="pe_data")
            
            if data_file is not None:
                # 读取数据
                df = pd.read_csv(data_file)
                st.subheader("数据预览")
                st.dataframe(df.head())
                
                # 选择特征列
                feature_columns = st.multiselect(
                    "选择特征列", 
                    df.columns.tolist(),
                    default=df.columns.tolist()[:5] if len(df.columns) >= 5 else df.columns.tolist()
                )
                
                if feature_columns:
                    # 选择样本索引
                    sample_idx = st.number_input(
                        "选择样本索引", 
                        min_value=0, 
                        max_value=len(df)-1, 
                        value=0
                    )
                    
                    if st.button("解释预测结果", type="primary"):
                        with st.spinner("正在解释预测结果..."):
                            try:
                                # 保存并加载模型
                                temp_model_path = f"temp_model_{model_file.name}"
                                with open(temp_model_path, "wb") as f:
                                    f.write(model_file.getbuffer())
                                
                                model = joblib.load(temp_model_path)
                                
                                # 准备数据
                                X = df[feature_columns]
                                
                                # 创建解释器并解释预测
                                explainer = ModelExplainer(model, feature_columns)
                                explanation = explainer.explain_prediction(X, sample_idx)
                                
                                # 显示结果
                                st.subheader("预测解释结果")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("预测值", f"{explanation.get('prediction', 'N/A'):.4f}")
                                    st.metric("基础值", f"{explanation.get('base_value', 0):.4f}")
                                    st.write(f"模型类型: {explanation.get('model_type', 'N/A')}")
                                
                                with col2:
                                    st.write("特征值:")
                                    feature_values = explanation.get('feature_values', [])
                                    for i, (feature, value) in enumerate(zip(feature_columns, feature_values)):
                                        st.write(f"**{feature}**: {value:.4f}")
                                
                                # SHAP值可视化
                                st.subheader("SHAP值分析")
                                shap_values = explanation.get('shap_values', [])
                                if shap_values:
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    
                                    y_pos = np.arange(len(feature_columns))
                                    colors = ['red' if val < 0 else 'blue' for val in shap_values]
                                    bars = ax.barh(y_pos, shap_values, color=colors, alpha=0.7)
                                    
                                    ax.set_yticks(y_pos)
                                    ax.set_yticklabels([f"{feat}\n({val:.4f})" for feat, val in zip(feature_columns, feature_values)])
                                    ax.set_xlabel('SHAP值')
                                    ax.set_title(f'预测解释 - 样本 {sample_idx}')
                                    ax.invert_yaxis()
                                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                                    
                                    st.pyplot(fig)
                                
                                # 清理临时文件
                                os.remove(temp_model_path)
                                
                            except Exception as e:
                                st.error(f"解释预测结果失败: {e}")
                
        except Exception as e:
            st.error(f"处理文件失败: {e}")

def batch_explanation_page():
    """批量解释页面"""
    st.header("🔄 批量预测解释")
    st.info("此功能支持对多个样本进行批量解释分析")
    
    # 上传模型文件
    model_file = st.file_uploader("上传模型文件", type=['pkl', 'joblib'], key="be_model")
    
    if model_file is not None:
        try:
            # 上传数据文件
            data_file = st.file_uploader("上传数据文件 (CSV)", type=['csv'], key="be_data")
            
            if data_file is not None:
                # 读取数据
                df = pd.read_csv(data_file)
                st.subheader("数据预览")
                st.dataframe(df.head())
                
                # 选择特征列
                feature_columns = st.multiselect(
                    "选择特征列", 
                    df.columns.tolist(),
                    default=df.columns.tolist()[:5] if len(df.columns) >= 5 else df.columns.tolist()
                )
                
                if feature_columns:
                    # 选择样本范围
                    col1, col2 = st.columns(2)
                    with col1:
                        start_idx = st.number_input(
                            "起始样本索引", 
                            min_value=0, 
                            max_value=len(df)-1, 
                            value=0
                        )
                    with col2:
                        end_idx = st.number_input(
                            "结束样本索引", 
                            min_value=start_idx+1, 
                            max_value=len(df), 
                            value=min(start_idx+10, len(df))
                        )
                    
                    if st.button("批量解释", type="primary"):
                        with st.spinner("正在批量解释预测结果..."):
                            try:
                                # 保存并加载模型
                                temp_model_path = f"temp_model_{model_file.name}"
                                with open(temp_model_path, "wb") as f:
                                    f.write(model_file.getbuffer())
                                
                                model = joblib.load(temp_model_path)
                                
                                # 准备数据
                                X = df[feature_columns].iloc[start_idx:end_idx]
                                
                                # 创建解释器并批量解释
                                explainer = ModelExplainer(model, feature_columns)
                                
                                explanations = []
                                for i in range(len(X)):
                                    explanation = explainer.explain_prediction(X, i)
                                    explanations.append(explanation)
                                
                                # 显示结果
                                st.subheader(f"批量解释结果 (样本 {start_idx} 到 {end_idx-1})")
                                
                                # 汇总统计
                                shap_values_list = [exp.get('shap_values', []) for exp in explanations]
                                if shap_values_list and shap_values_list[0]:
                                    avg_shap_values = np.mean(shap_values_list, axis=0)
                                    
                                    st.subheader("平均SHAP值")
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    bars = ax.barh(range(len(feature_columns)), avg_shap_values)
                                    ax.set_yticks(range(len(feature_columns)))
                                    ax.set_yticklabels(feature_columns)
                                    ax.set_xlabel('平均SHAP值')
                                    ax.set_title('批量样本平均特征贡献')
                                    ax.invert_yaxis()
                                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                                    st.pyplot(fig)
                                
                                # 显示详细结果
                                st.subheader("详细解释")
                                for i, explanation in enumerate(explanations):
                                    with st.expander(f"样本 {start_idx + i} - 预测值: {explanation.get('prediction', 'N/A'):.4f}"):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("SHAP值:")
                                            shap_values = explanation.get('shap_values', [])
                                            for feature, shap_val in zip(feature_columns, shap_values):
                                                st.write(f"**{feature}**: {shap_val:.4f}")
                                        with col2:
                                            st.write("特征值:")
                                            feature_values = explanation.get('feature_values', [])
                                            for feature, value in zip(feature_columns, feature_values):
                                                st.write(f"**{feature}**: {value:.4f}")
                                
                                # 清理临时文件
                                os.remove(temp_model_path)
                                
                            except Exception as e:
                                st.error(f"批量解释失败: {e}")
                
        except Exception as e:
            st.error(f"处理文件失败: {e}")

def model_comparison_page():
    """模型对比页面"""
    st.header("⚖️ 模型对比分析")
    st.info("此功能支持对比多个模型的特征重要性和预测解释")
    
    # 上传多个模型文件
    st.subheader("上传模型文件")
    model_files = st.file_uploader(
        "上传多个模型文件", 
        type=['pkl', 'joblib'], 
        accept_multiple_files=True,
        key="mc_models"
    )
    
    if len(model_files) >= 2:
        try:
            # 上传数据文件
            data_file = st.file_uploader("上传数据文件 (CSV)", type=['csv'], key="mc_data")
            
            if data_file is not None:
                # 读取数据
                df = pd.read_csv(data_file)
                st.subheader("数据预览")
                st.dataframe(df.head())
                
                # 选择特征列
                feature_columns = st.multiselect(
                    "选择特征列", 
                    df.columns.tolist(),
                    default=df.columns.tolist()[:5] if len(df.columns) >= 5 else df.columns.tolist()
                )
                
                if feature_columns:
                    # 选择解释方法
                    method = st.selectbox(
                        "选择解释方法", 
                        ["default", "permutation", "shap"],
                        format_func=lambda x: {
                            "default": "默认重要性",
                            "permutation": "排列重要性",
                            "shap": "SHAP值"
                        }[x]
                    )
                    
                    # 选择样本索引
                    sample_idx = st.number_input(
                        "选择样本索引", 
                        min_value=0, 
                        max_value=len(df)-1, 
                        value=0
                    )
                    
                    if st.button("对比分析", type="primary"):
                        with st.spinner("正在进行模型对比分析..."):
                            try:
                                # 加载所有模型
                                models = []
                                model_names = []
                                
                                for i, model_file in enumerate(model_files):
                                    temp_model_path = f"temp_model_{i}_{model_file.name}"
                                    with open(temp_model_path, "wb") as f:
                                        f.write(model_file.getbuffer())
                                    
                                    model = joblib.load(temp_model_path)
                                    models.append(model)
                                    model_names.append(f"模型 {i+1} ({type(model).__name__})")
                                    
                                # 准备数据
                                X = df[feature_columns]
                                
                                # 对比特征重要性
                                st.subheader("特征重要性对比")
                                
                                importance_data = []
                                for model, model_name in zip(models, model_names):
                                    explainer = ModelExplainer(model, feature_columns)
                                    importance = explainer.calculate_feature_importance(X, method=method)
                                    importance['model'] = model_name
                                    importance_data.append(importance)
                                
                                # 合并数据
                                combined_importance = pd.concat(importance_data, ignore_index=True)
                                
                                # 可视化对比
                                fig, ax = plt.subplots(figsize=(12, 8))
                                
                                # 只显示前10个特征
                                top_features = importance_data[0].head(10)['feature'].tolist()
                                
                                for model_name in model_names:
                                    model_importance = combined_importance[combined_importance['model'] == model_name]
                                    model_importance = model_importance[model_importance['feature'].isin(top_features)]
                                    model_importance = model_importance.set_index('feature').reindex(top_features)
                                    ax.plot(range(len(top_features)), model_importance['importance'], 
                                           marker='o', label=model_name, linewidth=2)
                                
                                ax.set_xticks(range(len(top_features)))
                                ax.set_xticklabels(top_features, rotation=45)
                                ax.set_xlabel('特征')
                                ax.set_ylabel('重要性')
                                ax.set_title(f'模型特征重要性对比 ({method.upper()}方法)')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                                
                                # 对比预测解释
                                st.subheader("预测解释对比")
                                
                                explanation_data = []
                                for model, model_name in zip(models, model_names):
                                    explainer = ModelExplainer(model, feature_columns)
                                    explanation = explainer.explain_prediction(X, sample_idx)
                                    explanation['model'] = model_name
                                    explanation_data.append(explanation)
                                
                                # 显示预测值对比
                                pred_values = [exp['prediction'] for exp in explanation_data]
                                pred_df = pd.DataFrame({
                                    '模型': model_names,
                                    '预测值': pred_values
                                })
                                
                                st.dataframe(pred_df)
                                
                                # 可视化预测值对比
                                fig2, ax2 = plt.subplots(figsize=(10, 6))
                                bars = ax2.bar(model_names, pred_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(model_names)])
                                ax2.set_ylabel('预测值')
                                ax2.set_title(f'不同模型预测值对比 (样本 {sample_idx})')
                                
                                # 添加数值标签
                                for bar, value in zip(bars, pred_values):
                                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                            f'{value:.4f}', ha='center', va='bottom')
                                
                                plt.xticks(rotation=45)
                                st.pyplot(fig2)
                                
                                # 清理临时文件
                                for i in range(len(model_files)):
                                    temp_model_path = f"temp_model_{i}_{model_files[i].name}"
                                    if os.path.exists(temp_model_path):
                                        os.remove(temp_model_path)
                                
                            except Exception as e:
                                st.error(f"模型对比分析失败: {e}")
                
        except Exception as e:
            st.error(f"处理文件失败: {e}")
    else:
        st.info("请至少上传两个模型文件进行对比分析")

if __name__ == "__main__":
    main()

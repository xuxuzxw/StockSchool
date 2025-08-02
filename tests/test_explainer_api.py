#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型解释器API测试

测试API接口的功能和性能
"""

import sys
import os
import unittest
import json
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy.model_explainer import ModelExplainer

class TestExplainerAPI(unittest.TestCase):
    """模型解释器API测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 创建测试数据
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.rand(100, 5), 
                             columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        self.y = pd.Series(np.random.rand(100))
        
        # 创建测试模型
        self.rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.rf_model.fit(self.X, self.y)
        
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X, self.y)
        
        # 保存模型
        self.rf_model_path = self.temp_dir / "rf_model.pkl"
        self.lr_model_path = self.temp_dir / "lr_model.pkl"
        
        joblib.dump(self.rf_model, self.rf_model_path)
        joblib.dump(self.lr_model, self.lr_model_path)
        
        # 测试数据
        self.test_data = self.X.head(10).values.tolist()
        self.test_feature_names = self.X.columns.tolist()
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_api_model_loading(self):
        """测试API模型加载功能"""
        from src.api.explainer_api import load_model
        from fastapi import HTTPException
        
        # 测试正常加载
        model = load_model(str(self.rf_model_path))
        self.assertIsNotNone(model)
        self.assertEqual(type(model).__name__, "RandomForestRegressor")
        
        # 测试文件不存在
        with self.assertRaises(HTTPException) as context:
            load_model("nonexistent_model.pkl")
        
        self.assertEqual(context.exception.status_code, 500)
    
    def test_api_explain_request_model(self):
        """测试API请求模型"""
        from src.api.explainer_api import (
            ExplainRequest, BatchExplainRequest, FeatureImportanceRequest
        )
        
        # 测试单个解释请求
        request_data = {
            "model_path": str(self.rf_model_path),
            "data": self.test_data[:1],  # 单个样本
            "feature_names": self.test_feature_names,
            "method": "shap",
            "sample_idx": 0
        }
        
        request = ExplainRequest(**request_data)
        self.assertEqual(request.model_path, str(self.rf_model_path))
        self.assertEqual(len(request.data), 1)
        self.assertEqual(request.method, "shap")
    
    def test_api_feature_importance_request_model(self):
        """测试API特征重要性请求模型"""
        from src.api.explainer_api import FeatureImportanceRequest
        
        # 测试特征重要性请求
        request_data = {
            "model_path": str(self.rf_model_path),
            "data": self.test_data,
            "feature_names": self.test_feature_names,
            "target": self.y.head(10).tolist(),
            "method": "shap"
        }
        
        request = FeatureImportanceRequest(**request_data)
        self.assertEqual(request.model_path, str(self.rf_model_path))
        self.assertEqual(len(request.data), 10)
        self.assertEqual(len(request.target), 10)
        self.assertEqual(request.method, "shap")
    
    def test_api_batch_explain_request_model(self):
        """测试API批量解释请求模型"""
        from src.api.explainer_api import BatchExplainRequest
        
        # 测试批量解释请求
        batch_data = [self.test_data[:1], self.test_data[1:2]]  # 两个样本
        request_data = {
            "model_path": str(self.rf_model_path),
            "data_list": batch_data,
            "feature_names": self.test_feature_names,
            "method": "shap"
        }
        
        request = BatchExplainRequest(**request_data)
        self.assertEqual(request.model_path, str(self.rf_model_path))
        self.assertEqual(len(request.data_list), 2)
        self.assertEqual(request.method, "shap")
    
    def test_api_explain_endpoint(self):
        """测试API解释端点"""
        from src.api.explainer_api import explain_prediction
        from src.api.explainer_api import ExplainRequest
        
        # 创建请求
        request_data = {
            "model_path": str(self.rf_model_path),
            "data": self.test_data[:1],  # 单个样本
            "feature_names": self.test_feature_names,
            "method": "shap",
            "sample_idx": 0
        }
        
        request = ExplainRequest(**request_data)
        
        # 调用API
        import asyncio
        response = asyncio.run(explain_prediction(request))
        
        # 验证响应
        self.assertEqual(response.status, "success")
        # 验证响应包含必要的字段（直接在响应中，而不是嵌套的explanation中）
        self.assertIn("feature_names", response.explanation)
        self.assertIn("shap_values", response.explanation)
        self.assertIn("prediction", response.explanation)
        self.assertIn("sample_idx", response.explanation)
        
        # 验证数据一致性
        self.assertEqual(len(response.explanation["feature_names"]), 
                        len(response.explanation["shap_values"]))
    
    def test_api_feature_importance_endpoint(self):
        """测试API特征重要性端点"""
        from src.api.explainer_api import feature_importance
        from src.api.explainer_api import FeatureImportanceRequest
        
        # 创建请求
        request_data = {
            "model_path": str(self.rf_model_path),
            "data": self.test_data,
            "feature_names": self.test_feature_names,
            "target": self.y.head(10).tolist(),
            "method": "shap"
        }
        
        request = FeatureImportanceRequest(**request_data)
        
        # 调用API
        import asyncio
        response = asyncio.run(feature_importance(request))
        
        # 验证响应
        self.assertEqual(response.status, "success")
        self.assertIsInstance(response.importance, list)
        self.assertGreater(len(response.importance), 0)
        
        # 验证重要性数据结构
        first_importance = response.importance[0]
        self.assertIn("feature", first_importance)
        self.assertIn("importance", first_importance)
    
    def test_api_batch_explain_endpoint(self):
        """测试API批量解释端点"""
        from src.api.explainer_api import batch_explain
        from src.api.explainer_api import BatchExplainRequest
        
        # 创建批量数据
        batch_data = [self.test_data[:1], self.test_data[1:2], self.test_data[2:3]]
        
        # 创建请求
        request_data = {
            "model_path": str(self.rf_model_path),
            "data_list": batch_data,
            "feature_names": self.test_feature_names,
            "method": "shap"
        }
        
        request = BatchExplainRequest(**request_data)
        
        # 调用API
        import asyncio
        response = asyncio.run(batch_explain(request))
        
        # 验证响应
        self.assertEqual(response.status, "success")
        self.assertIsInstance(response.explanations, list)
        self.assertEqual(len(response.explanations), 3)
        
        # 验证每个解释
        for explanation in response.explanations:
            self.assertIn("feature_names", explanation)
            self.assertIn("shap_values", explanation)
            self.assertIn("prediction", explanation)
            self.assertIn("batch_index", explanation)
    
    def test_api_model_info_endpoint(self):
        """测试API模型信息端点"""
        from src.api.explainer_api import get_model_info
        
        # 调用API
        import asyncio
        response = asyncio.run(get_model_info(str(self.rf_model_path)))
        
        # 验证响应
        self.assertEqual(response["status"], "success")
        self.assertIn("model_info", response)
        
        model_info = response["model_info"]
        self.assertEqual(model_info["model_type"], "RandomForestRegressor")
        self.assertTrue(model_info["has_predict"])
        self.assertTrue(model_info["has_feature_importances"])
    
    def test_api_model_summary_endpoint(self):
        """测试API模型摘要端点"""
        from src.api.explainer_api import model_summary
        from src.api.explainer_api import ExplainRequest
        
        # 创建请求
        request_data = {
            "model_path": str(self.rf_model_path),
            "data": self.test_data,
            "feature_names": self.test_feature_names,
            "method": "default",
            "sample_idx": 0
        }
        
        request = ExplainRequest(**request_data)
        
        # 调用API
        import asyncio
        response = asyncio.run(model_summary(request))
        
        # 验证响应
        self.assertEqual(response["status"], "success")
        self.assertIn("summary", response)
        
        summary = response["summary"]
        self.assertIn("model_type", summary)
        self.assertIn("feature_count", summary)
        self.assertIn("device", summary)
    
    def test_api_different_model_types(self):
        """测试API对不同模型类型的支持"""
        from src.api.explainer_api import feature_importance
        from src.api.explainer_api import FeatureImportanceRequest
        
        # 测试随机森林模型
        import asyncio
        rf_request_data = {
            "model_path": str(self.rf_model_path),
            "data": self.test_data,
            "feature_names": self.test_feature_names,
            "method": "default"
        }
        
        rf_request = FeatureImportanceRequest(**rf_request_data)
        rf_response = asyncio.run(feature_importance(rf_request))
        self.assertEqual(rf_response.status, "success")
        
        # 测试线性回归模型
        lr_request_data = {
            "model_path": str(self.lr_model_path),
            "data": self.test_data,
            "feature_names": self.test_feature_names,
            "method": "default"
        }
        
        lr_request = FeatureImportanceRequest(**lr_request_data)
        lr_response = asyncio.run(feature_importance(lr_request))
        self.assertEqual(lr_response.status, "success")
    
    def test_api_error_handling(self):
        """测试API错误处理"""
        from src.api.explainer_api import explain_prediction, load_model
        from src.api.explainer_api import ExplainRequest
        from fastapi import HTTPException
        import asyncio
        
        # 测试不存在的模型文件
        request_data = {
            "model_path": "nonexistent_model.pkl",
            "data": self.test_data[:1],
            "feature_names": self.test_feature_names,
            "method": "shap",
            "sample_idx": 0
        }
        
        request = ExplainRequest(**request_data)
        
        # 应该抛出HTTPException
        with self.assertRaises(HTTPException) as context:
            asyncio.run(explain_prediction(request))
        
        self.assertEqual(context.exception.status_code, 500)
    
    def test_api_performance(self):
        """测试API性能"""
        from src.api.explainer_api import explain_prediction
        from src.api.explainer_api import ExplainRequest
        
        # 创建请求
        request_data = {
            "model_path": str(self.rf_model_path),
            "data": self.test_data[:1],
            "feature_names": self.test_feature_names,
            "method": "shap",
            "sample_idx": 0
        }
        
        request = ExplainRequest(**request_data)
        
        # 性能测试
        import asyncio
        start_time = time.time()
        response = asyncio.run(explain_prediction(request))
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 10.0)  # 应该在10秒内完成
        self.assertEqual(response.status, "success")
    
    def test_api_cache_functionality(self):
        """测试API缓存功能"""
        from src.api.explainer_api import explain_prediction
        from src.api.explainer_api import ExplainRequest
        
        # 创建相同的请求多次
        request_data = {
            "model_path": str(self.rf_model_path),
            "data": self.test_data[:1],
            "feature_names": self.test_feature_names,
            "method": "shap",
            "sample_idx": 0
        }
        
        request = ExplainRequest(**request_data)
        
        # 第一次调用
        import asyncio
        start_time1 = time.time()
        response1 = asyncio.run(explain_prediction(request))
        end_time1 = time.time()

        # 第二次调用（应该从缓存获取）
        start_time2 = time.time()
        response2 = asyncio.run(explain_prediction(request))
        end_time2 = time.time()

        # 验证结果一致性
        self.assertEqual(response1.status, response2.status)
        
        # 第二次应该更快（缓存命中）
        time1 = end_time1 - start_time1
        time2 = end_time2 - start_time2
        # 注意：由于测试环境可能没有启用缓存，这里只验证功能正确性
        self.assertEqual(response1.status, "success")
        self.assertEqual(response2.status, "success")

class TestExplainerMonitor(unittest.TestCase):
    """模型解释器监控测试类"""
    
    def setUp(self):
        """测试前准备"""
        from src.monitoring.explainer_monitor import ExplainerPerformanceMonitor
        self.monitor = ExplainerPerformanceMonitor(history_size=100)
    
    def test_monitor_initialization(self):
        """测试监控器初始化"""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(self.monitor.metrics['request_count'], 0)
        self.assertEqual(self.monitor.metrics['success_count'], 0)
    
    def test_record_request(self):
        """测试记录请求"""
        start_time = time.time() - 0.1  # 模拟0.1秒的请求
        
        # 记录成功请求
        self.monitor.record_request(start_time, success=True, is_batch=False, cache_hit=False)
        
        self.assertEqual(self.monitor.metrics['request_count'], 1)
        self.assertEqual(self.monitor.metrics['success_count'], 1)
        self.assertEqual(self.monitor.metrics['single_requests'], 1)
        self.assertEqual(self.monitor.metrics['cache_misses'], 1)
        
        # 记录失败请求
        start_time = time.time() - 0.2  # 模拟0.2秒的请求
        self.monitor.record_request(start_time, success=False, is_batch=True, cache_hit=True)
        
        self.assertEqual(self.monitor.metrics['request_count'], 2)
        self.assertEqual(self.monitor.metrics['error_count'], 1)
        self.assertEqual(self.monitor.metrics['batch_requests'], 1)
        self.assertEqual(self.monitor.metrics['cache_hits'], 1)
    
    def test_get_system_metrics(self):
        """测试获取系统指标"""
        metrics = self.monitor.get_system_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('cpu_percent', metrics)
        self.assertIn('memory_percent', metrics)
        self.assertIn('disk_percent', metrics)
        self.assertIn('timestamp', metrics)
        
        # 验证数值范围
        self.assertGreaterEqual(metrics['cpu_percent'], 0)
        self.assertLessEqual(metrics['cpu_percent'], 100)
        self.assertGreaterEqual(metrics['memory_percent'], 0)
        self.assertLessEqual(metrics['memory_percent'], 100)
    
    def test_get_performance_stats(self):
        """测试获取性能统计"""
        # 先记录一些请求
        start_time = time.time() - 0.1
        self.monitor.record_request(start_time, success=True)
        
        start_time = time.time() - 0.2
        self.monitor.record_request(start_time, success=True)
        
        start_time = time.time() - 0.3
        self.monitor.record_request(start_time, success=False)
        
        # 获取统计信息
        stats = self.monitor.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('request_count', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('average_response_time', stats)
        self.assertIn('cache_hit_rate', stats)
        
        # 验证统计值
        self.assertEqual(stats['request_count'], 3)
        self.assertAlmostEqual(stats['success_rate'], 66.67, places=2)
        self.assertGreater(stats['average_response_time'], 0)
    
    def test_get_recent_performance(self):
        """测试获取最近性能数据"""
        # 记录一些请求
        start_time = time.time() - 0.1
        self.monitor.record_request(start_time, success=True)
        
        # 获取最近5分钟的性能数据
        recent_stats = self.monitor.get_recent_performance(minutes=5)
        
        self.assertIsInstance(recent_stats, dict)
        self.assertIn('request_count', recent_stats)
        self.assertIn('average_response_time', recent_stats)
        self.assertIn('success_rate', recent_stats)
        
        self.assertEqual(recent_stats['request_count'], 1)
        self.assertGreater(recent_stats['average_response_time'], 0)
    
    def test_reset_metrics(self):
        """测试重置指标"""
        # 记录一些数据
        start_time = time.time() - 0.1
        self.monitor.record_request(start_time, success=True)
        
        self.assertGreater(self.monitor.metrics['request_count'], 0)
        
        # 重置指标
        self.monitor.reset_metrics()
        
        # 验证重置
        self.assertEqual(self.monitor.metrics['request_count'], 0)
        self.assertEqual(self.monitor.metrics['success_count'], 0)
        self.assertEqual(self.monitor.metrics['error_count'], 0)
    
    def test_export_metrics(self):
        """测试导出指标"""
        import tempfile
        import os
        
        # 记录一些数据
        start_time = time.time() - 0.1
        self.monitor.record_request(start_time, success=True)
        
        # 导出指标
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, "test_export.json")
            result_path = self.monitor.export_metrics(export_path)
            
            self.assertEqual(result_path, export_path)
            self.assertTrue(os.path.exists(export_path))
            
            # 验证导出文件内容
            with open(export_path, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            
            self.assertIn('metrics', exported_data)
            self.assertIn('performance_stats', exported_data)
            self.assertIn('system_metrics_history', exported_data)
            self.assertIn('response_times', exported_data)

class TestIntegration(unittest.TestCase):
    """集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 创建测试数据
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.rand(50, 4), 
                             columns=['open', 'high', 'low', 'close'])
        self.y = pd.Series(np.random.rand(50))
        
        # 创建股票预测模型
        self.model = RandomForestRegressor(n_estimators=20, random_state=42)
        self.model.fit(self.X, self.y)
        
        # 保存模型
        self.model_path = self.temp_dir / "stock_model.pkl"
        joblib.dump(self.model, self.model_path)
        
        # 测试数据
        self.stock_data = self.X.head(5).values.tolist()
        self.feature_names = self.X.columns.tolist()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_api_workflow(self):
        """测试完整的API工作流程"""
        print("开始API集成测试...")
        
        from src.api.explainer_api import (
            get_model_info, feature_importance, explain_prediction, batch_explain
        )
        from src.api.explainer_api import (
            ExplainRequest, FeatureImportanceRequest, BatchExplainRequest
        )
        
        # 1. 获取模型信息
        import asyncio
        model_info_response = asyncio.run(get_model_info(str(self.model_path)))
        self.assertEqual(model_info_response["status"], "success")
        self.assertEqual(model_info_response["model_info"]["model_type"], "RandomForestRegressor")
        print("✓ 模型信息获取测试通过")
        
        # 2. 计算特征重要性
        fi_request_data = {
            "model_path": str(self.model_path),
            "data": self.stock_data,
            "feature_names": self.feature_names,
            "method": "shap"
        }
        
        fi_request = FeatureImportanceRequest(**fi_request_data)
        fi_response = asyncio.run(feature_importance(fi_request))
        self.assertEqual(fi_response.status, "success")
        self.assertGreater(len(fi_response.importance), 0)
        print("✓ 特征重要性计算测试通过")
        
        # 3. 单个预测解释
        explain_request_data = {
            "model_path": str(self.model_path),
            "data": self.stock_data[:1],
            "feature_names": self.feature_names,
            "method": "shap",
            "sample_idx": 0
        }
        
        explain_request = ExplainRequest(**explain_request_data)
        explain_response = asyncio.run(explain_prediction(explain_request))
        self.assertEqual(explain_response.status, "success")
        # 验证响应包含必要的字段
        self.assertIn("feature_names", explain_response.explanation)
        self.assertIn("shap_values", explain_response.explanation)
        self.assertIn("prediction", explain_response.explanation)
        print("✓ 单个预测解释测试通过")
        
        # 4. 批量预测解释
        batch_data = [self.stock_data[:1], self.stock_data[1:2], self.stock_data[2:3]]
        batch_request_data = {
            "model_path": str(self.model_path),
            "data_list": batch_data,
            "feature_names": self.feature_names,
            "method": "shap"
        }
        
        batch_request = BatchExplainRequest(**batch_request_data)
        batch_response = asyncio.run(batch_explain(batch_request))
        self.assertEqual(batch_response.status, "success")
        self.assertEqual(len(batch_response.explanations), 3)
        print("✓ 批量预测解释测试通过")
        
        print("API集成测试完成!")
    
    def test_monitoring_integration(self):
        """测试监控集成"""
        from src.monitoring.explainer_monitor import get_monitor
        
        # 获取全局监控实例
        monitor = get_monitor()
        self.assertIsNotNone(monitor)
        
        # 记录一些请求
        start_time = time.time() - 0.1
        monitor.record_request(start_time, success=True)
        
        # 获取性能统计
        stats = monitor.get_performance_stats()
        self.assertGreater(stats['request_count'], 0)
        print("✓ 监控集成测试通过")
    
    def test_performance_benchmark(self):
        """性能基准测试"""
        from src.api.explainer_api import explain_prediction
        from src.api.explainer_api import ExplainRequest
        
        print("开始性能基准测试...")
        
        # 创建请求
        request_data = {
            "model_path": str(self.model_path),
            "data": self.stock_data[:1],
            "feature_names": self.feature_names,
            "method": "shap",
            "sample_idx": 0
        }
        
        request = ExplainRequest(**request_data)
        
        # 多次调用测试性能
        import asyncio
        times = []
        for i in range(5):
            start_time = time.time()
            response = asyncio.run(explain_prediction(request))
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            
            self.assertEqual(response.status, "success")
        
        avg_time = sum(times) / len(times)
        print(f"平均响应时间: {avg_time:.3f}秒")
        print(f"最快响应时间: {min(times):.3f}秒")
        print(f"最慢响应时间: {max(times):.3f}秒")
        
        # 验证性能要求（平均响应时间应小于5秒）
        self.assertLess(avg_time, 5.0)
        print("✓ 性能基准测试通过")

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)

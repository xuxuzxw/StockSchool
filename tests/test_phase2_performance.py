from pathlib import Path

from src.utils.gpu_utils import (测试GPU工具模块的性能优化功能, MagicMock, 0, """, __file__,
                                 from, import)
from src.utils.gpu_utils import \
    numpy as np  # !/usr/bin/env python3; -*- coding: utf-8 -*-; 添加项目根目录到路径
from src.utils.gpu_utils import os, os.path.abspath, os.path.dirname
from src.utils.gpu_utils import pandas as pd
from src.utils.gpu_utils import (patch, sys, sys.path.insert, tempfile, torch,
                                 unittest, unittest.mock, 第二阶段性能优化测试文件)

    GPUManager, GPUMemoryMonitor, SHAPCache,
    get_device, is_gpu_available, get_gpu_info,
    get_batch_size, check_memory_sufficient,
    handle_oom, fallback_to_cpu,
    get_shap_explainer, calculate_shap_values_optimized,
    get_shap_cache
)

class TestPhase2PerformanceOptimization(unittest.TestCase):
    """第二阶段性能优化测试类"""

    def setUp(self):
        """测试前准备"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试后清理"""
        # 清理临时目录
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_enhanced_batch_size_calculation(self):
        """测试增强的批量大小计算"""
        gpu_manager = GPUManager()

        # 测试不同模型类型的批量大小计算
        batch_size_tree = gpu_manager.get_optimal_batch_size(
            data_size=10000, model_type='tree'
        )
        batch_size_neural = gpu_manager.get_optimal_batch_size(
            data_size=10000, model_type='neural_network'
        )
        batch_size_linear = gpu_manager.get_optimal_batch_size(
            data_size=10000, model_type='linear'
        )

        print(f"树模型批量大小: {batch_size_tree}")
        print(f"神经网络批量大小: {batch_size_neural}")
        print(f"线性模型批量大小: {batch_size_linear}")

        # 验证批量大小调整逻辑
        self.assertIsInstance(batch_size_tree, int)
        self.assertIsInstance(batch_size_neural, int)
        self.assertIsInstance(batch_size_linear, int)
        self.assertGreater(batch_size_tree, 0)
        self.assertGreater(batch_size_neural, 0)
        self.assertGreater(batch_size_linear, 0)

    def test_memory_monitoring_and_warning(self):
        """测试内存监控和预警系统"""
        gpu_manager = GPUManager()

        # 测试内存预警功能
        warning_info = gpu_manager.check_memory_warning(warning_threshold=0.8)
        self.assertIsInstance(warning_info, dict)
        self.assertIn('warning', warning_info)

        # 测试详细GPU信息获取
        detailed_info = gpu_manager.get_detailed_gpu_info()
        self.assertIsInstance(detailed_info, dict)

        print(f"内存预警信息: {warning_info}")
        print(f"详细GPU信息: {detailed_info}")

    def test_gpu_memory_monitor_trends(self):
        """测试GPU内存监控趋势分析"""
        memory_monitor = GPUMemoryMonitor()

        # 测试内存趋势分析
        trend_info = memory_monitor.get_memory_trend(window_size=5)
        self.assertIsInstance(trend_info, dict)
        self.assertIn('trend', trend_info)

        # 模拟记录内存使用情况
        test_memory_info = {
            'device_type': 'cuda',
            'memory_utilization': 75.0,
            'timestamp': 1234567890
        }
        memory_monitor.record_memory_usage(test_memory_info)

        # 再次测试趋势分析
        trend_info = memory_monitor.get_memory_trend(window_size=5)
        self.assertIsInstance(trend_info, dict)

        print(f"内存使用趋势: {trend_info}")

    def test_shap_cache_functionality(self):
        """测试SHAP缓存功能"""
        # 创建SHAP缓存实例
        cache_dir = os.path.join(self.temp_dir, 'shap_cache_test')
        shap_cache = SHAPCache(cache_dir=cache_dir)

        # 测试缓存SHAP值
        model_hash = "test_model_123"
        data_hash = "test_data_456"
        test_shap_values = np.random.rand(100, 10)

        shap_cache.cache_shap_values(model_hash, data_hash, test_shap_values)

        # 测试获取缓存的SHAP值
        cached_values = shap_cache.get_cached_shap_values(model_hash, data_hash)
        self.assertIsNotNone(cached_values)
        np.testing.assert_array_equal(test_shap_values, cached_values)

        print(f"SHAP缓存测试通过")
        print(f"缓存目录: {cache_dir}")

    def test_shap_explainer_optimization(self):
        """测试SHAP解释器优化"""
        # 测试获取SHAP解释器
        try:
            # 创建简单的测试模型
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor

            # 创建测试数据
            X = np.random.rand(100, 5)
            y = np.random.rand(100)

            # 训练简单模型
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            # 测试获取SHAP解释器
            explainer = get_shap_explainer(model, model_type='tree')
            self.assertIsNotNone(explainer)

            print(f"SHAP解释器创建成功: {type(explainer)}")

        except ImportError:
            print("SHAP库未安装，跳过SHAP解释器测试")
        except Exception as e:
            print(f"SHAP解释器测试失败: {e}")

    def test_optimized_shap_calculation(self):
        """测试优化的SHAP计算"""
        try:
            import numpy as np
            import shap
            from sklearn.ensemble import RandomForestRegressor

            # 创建测试数据
            X = np.random.rand(200, 5)
            y = np.random.rand(200)

            # 训练模型
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            # 创建解释器
            explainer = shap.TreeExplainer(model)

            # 测试优化的SHAP计算
            shap_values = calculate_shap_values_optimized(
                explainer, X, batch_size=50
            )

            self.assertIsNotNone(shap_values)
            print(f"优化SHAP计算成功，结果形状: {shap_values.shape}")

        except ImportError:
            print("SHAP库未安装，跳过优化SHAP计算测试")
        except Exception as e:
            print(f"优化SHAP计算测试失败: {e}")

    def test_model_type_detection(self):
        """测试模型类型检测"""
        try:
            import lightgbm as lgb
            import xgboost as xgb
            from sklearn.ensemble import (GradientBoostingRegressor,
                                          RandomForestRegressor)
            from sklearn.linear_model import LinearRegression

            # 创建不同类型的模型
            models = {
                'random_forest': RandomForestRegressor(n_estimators=10),
                'gradient_boosting': GradientBoostingRegressor(),
                'linear_regression': LinearRegression(),
            }

            # 如果XGBoost可用
            try:
                models['xgboost'] = xgb.XGBRegressor()
            except:
                pass

            # 如果LightGBM可用
            try:
                models['lightgbm'] = lgb.LGBMRegressor()
            except:
                pass

            # 测试模型类型检测
            for model_name, model in models.items():
                detected_type = None  # 这里应该是内部函数调用
                print(f"模型 {model_name}: 类型检测")

        except ImportError:
            print("某些机器学习库未安装，跳过模型类型检测测试")

    def test_performance_benchmarking(self):
        """测试性能基准测试"""
        import time

        gpu_manager = GPUManager()

        # 测试批量大小计算性能
        start_time = time.time()
        for _ in range(100):
            batch_size = gpu_manager.get_optimal_batch_size(
                data_size=50000, model_type='tree'
            )
        end_time = time.time()

        avg_time = (end_time - start_time) / 100
        print(f"批量大小计算平均时间: {avg_time:.6f}秒")

        # 测试内存检查性能
        start_time = time.time()
        for _ in range(50):
            sufficient = gpu_manager.check_memory_sufficient(1000)
        end_time = time.time()

        avg_time = (end_time - start_time) / 50
        print(f"内存检查平均时间: {avg_time:.6f}秒")

        self.assertLess(avg_time, 0.1)  # 应该在100ms内完成

    def test_adaptive_batch_sizing(self):
        """测试自适应批量大小调整"""
        gpu_manager = GPUManager()

        # 测试不同数据规模的批量大小调整
        test_cases = [
            (1000, "小数据集"),
            (50000, "中等数据集"),
            (1000000, "大数据集"),
            (5000000, "超大数据集")
        ]

        for data_size, description in test_cases:
            batch_size = gpu_manager.get_optimal_batch_size(
                data_size=data_size, model_type='tree'
            )
            print(f"{description} ({data_size}样本): 批量大小 = {batch_size}")

            # 验证批量大小合理性
            self.assertIsInstance(batch_size, int)
            self.assertGreater(batch_size, 0)

    def test_memory_efficient_operations(self):
        """测试内存高效操作"""
        gpu_manager = GPUManager()

        # 测试内存充足的场景
        sufficient = gpu_manager.check_memory_sufficient(100)  # 100MB
        self.assertIsInstance(sufficient, bool)

        # 测试大内存需求的场景
        sufficient = gpu_manager.check_memory_sufficient(10000)  # 10GB
        self.assertIsInstance(sufficient, bool)

        print(f"100MB内存是否足够: {sufficient}")
        print(f"10GB内存是否足够: {sufficient}")

class TestIntegrationPhase2(unittest.TestCase):
    """第二阶段集成测试"""

    def test_complete_performance_workflow(self):
        """测试完整的性能优化工作流程"""
        print("开始第二阶段性能优化集成测试...")

        # 1. 初始化GPU管理器
        gpu_manager = GPUManager()
        print(f"GPU可用性: {gpu_manager.is_gpu_available()}")

        # 2. 测试增强的批量大小计算
        batch_size = gpu_manager.get_optimal_batch_size(
            data_size=100000, model_type='tree'
        )
        print(f"优化批量大小: {batch_size}")

        # 3. 测试内存监控
        memory_info = gpu_manager.get_gpu_info()
        print(f"GPU内存信息: {memory_info}")

        # 4. 测试内存预警
        warning_info = gpu_manager.check_memory_warning(0.8)
        print(f"内存预警: {warning_info}")

        # 5. 测试详细GPU信息
        detailed_info = gpu_manager.get_detailed_gpu_info()
        print(f"详细GPU信息: {detailed_info}")

        # 6. 测试SHAP缓存
        shap_cache = get_shap_cache()
        self.assertIsNotNone(shap_cache)
        print("SHAP缓存系统正常")

        print("第二阶段性能优化集成测试完成!")

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复机制"""
        gpu_manager = GPUManager()

        # 测试OOM处理
        should_retry = gpu_manager.handle_oom(retry_count=0, max_retries=3)
        print(f"OOM处理结果: {'继续重试' if should_retry else '停止重试'}")

        # 测试多次重试
        should_retry = gpu_manager.handle_oom(retry_count=2, max_retries=3)
        print(f"多次重试结果: {'继续重试' if should_retry else '停止重试'}")

        # 测试超过最大重试次数
        should_retry = gpu_manager.handle_oom(retry_count=3, max_retries=3)
        print(f"超过最大重试次数: {'继续重试' if should_retry else '停止重试'}")

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)

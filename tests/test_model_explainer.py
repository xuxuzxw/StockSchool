from pathlib import Path

from src.strategy.model_explainer import (MagicMock, ModelExplainer模块测试文件, 0,
                                          """, __file__, from, import)
from src.strategy.model_explainer import \
    numpy as np  # !/usr/bin/env python3; -*- coding: utf-8 -*-; 添加项目根目录到路径
from src.strategy.model_explainer import os, os.path.abspath, os.path.dirname
from src.strategy.model_explainer import pandas as pd
from src.strategy.model_explainer import (patch, sys, sys.path.insert,
                                          tempfile, torch, unittest,
                                          unittest.mock, 测试模型解释器的各项功能)

    ModelExplainer, ModelExplainerError,
    create_model_explainer, explain_model_predictions
)

class TestModelExplainer(unittest.TestCase):
    """ModelExplainer测试类"""

    def setUp(self):
        """测试前准备"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()

        # 创建测试数据
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.rand(100, 5),
                             columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        self.y = pd.Series(np.random.rand(100))

        # 创建测试模型
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)

        # 创建解释器
        self.explainer = ModelExplainer(self.model, self.X.columns.tolist())

    def tearDown(self):
        """测试后清理"""
        # 清理临时目录
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """测试初始化"""
        explainer = ModelExplainer(self.model, self.X.columns.tolist())
        self.assertIsNotNone(explainer)
        self.assertEqual(explainer.feature_names, self.X.columns.tolist())
        self.assertIsNotNone(explainer.device)

    def test_get_model_type(self):
        """测试模型类型检测"""
        model_type = self.explainer._get_model_type()
        self.assertEqual(model_type, 'tree')

    def test_default_feature_importance(self):
        """测试默认特征重要性计算"""
        importance = self.explainer.calculate_feature_importance(self.X, method='default')
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)
        self.assertEqual(len(importance), len(self.X.columns))
        self.assertTrue((importance['importance'] >= 0).all())

    def test_permutation_feature_importance(self):
        """测试排列特征重要性计算"""
        importance = self.explainer.calculate_feature_importance(self.X, self.y, method='permutation')
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)
        self.assertIn('std', importance.columns)
        self.assertEqual(len(importance), len(self.X.columns))
        self.assertTrue((importance['importance'] >= 0).all())

    def test_shap_feature_importance(self):
        """测试SHAP特征重要性计算"""
        importance = self.explainer.calculate_feature_importance(self.X, method='shap')
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)
        self.assertEqual(len(importance), len(self.X.columns))
        self.assertTrue((importance['importance'] >= 0).all())

    def test_invalid_method(self):
        """测试无效方法"""
        with self.assertRaises(ModelExplainerError):
            self.explainer.calculate_feature_importance(self.X, method='invalid')

    def test_permutation_without_y(self):
        """测试不提供y的排列重要性计算"""
        # 清除缓存以确保重新计算
        cache_key = self.explainer._get_cache_key('importance_permutation', str(hash(str(self.X.shape) + str(self.X.columns.tolist()))))
        cache_file = self.explainer.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_file.unlink()

        with self.assertRaises(ModelExplainerError):
            self.explainer.calculate_feature_importance(self.X, method='permutation')

    def test_prediction_explanation(self):
        """测试预测解释"""
        explanation = self.explainer.explain_prediction(self.X, sample_idx=0)
        self.assertIsInstance(explanation, dict)
        self.assertIn('sample_idx', explanation)
        self.assertIn('feature_names', explanation)
        self.assertIn('feature_values', explanation)
        self.assertIn('shap_values', explanation)
        self.assertIn('base_value', explanation)
        self.assertIn('prediction', explanation)
        self.assertIn('model_type', explanation)

        # 验证数据一致性
        self.assertEqual(len(explanation['feature_names']), len(explanation['feature_values']))
        self.assertEqual(len(explanation['feature_names']), len(explanation['shap_values']))

    def test_feature_interactions(self):
        """测试特征交互分析"""
        interactions = self.explainer.analyze_feature_interactions(self.X)
        self.assertIsInstance(interactions, pd.DataFrame)
        if len(interactions) > 0:
            self.assertIn('feature1', interactions.columns)
            self.assertIn('feature2', interactions.columns)
            self.assertIn('correlation', interactions.columns)
            self.assertIn('interaction_strength', interactions.columns)

    def test_model_summary(self):
        """测试模型摘要生成"""
        summary = self.explainer.generate_model_summary(self.X, self.y)
        self.assertIsInstance(summary, dict)
        self.assertIn('model_type', summary)
        self.assertIn('feature_count', summary)
        self.assertIn('feature_names', summary)
        self.assertIn('device', summary)
        self.assertIn('cache_enabled', summary)
        self.assertIn('samples_count', summary)
        self.assertIn('target_stats', summary)

    def test_model_summary_without_y(self):
        """测试不提供y的模型摘要生成"""
        summary = self.explainer.generate_model_summary(self.X)
        self.assertIsInstance(summary, dict)
        self.assertIn('model_type', summary)
        self.assertIn('feature_count', summary)
        self.assertIn('feature_names', summary)
        self.assertIn('device', summary)
        self.assertIn('cache_enabled', summary)
        self.assertIn('samples_count', summary)
        self.assertNotIn('target_stats', summary)

    def test_cache_functionality(self):
        """测试缓存功能"""
        # 第一次计算
        importance1 = self.explainer.calculate_feature_importance(self.X, method='default')

        # 第二次计算（应该从缓存获取）
        importance2 = self.explainer.calculate_feature_importance(self.X, method='default')

        # 验证结果一致性
        pd.testing.assert_frame_equal(importance1, importance2)

    def test_convenience_functions(self):
        """测试便捷函数"""
        # 测试create_model_explainer
        explainer = create_model_explainer(self.model, self.X.columns.tolist())
        self.assertIsInstance(explainer, ModelExplainer)

        # 测试explain_model_predictions
        results = explain_model_predictions(
            self.model, self.X, self.X.columns.tolist(),
            method='default', sample_idx=0
        )
        self.assertIsInstance(results, dict)
        self.assertIn('feature_importance', results)
        self.assertIn('prediction_explanation', results)
        self.assertIn('model_summary', results)

class TestModelExplainerErrorHandling(unittest.TestCase):
    """ModelExplainer错误处理测试"""

    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.rand(100, 5),
                             columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        self.y = pd.Series(np.random.rand(100))

        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)

    def test_shap_initialization_failure(self):
        """测试SHAP初始化失败处理"""
        # 创建一个会导致SHAP初始化失败的解释器
        explainer = ModelExplainer(self.model, self.X.columns.tolist())

        # 清除缓存以确保重新计算
        cache_key = explainer._get_cache_key('importance_shap', str(hash(str(self.X.shape) + str(self.X.columns.tolist()))))
        cache_file = explainer.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_file.unlink()

        # 模拟SHAP初始化失败
        with patch.object(explainer, '_initialize_shap_explainer') as mock_init:
            mock_init.side_effect = Exception("SHAP初始化失败")

            with self.assertRaises(ModelExplainerError):
                explainer.calculate_feature_importance(self.X, method='shap')

    def test_oom_handling(self):
        """测试OOM处理"""
        explainer = ModelExplainer(self.model, self.X.columns.tolist())

        # 模拟OOM错误
        with patch.object(explainer, '_initialize_shap_explainer') as mock_init:
            mock_init.side_effect = torch.cuda.OutOfMemoryError("显存不足")

            # 这应该触发降级到CPU
            try:
                explainer.calculate_feature_importance(self.X, method='shap')
            except ModelExplainerError:
                pass  # 期望的错误处理

    def test_large_data_handling(self):
        """测试大数据处理"""
        # 创建大型数据集
        large_X = pd.DataFrame(np.random.rand(15000, 10),
                              columns=[f'feature_{i}' for i in range(10)])
        large_y = pd.Series(np.random.rand(15000))

        explainer = ModelExplainer(self.model, large_X.columns.tolist())

        # 测试SHAP重要性计算（应该自动采样）
        importance = explainer.calculate_feature_importance(large_X, method='shap')
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertEqual(len(importance), len(large_X.columns))

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_complete_workflow(self):
        """测试完整工作流程"""
        print("开始ModelExplainer集成测试...")

        # 创建测试数据
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(200, 8),
                        columns=[f'feature_{i}' for i in range(8)])
        y = pd.Series(np.random.rand(200))

        # 创建不同类型的模型
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        # 测试树模型
        tree_model = RandomForestRegressor(n_estimators=20, random_state=42)
        tree_model.fit(X, y)

        tree_explainer = ModelExplainer(tree_model, X.columns.tolist())

        # 测试所有解释方法
        methods = ['default', 'permutation', 'shap']
        for method in methods:
            if method == 'permutation':
                importance = tree_explainer.calculate_feature_importance(X, y, method=method)
            else:
                importance = tree_explainer.calculate_feature_importance(X, method=method)

            self.assertIsInstance(importance, pd.DataFrame)
            print(f"树模型 {method} 方法测试通过，特征数: {len(importance)}")

        # 测试线性模型
        linear_model = LinearRegression()
        linear_model.fit(X, y)

        linear_explainer = ModelExplainer(linear_model, X.columns.tolist())

        # 测试默认和SHAP方法
        for method in ['default', 'shap']:
            importance = linear_explainer.calculate_feature_importance(X, method=method)
            self.assertIsInstance(importance, pd.DataFrame)
            print(f"线性模型 {method} 方法测试通过，特征数: {len(importance)}")

        # 测试预测解释
        explanation = tree_explainer.explain_prediction(X, sample_idx=0)
        self.assertIsInstance(explanation, dict)
        print(f"预测解释测试通过，特征数: {len(explanation['feature_names'])}")

        # 测试特征交互分析
        interactions = tree_explainer.analyze_feature_interactions(X)
        self.assertIsInstance(interactions, pd.DataFrame)
        print(f"特征交互分析测试通过")

        # 测试模型摘要
        summary = tree_explainer.generate_model_summary(X, y)
        self.assertIsInstance(summary, dict)
        print(f"模型摘要测试通过")

        print("ModelExplainer集成测试完成!")

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)

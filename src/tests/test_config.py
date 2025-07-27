import unittest
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config_loader import Config, config


class TestConfig(unittest.TestCase):
    """测试配置加载器"""
    
    def setUp(self):
        """测试前准备"""
        self.config = Config()
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        config1 = Config()
        config2 = Config()
        self.assertIs(config1, config2)
        self.assertIs(config1, config)
    
    def test_config_loading(self):
        """测试配置文件加载"""
        # 测试配置文件是否成功加载
        self.assertIsNotNone(self.config.all_config)
        self.assertIsInstance(self.config.all_config, dict)
    
    def test_get_existing_config(self):
        """测试获取存在的配置项"""
        # 测试获取因子参数
        rsi_window = self.config.get('factor_params.rsi.window')
        self.assertEqual(rsi_window, 14)
        
        # 测试获取嵌套配置
        ma_windows = self.config.get('factor_params.ma.windows')
        self.assertEqual(ma_windows, [5, 10, 20, 60])
        
        # 测试获取策略参数
        risk_free_rate = self.config.get('strategy_params.risk_free_rate')
        self.assertEqual(risk_free_rate, 0.03)
        
        # 测试获取数据同步参数
        batch_size = self.config.get('data_sync_params.batch_size')
        self.assertEqual(batch_size, 1000)
    
    def test_get_nonexistent_config(self):
        """测试获取不存在的配置项"""
        # 测试不存在的配置项返回None
        result = self.config.get('nonexistent.config.key')
        self.assertIsNone(result)
        
        # 测试不存在的配置项返回默认值
        result = self.config.get('nonexistent.config.key', 'default_value')
        self.assertEqual(result, 'default_value')
        
        # 测试部分路径存在但最终键不存在
        result = self.config.get('factor_params.nonexistent_key')
        self.assertIsNone(result)
        
        result = self.config.get('factor_params.nonexistent_key', 100)
        self.assertEqual(result, 100)
    
    def test_get_with_default_values(self):
        """测试带默认值的获取"""
        # 测试存在的配置项，默认值应该被忽略
        result = self.config.get('factor_params.rsi.window', 999)
        self.assertEqual(result, 14)
        
        # 测试不存在的配置项，应该返回默认值
        result = self.config.get('factor_params.unknown_param', 42)
        self.assertEqual(result, 42)
    
    def test_nested_config_access(self):
        """测试嵌套配置访问"""
        # 测试多层嵌套
        lgbm_learning_rate = self.config.get('training_params.lgbm_params.learning_rate')
        self.assertEqual(lgbm_learning_rate, 0.05)
        
        # 测试数组配置
        ma_windows = self.config.get('factor_params.ma.windows')
        self.assertIsInstance(ma_windows, list)
        self.assertIn(20, ma_windows)
    
    def test_config_data_types(self):
        """测试配置数据类型"""
        # 测试整数
        rsi_window = self.config.get('factor_params.rsi.window')
        self.assertIsInstance(rsi_window, int)
        
        # 测试浮点数
        risk_free_rate = self.config.get('strategy_params.risk_free_rate')
        self.assertIsInstance(risk_free_rate, float)
        
        # 测试列表
        ma_windows = self.config.get('factor_params.ma.windows')
        self.assertIsInstance(ma_windows, list)
        
        # 测试字符串
        model_name = self.config.get('training_params.model_name')
        self.assertIsInstance(model_name, str)
    
    def test_invalid_key_path(self):
        """测试无效的键路径"""
        # 测试空字符串
        result = self.config.get('')
        self.assertEqual(result, self.config.all_config)
        
        # 测试单个键
        result = self.config.get('factor_params')
        self.assertIsInstance(result, dict)
        
        # 测试访问非字典类型的子键
        result = self.config.get('factor_params.rsi.window.invalid')
        self.assertIsNone(result)
    
    def test_global_config_instance(self):
        """测试全局配置实例"""
        from src.utils.config_loader import config as global_config
        
        # 测试全局实例可用
        self.assertIsNotNone(global_config)
        
        # 测试全局实例与类实例相同
        self.assertIs(global_config, self.config)
        
        # 测试全局实例功能正常
        rsi_window = global_config.get('factor_params.rsi.window')
        self.assertEqual(rsi_window, 14)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
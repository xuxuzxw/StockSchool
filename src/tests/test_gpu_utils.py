from src.config.unified_config import config
from src.utils.gpu_utils import GPU工具模块测试文件, 测试GPU工具模块的各项功能, MagicMock, 0, """, __file__, from, import
from src.utils.gpu_utils import numpy as np  # !/usr/bin/env python3; -*- coding: utf-8 -*-; 添加项目根目录到路径
from src.utils.gpu_utils import os, os.path.abspath, os.path.dirname
from src.utils.gpu_utils import pandas as pd
from src.utils.gpu_utils import patch, sys, sys.path.insert, torch, unittest, unittest.mock

    GPUManager, get_device, is_gpu_available, get_gpu_info,
    get_batch_size, check_memory_sufficient, handle_oom, fallback_to_cpu
)

class TestGPUManager(unittest.TestCase):
    """GPU管理器测试类"""

    def setUp(self):
        """测试前准备"""
        # 重置配置
        config.reload()

    def test_gpu_manager_initialization(self):
        """测试GPU管理器初始化"""
        gpu_manager = GPUManager()
        self.assertIsNotNone(gpu_manager)
        self.assertTrue(gpu_manager.initialized)

    @patch('torch.cuda.is_available')
    def test_get_device_with_cuda_available(self, mock_is_available):
        """测试获取设备 - CUDA可用"""
        mock_is_available.return_value = True
        with patch('torch.cuda.get_device_name') as mock_get_name:
            mock_get_name.return_value = 'Test GPU'
            gpu_manager = GPUManager()
            device = gpu_manager.get_device()
            self.assertEqual(device.type, 'cuda')

    @patch('torch.cuda.is_available')
    def test_get_device_with_cuda_unavailable(self, mock_is_available):
        """测试获取设备 - CUDA不可用"""
        mock_is_available.return_value = False
        gpu_manager = GPUManager()
        device = gpu_manager.get_device()
        self.assertEqual(device.type, 'cpu')

    @patch('torch.cuda.is_available')
    def test_is_gpu_available(self, mock_is_available):
        """测试GPU可用性检测"""
        # CUDA可用
        mock_is_available.return_value = True
        gpu_manager = GPUManager()
        self.assertTrue(gpu_manager.is_gpu_available())

        # CUDA不可用
        mock_is_available.return_value = False
        gpu_manager = GPUManager()
        self.assertFalse(gpu_manager.is_gpu_available())

    @patch('torch.cuda.is_available')
    def test_get_gpu_info_cpu_mode(self, mock_is_available):
        """测试获取GPU信息 - CPU模式"""
        mock_is_available.return_value = False
        gpu_manager = GPUManager()
        info = gpu_manager.get_gpu_info()
        self.assertEqual(info['device_type'], 'cpu')
        self.assertIn('cpu_count', info)
        self.assertIn('cpu_memory_total', info)

    @patch('torch.cuda.is_available')
    def test_get_gpu_info_cuda_mode(self, mock_is_available):
        """测试获取GPU信息 - CUDA模式"""
        mock_is_available.return_value = True
        with patch('torch.cuda.get_device_name') as mock_get_name:
            mock_get_name.return_value = 'Test GPU'
            with patch('torch.cuda.device_count') as mock_device_count:
                mock_device_count.return_value = 1
                gpu_manager = GPUManager()
                info = gpu_manager.get_gpu_info()
                self.assertEqual(info['device_type'], 'cuda')
                self.assertEqual(info['device_name'], 'Test GPU')
                self.assertEqual(info['gpu_count'], 1)

    def test_get_optimal_batch_size_cpu_mode(self):
        """测试获取最优批量大小 - CPU模式"""
        with patch('src.utils.gpu_utils.is_gpu_available', return_value=False):
            gpu_manager = GPUManager()
            batch_size = gpu_manager.get_optimal_batch_size()
            self.assertIsInstance(batch_size, int)
            self.assertGreater(batch_size, 0)

    @patch('torch.cuda.is_available')
    def test_get_optimal_batch_size_cuda_mode(self, mock_is_available):
        """测试获取最优批量大小 - CUDA模式"""
        mock_is_available.return_value = True
        gpu_manager = GPUManager()
        batch_size = gpu_manager.get_optimal_batch_size()
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)

    def test_check_memory_sufficient_cpu_mode(self):
        """测试内存检查 - CPU模式"""
        with patch('src.utils.gpu_utils.is_gpu_available', return_value=False):
            gpu_manager = GPUManager()
            sufficient = gpu_manager.check_memory_sufficient(100)
            self.assertIsInstance(sufficient, bool)

    def test_handle_oom(self):
        """测试内存不足处理"""
        gpu_manager = GPUManager()
        # 测试第一次重试
        result = gpu_manager.handle_oom(retry_count=0, max_retries=3)
        self.assertTrue(result)

        # 测试超过最大重试次数
        result = gpu_manager.handle_oom(retry_count=3, max_retries=3)
        self.assertFalse(result)

    @patch('torch.cuda.is_available')
    def test_fallback_to_cpu(self, mock_is_available):
        """测试降级到CPU"""
        mock_is_available.return_value = True
        gpu_manager = GPUManager()
        original_device = gpu_manager.get_device()

        # 降级到CPU
        new_device = gpu_manager.fallback_to_cpu()
        self.assertEqual(new_device.type, 'cpu')
        self.assertFalse(config.get('feature_params.use_cuda', True))

class TestConvenienceFunctions(unittest.TestCase):
    """便捷函数测试类"""

    def setUp(self):
        """测试前准备"""
        config.reload()

    def test_get_device_function(self):
        """测试获取设备便捷函数"""
        device = get_device()
        self.assertIsInstance(device, torch.device)

    def test_is_gpu_available_function(self):
        """测试GPU可用性检测便捷函数"""
        available = is_gpu_available()
        self.assertIsInstance(available, bool)

    def test_get_gpu_info_function(self):
        """测试获取GPU信息便捷函数"""
        info = get_gpu_info()
        self.assertIsInstance(info, dict)

    def test_get_batch_size_function(self):
        """测试获取批量大小便捷函数"""
        batch_size = get_batch_size()
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)

    def test_check_memory_sufficient_function(self):
        """测试内存检查便捷函数"""
        sufficient = check_memory_sufficient(100)
        self.assertIsInstance(sufficient, bool)

    def test_handle_oom_function(self):
        """测试内存不足处理便捷函数"""
        result = handle_oom(retry_count=0, max_retries=3)
        self.assertTrue(result)

    def test_fallback_to_cpu_function(self):
        """测试降级到CPU便捷函数"""
        device = fallback_to_cpu()
        self.assertEqual(device.type, 'cpu')

class TestIntegration(unittest.TestCase):
    """集成测试类"""

    def setUp(self):
        """测试前准备"""
        config.reload()

    def test_full_workflow(self):
        """测试完整工作流程"""
        # 1. 初始化GPU管理器
        gpu_manager = GPUManager()

        # 2. 检查设备
        device = gpu_manager.get_device()
        self.assertIsInstance(device, torch.device)

        # 3. 检查GPU可用性
        gpu_available = gpu_manager.is_gpu_available()
        self.assertIsInstance(gpu_available, bool)

        # 4. 获取GPU信息
        gpu_info = gpu_manager.get_gpu_info()
        self.assertIsInstance(gpu_info, dict)

        # 5. 获取批量大小
        batch_size = gpu_manager.get_optimal_batch_size()
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)

        # 6. 检查内存
        memory_sufficient = gpu_manager.check_memory_sufficient(100)
        self.assertIsInstance(memory_sufficient, bool)

        # 7. 测试内存不足处理
        handle_result = gpu_manager.handle_oom(retry_count=0, max_retries=3)
        self.assertTrue(handle_result)

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)

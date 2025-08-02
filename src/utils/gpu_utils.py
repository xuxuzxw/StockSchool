#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU工具模块

该模块提供GPU相关的工具函数，包括：
1. GPU可用性检测
2. 动态批量大小计算
3. GPU内存监控
4. 自动降级策略
5. 性能优化和监控

作者: StockSchool Team
创建时间: 2025-07-31
"""

import torch
import logging
from typing import Optional, Dict, Any
import psutil
import time
import numpy as np
from pathlib import Path
from src.utils.config_loader import config
from src.monitoring.logger import get_logger

logger = get_logger()

# 尝试导入pynvml用于GPU监控
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("pynvml未安装，GPU监控功能不可用")

class GPUMemoryMonitor:
    """GPU内存监控器"""
    
    def __init__(self):
        self.memory_history = []
        self.max_history_size = 1000
    
    def record_memory_usage(self, memory_info: Dict[str, Any]):
        """记录内存使用情况"""
        if memory_info.get('device_type') == 'cuda':
            memory_info['timestamp'] = time.time()
            self.memory_history.append(memory_info)
            
            # 限制历史记录大小
            if len(self.memory_history) > self.max_history_size:
                self.memory_history.pop(0)
    
    def get_memory_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """获取内存使用趋势"""
        if len(self.memory_history) < window_size:
            return {'trend': 'insufficient_data'}
        
        recent_data = self.memory_history[-window_size:]
        utilizations = [data.get('memory_utilization', 0) for data in recent_data if 'memory_utilization' in data]
        
        if not utilizations:
            return {'trend': 'no_data'}
        
        # 计算趋势
        if len(utilizations) >= 2:
            x = np.arange(len(utilizations))
            slope = np.polyfit(x, utilizations, 1)[0]
        else:
            slope = 0
        
        return {
            'trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'slope': slope,
            'average_utilization': sum(utilizations) / len(utilizations),
            'recent_utilizations': utilizations[-5:] if len(utilizations) >= 5 else utilizations
        }

class GPUManager:
    """GPU管理器"""
    
    def __init__(self):
        """初始化GPU管理器"""
        self.device = self._get_device()
        self.initialized = True
        self.memory_monitor = GPUMemoryMonitor()
        
        # 初始化NVML（如果可用）
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except Exception as e:
                logger.warning(f"NVML初始化失败: {e}")
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False
    
    def _get_device(self) -> torch.device:
        """获取计算设备"""
        use_cuda = config.get('feature_params.use_cuda', True)
        
        if use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"使用GPU设备: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            if use_cuda:
                logger.warning("CUDA不可用，降级到CPU")
            else:
                logger.info("使用CPU设备（CUDA已禁用）")
        
        return device
    
    def is_gpu_available(self) -> bool:
        """检查GPU是否可用
        
        Returns:
            bool: GPU是否可用
        """
        return self.device.type == "cuda"
    
    def get_device(self) -> torch.device:
        """获取当前计算设备
        
        Returns:
            torch.device: 当前计算设备
        """
        return self.device
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息
        
        Returns:
            Dict[str, Any]: GPU信息字典
        """
        if not self.is_gpu_available():
            return {
                'device_type': 'cpu',
                'cpu_count': psutil.cpu_count(),
                'cpu_memory_total': psutil.virtual_memory().total // (1024**3),
                'cpu_memory_available': psutil.virtual_memory().available // (1024**3)
            }
        
        try:
            if self.nvml_initialized:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                return {
                    'device_type': 'cuda',
                    'device_name': torch.cuda.get_device_name(0),
                    'gpu_count': torch.cuda.device_count(),
                    'memory_total': memory_info.total // (1024**2),  # MB
                    'memory_used': memory_info.used // (1024**2),    # MB
                    'memory_free': memory_info.free // (1024**2),    # MB
                    'gpu_utilization': utilization.gpu,
                    'memory_utilization': utilization.memory,
                    'temperature': temperature
                }
            else:
                # 使用PyTorch获取基本信息
                return {
                    'device_type': 'cuda',
                    'device_name': torch.cuda.get_device_name(0),
                    'gpu_count': torch.cuda.device_count(),
                    'memory_allocated': torch.cuda.memory_allocated(0) // (1024**2),  # MB
                    'memory_reserved': torch.cuda.memory_reserved(0) // (1024**2),    # MB
                    'memory_cached': torch.cuda.memory_cached(0) // (1024**2)         # MB
                }
        except Exception as e:
            logger.error(f"获取GPU信息失败: {e}")
            return {
                'device_type': 'cuda' if self.is_gpu_available() else 'cpu',
                'error': str(e)
            }
    
    def check_memory_warning(self, warning_threshold: float = 0.8) -> Dict[str, Any]:
        """检查内存使用预警
        
        Args:
            warning_threshold: 预警阈值 (0.0-1.0)
        
        Returns:
            Dict[str, Any]: 预警信息
        """
        memory_info = self.get_gpu_info()
        if memory_info.get('device_type') == 'cuda':
            memory_utilization = memory_info.get('memory_utilization', 0)
            if memory_utilization > (warning_threshold * 100):
                return {
                    'warning': True,
                    'utilization': memory_utilization,
                    'threshold': warning_threshold * 100,
                    'message': f"GPU内存使用率 {memory_utilization}% 超过预警阈值 {warning_threshold * 100}%"
                }
        
        return {'warning': False}
    
    def get_detailed_gpu_info(self) -> Dict[str, Any]:
        """获取详细的GPU信息（包括功耗、温度等）
        
        Returns:
            Dict[str, Any]: 详细的GPU信息
        """
        if not self.is_gpu_available() or not self.nvml_initialized:
            return self.get_gpu_info()
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            
            detailed_info = {
                'device_type': 'cuda',
                'device_name': torch.cuda.get_device_name(0),
                'gpu_count': torch.cuda.device_count(),
                'memory_total': memory_info.total // (1024**2),  # MB
                'memory_used': memory_info.used // (1024**2),    # MB
                'memory_free': memory_info.free // (1024**2),    # MB
                'gpu_utilization': utilization.gpu,
                'memory_utilization': utilization.memory,
                'temperature': temperature,
                'temperature_unit': 'C',
                'power_usage': power // 1000,  # W
                'power_limit': power_limit // 1000,  # W
                'power_utilization': (power / power_limit) * 100 if power_limit > 0 else 0,
                'timestamp': time.time()
            }
            
            # 记录到内存监控器
            self.memory_monitor.record_memory_usage(detailed_info)
            
            return detailed_info
            
        except Exception as e:
            logger.warning(f"获取详细GPU信息失败: {e}")
            return self.get_gpu_info()
    
    def get_optimal_batch_size(self, data_size: int = None, 
                              model_type: str = None,
                              model_memory_estimate: int = None) -> int:
        """计算最优批量大小
        
        Args:
            data_size: 数据大小（可选）
            model_type: 模型类型 ('tree', 'linear', 'neural_network', 'ensemble')
            model_memory_estimate: 模型内存估计（MB，可选）
        
        Returns:
            int: 最优批量大小
        """
        # 从配置获取默认批量大小
        default_batch_size = config.get('feature_params.shap_batch_size', 500)
        
        # 根据模型类型调整批量大小策略
        if model_type == 'tree':
            # 树模型可以处理更大批量
            base_batch = default_batch_size * 2
        elif model_type == 'neural_network':
            # 神经网络需要更小批量以避免内存溢出
            base_batch = default_batch_size // 2
        elif model_type == 'linear':
            # 线性模型适中
            base_batch = default_batch_size
        elif model_type == 'ensemble':
            # 集成模型需要更小批量
            base_batch = default_batch_size // 3
        else:
            base_batch = default_batch_size
        
        # 根据数据规模调整
        if data_size:
            if data_size > 1000000:  # 1M+ 数据
                base_batch = max(50, base_batch // 4)
                logger.info(f"大数据集({data_size})调整批量大小: {base_batch}")
            elif data_size > 100000:  # 100K+ 数据
                base_batch = max(100, base_batch // 2)
                logger.info(f"中等数据集({data_size})调整批量大小: {base_batch}")
            elif data_size > 10000:  # 10K+ 数据
                base_batch = max(200, base_batch)
                logger.info(f"小数据集({data_size})调整批量大小: {base_batch}")
        
        if not self.is_gpu_available():
            # CPU模式，根据内存调整
            memory_available = psutil.virtual_memory().available // (1024**2)  # MB
            if memory_available < 4096:  # 4GB
                final_batch = max(50, base_batch // 4)
            elif memory_available < 8192:  # 8GB
                final_batch = max(100, base_batch // 2)
            else:
                final_batch = base_batch
            
            if final_batch != base_batch:
                logger.info(f"CPU模式内存调整: {base_batch} -> {final_batch}")
            return final_batch
        
        # GPU模式，根据显存调整
        try:
            if self.nvml_initialized:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_free = memory_info.free // (1024**2)  # MB
                
                # 从配置获取最大GPU内存限制
                max_gpu_memory = config.get('feature_params.max_gpu_memory', 20480)  # 20GB
                
                # 根据可用显存调整批量大小
                if memory_free < 1024:  # 1GB
                    final_batch = max(50, base_batch // 8)
                elif memory_free < 2048:  # 2GB
                    final_batch = max(100, base_batch // 4)
                elif memory_free < 4096:  # 4GB
                    final_batch = max(200, base_batch // 2)
                elif memory_free > max_gpu_memory * 0.8:  # 80%以上可用
                    final_batch = min(base_batch * 2, 2000)
                else:
                    final_batch = base_batch
            else:
                # 没有NVML，使用PyTorch内存信息
                memory_reserved = torch.cuda.memory_reserved(0) // (1024**2)  # MB
                if memory_reserved > 10000:  # 10GB
                    final_batch = max(100, base_batch // 2)
                elif memory_reserved > 5000:  # 5GB
                    final_batch = base_batch
                else:
                    final_batch = min(base_batch * 2, 1000)
                    
            if final_batch != base_batch:
                logger.info(f"GPU内存调整: {base_batch} -> {final_batch}")
            return final_batch
                    
        except Exception as e:
            logger.warning(f"计算最优批量大小时出错: {e}")
            return base_batch
    
    def check_memory_sufficient(self, required_memory: int) -> bool:
        """检查内存是否足够
        
        Args:
            required_memory: 所需内存（MB）
        
        Returns:
            bool: 内存是否足够
        """
        if not self.is_gpu_available():
            memory_available = psutil.virtual_memory().available // (1024**2)  # MB
            return memory_available > required_memory * 1.2  # 预留20%缓冲
        else:
            try:
                if self.nvml_initialized:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_free = memory_info.free // (1024**2)  # MB
                    return memory_free > required_memory * 1.2  # 预留20%缓冲
                else:
                    # 使用PyTorch估算
                    memory_reserved = torch.cuda.memory_reserved(0)
                    memory_allocated = torch.cuda.memory_allocated(0)
                    memory_free = (memory_reserved - memory_allocated) // (1024**2)  # MB
                    return memory_free > required_memory * 1.2
            except Exception as e:
                logger.warning(f"检查内存时出错: {e}")
                return True  # 默认返回True以避免阻塞
    
    def handle_oom(self, retry_count: int = 0, max_retries: int = 3) -> bool:
        """处理内存不足情况
        
        Args:
            retry_count: 当前重试次数
            max_retries: 最大重试次数
        
        Returns:
            bool: 是否应该继续重试
        """
        logger.warning(f"检测到内存不足，重试次数: {retry_count + 1}/{max_retries}")
        
        # 清理GPU缓存
        if self.is_gpu_available():
            torch.cuda.empty_cache()
            logger.info("已清理GPU缓存")
        
        # 从配置获取最大重试次数
        max_oom_retries = config.get('feature_params.gpu_oom_retry', 3)
        
        if retry_count < min(max_retries, max_oom_retries):
            # 降低批量大小
            current_batch_size = config.get('feature_params.shap_batch_size', 500)
            new_batch_size = max(50, current_batch_size // 2)
            config.set('feature_params.shap_batch_size', new_batch_size)
            logger.info(f"批量大小已调整: {current_batch_size} -> {new_batch_size}")
            
            # 等待一段时间让内存释放
            import time
            time.sleep(1)
            
            return True
        else:
            logger.error("达到最大重试次数，无法继续处理")
            return False
    
    def fallback_to_cpu(self) -> torch.device:
        """降级到CPU设备
        
        Returns:
            torch.device: CPU设备
        """
        logger.warning("降级到CPU模式")
        config.set('feature_params.use_cuda', False)
        self.device = torch.device("cpu")
        return self.device

# 全局GPU管理器实例
_gpu_manager = None

def get_gpu_manager() -> GPUManager:
    """获取全局GPU管理器实例
    
    Returns:
        GPUManager: GPU管理器实例
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager

def get_device() -> torch.device:
    """获取计算设备的便捷函数
    
    Returns:
        torch.device: 计算设备
    """
    return get_gpu_manager().get_device()

def is_gpu_available() -> bool:
    """检查GPU是否可用的便捷函数
    
    Returns:
        bool: GPU是否可用
    """
    return get_gpu_manager().is_gpu_available()

def get_gpu_info() -> Dict[str, Any]:
    """获取GPU信息的便捷函数
    
    Returns:
        Dict[str, Any]: GPU信息
    """
    return get_gpu_manager().get_gpu_info()

def get_batch_size(data_size: int = None, model_memory_estimate: int = None) -> int:
    """获取最优批量大小的便捷函数
    
    Args:
        data_size: 数据大小（可选）
        model_memory_estimate: 模型内存估计（MB，可选）
    
    Returns:
        int: 最优批量大小
    """
    return get_gpu_manager().get_optimal_batch_size(data_size, model_memory_estimate)

def check_memory_sufficient(required_memory: int) -> bool:
    """检查内存是否足够的便捷函数
    
    Args:
        required_memory: 所需内存（MB）
    
    Returns:
        bool: 内存是否足够
    """
    return get_gpu_manager().check_memory_sufficient(required_memory)

def handle_oom(retry_count: int = 0, max_retries: int = 3) -> bool:
    """处理内存不足的便捷函数
    
    Args:
        retry_count: 当前重试次数
        max_retries: 最大重试次数
    
    Returns:
        bool: 是否应该继续重试
    """
    return get_gpu_manager().handle_oom(retry_count, max_retries)

def fallback_to_cpu() -> torch.device:
    """降级到CPU的便捷函数
    
    Returns:
        torch.device: CPU设备
    """
    return get_gpu_manager().fallback_to_cpu()

# 初始化GPU管理器
def initialize_gpu_manager():
    """初始化GPU管理器"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager

# SHAP缓存类
class SHAPCache:
    """SHAP计算缓存"""
    
    def __init__(self, cache_dir: str = './shap_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = {}
    
    def get_cached_shap_values(self, model_hash: str, data_hash: str) -> Optional[np.ndarray]:
        """获取缓存的SHAP值"""
        cache_key = f"{model_hash}_{data_hash}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 检查磁盘缓存
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                shap_values = np.load(cache_file)
                self.cache[cache_key] = shap_values
                return shap_values
            except Exception as e:
                logger.warning(f"加载SHAP缓存失败: {e}")
        
        return None
    
    def cache_shap_values(self, model_hash: str, data_hash: str, shap_values: np.ndarray):
        """缓存SHAP值"""
        cache_key = f"{model_hash}_{data_hash}"
        self.cache[cache_key] = shap_values
        
        # 保存到磁盘
        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, shap_values)
        except Exception as e:
            logger.warning(f"保存SHAP缓存失败: {e}")

# 全局SHAP缓存实例
_shap_cache = None

def get_shap_cache() -> SHAPCache:
    """获取全局SHAP缓存实例"""
    global _shap_cache
    if _shap_cache is None:
        _shap_cache = SHAPCache()
    return _shap_cache

def get_shap_explainer(model, model_type: str = 'auto'):
    """获取优化的SHAP解释器
    
    Args:
        model: 机器学习模型
        model_type: 模型类型 ('auto', 'tree', 'linear', 'neural_network')
    
    Returns:
        SHAP解释器
    """
    try:
        import shap
        
        if model_type == 'auto':
            model_type = _detect_model_type(model)
        
        if model_type in ['tree', 'xgboost', 'lightgbm', 'catboost']:
            # 使用TreeSHAP
            return shap.TreeExplainer(model, model_output='raw_values')
        elif model_type in ['linear', 'logistic']:
            # 使用LinearSHAP
            return shap.LinearExplainer(model)
        else:
            # 使用通用解释器
            return shap.Explainer(model)
    except ImportError:
        logger.error("SHAP库未安装")
        return None
    except Exception as e:
        logger.error(f"创建SHAP解释器失败: {e}")
        return None

def _detect_model_type(model) -> str:
    """检测模型类型"""
    model_class = model.__class__.__name__.lower()
    
    if any(x in model_class for x in ['xgb', 'xgboost', 'lgb', 'lightgbm', 'catboost']):
        return 'tree'
    elif any(x in model_class for x in ['linear', 'logistic', 'ridge', 'lasso']):
        return 'linear'
    elif any(x in model_class for x in ['neural', 'mlp', 'cnn', 'rnn']):
        return 'neural_network'
    else:
        return 'unknown'

def calculate_shap_values_optimized(explainer, data, batch_size: int = None):
    """优化的大数据集SHAP计算
    
    Args:
        explainer: SHAP解释器
        data: 输入数据
        batch_size: 批量大小
    
    Returns:
        SHAP值数组
    """
    if batch_size is None:
        batch_size = get_batch_size(len(data) if hasattr(data, '__len__') else None)
    
    # 小数据集直接计算
    if hasattr(data, '__len__') and len(data) <= batch_size:
        return explainer.shap_values(data)
    
    # 大数据集分批计算
    all_shap_values = []
    data_length = len(data) if hasattr(data, '__len__') else data.shape[0]
    
    for i in range(0, data_length, batch_size):
        if hasattr(data, 'iloc'):
            # Pandas DataFrame
            batch_data = data.iloc[i:i+batch_size]
        elif hasattr(data, 'shape'):
            # NumPy array
            batch_data = data[i:i+batch_size]
        else:
            # List or other iterable
            batch_data = data[i:i+batch_size]
        
        batch_shap_values = explainer.shap_values(batch_data)
        all_shap_values.append(batch_shap_values)
    
    # 合并结果
    if isinstance(all_shap_values[0], list):
        # 多输出情况
        merged_values = []
        for output_idx in range(len(all_shap_values[0])):
            output_values = [batch[output_idx] for batch in all_shap_values]
            merged_values.append(np.concatenate(output_values, axis=0))
        return merged_values
    else:
        # 单输出情况
        return np.concatenate(all_shap_values, axis=0)

if __name__ == '__main__':
    # 测试代码
    print("测试GPU工具模块...")
    
    # 初始化GPU管理器
    gpu_manager = initialize_gpu_manager()
    
    # 测试GPU可用性
    print(f"GPU可用: {gpu_manager.is_gpu_available()}")
    
    # 测试设备获取
    device = gpu_manager.get_device()
    print(f"当前设备: {device}")
    
    # 测试GPU信息
    gpu_info = gpu_manager.get_gpu_info()
    print(f"GPU信息: {gpu_info}")
    
    # 测试批量大小计算
    batch_size = gpu_manager.get_optimal_batch_size()
    print(f"最优批量大小: {batch_size}")
    
    # 测试内存检查
    memory_sufficient = gpu_manager.check_memory_sufficient(1000)
    print(f"1000MB内存是否足够: {memory_sufficient}")
    
    # 测试内存预警
    warning_info = gpu_manager.check_memory_warning(0.8)
    print(f"内存预警信息: {warning_info}")
    
    # 测试详细GPU信息
    detailed_info = gpu_manager.get_detailed_gpu_info()
    print(f"详细GPU信息: {detailed_info}")
    
    print("GPU工具模块测试完成!")

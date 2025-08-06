"""
测试配置管理 - 提供类型安全的配置管理
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


@dataclass
class DateRange:
    """日期范围配置"""
    start: str
    end: str
    
    def __post_init__(self):
        """验证日期格式"""
        try:
            datetime.strptime(self.start, '%Y-%m-%d')
            datetime.strptime(self.end, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"日期格式错误: {e}")


@dataclass
class ConcurrentTestConfig:
    """并发测试配置"""
    concurrent_users: int = 5
    concurrent_operations: int = 10
    max_error_rate: float = 0.1
    
    def __post_init__(self):
        if self.concurrent_users <= 0:
            raise ValueError("并发用户数必须大于0")
        if self.concurrent_operations <= 0:
            raise ValueError("并发操作数必须大于0")
        if not 0 <= self.max_error_rate <= 1:
            raise ValueError("错误率必须在0-1之间")


@dataclass
class PerformanceThresholds:
    """性能阈值配置"""
    max_sync_time_per_stock: float = 1.0  # 秒
    max_factor_calc_time: float = 5.0     # 秒
    max_model_training_time: float = 300.0  # 秒
    max_prediction_time: float = 1.0      # 秒
    min_accuracy_threshold: float = 0.6   # 最小准确率


@dataclass
class IntegrationTestConfig:
    """集成测试配置"""
    test_stocks: List[str] = field(default_factory=lambda: ['000001.SZ', '000002.SZ', '600000.SH'])
    test_date_range: DateRange = field(default_factory=lambda: DateRange('2024-01-01', '2024-01-31'))
    timeout_seconds: int = 300
    concurrent_config: ConcurrentTestConfig = field(default_factory=ConcurrentTestConfig)
    performance_thresholds: PerformanceThresholds = field(default_factory=PerformanceThresholds)
    
    # 测试开关
    enable_e2e_test: bool = True
    enable_concurrent_test: bool = True
    enable_fault_recovery_test: bool = True
    enable_long_running_test: bool = False  # 默认关闭长时间测试
    enable_business_scenario_test: bool = True
    
    # 数据库配置
    use_test_database: bool = True
    cleanup_after_test: bool = True
    
    def __post_init__(self):
        """配置验证"""
        if self.timeout_seconds <= 0:
            raise ValueError("超时时间必须大于0")
        
        if not self.test_stocks:
            raise ValueError("测试股票列表不能为空")
        
        # 验证股票代码格式
        for stock in self.test_stocks:
            if not self._is_valid_stock_code(stock):
                raise ValueError(f"无效的股票代码: {stock}")
    
    @staticmethod
    def _is_valid_stock_code(code: str) -> bool:
        """验证股票代码格式"""
        if not isinstance(code, str) or len(code) != 9:
            return False
        
        # 检查格式：6位数字.2位交易所代码
        parts = code.split('.')
        if len(parts) != 2:
            return False
        
        stock_num, exchange = parts
        if not stock_num.isdigit() or len(stock_num) != 6:
            return False
        
        if exchange not in ['SZ', 'SH']:
            return False
        
        return True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IntegrationTestConfig':
        """从字典创建配置"""
        # 处理嵌套配置
        if 'test_date_range' in config_dict and isinstance(config_dict['test_date_range'], dict):
            config_dict['test_date_range'] = DateRange(**config_dict['test_date_range'])
        
        if 'concurrent_config' in config_dict and isinstance(config_dict['concurrent_config'], dict):
            config_dict['concurrent_config'] = ConcurrentTestConfig(**config_dict['concurrent_config'])
        
        if 'performance_thresholds' in config_dict and isinstance(config_dict['performance_thresholds'], dict):
            config_dict['performance_thresholds'] = PerformanceThresholds(**config_dict['performance_thresholds'])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._config_cache = {}
    
    def get_integration_config(self, config_dict: Optional[Dict[str, Any]] = None) -> IntegrationTestConfig:
        """获取集成测试配置"""
        if config_dict is None:
            config_dict = {}
        
        # 使用缓存键
        cache_key = str(sorted(config_dict.items()))
        
        if cache_key not in self._config_cache:
            self._config_cache[cache_key] = IntegrationTestConfig.from_dict(config_dict)
        
        return self._config_cache[cache_key]
    
    def validate_config(self, config: IntegrationTestConfig) -> List[str]:
        """验证配置并返回警告信息"""
        warnings = []
        
        # 检查测试股票数量
        if len(config.test_stocks) > 10:
            warnings.append(f"测试股票数量较多({len(config.test_stocks)})，可能影响测试性能")
        
        # 检查日期范围
        start_date = datetime.strptime(config.test_date_range.start, '%Y-%m-%d')
        end_date = datetime.strptime(config.test_date_range.end, '%Y-%m-%d')
        date_diff = (end_date - start_date).days
        
        if date_diff > 365:
            warnings.append(f"测试日期范围过长({date_diff}天)，建议缩短以提高测试效率")
        
        # 检查并发配置
        if config.concurrent_config.concurrent_users > 20:
            warnings.append("并发用户数过高，可能导致系统资源不足")
        
        return warnings


# 全局配置管理器
config_manager = ConfigManager()
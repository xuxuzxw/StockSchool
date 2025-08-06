"""
性能测试专用异常类
"""


class PerformanceTestError(Exception):
    """性能测试基础异常"""
    pass


class SystemMetricsError(PerformanceTestError):
    """系统指标获取异常"""
    pass


class LoadTestError(PerformanceTestError):
    """负载测试异常"""
    pass


class StressTestError(PerformanceTestError):
    """压力测试异常"""
    pass


class DatabasePerformanceError(PerformanceTestError):
    """数据库性能测试异常"""
    pass


class GPUPerformanceError(PerformanceTestError):
    """GPU性能测试异常"""
    pass


class ConfigurationError(PerformanceTestError):
    """配置错误异常"""
    pass


class ThresholdExceededError(PerformanceTestError):
    """性能阈值超限异常"""
    
    def __init__(self, metric_name: str, actual_value: float, threshold: float):
        self.metric_name = metric_name
        self.actual_value = actual_value
        self.threshold = threshold
        super().__init__(
            f"{metric_name} 超过阈值: {actual_value} > {threshold}"
        )
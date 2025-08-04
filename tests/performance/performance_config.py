#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试配置管理
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import date, timedelta

@dataclass
class TestDataConfig:
    """测试数据配置"""
    stock_count: int = 20
    days_range: int = 90
    initial_price: float = 100.0
    volatility: float = 0.02
    
    @property
    def end_date(self) -> date:
        return date.today() - timedelta(days=1)
    
    @property
    def start_date(self) -> date:
        return self.end_date - timedelta(days=self.days_range)

@dataclass
class PerformanceBenchmark:
    """性能基准配置"""
    max_execution_time: float
    max_memory_mb: float
    min_throughput: float
    max_avg_time_per_item: float
    
    def validate_metrics(self, execution_time: float, memory_mb: float, 
                        data_count: int) -> List[str]:
        """验证性能指标，返回违规项列表"""
        violations = []
        
        if execution_time > self.max_execution_time:
            violations.append(f"执行时间超限: {execution_time:.2f}s > {self.max_execution_time}s")
        
        if memory_mb > self.max_memory_mb:
            violations.append(f"内存使用超限: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
        
        if data_count > 0:
            throughput = data_count / execution_time
            if throughput < self.min_throughput:
                violations.append(f"吞吐量过低: {throughput:.2f} < {self.min_throughput}")
            
            avg_time = execution_time / data_count
            if avg_time > self.max_avg_time_per_item:
                violations.append(f"单项平均时间过长: {avg_time:.4f}s > {self.max_avg_time_per_item}s")
        
        return violations

class PerformanceTestConfig:
    """性能测试配置管理器"""
    
    DEFAULT_CONFIG = {
        'test_data': {
            'stock_count': 20,
            'days_range': 90,
            'initial_price': 100.0,
            'volatility': 0.02
        },
        'benchmarks': {
            'single_factor': {
                'max_execution_time': 5.0,
                'max_memory_mb': 200.0,
                'min_throughput': 100.0,
                'max_avg_time_per_item': 0.01
            },
            'multi_factor': {
                'max_execution_time': 30.0,
                'max_memory_mb': 500.0,
                'min_throughput': 50.0,
                'max_avg_time_per_item': 0.05
            },
            'concurrent': {
                'max_execution_time': 20.0,
                'max_memory_mb': 300.0,
                'min_throughput': 200.0,
                'max_avg_time_per_item': 0.02
            },
            'large_dataset': {
                'max_execution_time': 60.0,
                'max_memory_mb': 1000.0,
                'min_throughput': 20.0,
                'max_avg_time_per_item': 0.1
            }
        },
        'test_stocks': [
            '000001.SZ', '000002.SZ', '000858.SZ', '000876.SZ',
            '600000.SH', '600036.SH', '600519.SH', '600887.SH',
            '000063.SZ', '000166.SZ', '000338.SZ', '000725.SZ'
        ],
        'factor_configs': [
            {'name': 'sma_5', 'window': 5, 'type': 'technical'},
            {'name': 'sma_20', 'window': 20, 'type': 'technical'},
            {'name': 'rsi_14', 'window': 14, 'type': 'technical'},
            {'name': 'macd', 'fast': 12, 'slow': 26, 'signal': 9, 'type': 'technical'}
        ]
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化配置管理器"""
        self.config = self._load_config(config_file)
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                # 合并默认配置和文件配置
                config = self.DEFAULT_CONFIG.copy()
                config.update(file_config)
                return config
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
        
        return self.DEFAULT_CONFIG.copy()
    
    def get_test_data_config(self) -> TestDataConfig:
        """获取测试数据配置"""
        data_config = self.config.get('test_data', {})
        return TestDataConfig(**data_config)
    
    def get_benchmark(self, test_type: str) -> PerformanceBenchmark:
        """获取性能基准"""
        benchmark_config = self.config.get('benchmarks', {}).get(test_type, {})
        if not benchmark_config:
            # 使用默认基准
            benchmark_config = self.DEFAULT_CONFIG['benchmarks']['single_factor']
        
        return PerformanceBenchmark(**benchmark_config)
    
    def get_test_stocks(self, count: Optional[int] = None) -> List[str]:
        """获取测试股票列表"""
        stocks = self.config.get('test_stocks', [])
        if count:
            return stocks[:count]
        return stocks
    
    def get_factor_configs(self, factor_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取因子配置"""
        factor_configs = self.config.get('factor_configs', [])
        if factor_type:
            return [config for config in factor_configs if config.get('type') == factor_type]
        return factor_configs
    
    def save_config(self, config_file: str):
        """保存配置到文件"""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"保存配置文件失败: {e}")

# 全局配置实例
performance_config = PerformanceTestConfig(
    os.path.join(os.path.dirname(__file__), 'performance_test_config.yml')
)
import logging
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Any, Dict, List

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试策略模式实现
"""


logger = logging.getLogger(__name__)

class PerformanceTestStrategy(ABC):
    """性能测试策略接口"""

    @abstractmethod
    def execute_test(self, engine, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """执行性能测试"""
        pass

    @abstractmethod
    def get_test_name(self) -> str:
        """获取测试名称"""
        pass

class SingleFactorTestStrategy(PerformanceTestStrategy):
    """单因子性能测试策略"""

    def execute_test(self, engine, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """执行单因子性能测试"""
        ts_codes = test_params.get('ts_codes', ['000001.SZ'])
        factor_name = test_params.get('factor_name', 'sma_5')
        window = test_params.get('window', 5)

        if factor_name.startswith('sma'):
            result = engine.calculate_sma(
                ts_codes=ts_codes,
                start_date=date.today() - timedelta(days=30),
                end_date=date.today() - timedelta(days=1),
                window=window
            )
        elif factor_name.startswith('rsi'):
            result = engine.calculate_rsi(
                ts_codes=ts_codes,
                start_date=date.today() - timedelta(days=30),
                end_date=date.today() - timedelta(days=1),
                window=window
            )
        else:
            raise ValueError(f"不支持的因子类型: {factor_name}")

        return {
            'result': result,
            'factor_name': factor_name,
            'stock_count': len(ts_codes),
            'data_count': len(result) if not result.empty else 0
        }

    def get_test_name(self) -> str:
        """方法描述"""

class MultipleFactor TestStrategy(PerformanceTestStrategy):
    """多因子性能测试策略"""

    def execute_test(self, engine, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """执行多因子性能测试"""
        ts_codes = test_params.get('ts_codes', ['000001.SZ'])
        factor_configs = test_params.get('factor_configs', [
            {'name': 'sma_5', 'window': 5},
            {'name': 'sma_20', 'window': 20},
            {'name': 'rsi_14', 'window': 14}
        ])

        results = {}
        total_data_count = 0

        for config in factor_configs:
            factor_name = config['name']
            window = config['window']

            try:
                if factor_name.startswith('sma'):
                    result = engine.calculate_sma(ts_codes,
                                                date.today() - timedelta(days=30),
                                                date.today() - timedelta(days=1),
                                                window)
                elif factor_name.startswith('rsi'):
                    result = engine.calculate_rsi(ts_codes,
                                                date.today() - timedelta(days=30),
                                                date.today() - timedelta(days=1),
                                                window)
                else:
                    continue

                results[factor_name] = result
                total_data_count += len(result) if not result.empty else 0

            except Exception as e:
                logger.warning(f"因子 {factor_name} 计算失败: {e}")
                results[factor_name] = None

        return {
            'results': results,
            'factor_count': len(factor_configs),
            'successful_factors': len([r for r in results.values() if r is not None]),
            'total_data_count': total_data_count
        }

    def get_test_name(self) -> str:
        """方法描述"""

class ConcurrentTestStrategy(PerformanceTestStrategy):
    """并发性能测试策略"""

    def execute_test(self, engine, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """执行并发性能测试"""
        from concurrent.futures import ThreadPoolExecutor

        ts_codes = test_params.get('ts_codes', ['000001.SZ', '000002.SZ'])
        max_workers = test_params.get('max_workers', 4)
        factor_name = test_params.get('factor_name', 'sma_5')
        window = test_params.get('window', 5)

        def calculate_single_stock(ts_code):
            """单股票计算函数"""
            try:
                if factor_name.startswith('sma'):
                    result = engine.calculate_sma([ts_code],
                                                date.today() - timedelta(days=20),
                                                date.today() - timedelta(days=1),
                                                window)
                else:
                    result = engine.calculate_rsi([ts_code],
                                                date.today() - timedelta(days=20),
                                                date.today() - timedelta(days=1),
                                                window)

                return {
                    'ts_code': ts_code,
                    'success': True,
                    'data_count': len(result) if not result.empty else 0
                }
            except Exception as e:
                return {
                    'ts_code': ts_code,
                    'success': False,
                    'error': str(e),
                    'data_count': 0
                }

        # 并发执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            concurrent_results = list(executor.map(calculate_single_stock, ts_codes))

        successful_count = sum(1 for r in concurrent_results if r['success'])
        total_data_count = sum(r['data_count'] for r in concurrent_results)

        return {
            'concurrent_results': concurrent_results,
            'total_stocks': len(ts_codes),
            'successful_stocks': successful_count,
            'total_data_count': total_data_count,
            'max_workers': max_workers
        }

    def get_test_name(self) -> str:
        """方法描述"""

class PerformanceTestContext:
    """性能测试上下文 - 策略模式的上下文类"""

    def __init__(self, strategy: PerformanceTestStrategy):
        """方法描述"""

    def set_strategy(self, strategy: PerformanceTestStrategy):
        """设置测试策略"""
        self._strategy = strategy

    def execute_performance_test(self, engine, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """执行性能测试"""
        logger.info(f"执行 {self._strategy.get_test_name()}")
        return self._strategy.execute_test(engine, test_params)
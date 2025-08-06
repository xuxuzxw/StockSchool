"""
性能测试工厂
"""
from typing import Dict, Any, Type
from .strategies import (
    PerformanceTestStrategy,
    SystemBaselineStrategy,
    LoadTestStrategy,
    PerformanceTestError
)


class PerformanceTestFactory:
    """性能测试策略工厂"""
    
    _strategies: Dict[str, Type[PerformanceTestStrategy]] = {
        'system_baseline': SystemBaselineStrategy,
        'load_test': LoadTestStrategy,
        # 可以继续添加其他策略
    }
    
    @classmethod
    def create_strategy(
        self, 
        strategy_name: str, 
        config: Dict[str, Any]
    ) -> PerformanceTestStrategy:
        """创建性能测试策略"""
        if strategy_name not in self._strategies:
            raise PerformanceTestError(f"未知的性能测试策略: {strategy_name}")
        
        strategy_class = self._strategies[strategy_name]
        return strategy_class(config)
    
    @classmethod
    def register_strategy(
        cls, 
        name: str, 
        strategy_class: Type[PerformanceTestStrategy]
    ):
        """注册新的性能测试策略"""
        cls._strategies[name] = strategy_class
    
    @classmethod
    def get_available_strategies(cls) -> list:
        """获取可用的策略列表"""
        return list(cls._strategies.keys())


class PerformanceTestExecutor:
    """性能测试执行器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.factory = PerformanceTestFactory()
    
    def execute_test(self, strategy_name: str) -> Dict[str, Any]:
        """执行指定的性能测试"""
        try:
            # 创建策略
            strategy = self.factory.create_strategy(strategy_name, self.config)
            
            # 执行测试
            results = strategy.execute_test()
            
            # 验证结果
            validation = strategy.validate_results(results)
            
            return {
                'strategy': strategy_name,
                'results': results,
                'validation': validation,
                'success': validation['passed']
            }
            
        except PerformanceTestError as e:
            return {
                'strategy': strategy_name,
                'error': str(e),
                'success': False
            }
        except Exception as e:
            return {
                'strategy': strategy_name,
                'error': f"未预期的错误: {e}",
                'success': False
            }
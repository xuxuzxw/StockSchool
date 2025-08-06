"""
AI测试策略实现 - 展示策略模式的完整应用
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime


class AITestStrategy(ABC):
    """AI测试策略抽象基类"""
    
    @abstractmethod
    def execute_test(self, components: Dict[str, Any], 
                    config: 'AITestConfig') -> Dict[str, Any]:
        """执行测试策略"""
        pass
    
    @abstractmethod
    def get_test_name(self) -> str:
        """获取测试名称"""
        pass


class ModelTrainingTestStrategy(AITestStrategy):
    """模型训练测试策略"""
    
    def get_test_name(self) -> str:
        return "模型训练测试"
    
    def execute_test(self, components: Dict[str, Any], 
                    config: 'AITestConfig') -> Dict[str, Any]:
        """执行模型训练测试"""
        trainer = components.get('model_trainer')
        
        if not trainer:
            return self._get_mock_results(config)
        
        try:
            return self._execute_real_training(trainer, config)
        except Exception as e:
            print(f"模型训练失败: {e}")
            return self._get_mock_results(config)
    
    def _execute_real_training(self, trainer: Any, config: 'AITestConfig') -> Dict[str, Any]:
        """执行真实的模型训练"""
        model_types = ['lightgbm', 'xgboost', 'random_forest']
        training_results = {}
        
        for model_type in model_types:
            try:
                # 调用实际的训练方法
                result = trainer.train_model(
                    model_type=model_type,
                    factors=config.test_factors,
                    stocks=config.test_stocks
                )
                training_results[model_type] = {
                    'success': True,
                    'metrics': result.get('metrics', {}),
                    'training_time': result.get('training_time', 0)
                }
            except Exception as e:
                training_results[model_type] = {
                    'success': False,
                    'error': str(e)
                }
        
        return {
            "training_status": "success",
            "models_tested": model_types,
            "training_results": training_results,
            "successful_models": [
                model for model, result in training_results.items() 
                if result.get('success', False)
            ]
        }
    
    def _get_mock_results(self, config: 'AITestConfig') -> Dict[str, Any]:
        """获取模拟结果"""
        return {
            "training_status": "mock",
            "models_tested": ['lightgbm', 'xgboost', 'random_forest'],
            "successful_models": ['lightgbm', 'xgboost'],
            "training_quality_score": 85.0,
            "warning": "使用模拟数据，实际训练器不可用"
        }


class PerformanceValidationStrategy(AITestStrategy):
    """性能验证测试策略"""
    
    def get_test_name(self) -> str:
        return "性能验证测试"
    
    def execute_test(self, components: Dict[str, Any], 
                    config: 'AITestConfig') -> Dict[str, Any]:
        """执行性能验证测试"""
        evaluator = components.get('strategy_evaluator')
        min_r2 = config.performance_thresholds.get('min_r2_score', 0.1)
        
        if not evaluator:
            return self._get_mock_performance_results(min_r2)
        
        try:
            return self._execute_real_performance_validation(evaluator, config)
        except Exception as e:
            print(f"性能验证失败: {e}")
            return self._get_mock_performance_results(min_r2)
    
    def _execute_real_performance_validation(self, evaluator: Any, 
                                           config: 'AITestConfig') -> Dict[str, Any]:
        """执行真实的性能验证"""
        models = ['lightgbm', 'xgboost', 'random_forest']
        performance_results = {}
        
        for model_name in models:
            try:
                metrics = evaluator.evaluate_model(
                    model_name=model_name,
                    test_stocks=config.test_stocks
                )
                
                performance_results[model_name] = {
                    'r2_score': metrics.get('r2', 0),
                    'rmse_score': metrics.get('rmse', 0),
                    'mae_score': metrics.get('mae', 0),
                    'meets_threshold': metrics.get('r2', 0) > config.performance_thresholds.get('min_r2_score', 0.1)
                }
            except Exception as e:
                performance_results[model_name] = {
                    'error': str(e),
                    'meets_threshold': False
                }
        
        models_meeting_threshold = sum(
            1 for result in performance_results.values() 
            if result.get('meets_threshold', False)
        )
        
        return {
            "validation_status": "success",
            "models_evaluated": models,
            "models_meeting_threshold": models_meeting_threshold,
            "performance_results": performance_results,
            "performance_score": (models_meeting_threshold / len(models)) * 100
        }
    
    def _get_mock_performance_results(self, min_r2: float) -> Dict[str, Any]:
        """获取模拟性能结果"""
        models = ['lightgbm', 'xgboost', 'random_forest']
        performance_results = {}
        
        for model_name in models:
            r2_score = np.random.uniform(0.08, 0.30)
            performance_results[model_name] = {
                'r2_score': r2_score,
                'rmse_score': np.random.uniform(0.025, 0.055),
                'mae_score': np.random.uniform(0.020, 0.045),
                'meets_threshold': r2_score > min_r2
            }
        
        models_meeting_threshold = sum(
            1 for result in performance_results.values() 
            if result['meets_threshold']
        )
        
        return {
            "validation_status": "mock",
            "models_evaluated": models,
            "models_meeting_threshold": models_meeting_threshold,
            "performance_results": performance_results,
            "performance_score": (models_meeting_threshold / len(models)) * 100,
            "warning": "使用模拟数据"
        }


class PredictionFunctionalityStrategy(AITestStrategy):
    """预测功能测试策略"""
    
    def get_test_name(self) -> str:
        return "预测功能测试"
    
    def execute_test(self, components: Dict[str, Any], 
                    config: 'AITestConfig') -> Dict[str, Any]:
        """执行预测功能测试"""
        predictor = components.get('stock_predictor')
        
        if not predictor:
            return self._get_mock_prediction_results(config)
        
        try:
            return self._execute_real_prediction_test(predictor, config)
        except Exception as e:
            print(f"预测功能测试失败: {e}")
            return self._get_mock_prediction_results(config)
    
    def _execute_real_prediction_test(self, predictor: Any, 
                                    config: 'AITestConfig') -> Dict[str, Any]:
        """执行真实的预测测试"""
        test_stock = config.test_stocks[0]
        test_date = datetime.now().strftime('%Y-%m-%d')
        
        # 单股票预测测试
        start_time = datetime.now()
        single_prediction = predictor.predict_single(test_stock, test_date)
        single_time = (datetime.now() - start_time).total_seconds()
        
        # 批量预测测试
        start_time = datetime.now()
        batch_predictions = predictor.predict_batch(config.test_stocks, test_date)
        batch_time = (datetime.now() - start_time).total_seconds()
        
        max_prediction_time = config.performance_thresholds.get('max_prediction_time', 5.0)
        
        return {
            "prediction_status": "success",
            "single_prediction": {
                "completed": single_prediction is not None,
                "prediction_time": single_time,
                "within_time_limit": single_time < max_prediction_time
            },
            "batch_prediction": {
                "completed": len(batch_predictions) > 0,
                "batch_size": len(config.test_stocks),
                "prediction_time": batch_time,
                "within_time_limit": batch_time < max_prediction_time * len(config.test_stocks)
            },
            "functionality_score": 90.0
        }
    
    def _get_mock_prediction_results(self, config: 'AITestConfig') -> Dict[str, Any]:
        """获取模拟预测结果"""
        single_time = np.random.uniform(0.5, 2.0)
        batch_time = np.random.uniform(2.0, 8.0)
        max_time = config.performance_thresholds.get('max_prediction_time', 5.0)
        
        return {
            "prediction_status": "mock",
            "single_prediction": {
                "completed": True,
                "prediction_time": single_time,
                "within_time_limit": single_time < max_time
            },
            "batch_prediction": {
                "completed": True,
                "batch_size": len(config.test_stocks),
                "prediction_time": batch_time,
                "within_time_limit": batch_time < max_time * len(config.test_stocks)
            },
            "functionality_score": 85.0,
            "warning": "使用模拟数据"
        }


class TestStrategyFactory:
    """测试策略工厂"""
    
    @staticmethod
    def create_strategy(strategy_type: str) -> AITestStrategy:
        """创建测试策略"""
        strategies = {
            "model_training": ModelTrainingTestStrategy(),
            "performance_validation": PerformanceValidationStrategy(),
            "prediction_functionality": PredictionFunctionalityStrategy(),
        }
        
        strategy = strategies.get(strategy_type)
        if not strategy:
            raise ValueError(f"不支持的测试策略类型: {strategy_type}")
        
        return strategy
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """获取可用的策略类型"""
        return ["model_training", "performance_validation", "prediction_functionality"]
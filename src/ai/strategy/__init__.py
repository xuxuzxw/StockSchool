"""AI策略系统核心模块

本模块包含AI策略系统的核心组件：
- AIModelManager: AI模型管理器，负责模型版本管理、训练、部署
- FactorWeightEngine: 因子权重引擎，基于SHAP值计算因子重要性
- StockScoringEngine: 股票评分引擎，计算股票综合评分和排名
- PredictionService: 预测服务，提供实时预测API

后续将添加：
- StrategyCustomizer: 策略定制器，根据用户偏好定制投资策略
- ModelExplainer: 模型解释器，提供模型决策解释
- BacktestEngine: 回测引擎，进行策略回测和性能评估
- ModelMonitor: 模型监控器，监控模型性能和数据漂移
"""

from .model_manager import AIModelManager
from .factor_weight_engine import FactorWeightEngine
from .stock_scoring_engine import StockScoringEngine
from .prediction_service import PredictionService
from .model_explainer import ModelExplainer
from .backtest_engine import BacktestEngine
from .strategy_customizer import StrategyCustomizer

__all__ = [
    'AIModelManager',
    'FactorWeightEngine', 
    'StockScoringEngine',
    'PredictionService',
    'ModelExplainer',
    'BacktestEngine',
    'StrategyCustomizer'
]
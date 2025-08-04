#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术面因子计算引擎
实现各类技术指标因子的计算
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import List, Dict, Optional, Any
from loguru import logger
from sqlalchemy import text

from .base_factor_engine import BaseFactorEngine, BaseFactorCalculator
from .factor_models import (
    FactorType, FactorCategory, FactorResult, FactorValue, 
    FactorMetadata, FactorConfig, CalculationStatus,
    create_factor_metadata, create_factor_config
)
from .abstract_factor_calculator import AbstractFactorCalculator
from .factor_calculation_config import TechnicalFactorCalculationConfig
from .data_validation_mixin import DataValidationError, InsufficientDataError
from .indicators import TechnicalIndicators


class MomentumFactorCalculator(AbstractFactorCalculator):
    """
    动量类因子计算器
    
    计算包括RSI、威廉指标、动量指标、变化率等动量类技术指标
    """
    
    def __init__(self, config: Optional[TechnicalFactorCalculationConfig] = None):
        """
        初始化动量因子计算器
        
        Args:
            config: 技术面因子计算配置
        """
        super().__init__(config)
        self.momentum_config = self.config.momentum
    
    def _get_required_columns(self) -> List[str]:
        """获取计算所需的数据列"""
        return ['close']
    
    def _get_minimum_data_length(self) -> int:
        """获取最小数据长度要求"""
        # 返回所有窗口期中的最大值
        max_window = max(
            max(self.momentum_config.rsi_windows, default=0),
            max(self.momentum_config.momentum_windows, default=0),
            max(self.momentum_config.roc_windows, default=0),
            max(self.momentum_config.williams_r_windows, default=0)
        )
        return max_window
    
    def _validate_specific_requirements(self, data: pd.DataFrame, **kwargs) -> None:
        """验证动量因子特定要求"""
        # 检查威廉指标所需的额外列
        if any(f'williams_r_{w}' in kwargs.get('factor_names', []) 
               for w in self.momentum_config.williams_r_windows):
            required_cols = ['high', 'low', 'close']
            self.validate_required_columns(data, required_cols)
    
    def _execute_calculation(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """执行动量类因子计算"""
        results = {}
        factor_names = kwargs.get('factor_names', None)
        
        # RSI指标
        for window in self.momentum_config.rsi_windows:
            factor_name = f'rsi_{window}'
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self._calculate_rsi(data, window)
        
        # 威廉指标（需要额外的high、low列）
        if all(col in data.columns for col in ['high', 'low', 'close']):
            for window in self.momentum_config.williams_r_windows:
                factor_name = f'williams_r_{window}'
                if factor_names is None or factor_name in factor_names:
                    results[factor_name] = self._calculate_williams_r(data, window)
        
        # 动量指标
        for window in self.momentum_config.momentum_windows:
            factor_name = f'momentum_{window}'
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self._calculate_momentum(data, window)
        
        # 变化率指标
        for window in self.momentum_config.roc_windows:
            factor_name = f'roc_{window}'
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self._calculate_roc(data, window)
        
        return results
    
    def _calculate_rsi(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        计算RSI指标
        
        Args:
            data: 包含close列的价格数据
            window: 计算窗口期
            
        Returns:
            RSI值序列
        """
        try:
            return TechnicalIndicators.rsi(data['close'], window)
        except Exception as e:
            logger.error(f"计算RSI({window})时出错: {e}")
            raise
    
    def _calculate_williams_r(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        计算威廉指标
        
        Args:
            data: 包含high、low、close列的价格数据
            window: 计算窗口期
            
        Returns:
            威廉指标值序列
        """
        try:
            return TechnicalIndicators.williams_r(
                data['high'], data['low'], data['close'], window
            )
        except Exception as e:
            logger.error(f"计算威廉指标({window})时出错: {e}")
            raise
    
    def _calculate_momentum(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        计算动量指标
        
        Args:
            data: 包含close列的价格数据
            window: 计算窗口期
            
        Returns:
            动量指标值序列
        """
        try:
            return TechnicalIndicators.momentum(data['close'], window)
        except Exception as e:
            logger.error(f"计算动量指标({window})时出错: {e}")
            raise
    
    def _calculate_roc(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        计算变化率指标
        
        Args:
            data: 包含close列的价格数据
            window: 计算窗口期
            
        Returns:
            变化率指标值序列
        """
        try:
            return data['close'].pct_change(window) * 100
        except Exception as e:
            logger.error(f"计算变化率指标({window})时出错: {e}")
            raise


class TrendFactorCalculator(AbstractFactorCalculator):
    """
    趋势类因子计算器
    
    计算包括移动平均线、MACD等趋势类技术指标
    """
    
    def __init__(self, config: Optional[TechnicalFactorCalculationConfig] = None):
        """
        初始化趋势因子计算器
        
        Args:
            config: 技术面因子计算配置
        """
        super().__init__(config)
        self.trend_config = self.config.trend
    
    def _get_required_columns(self) -> List[str]:
        """获取计算所需的数据列"""
        return ['close']
    
    def _get_minimum_data_length(self) -> int:
        """获取最小数据长度要求"""
        # 返回所有窗口期中的最大值
        max_window = max(
            max(self.trend_config.sma_windows, default=0),
            max(self.trend_config.ema_windows, default=0),
            self.trend_config.macd_slow_period + self.trend_config.macd_signal_period
        )
        return max_window
    
    def _execute_calculation(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """执行趋势类因子计算"""
        results = {}
        factor_names = kwargs.get('factor_names', None)
        
        # 简单移动平均
        for window in self.trend_config.sma_windows:
            factor_name = f'sma_{window}'
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self._calculate_sma(data, window)
        
        # 指数移动平均
        for window in self.trend_config.ema_windows:
            factor_name = f'ema_{window}'
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self._calculate_ema(data, window)
        
        # MACD指标
        macd_factors = ['macd', 'macd_signal', 'macd_histogram']
        if factor_names is None or any(f in factor_names for f in macd_factors):
            macd_results = self._calculate_macd(data)
            for factor_name, factor_series in macd_results.items():
                if factor_names is None or factor_name in factor_names:
                    results[factor_name] = factor_series
        
        return results
    
    def _postprocess_specific(self, results: Dict[str, pd.Series], 
                            data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """趋势因子特定的后处理"""
        # 计算价格相对均线的比率
        if 'close' in data.columns:
            for window in [20, 60]:  # 常用的均线周期
                sma_key = f'sma_{window}'
                ratio_key = f'price_to_sma{window}'
                
                if sma_key in results:
                    results[ratio_key] = data['close'] / results[sma_key]
        
        return results
    
    def _calculate_sma(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        计算简单移动平均
        
        Args:
            data: 包含close列的价格数据
            window: 计算窗口期
            
        Returns:
            SMA值序列
        """
        try:
            return TechnicalIndicators.sma(data['close'], window)
        except Exception as e:
            logger.error(f"计算SMA({window})时出错: {e}")
            raise
    
    def _calculate_ema(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        计算指数移动平均
        
        Args:
            data: 包含close列的价格数据
            window: 计算窗口期
            
        Returns:
            EMA值序列
        """
        try:
            return TechnicalIndicators.ema(data['close'], window)
        except Exception as e:
            logger.error(f"计算EMA({window})时出错: {e}")
            raise
    
    def _calculate_macd(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算MACD指标
        
        Args:
            data: 包含close列的价格数据
            
        Returns:
            MACD指标字典
        """
        try:
            macd_df = TechnicalIndicators.macd(
                data['close'], 
                self.trend_config.macd_fast_period,
                self.trend_config.macd_slow_period,
                self.trend_config.macd_signal_period
            )
            
            return {
                'macd': macd_df['MACD'],
                'macd_signal': macd_df['Signal'],
                'macd_histogram': macd_df['Histogram']
            }
        except Exception as e:
            logger.error(f"计算MACD时出错: {e}")
            raise


class VolatilityFactorCalculator(BaseFactorCalculator):
    """波动率类因子计算器"""
    
    def calculate_historical_volatility(self, data: pd.DataFrame, window: int) -> pd.Series:
        """计算历史波动率"""
        if 'close' not in data.columns:
            raise ValueError("缺少收盘价数据")
        
        returns = data['close'].pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)  # 年化波动率
    
    def calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算平均真实波幅"""
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"缺少{col}数据")
        
        return TechnicalIndicators.atr(data['high'], data['low'], data['close'], window)
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, 
                                 window: int = 20, 
                                 num_std: float = 2.0) -> Dict[str, pd.Series]:
        """计算布林带指标"""
        if 'close' not in data.columns:
            raise ValueError("缺少收盘价数据")
        
        bb_df = TechnicalIndicators.bollinger_bands(data['close'], window, num_std)
        
        # 计算布林带宽度和位置
        bb_width = (bb_df['Upper'] - bb_df['Lower']) / bb_df['Middle']
        bb_position = (data['close'] - bb_df['Lower']) / (bb_df['Upper'] - bb_df['Lower'])
        
        return {
            'bb_upper': bb_df['Upper'],
            'bb_middle': bb_df['Middle'],
            'bb_lower': bb_df['Lower'],
            'bb_width': bb_width,
            'bb_position': bb_position
        }
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """计算所有波动率类因子"""
        results = {}
        
        try:
            # 历史波动率
            results['volatility_5'] = self.calculate_historical_volatility(data, 5)
            results['volatility_20'] = self.calculate_historical_volatility(data, 20)
            results['volatility_60'] = self.calculate_historical_volatility(data, 60)
            
            # ATR指标
            if all(col in data.columns for col in ['high', 'low', 'close']):
                results['atr_14'] = self.calculate_atr(data, 14)
            
            # 布林带指标
            bb_results = self.calculate_bollinger_bands(data)
            results.update(bb_results)
            
        except Exception as e:
            logger.error(f"计算波动率类因子时出错: {e}")
            raise
        
        return results


class VolumeFactorCalculator(BaseFactorCalculator):
    """成交量类因子计算器"""
    
    def calculate_volume_sma(self, data: pd.DataFrame, window: int) -> pd.Series:
        """计算成交量移动平均"""
        if 'vol' not in data.columns:
            raise ValueError("缺少成交量数据")
        
        return TechnicalIndicators.sma(data['vol'], window)
    
    def calculate_volume_ratio(self, data: pd.DataFrame, window: int) -> pd.Series:
        """计算量比"""
        volume_sma = self.calculate_volume_sma(data, window)
        return data['vol'] / volume_sma
    
    def calculate_vpt(self, data: pd.DataFrame) -> pd.Series:
        """计算量价趋势指标"""
        required_cols = ['close', 'vol']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"缺少{col}数据")
        
        return TechnicalIndicators.vpt(data['close'], data['vol'])
    
    def calculate_mfi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算资金流量指标"""
        required_cols = ['high', 'low', 'close', 'vol']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"缺少{col}数据")
        
        return TechnicalIndicators.mfi(data['high'], data['low'], data['close'], data['vol'], window)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """计算所有成交量类因子"""
        results = {}
        
        try:
            # 成交量移动平均
            if 'vol' in data.columns:
                results['volume_sma_5'] = self.calculate_volume_sma(data, 5)
                results['volume_sma_20'] = self.calculate_volume_sma(data, 20)
                
                # 量比
                results['volume_ratio_5'] = self.calculate_volume_ratio(data, 5)
                results['volume_ratio_20'] = self.calculate_volume_ratio(data, 20)
            
            # VPT指标
            if all(col in data.columns for col in ['close', 'vol']):
                results['vpt'] = self.calculate_vpt(data)
            
            # MFI指标
            if all(col in data.columns for col in ['high', 'low', 'close', 'vol']):
                results['mfi'] = self.calculate_mfi(data, 14)
            
        except Exception as e:
            logger.error(f"计算成交量类因子时出错: {e}")
            raise
        
        return results


class TechnicalFactorEngine(BaseFactorEngine):
    """技术面因子计算引擎"""
    
    def __init__(self, engine):
        """初始化技术面因子引擎"""
        super().__init__(engine, FactorType.TECHNICAL)
        
        # 初始化各类因子计算器
        self.momentum_calculator = MomentumFactorCalculator()
        self.trend_calculator = TrendFactorCalculator()
        self.volatility_calculator = VolatilityFactorCalculator()
        self.volume_calculator = VolumeFactorCalculator()
        
        logger.info("技术面因子引擎初始化完成")
    
    def _initialize_factors(self):
        """初始化因子配置和元数据"""
        # 动量类因子
        momentum_factors = [
            ("rsi_14", "14日相对强弱指数", {"window": 14}),
            ("rsi_6", "6日相对强弱指数", {"window": 6}),
            ("williams_r_14", "14日威廉指标", {"window": 14}),
            ("momentum_5", "5日动量", {"window": 5}),
            ("momentum_10", "10日动量", {"window": 10}),
            ("momentum_20", "20日动量", {"window": 20}),
            ("roc_5", "5日变化率", {"window": 5}),
            ("roc_10", "10日变化率", {"window": 10}),
            ("roc_20", "20日变化率", {"window": 20}),
        ]
        
        for name, desc, params in momentum_factors:
            metadata = create_factor_metadata(
                name=name,
                description=desc,
                factor_type=FactorType.TECHNICAL,
                category=FactorCategory.MOMENTUM,
                parameters=params,
                data_requirements=["close"] if "williams" not in name else ["high", "low", "close"],
                min_periods=params.get("window", 1)
            )
            config = create_factor_config(name=name, parameters=params)
            self.register_factor(metadata, config)
        
        # 趋势类因子
        trend_factors = [
            ("sma_5", "5日简单移动平均", {"window": 5}),
            ("sma_10", "10日简单移动平均", {"window": 10}),
            ("sma_20", "20日简单移动平均", {"window": 20}),
            ("sma_60", "60日简单移动平均", {"window": 60}),
            ("ema_12", "12日指数移动平均", {"window": 12}),
            ("ema_26", "26日指数移动平均", {"window": 26}),
            ("macd", "MACD线", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
            ("macd_signal", "MACD信号线", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
            ("macd_histogram", "MACD柱状图", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
            ("price_to_sma20", "价格相对20日均线", {"window": 20}),
            ("price_to_sma60", "价格相对60日均线", {"window": 60}),
        ]
        
        for name, desc, params in trend_factors:
            metadata = create_factor_metadata(
                name=name,
                description=desc,
                factor_type=FactorType.TECHNICAL,
                category=FactorCategory.TREND,
                parameters=params,
                data_requirements=["close"],
                min_periods=params.get("window", params.get("slow_period", 1))
            )
            config = create_factor_config(name=name, parameters=params)
            self.register_factor(metadata, config)
        
        # 波动率类因子
        volatility_factors = [
            ("volatility_5", "5日历史波动率", {"window": 5}),
            ("volatility_20", "20日历史波动率", {"window": 20}),
            ("volatility_60", "60日历史波动率", {"window": 60}),
            ("atr_14", "14日平均真实波幅", {"window": 14}),
            ("bb_upper", "布林带上轨", {"window": 20, "num_std": 2.0}),
            ("bb_middle", "布林带中轨", {"window": 20, "num_std": 2.0}),
            ("bb_lower", "布林带下轨", {"window": 20, "num_std": 2.0}),
            ("bb_width", "布林带宽度", {"window": 20, "num_std": 2.0}),
            ("bb_position", "布林带位置", {"window": 20, "num_std": 2.0}),
        ]
        
        for name, desc, params in volatility_factors:
            data_req = ["close"] if "atr" not in name else ["high", "low", "close"]
            metadata = create_factor_metadata(
                name=name,
                description=desc,
                factor_type=FactorType.TECHNICAL,
                category=FactorCategory.VOLATILITY,
                parameters=params,
                data_requirements=data_req,
                min_periods=params.get("window", 1)
            )
            config = create_factor_config(name=name, parameters=params)
            self.register_factor(metadata, config)
        
        # 成交量类因子
        volume_factors = [
            ("volume_sma_5", "5日成交量均值", {"window": 5}),
            ("volume_sma_20", "20日成交量均值", {"window": 20}),
            ("volume_ratio_5", "5日量比", {"window": 5}),
            ("volume_ratio_20", "20日量比", {"window": 20}),
            ("vpt", "量价趋势", {}),
            ("mfi", "资金流量指标", {"window": 14}),
        ]
        
        for name, desc, params in volume_factors:
            if name == "vpt":
                data_req = ["close", "vol"]
            elif name == "mfi":
                data_req = ["high", "low", "close", "vol"]
            else:
                data_req = ["vol"]
            
            metadata = create_factor_metadata(
                name=name,
                description=desc,
                factor_type=FactorType.TECHNICAL,
                category=FactorCategory.VOLUME,
                parameters=params,
                data_requirements=data_req,
                min_periods=params.get("window", 1)
            )
            config = create_factor_config(name=name, parameters=params)
            self.register_factor(metadata, config)
    
    def get_required_data(self, 
                         ts_code: str,
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None) -> pd.DataFrame:
        """获取计算所需的股票数据"""
        query = """
            SELECT trade_date, ts_code, open, high, low, close, vol, amount
            FROM stock_daily 
            WHERE ts_code = :ts_code
        """
        
        params = {'ts_code': ts_code}
        
        if start_date:
            query += " AND trade_date >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND trade_date <= :end_date"
            params['end_date'] = end_date
            
        query += " ORDER BY trade_date"
        
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)
            
        if df.empty:
            logger.warning(f"未找到股票 {ts_code} 的数据")
            return df
            
        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        logger.info(f"获取股票 {ts_code} 数据 {len(df)} 条")
        return df
    
    def calculate_factors(self, 
                         ts_code: str,
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None,
                         factor_names: Optional[List[str]] = None) -> FactorResult:
        """计算技术面因子"""
        start_time = datetime.now()
        
        try:
            # 获取数据
            data = self.get_required_data(ts_code, start_date, end_date)
            
            if data.empty:
                return FactorResult(
                    ts_code=ts_code,
                    calculation_date=start_time,
                    factor_type=self.factor_type,
                    status=CalculationStatus.SKIPPED,
                    error_message="无可用数据"
                )
            
            # 验证数据
            required_columns = ['trade_date', 'close']
            if not self.validate_data(data, required_columns):
                return self.handle_calculation_error(ts_code, ValueError("数据验证失败"))
            
            # 计算各类因子
            all_factors = {}
            
            # 动量类因子
            momentum_factors = self.momentum_calculator.calculate(data)
            all_factors.update(momentum_factors)
            
            # 趋势类因子
            trend_factors = self.trend_calculator.calculate(data)
            all_factors.update(trend_factors)
            
            # 波动率类因子
            volatility_factors = self.volatility_calculator.calculate(data)
            all_factors.update(volatility_factors)
            
            # 成交量类因子
            volume_factors = self.volume_calculator.calculate(data)
            all_factors.update(volume_factors)
            
            # 创建因子结果
            result = FactorResult(
                ts_code=ts_code,
                calculation_date=start_time,
                factor_type=self.factor_type,
                status=CalculationStatus.SUCCESS,
                data_points=len(data)
            )
            
            # 添加因子值
            for factor_name, factor_series in all_factors.items():
                if factor_names and factor_name not in factor_names:
                    continue
                
                for i, (trade_date, value) in enumerate(zip(data['trade_date'], factor_series)):
                    if pd.notna(value):
                        factor_value = FactorValue(
                            ts_code=ts_code,
                            trade_date=trade_date.date(),
                            factor_name=factor_name,
                            raw_value=float(value)
                        )
                        result.add_factor(factor_value)
            
            # 计算执行时间
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            logger.info(f"股票 {ts_code} 技术面因子计算完成，"
                       f"计算了 {len(all_factors)} 个因子，耗时 {execution_time:.2f}秒")
            
            return result
            
        except Exception as e:
            return self.handle_calculation_error(ts_code, e)


if __name__ == "__main__":
    # 测试代码
    print("技术面因子引擎测试")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'ts_code': ['000001.SZ'] * 100,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'vol': np.random.randint(1000000, 10000000, 100),
        'amount': np.random.randint(100000000, 1000000000, 100)
    })
    
    # 确保high >= low
    test_data['high'] = np.maximum(test_data['high'], test_data['low'])
    
    # 测试各类计算器
    momentum_calc = MomentumFactorCalculator()
    trend_calc = TrendFactorCalculator()
    volatility_calc = VolatilityFactorCalculator()
    volume_calc = VolumeFactorCalculator()
    
    print("测试动量类因子计算...")
    momentum_results = momentum_calc.calculate(test_data)
    print(f"动量类因子数量: {len(momentum_results)}")
    
    print("测试趋势类因子计算...")
    trend_results = trend_calc.calculate(test_data)
    print(f"趋势类因子数量: {len(trend_results)}")
    
    print("测试波动率类因子计算...")
    volatility_results = volatility_calc.calculate(test_data)
    print(f"波动率类因子数量: {len(volatility_results)}")
    
    print("测试成交量类因子计算...")
    volume_results = volume_calc.calculate(test_data)
    print(f"成交量类因子数量: {len(volume_results)}")
    
    print("技术面因子引擎测试完成")
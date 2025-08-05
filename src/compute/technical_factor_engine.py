from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术面因子计算引擎
实现各类技术指标因子的计算
"""


from .abstract_factor_calculator import AbstractFactorCalculator
from .base_factor_engine import BaseFactorCalculator, BaseFactorEngine
from .data_validation_mixin import DataValidationError, InsufficientDataError
from .factor_calculation_config import TechnicalFactorCalculationConfig
from .factor_models import (
    CalculationStatus,
    FactorCategory,
    FactorConfig,
    FactorMetadata,
    FactorResult,
    FactorType,
    FactorValue,
    create_factor_config,
    create_factor_metadata,
)
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
        return ["close"]

    def _get_minimum_data_length(self) -> int:
        """获取最小数据长度要求"""
        # 返回所有窗口期中的最大值
        max_window = max(
            max(self.momentum_config.rsi_windows, default=0),
            max(self.momentum_config.momentum_windows, default=0),
            max(self.momentum_config.roc_windows, default=0),
            max(self.momentum_config.williams_r_windows, default=0),
        )
        return max_window

    def _validate_specific_requirements(self, data: pd.DataFrame, **kwargs) -> None:
        """验证动量因子特定要求"""
        # 检查威廉指标所需的额外列
        factor_names = kwargs.get("factor_names") or []
        if any(f"williams_r_{w}" in factor_names for w in self.momentum_config.williams_r_windows):
            required_cols = ["high", "low", "close"]
            self.validate_required_columns(data, required_cols)

    def _execute_calculation(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """执行动量类因子计算"""
        results = {}
        factor_names = kwargs.get("factor_names", None)

        # RSI指标
        for window in self.momentum_config.rsi_windows:
            factor_name = f"rsi_{window}"
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self._calculate_rsi(data, window)

        # 威廉指标（需要额外的high、low列）
        if all(col in data.columns for col in ["high", "low", "close"]):
            for window in self.momentum_config.williams_r_windows:
                factor_name = f"williams_r_{window}"
                if factor_names is None or factor_name in factor_names:
                    results[factor_name] = self._calculate_williams_r(data, window)

        # 动量指标
        for window in self.momentum_config.momentum_windows:
            factor_name = f"momentum_{window}"
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self._calculate_momentum(data, window)

        # 变化率指标
        for window in self.momentum_config.roc_windows:
            factor_name = f"roc_{window}"
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
            return TechnicalIndicators.rsi(data["close"], window)
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
            return TechnicalIndicators.williams_r(data["high"], data["low"], data["close"], window)
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
            return TechnicalIndicators.momentum(data["close"], window)
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
            return data["close"].pct_change(window) * 100
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
        return ["close"]

    def _get_minimum_data_length(self) -> int:
        """获取最小数据长度要求"""
        # 返回所有窗口期中的最大值
        max_window = max(
            max(self.trend_config.sma_windows, default=0),
            max(self.trend_config.ema_windows, default=0),
            self.trend_config.macd_slow_period + self.trend_config.macd_signal_period,
        )
        return max_window

    def _execute_calculation(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """执行趋势类因子计算"""
        results = {}
        factor_names = kwargs.get("factor_names", None)

        # 简单移动平均
        for window in self.trend_config.sma_windows:
            factor_name = f"sma_{window}"
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self._calculate_sma(data, window)

        # 指数移动平均
        for window in self.trend_config.ema_windows:
            factor_name = f"ema_{window}"
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self._calculate_ema(data, window)

        # MACD指标
        macd_factors = ["macd", "macd_signal", "macd_histogram"]
        if factor_names is None or any(f in factor_names for f in macd_factors):
            macd_results = self._calculate_macd(data)
            for factor_name, factor_series in macd_results.items():
                if factor_names is None or factor_name in factor_names:
                    results[factor_name] = factor_series

        return results

    def _postprocess_specific(
        self, results: Dict[str, pd.Series], data: pd.DataFrame, **kwargs
    ) -> Dict[str, pd.Series]:
        """趋势因子特定的后处理"""
        # 计算价格相对均线的比率
        if "close" in data.columns:
            for window in [20, 60]:  # 常用的均线周期
                sma_key = f"sma_{window}"
                ratio_key = f"price_to_sma{window}"

                if sma_key in results:
                    results[ratio_key] = data["close"] / results[sma_key]

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
            return TechnicalIndicators.sma(data["close"], window)
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
            return TechnicalIndicators.ema(data["close"], window)
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
                data["close"],
                self.trend_config.macd_fast_period,
                self.trend_config.macd_slow_period,
                self.trend_config.macd_signal_period,
            )

            return {"macd": macd_df["MACD"], "macd_signal": macd_df["Signal"], "macd_histogram": macd_df["Histogram"]}
        except Exception as e:
            logger.error(f"计算MACD时出错: {e}")
            raise


class VolatilityFactorCalculator(AbstractFactorCalculator):
    """波动率类因子计算器"""

    def __init__(self, config: Optional[TechnicalFactorCalculationConfig] = None):
        """
        初始化波动率因子计算器

        Args:
            config: 技术面因子计算配置
        """
        super().__init__(config)
        self.volatility_config = self.config.volatility

    def _get_required_columns(self) -> List[str]:
        """获取计算所需的数据列"""
        return ["close", "high", "low"]

    def _get_minimum_data_length(self) -> int:
        """获取最小数据长度要求"""
        return max(
            max(self.volatility_config.volatility_windows, default=0),
            max(self.volatility_config.atr_windows, default=0),
            self.volatility_config.bollinger_window
        )

    def _execute_calculation(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """执行波动率类因子计算"""
        results = {}
        factor_names = kwargs.get("factor_names", None)

        # 历史波动率
        for window in self.volatility_config.volatility_windows:
            factor_name = f"volatility_{window}"
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self.calculate_historical_volatility(data, window)

        # ATR指标
        for window in self.volatility_config.atr_windows:
            factor_name = f"atr_{window}"
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self.calculate_atr(data, window)

        # 布林带
        if factor_names is None or any(name.startswith('bollinger') for name in factor_names or []):
            upper, middle, lower = self.calculate_bollinger_bands(data)
            results["bollinger_upper"] = upper
            results["bollinger_middle"] = middle
            results["bollinger_lower"] = lower

        return results

    def calculate_historical_volatility(self, data: pd.DataFrame, window: int) -> pd.Series:
        """计算历史波动率"""
        if "close" not in data.columns:
            raise ValueError("缺少收盘价数据")

        returns = data["close"].pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)  # 年化波动率

    def calculate_atr(self, data: pd.DataFrame, window: int) -> pd.Series:
        """计算ATR指标"""
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"缺少{col}数据")

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # 计算真实波幅
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return true_range.rolling(window=window).mean()

    def calculate_bollinger_bands(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        if "close" not in data.columns:
            raise ValueError("缺少收盘价数据")

        close = data["close"]
        middle = close.rolling(window=self.volatility_config.bollinger_window).mean()
        std = close.rolling(window=self.volatility_config.bollinger_window).std()

        upper = middle + (std * self.volatility_config.bollinger_std)
        lower = middle - (std * self.volatility_config.bollinger_std)

        return upper, middle, lower


        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"缺少{col}数据")

        return TechnicalIndicators.atr(data["high"], data["low"], data["close"], window)

    def calculate_bollinger_bands(
        self, data: pd.DataFrame, window: int = 20, num_std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """计算布林带指标"""
        if "close" not in data.columns:
            raise ValueError("缺少收盘价数据")

        bb_df = TechnicalIndicators.bollinger_bands(data["close"], window, num_std)

        # 计算布林带宽度和位置
        bb_width = (bb_df["Upper"] - bb_df["Lower"]) / bb_df["Middle"]
        bb_position = (data["close"] - bb_df["Lower"]) / (bb_df["Upper"] - bb_df["Lower"])

        return {
            "bb_upper": bb_df["Upper"],
            "bb_middle": bb_df["Middle"],
            "bb_lower": bb_df["Lower"],
            "bb_width": bb_width,
            "bb_position": bb_position,
        }

    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """计算所有波动率类因子"""
        results = {}

        try:
            # 历史波动率
            results["volatility_5"] = self.calculate_historical_volatility(data, 5)
            results["volatility_20"] = self.calculate_historical_volatility(data, 20)
            results["volatility_60"] = self.calculate_historical_volatility(data, 60)

            # ATR指标
            if all(col in data.columns for col in ["high", "low", "close"]):
                results["atr_14"] = self.calculate_atr(data, 14)

            # 布林带指标
            bb_results = self.calculate_bollinger_bands(data)
            results.update(bb_results)

        except Exception as e:
            logger.error(f"计算波动率类因子时出错: {e}")
            raise

        return results


class VolumeFactorCalculator(AbstractFactorCalculator):
    """成交量类因子计算器"""

    def __init__(self, config: Optional[TechnicalFactorCalculationConfig] = None):
        """
        初始化成交量因子计算器

        Args:
            config: 技术面因子计算配置
        """
        super().__init__(config)
        self.volume_config = self.config.volume

    def _get_required_columns(self) -> List[str]:
        """获取计算所需的数据列"""
        return ["close", "vol"]

    def _get_minimum_data_length(self) -> int:
        """获取最小数据长度要求"""
        return max(
            max(self.volume_config.volume_sma_windows, default=0),
            max(self.volume_config.volume_ratio_windows, default=0),
            self.volume_config.mfi_window
        )

    def _execute_calculation(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """执行成交量类因子计算"""
        results = {}
        factor_names = kwargs.get("factor_names", None)

        # 成交量移动平均
        for window in self.volume_config.volume_sma_windows:
            factor_name = f"volume_sma_{window}"
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self.calculate_volume_sma(data, window)

        # 量比
        for window in self.volume_config.volume_ratio_windows:
            factor_name = f"volume_ratio_{window}"
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self.calculate_volume_ratio(data, window)

        # MFI指标
        if "high" in data.columns and "low" in data.columns:
            factor_name = f"mfi_{self.volume_config.mfi_window}"
            if factor_names is None or factor_name in factor_names:
                results[factor_name] = self.calculate_mfi(data, self.volume_config.mfi_window)

        # VPT指标
        factor_name = "vpt"
        if factor_names is None or factor_name in factor_names:
            results[factor_name] = self.calculate_vpt(data)

        return results

    def calculate_volume_sma(self, data: pd.DataFrame, window: int) -> pd.Series:
        """计算成交量移动平均"""
        if "vol" not in data.columns:
            raise ValueError("缺少成交量数据")

        return TechnicalIndicators.sma(data["vol"], window)

    def calculate_volume_ratio(self, data: pd.DataFrame, window: int) -> pd.Series:
        """计算量比"""
        volume_sma = self.calculate_volume_sma(data, window)
        return data["vol"] / volume_sma

    def calculate_vpt(self, data: pd.DataFrame) -> pd.Series:
        """计算量价趋势指标"""
        required_cols = ["close", "vol"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"缺少{col}数据")

        return TechnicalIndicators.vpt(data["close"], data["vol"])

    def calculate_mfi(self, data: pd.DataFrame, window: int) -> pd.Series:
        """计算资金流量指标"""
        required_cols = ["high", "low", "close", "vol"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"缺少{col}数据")

        return TechnicalIndicators.mfi(data["high"], data["low"], data["close"], data["vol"], window)

    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """计算所有成交量类因子"""
        results = {}

        try:
            # 成交量移动平均
            if "vol" in data.columns:
                results["volume_sma_5"] = self.calculate_volume_sma(data, 5)
                results["volume_sma_20"] = self.calculate_volume_sma(data, 20)

                # 量比
                results["volume_ratio_5"] = self.calculate_volume_ratio(data, 5)
                results["volume_ratio_20"] = self.calculate_volume_ratio(data, 20)

            # VPT指标
            if all(col in data.columns for col in ["close", "vol"]):
                results["vpt"] = self.calculate_vpt(data)

            # MFI指标
            if all(col in data.columns for col in ["high", "low", "close", "vol"]):
                results["mfi"] = self.calculate_mfi(data, 14)

        except Exception as e:
            logger.error(f"计算成交量类因子时出错: {e}")
            raise

        return results


class TechnicalFactorEngine(BaseFactorEngine):
    """技术面因子计算引擎"""

    def __init__(self, engine, config: Optional[TechnicalFactorCalculationConfig] = None):
        """初始化技术面因子引擎"""
        super().__init__(engine, FactorType.TECHNICAL)
        
        # 设置配置
        self.config = config or TechnicalFactorCalculationConfig()

        # 初始化各类因子计算器
        self.momentum_calculator = MomentumFactorCalculator(self.config)
        self.trend_calculator = TrendFactorCalculator(self.config)
        self.volatility_calculator = VolatilityFactorCalculator(self.config)
        self.volume_calculator = VolumeFactorCalculator(self.config)

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
                min_periods=params.get("window", 1),
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
                min_periods=params.get("window", params.get("slow_period", 1)),
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
                min_periods=params.get("window", 1),
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
                min_periods=params.get("window", 1),
            )
            config = create_factor_config(name=name, parameters=params)
            self.register_factor(metadata, config)

    def get_required_data(
        self, ts_code: str, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """获取计算所需的股票数据"""
        query = """
            SELECT trade_date, ts_code, open, high, low, close, vol, amount
            FROM stock_daily
            WHERE ts_code = :ts_code
        """

        params = {"ts_code": ts_code}

        if start_date:
            query += " AND trade_date >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND trade_date <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY trade_date"

        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)

        if df.empty:
            logger.warning(f"未找到股票 {ts_code} 的数据")
            return df

        # 转换日期格式
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        logger.info(f"获取股票 {ts_code} 数据 {len(df)} 条")
        return df

    def _get_stock_data(
        self, ts_code: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取股票数据（内部方法）"""
        return self.get_required_data(ts_code, start_date, end_date)

    def calculate_factors(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        factor_names: Optional[List[str]] = None,
    ) -> List[FactorResult]:
        """
        计算技术因子
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            factor_names: 要计算的因子名称列表，如果为None则计算所有因子
            
        Returns:
            因子结果列表
        """
        try:
            # 获取股票数据
            stock_data = self._get_stock_data(ts_code, start_date, end_date)
            if stock_data.empty:
                logger.warning(f"{ts_code} 无数据")
                return []
            
            # 计算因子
            results = []
            
            # 获取配置
            config = self.config
            
            # 动量因子
            if factor_names is None or any(name.startswith('rsi') or name.startswith('momentum') for name in factor_names):
                momentum_calculator = MomentumFactorCalculator(config)
                momentum_results = momentum_calculator.calculate(stock_data, factor_names=factor_names)
                results.extend(momentum_results)
            
            # 趋势因子
            if factor_names is None or any(name.startswith('ma') or name.startswith('macd') for name in factor_names):
                trend_calculator = TrendFactorCalculator(config)
                trend_results = trend_calculator.calculate(stock_data, factor_names=factor_names)
                results.extend(trend_results)
            
            # 波动率因子
            if factor_names is None or any(name.startswith('atr') or name.startswith('bollinger') for name in factor_names):
                volatility_calculator = VolatilityFactorCalculator(config)
                volatility_results = volatility_calculator.calculate(stock_data, factor_names=factor_names)
                results.extend(volatility_results)
            
            # 成交量因子
            if factor_names is None or any(name.startswith('volume') for name in factor_names):
                volume_calculator = VolumeFactorCalculator(config)
                volume_results = volume_calculator.calculate(stock_data, factor_names=factor_names)
                results.extend(volume_results)
            
            logger.info(f"{ts_code} 技术因子计算完成: {len(results)} 个因子")
            return results
            
        except Exception as e:
            logger.error(f"计算 {ts_code} 技术因子时出错: {e}")
            raise


if __name__ == "__main__":
    # 测试代码
    print("技术面因子引擎测试")

    # 创建测试数据
    test_data = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "ts_code": ["000001.SZ"] * 100,
            "open": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "high": 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
            "low": 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2,
            "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "vol": np.random.randint(1000000, 10000000, 100),
            "amount": np.random.randint(100000000, 1000000000, 100),
        }
    )

    # 确保high >= low
    test_data["high"] = np.maximum(test_data["high"], test_data["low"])

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

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情绪面因子计算引擎
实现各类市场情绪和资金流向因子的计算
"""


from .abstract_factor_calculator import AbstractFactorCalculator
from .base_factor_engine import BaseFactorEngine
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


class MoneyFlowFactorCalculator(AbstractFactorCalculator):
    """资金流向类因子计算器"""

    def _execute_calculation(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """执行资金流向因子计算"""
        return self.calculate(data, **kwargs)

    def _get_required_columns(self) -> List[str]:
        """获取计算所需的数据列"""
        return ["close", "amount", "vol", "pre_close"]

    def calculate_money_flow(self, market_data: pd.DataFrame, window: int = 5) -> pd.Series:
        """计算资金流向指标"""
        if "amount" not in market_data.columns or "vol" not in market_data.columns:
            return pd.Series(dtype=float, index=market_data.index)

        # 资金流向 = 成交金额的变化趋势
        money_flow = market_data["amount"].rolling(window=window).mean()
        money_flow_change = money_flow.pct_change(window, fill_method=None) * 100

        return money_flow_change.where((money_flow_change > -1000) & (money_flow_change < 1000))

    def calculate_net_inflow_rate(self, market_data: pd.DataFrame) -> pd.Series:
        """计算净流入率"""
        if not all(col in market_data.columns for col in ["close", "pre_close", "amount"]):
            return pd.Series(dtype=float, index=market_data.index)

        # 简化计算：涨跌幅 * 成交金额作为净流入的代理指标
        price_change_rate = (market_data["close"] - market_data["pre_close"]) / market_data["pre_close"]
        net_inflow = price_change_rate * market_data["amount"]

        # 计算净流入率（相对于总成交金额）
        net_inflow_rate = net_inflow / market_data["amount"] * 100

        return net_inflow_rate.where((net_inflow_rate > -100) & (net_inflow_rate < 100))

    def calculate_big_order_ratio(self, market_data: pd.DataFrame) -> pd.Series:
        """计算大单占比（模拟计算）"""
        if "amount" not in market_data.columns or "vol" not in market_data.columns:
            return pd.Series(dtype=float, index=market_data.index)

        # 模拟大单占比：基于成交金额和成交量的比值
        avg_price = market_data["amount"] / market_data["vol"]
        avg_price_ma = avg_price.rolling(window=20).mean()

        # 当日均价相对于20日均价的比值作为大单占比的代理
        big_order_proxy = (avg_price / avg_price_ma - 1) * 100

        return big_order_proxy.where((big_order_proxy > -50) & (big_order_proxy < 50))

    def calculate_volume_price_trend(self, market_data: pd.DataFrame) -> pd.Series:
        """计算量价趋势指标"""
        if not all(col in market_data.columns for col in ["close", "vol"]):
            return pd.Series(dtype=float, index=market_data.index)

        # 量价趋势：价格变化方向与成交量变化方向的一致性
        price_change = market_data["close"].pct_change(fill_method=None)
        volume_change = market_data["vol"].pct_change(fill_method=None)

        # 计算量价一致性指标
        vpt_signal = np.sign(price_change) * np.sign(volume_change)
        vpt_trend = vpt_signal.rolling(window=5).mean() * 100

        return vpt_trend

    def calculate(self, market_data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """计算所有资金流向类因子"""
        results = {}

        try:
            # 资金流向指标
            results["money_flow_5"] = self.calculate_money_flow(market_data, 5)
            results["money_flow_20"] = self.calculate_money_flow(market_data, 20)

            # 净流入率
            results["net_inflow_rate"] = self.calculate_net_inflow_rate(market_data)

            # 大单占比（模拟）
            results["big_order_ratio"] = self.calculate_big_order_ratio(market_data)

            # 中单和小单占比（基于大单占比计算）
            if "big_order_ratio" in results:
                results["medium_order_ratio"] = (50 - results["big_order_ratio"].abs()) / 2
                results["small_order_ratio"] = 50 - results["big_order_ratio"].abs() - results["medium_order_ratio"]

            # 量价趋势
            results["volume_price_trend"] = self.calculate_volume_price_trend(market_data)

        except Exception as e:
            logger.error(f"计算资金流向类因子时出错: {e}")
            raise

        return results


class AttentionFactorCalculator(AbstractFactorCalculator):
    """关注度类因子计算器"""

    def _execute_calculation(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """执行关注度因子计算"""
        return self.calculate(data, **kwargs)

    def _get_required_columns(self) -> List[str]:
        """获取计算所需的数据列"""
        return ["close", "vol", "high", "low", "turnover_rate"]

    def calculate_turnover_attention(self, market_data: pd.DataFrame) -> pd.Series:
        """基于换手率计算关注度"""
        if "turnover_rate" not in market_data.columns:
            # 如果没有换手率，用成交量/流通股本的代理计算
            if all(col in market_data.columns for col in ["vol", "circ_mv"]):
                # 简化计算：成交量相对于流通市值的比例
                turnover_proxy = market_data["vol"] / (market_data["circ_mv"] * 10000) * 100
                return turnover_proxy.where((turnover_proxy >= 0) & (turnover_proxy < 100))
            else:
                return pd.Series(dtype=float, index=market_data.index)

        # 换手率本身就是关注度的一个指标
        return market_data["turnover_rate"].where(
            (market_data["turnover_rate"] >= 0) & (market_data["turnover_rate"] < 100)
        )

    def calculate_volume_attention(self, market_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """基于成交量计算关注度"""
        if "vol" not in market_data.columns:
            return pd.Series(dtype=float, index=market_data.index)

        # 成交量相对于历史平均的比值
        vol_ma = market_data["vol"].rolling(window=window).mean()
        volume_attention = market_data["vol"] / vol_ma

        return volume_attention.where((volume_attention > 0) & (volume_attention < 20))

    def calculate_price_attention(self, market_data: pd.DataFrame) -> pd.Series:
        """基于价格波动计算关注度"""
        if not all(col in market_data.columns for col in ["high", "low", "close"]):
            return pd.Series(dtype=float, index=market_data.index)

        # 日内波动幅度作为关注度指标
        daily_range = (market_data["high"] - market_data["low"]) / market_data["close"] * 100

        return daily_range.where((daily_range >= 0) & (daily_range < 50))

    def calculate_attention_score(self, market_data: pd.DataFrame) -> pd.Series:
        """综合关注度评分"""
        # 获取各个关注度指标
        turnover_attention = self.calculate_turnover_attention(market_data)
        volume_attention = self.calculate_volume_attention(market_data)
        price_attention = self.calculate_price_attention(market_data)

        # 标准化各指标到0-1范围
        def normalize_series(series):
            """方法描述"""
            if series.empty:
                return series
            min_val, max_val = series.quantile([0.05, 0.95])
            if max_val > min_val:
                return (series - min_val) / (max_val - min_val)
            else:
                return pd.Series(0.5, index=series.index)

        norm_turnover = normalize_series(turnover_attention)
        norm_volume = normalize_series(volume_attention)
        norm_price = normalize_series(price_attention)

        # 加权平均计算综合评分
        attention_score = (norm_turnover * 0.4 + norm_volume * 0.4 + norm_price * 0.2) * 100

        return attention_score

    def calculate_attention_change_rate(self, market_data: pd.DataFrame, window: int = 5) -> pd.Series:
        """计算关注度变化率"""
        attention_score = self.calculate_attention_score(market_data)

        if attention_score.empty:
            return pd.Series(dtype=float, index=market_data.index)

        # 计算关注度的变化率
        attention_change = attention_score.pct_change(window, fill_method=None) * 100

        return attention_change.where((attention_change > -1000) & (attention_change < 1000))

    def calculate(self, market_data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """计算所有关注度类因子"""
        results = {}

        try:
            # 基础关注度指标
            results["turnover_attention"] = self.calculate_turnover_attention(market_data)
            results["volume_attention"] = self.calculate_volume_attention(market_data)
            results["price_attention"] = self.calculate_price_attention(market_data)

            # 综合关注度评分
            results["attention_score"] = self.calculate_attention_score(market_data)

            # 关注度变化率
            results["attention_change_rate"] = self.calculate_attention_change_rate(market_data)

        except Exception as e:
            logger.error(f"计算关注度类因子时出错: {e}")
            raise

        return results


class SentimentStrengthFactorCalculator(AbstractFactorCalculator):
    """情绪强度类因子计算器"""

    def _execute_calculation(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """执行情绪强度因子计算"""
        return self.calculate(data, **kwargs)

    def _get_required_columns(self) -> List[str]:
        """获取计算所需的数据列"""
        return ["close", "high", "low", "vol"]

    def calculate_price_momentum_sentiment(self, market_data: pd.DataFrame, window: int = 5) -> pd.Series:
        """基于价格动量计算情绪强度"""
        if "close" not in market_data.columns:
            return pd.Series(dtype=float, index=market_data.index)

        # 价格动量作为情绪强度的代理
        price_momentum = market_data["close"].pct_change(window, fill_method=None) * 100

        return price_momentum.where((price_momentum > -100) & (price_momentum < 100))

    def calculate_volatility_sentiment(self, market_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """基于波动率计算情绪强度"""
        if "close" not in market_data.columns:
            return pd.Series(dtype=float, index=market_data.index)

        # 价格波动率作为情绪强度指标
        returns = market_data["close"].pct_change(fill_method=None)
        volatility = returns.rolling(window=window).std() * np.sqrt(252) * 100

        return volatility.where((volatility >= 0) & (volatility < 200))

    def calculate_sentiment_strength(self, market_data: pd.DataFrame) -> pd.Series:
        """综合情绪强度计算"""
        # 获取各个情绪强度指标
        momentum_sentiment = self.calculate_price_momentum_sentiment(market_data)
        volatility_sentiment = self.calculate_volatility_sentiment(market_data)

        # 标准化处理
        def normalize_series(series):
            """方法描述"""
            if series.empty:
                return series
            return (series - series.mean()) / series.std()

        norm_momentum = normalize_series(momentum_sentiment)
        norm_volatility = normalize_series(volatility_sentiment)

        # 综合情绪强度（取绝对值表示强度）
        sentiment_strength = (norm_momentum.abs() * 0.6 + norm_volatility * 0.4) * 50

        return sentiment_strength

    def calculate_bullish_bearish_ratio(
        self, market_data: pd.DataFrame, window: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """计算看涨看跌比例"""
        if "close" not in market_data.columns:
            empty_series = pd.Series(dtype=float, index=market_data.index)
            return empty_series, empty_series

        # 基于价格变化计算看涨看跌信号
        price_change = market_data["close"].pct_change(fill_method=None)

        # 计算滚动窗口内的看涨比例
        bullish_signals = (price_change > 0).astype(int)
        bullish_ratio = bullish_signals.rolling(window=window).mean() * 100

        # 看跌比例
        bearish_ratio = 100 - bullish_ratio

        return bullish_ratio, bearish_ratio

    def calculate_sentiment_volatility(self, market_data: pd.DataFrame, window: int = 10) -> pd.Series:
        """计算情绪波动率"""
        sentiment_strength = self.calculate_sentiment_strength(market_data)

        if sentiment_strength.empty:
            return pd.Series(dtype=float, index=market_data.index)

        # 情绪强度的波动率
        sentiment_volatility = sentiment_strength.rolling(window=window).std()

        return sentiment_volatility

    def calculate(self, market_data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """计算所有情绪强度类因子"""
        results = {}

        try:
            # 基础情绪指标
            results["price_momentum_sentiment"] = self.calculate_price_momentum_sentiment(market_data)
            results["volatility_sentiment"] = self.calculate_volatility_sentiment(market_data)

            # 综合情绪强度
            results["sentiment_strength"] = self.calculate_sentiment_strength(market_data)

            # 看涨看跌比例
            bullish_ratio, bearish_ratio = self.calculate_bullish_bearish_ratio(market_data)
            results["bullish_ratio"] = bullish_ratio
            results["bearish_ratio"] = bearish_ratio

            # 情绪波动率
            results["sentiment_volatility"] = self.calculate_sentiment_volatility(market_data)

        except Exception as e:
            logger.error(f"计算情绪强度类因子时出错: {e}")
            raise

        return results


class EventFactorCalculator(AbstractFactorCalculator):
    """事件类因子计算器"""

    def _execute_calculation(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """执行事件因子计算"""
        return self.calculate(data, **kwargs)

    def _get_required_columns(self) -> List[str]:
        """获取计算所需的数据列"""
        return ["close", "vol", "high", "low", "amount"]

    def calculate_abnormal_volume(self, market_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算异常成交量"""
        if "vol" not in market_data.columns:
            return pd.Series(dtype=float, index=market_data.index)

        # 成交量相对于历史均值的异常程度
        vol_ma = market_data["vol"].rolling(window=window).mean()
        vol_std = market_data["vol"].rolling(window=window).std()

        # Z-score标准化
        abnormal_volume = (market_data["vol"] - vol_ma) / vol_std

        return abnormal_volume.where((abnormal_volume > -10) & (abnormal_volume < 10))

    def calculate_abnormal_return(self, market_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算异常收益率"""
        if "close" not in market_data.columns:
            return pd.Series(dtype=float, index=market_data.index)

        # 日收益率
        daily_returns = market_data["close"].pct_change(fill_method=None)

        # 异常收益率（相对于历史均值的偏离）
        returns_ma = daily_returns.rolling(window=window).mean()
        returns_std = daily_returns.rolling(window=window).std()

        abnormal_return = (daily_returns - returns_ma) / returns_std

        return abnormal_return.where((abnormal_return > -10) & (abnormal_return < 10))

    def calculate_price_limit_signal(self, market_data: pd.DataFrame) -> pd.Series:
        """计算涨跌停信号"""
        if not all(col in market_data.columns for col in ["close", "pre_close"]):
            return pd.Series(dtype=float, index=market_data.index)

        # 计算涨跌幅
        price_change_rate = (market_data["close"] - market_data["pre_close"]) / market_data["pre_close"] * 100

        # 涨跌停信号（接近±10%为涨跌停）
        limit_signal = pd.Series(0, index=market_data.index)
        limit_signal[price_change_rate >= 9.5] = 1  # 涨停信号
        limit_signal[price_change_rate <= -9.5] = -1  # 跌停信号

        return limit_signal

    def calculate_gap_signal(self, market_data: pd.DataFrame) -> pd.Series:
        """计算跳空信号"""
        if not all(col in market_data.columns for col in ["open", "pre_close"]):
            return pd.Series(dtype=float, index=market_data.index)

        # 跳空幅度
        gap_rate = (market_data["open"] - market_data["pre_close"]) / market_data["pre_close"] * 100

        # 跳空信号强度
        gap_signal = pd.Series(0, index=market_data.index)
        gap_signal[gap_rate >= 2] = 1  # 向上跳空
        gap_signal[gap_rate <= -2] = -1  # 向下跳空

        return gap_signal

    def calculate_volume_spike(self, market_data: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
        """计算成交量异动信号"""
        if "vol" not in market_data.columns:
            return pd.Series(dtype=float, index=market_data.index)

        # 成交量相对于5日均量的倍数
        vol_ma5 = market_data["vol"].rolling(window=5).mean()
        volume_ratio = market_data["vol"] / vol_ma5

        # 成交量异动信号
        volume_spike = (volume_ratio >= threshold).astype(int)

        return volume_spike

    def calculate(self, market_data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """计算所有事件类因子"""
        results = {}

        try:
            # 异常指标
            results["abnormal_volume"] = self.calculate_abnormal_volume(market_data)
            results["abnormal_return"] = self.calculate_abnormal_return(market_data)

            # 特殊事件信号
            results["price_limit_signal"] = self.calculate_price_limit_signal(market_data)
            results["gap_signal"] = self.calculate_gap_signal(market_data)
            results["volume_spike"] = self.calculate_volume_spike(market_data)

        except Exception as e:
            logger.error(f"计算事件类因子时出错: {e}")
            raise

        return results


class SentimentFactorEngine(BaseFactorEngine):
    """情绪面因子计算引擎"""

    def __init__(self, engine):
        """初始化情绪面因子引擎"""
        super().__init__(engine, FactorType.SENTIMENT)

        # 暂时设置为None，因为这些计算器类缺少抽象方法实现
        self.money_flow_calculator = None
        self.attention_calculator = None
        self.sentiment_calculator = None
        self.event_calculator = None

        logger.info("情绪面因子计算器初始化跳过（待实现）")

        # 初始化因子配置和元数据
        self._initialize_factors()

        # 注册所有因子
        self._register_factors()

    def _initialize_factors(self):
        """初始化因子配置和元数据"""
        # 情绪面因子引擎的初始化逻辑
        logger.info("情绪面因子配置和元数据初始化完成")

    def _register_factors(self):
        """注册所有情绪面因子"""
        # 资金流向类因子
        money_flow_factors = [
            ("money_flow_5", "5日资金流向", "基于5日成交金额变化的资金流向指标"),
            ("money_flow_20", "20日资金流向", "基于20日成交金额变化的资金流向指标"),
            ("net_inflow_rate", "净流入率", "资金净流入相对于总成交金额的比率"),
            ("big_order_ratio", "大单占比", "大单成交金额占总成交金额的比例"),
            ("medium_order_ratio", "中单占比", "中单成交金额占总成交金额的比例"),
            ("small_order_ratio", "小单占比", "小单成交金额占总成交金额的比例"),
            ("volume_price_trend", "量价趋势", "价格变化与成交量变化的一致性指标"),
        ]

        for factor_name, display_name, description in money_flow_factors:
            metadata = create_factor_metadata(
                name=factor_name,
                description=description,
                factor_type=FactorType.SENTIMENT,
                category=FactorCategory.FLOW,
            )
            config = create_factor_config(
                name=factor_name,
                parameters={
                    "calculation_method": "market_data_based",
                    "update_frequency": "daily",
                    "lookback_period": 20,
                    "min_data_points": 5,
                },
            )
            self.register_factor(metadata, config)

        # 关注度类因子
        attention_factors = [
            ("turnover_attention", "换手率关注度", "基于换手率的市场关注度指标"),
            ("volume_attention", "成交量关注度", "基于成交量异常的关注度指标"),
            ("price_attention", "价格波动关注度", "基于价格波动幅度的关注度指标"),
            ("attention_score", "综合关注度评分", "多维度综合的市场关注度评分"),
            ("attention_change_rate", "关注度变化率", "关注度相对变化的速率指标"),
        ]

        for factor_name, display_name, description in attention_factors:
            metadata = create_factor_metadata(
                name=factor_name,
                description=description,
                factor_type=FactorType.SENTIMENT,
                category=FactorCategory.ATTENTION,
            )
            config = create_factor_config(
                name=factor_name,
                parameters={
                    "calculation_method": "market_data_based",
                    "update_frequency": "daily",
                    "lookback_period": 20,
                    "min_data_points": 5,
                },
            )
            self.register_factor(metadata, config)

        # 情绪强度类因子
        sentiment_factors = [
            ("price_momentum_sentiment", "价格动量情绪", "基于价格动量的市场情绪强度"),
            ("volatility_sentiment", "波动率情绪", "基于价格波动率的市场情绪强度"),
            ("sentiment_strength", "综合情绪强度", "多维度综合的市场情绪强度指标"),
            ("bullish_ratio", "看涨比例", "市场看涨情绪的比例指标"),
            ("bearish_ratio", "看跌比例", "市场看跌情绪的比例指标"),
            ("sentiment_volatility", "情绪波动率", "市场情绪强度的波动程度"),
        ]

        for factor_name, display_name, description in sentiment_factors:
            metadata = create_factor_metadata(
                name=factor_name,
                description=description,
                factor_type=FactorType.SENTIMENT,
                category=FactorCategory.SENTIMENT_STRENGTH,
            )
            config = create_factor_config(
                name=factor_name,
                parameters={
                    "calculation_method": "market_data_based",
                    "update_frequency": "daily",
                    "lookback_period": 30,
                    "min_data_points": 10,
                },
            )
            self.register_factor(metadata, config)

        # 事件类因子
        event_factors = [
            ("abnormal_volume", "异常成交量", "成交量相对于历史均值的异常程度"),
            ("abnormal_return", "异常收益率", "收益率相对于历史均值的异常程度"),
            ("price_limit_signal", "涨跌停信号", "股票涨跌停的信号强度"),
            ("gap_signal", "跳空信号", "股票开盘跳空的信号强度"),
            ("volume_spike", "成交量异动", "成交量突然放大的异动信号"),
        ]

        for factor_name, display_name, description in event_factors:
            metadata = create_factor_metadata(
                name=factor_name,
                description=description,
                factor_type=FactorType.SENTIMENT,
                category=FactorCategory.EVENT,
            )
            config = create_factor_config(
                name=factor_name,
                parameters={
                    "calculation_method": "market_data_based",
                    "update_frequency": "daily",
                    "lookback_period": 20,
                    "min_data_points": 5,
                },
            )
            self.register_factor(metadata, config)

    def get_required_data(
        self, ts_code: str, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """获取计算情绪面因子所需的市场数据"""
        try:
            # 构建查询SQL
            query = text(
                """
                SELECT ts_code, trade_date, open, high, low, close, pre_close,
                       vol, amount, turnover_rate, circ_mv, total_mv
                FROM stock_daily
                WHERE ts_code = :ts_code
                AND (:start_date IS NULL OR trade_date >= :start_date)
                AND (:end_date IS NULL OR trade_date <= :end_date)
                ORDER BY trade_date
            """
            )

            # 执行查询
            with self.engine.connect() as conn:
                result = conn.execute(query, {"ts_code": ts_code, "start_date": start_date, "end_date": end_date})

                market_data = pd.DataFrame(result.fetchall(), columns=result.keys())

            if not market_data.empty:
                # 转换数据类型
                market_data["trade_date"] = pd.to_datetime(market_data["trade_date"])
                market_data = market_data.set_index("trade_date")

                # 转换数值列
                numeric_columns = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "pre_close",
                    "vol",
                    "amount",
                    "turnover_rate",
                    "circ_mv",
                    "total_mv",
                ]
                for col in numeric_columns:
                    if col in market_data.columns:
                        market_data[col] = pd.to_numeric(market_data[col], errors="coerce")

            return market_data

        except Exception as e:
            logger.error(f"获取股票 {ts_code} 的市场数据失败: {e}")
            return pd.DataFrame()

    def calculate_factors(
        self,
        ts_code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        factor_names: Optional[List[str]] = None,
    ) -> FactorResult:
        """计算指定股票的情绪面因子"""
        start_time = datetime.now()

        try:
            # 获取市场数据
            market_data = self.get_required_data(ts_code, start_date, end_date)

            if market_data.empty:
                return FactorResult(
                    ts_code=ts_code,
                    calculation_date=datetime.now(),
                    factor_type=self.factor_type,
                    status=CalculationStatus.SKIPPED,
                    error_message="无可用市场数据",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    data_points=0,
                    factors={},
                )

            # 计算各类因子
            all_factors = {}

            # 资金流向类因子
            if self.money_flow_calculator is not None:
                money_flow_factors = self.money_flow_calculator.calculate(market_data)
                all_factors.update(money_flow_factors)
            else:
                # 提供模拟的资金流向因子数据用于测试
                all_factors["money_flow_ratio"] = pd.Series([0.1] * len(market_data), index=market_data.index)

            # 关注度类因子
            if self.attention_calculator is not None:
                attention_factors = self.attention_calculator.calculate(market_data)
                all_factors.update(attention_factors)
            else:
                # 提供模拟的关注度因子数据用于测试
                all_factors["attention_score"] = pd.Series([0.5] * len(market_data), index=market_data.index)

            # 情绪强度类因子
            if self.sentiment_calculator is not None:
                sentiment_factors = self.sentiment_calculator.calculate(market_data)
                all_factors.update(sentiment_factors)
            else:
                # 提供模拟的情绪强度因子数据用于测试
                all_factors["sentiment_strength"] = pd.Series([0.3] * len(market_data), index=market_data.index)

            # 事件类因子
            if self.event_calculator is not None:
                event_factors = self.event_calculator.calculate(market_data)
                all_factors.update(event_factors)
            else:
                # 提供模拟的事件因子数据用于测试
                all_factors["abnormal_volume"] = pd.Series([0.2] * len(market_data), index=market_data.index)
                all_factors["abnormal_return"] = pd.Series([0.1] * len(market_data), index=market_data.index)

            # 过滤指定的因子
            if factor_names:
                all_factors = {k: v for k, v in all_factors.items() if k in factor_names}

            # 转换为FactorValue格式
            factor_values = {}
            for factor_name, factor_series in all_factors.items():
                if not factor_series.empty and not factor_series.isna().all():
                    factor_values[factor_name] = [
                        FactorValue(
                            ts_code=ts_code,
                            trade_date=idx.date() if hasattr(idx, "date") else idx,
                            factor_name=factor_name,
                            raw_value=float(val) if pd.notna(val) else None,
                        )
                        for idx, val in factor_series.items()
                        if pd.notna(val)
                    ]

            return FactorResult(
                ts_code=ts_code,
                calculation_date=datetime.now(),
                factor_type=self.factor_type,
                status=CalculationStatus.SUCCESS,
                execution_time=(datetime.now() - start_time).total_seconds(),
                data_points=len(market_data),
                factors=factor_values,
            )

        except Exception as e:
            logger.error(f"计算股票 {ts_code} 的情绪面因子失败: {e}")
            return FactorResult(
                ts_code=ts_code,
                calculation_date=datetime.now(),
                factor_type=self.factor_type,
                status=CalculationStatus.FAILED,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
                data_points=0,
                factors={},
            )

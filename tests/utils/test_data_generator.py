import random
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据自动生成工具
为单元测试和集成测试提供各种类型的模拟数据
"""


class MarketRegime(Enum):
    """市场状态枚举"""

    BULL = "bull"  # 牛市
    BEAR = "bear"  # 熊市
    SIDEWAYS = "sideways"  # 震荡市
    VOLATILE = "volatile"  # 高波动市


class TestDataGenerator:
    """测试数据生成器"""

    def __init__(self, seed: int = 42):
        """
        初始化数据生成器

        Args:
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_stock_list(self, count: int = 100, markets: List[str] = None) -> List[str]:
        """
        生成股票代码列表

        Args:
            count: 股票数量
            markets: 市场列表，如['SZ', 'SH']

        Returns:
            List[str]: 股票代码列表
        """
        if markets is None:
            markets = ["SZ", "SH"]

        stock_codes = []
        for i in range(count):
            market = random.choice(markets)
            if market == "SZ":
                # 深市股票代码
                code_num = random.randint(1, 999999)
                if code_num < 300000:
                    code_num = random.randint(1, 2999) if code_num < 3000 else code_num
                else:
                    code_num = random.randint(300001, 399999)
            else:
                # 沪市股票代码
                code_num = random.randint(600001, 688999)

            stock_code = f"{str(code_num).zfill(6)}.{market}"
            stock_codes.append(stock_code)

        return list(set(stock_codes))  # 去重

    def generate_trading_calendar(self, start_date: str = "2023-01-01", end_date: str = "2023-12-31") -> List[date]:
        """
        生成交易日历（排除周末和节假日）

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            List[date]: 交易日列表
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        # 简化处理：只排除周末
        trading_days = []
        current_date = start

        while current_date <= end:
            # 排除周末
            if current_date.weekday() < 5:  # 0-4是周一到周五
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        return trading_days

    def generate_price_series(
        self,
        length: int,
        start_price: float = 100.0,
        regime: MarketRegime = MarketRegime.SIDEWAYS,
        volatility: float = 0.02,
    ) -> np.ndarray:
        """
        生成价格序列

        Args:
            length: 序列长度
            start_price: 起始价格
            regime: 市场状态
            volatility: 波动率

        Returns:
            np.ndarray: 价格序列
        """
        if regime == MarketRegime.BULL:
            # 牛市：正向趋势
            trend = np.linspace(0, 0.5, length)
            noise = np.random.normal(0, volatility, length)
            returns = trend / length + noise
        elif regime == MarketRegime.BEAR:
            # 熊市：负向趋势
            trend = np.linspace(0, -0.3, length)
            noise = np.random.normal(0, volatility, length)
            returns = trend / length + noise
        elif regime == MarketRegime.VOLATILE:
            # 高波动市：增加波动率
            returns = np.random.normal(0, volatility * 2, length)
        else:
            # 震荡市：无明显趋势
            returns = np.random.normal(0, volatility, length)

        # 生成价格序列
        prices = start_price * np.exp(np.cumsum(returns))
        return prices

    def generate_ohlc_data(self, prices: np.ndarray, intraday_volatility: float = 0.01) -> Dict[str, np.ndarray]:
        """
        基于收盘价生成OHLC数据

        Args:
            prices: 收盘价序列
            intraday_volatility: 日内波动率

        Returns:
            Dict[str, np.ndarray]: OHLC数据
        """
        length = len(prices)

        # 生成开盘价（基于前一日收盘价）
        open_prices = np.zeros(length)
        open_prices[0] = prices[0]
        for i in range(1, length):
            gap = np.random.normal(0, intraday_volatility * 0.5)
            open_prices[i] = prices[i - 1] * (1 + gap)

        # 生成最高价和最低价
        high_prices = np.zeros(length)
        low_prices = np.zeros(length)

        for i in range(length):
            # 日内波动
            intraday_high = max(open_prices[i], prices[i]) * (1 + abs(np.random.normal(0, intraday_volatility)))
            intraday_low = min(open_prices[i], prices[i]) * (1 - abs(np.random.normal(0, intraday_volatility)))

            high_prices[i] = max(intraday_high, open_prices[i], prices[i])
            low_prices[i] = min(intraday_low, open_prices[i], prices[i])

        return {"open": open_prices, "high": high_prices, "low": low_prices, "close": prices}

    def generate_volume_data(
        self, prices: np.ndarray, base_volume: int = 5000000, price_volume_correlation: float = 0.3
    ) -> np.ndarray:
        """
        生成成交量数据

        Args:
            prices: 价格序列
            base_volume: 基础成交量
            price_volume_correlation: 价量相关性

        Returns:
            np.ndarray: 成交量序列
        """
        length = len(prices)

        # 计算价格变化率
        price_changes = np.diff(prices) / prices[:-1]
        price_changes = np.concatenate([[0], price_changes])

        # 生成与价格变化相关的成交量
        volume_factor = 1 + price_volume_correlation * np.abs(price_changes)
        random_factor = np.random.lognormal(0, 0.3, length)

        volumes = base_volume * volume_factor * random_factor
        volumes = np.maximum(volumes, base_volume * 0.1)  # 设置最小成交量

        return volumes.astype(int)

    def generate_stock_daily_data(
        self,
        stock_codes: List[str],
        trading_days: List[date],
        regime: MarketRegime = MarketRegime.SIDEWAYS,
        start_price_range: Tuple[float, float] = (10.0, 200.0),
        volatility_range: Tuple[float, float] = (0.015, 0.035),
    ) -> pd.DataFrame:
        """
        生成股票日线数据

        Args:
            stock_codes: 股票代码列表
            trading_days: 交易日列表
            regime: 市场状态
            start_price_range: 起始价格范围
            volatility_range: 波动率范围

        Returns:
            pd.DataFrame: 股票日线数据
        """
        data_list = []

        for stock_code in stock_codes:
            # 为每只股票生成随机参数
            start_price = np.random.uniform(*start_price_range)
            volatility = np.random.uniform(*volatility_range)
            base_volume = np.random.randint(1000000, 20000000)

            # 生成价格序列
            prices = self.generate_price_series(len(trading_days), start_price, regime, volatility)

            # 生成OHLC数据
            ohlc_data = self.generate_ohlc_data(prices)

            # 生成成交量数据
            volumes = self.generate_volume_data(prices, base_volume)

            # 生成成交额数据
            amounts = prices * volumes * (1 + np.random.normal(0, 0.1, len(prices)))

            # 计算涨跌幅
            pre_close = np.concatenate([[start_price], prices[:-1]])
            pct_chg = (prices - pre_close) / pre_close * 100

            # 组装数据
            for i, trade_date in enumerate(trading_days):
                data_list.append(
                    {
                        "ts_code": stock_code,
                        "trade_date": trade_date,
                        "open": round(ohlc_data["open"][i], 2),
                        "high": round(ohlc_data["high"][i], 2),
                        "low": round(ohlc_data["low"][i], 2),
                        "close": round(prices[i], 2),
                        "pre_close": round(pre_close[i], 2),
                        "change": round(prices[i] - pre_close[i], 2),
                        "pct_chg": round(pct_chg[i], 2),
                        "vol": volumes[i],
                        "amount": round(amounts[i], 2),
                    }
                )

        return pd.DataFrame(data_list)

    def generate_financial_data(self, stock_codes: List[str], report_dates: List[str] = None) -> pd.DataFrame:
        """
        生成财务数据

        Args:
            stock_codes: 股票代码列表
            report_dates: 报告期列表

        Returns:
            pd.DataFrame: 财务数据
        """
        if report_dates is None:
            report_dates = ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"]

        data_list = []

        for stock_code in stock_codes:
            # 生成基础财务指标
            base_revenue = np.random.uniform(100000000, 10000000000)  # 1亿到100亿
            base_assets = base_revenue * np.random.uniform(2, 8)

            for report_date in report_dates:
                # 季度增长
                quarter_growth = np.random.normal(0.05, 0.15)  # 5%±15%增长

                revenue = base_revenue * (1 + quarter_growth)
                net_profit = revenue * np.random.uniform(0.02, 0.15)  # 2%-15%净利率
                total_assets = base_assets * (1 + quarter_growth * 0.5)
                total_equity = total_assets * np.random.uniform(0.3, 0.7)
                total_liab = total_assets - total_equity

                data_list.append(
                    {
                        "ts_code": stock_code,
                        "end_date": report_date,
                        "revenue": round(revenue, 2),
                        "net_profit": round(net_profit, 2),
                        "total_assets": round(total_assets, 2),
                        "total_equity": round(total_equity, 2),
                        "total_liab": round(total_liab, 2),
                        "gross_profit": round(revenue * np.random.uniform(0.15, 0.4), 2),
                        "operating_profit": round(net_profit * np.random.uniform(1.2, 2.0), 2),
                        "ebitda": round(net_profit * np.random.uniform(1.5, 3.0), 2),
                        "cash_flow_ops": round(net_profit * np.random.uniform(0.8, 1.5), 2),
                    }
                )

        return pd.DataFrame(data_list)

    def generate_industry_data(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        生成行业分类数据

        Args:
            stock_codes: 股票代码列表

        Returns:
            pd.DataFrame: 行业数据
        """
        industries = [
            "银行",
            "保险",
            "证券",
            "地产",
            "建筑",
            "钢铁",
            "煤炭",
            "有色",
            "化工",
            "石化",
            "电力",
            "公用",
            "交通",
            "汽车",
            "机械",
            "电子",
            "通信",
            "计算机",
            "传媒",
            "医药",
            "食品",
            "纺织",
            "轻工",
            "商贸",
            "农业",
            "综合",
        ]

        data_list = []
        for stock_code in stock_codes:
            industry = np.random.choice(industries)
            data_list.append(
                {"ts_code": stock_code, "industry": industry, "industry_code": f"SW{industries.index(industry)+1:02d}"}
            )

        return pd.DataFrame(data_list)

    def generate_factor_data(
        self,
        stock_codes: List[str],
        trading_days: List[date],
        factor_names: List[str],
        factor_correlations: Dict[str, float] = None,
    ) -> pd.DataFrame:
        """
        生成因子数据

        Args:
            stock_codes: 股票代码列表
            trading_days: 交易日列表
            factor_names: 因子名称列表
            factor_correlations: 因子间相关性

        Returns:
            pd.DataFrame: 因子数据
        """
        if factor_correlations is None:
            factor_correlations = {}

        data_list = []

        for stock_code in stock_codes:
            # 为每只股票生成基础因子
            base_factor = np.random.normal(0, 1, len(trading_days))

            stock_factors = {}
            for i, factor_name in enumerate(factor_names):
                if i == 0:
                    # 第一个因子作为基础
                    stock_factors[factor_name] = base_factor
                else:
                    # 其他因子与基础因子有一定相关性
                    correlation = factor_correlations.get(factor_name, 0.2)
                    noise = np.random.normal(0, 1, len(trading_days))
                    stock_factors[factor_name] = base_factor * correlation + noise * np.sqrt(1 - correlation**2)

            # 组装数据
            for i, trade_date in enumerate(trading_days):
                row_data = {"ts_code": stock_code, "factor_date": trade_date}

                for factor_name in factor_names:
                    row_data[factor_name] = round(stock_factors[factor_name][i], 6)

                data_list.append(row_data)

        return pd.DataFrame(data_list)

    def generate_return_data(
        self, stock_codes: List[str], trading_days: List[date], return_periods: List[int] = None
    ) -> pd.DataFrame:
        """
        生成收益率数据

        Args:
            stock_codes: 股票代码列表
            trading_days: 交易日列表
            return_periods: 收益率周期列表

        Returns:
            pd.DataFrame: 收益率数据
        """
        if return_periods is None:
            return_periods = [1, 5, 10, 20]

        data_list = []

        for stock_code in stock_codes:
            # 生成日收益率序列
            daily_returns = np.random.normal(0.001, 0.02, len(trading_days))

            for i, trade_date in enumerate(trading_days):
                row_data = {"ts_code": stock_code, "trade_date": trade_date}

                # 计算不同周期的收益率
                for period in return_periods:
                    if i >= period - 1:
                        period_return = np.sum(daily_returns[i - period + 1 : i + 1])
                        row_data[f"return_{period}d"] = round(period_return * 100, 4)
                    else:
                        row_data[f"return_{period}d"] = None

                data_list.append(row_data)

        return pd.DataFrame(data_list)

    def generate_benchmark_data(self, trading_days: List[date], benchmark_name: str = "000300.SH") -> pd.DataFrame:
        """
        生成基准指数数据

        Args:
            trading_days: 交易日列表
            benchmark_name: 基准名称

        Returns:
            pd.DataFrame: 基准数据
        """
        # 生成基准价格序列（相对平稳）
        prices = self.generate_price_series(
            len(trading_days), start_price=4000.0, regime=MarketRegime.SIDEWAYS, volatility=0.015
        )

        data_list = []
        for i, trade_date in enumerate(trading_days):
            if i == 0:
                pct_chg = 0.0
            else:
                pct_chg = (prices[i] - prices[i - 1]) / prices[i - 1] * 100

            data_list.append(
                {
                    "ts_code": benchmark_name,
                    "trade_date": trade_date,
                    "close": round(prices[i], 2),
                    "pct_chg": round(pct_chg, 2),
                }
            )

        return pd.DataFrame(data_list)

    def create_test_dataset(
        self,
        n_stocks: int = 50,
        n_days: int = 252,
        start_date: str = "2023-01-01",
        regime: MarketRegime = MarketRegime.SIDEWAYS,
        include_financial: bool = True,
        include_factors: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        创建完整的测试数据集

        Args:
            n_stocks: 股票数量
            n_days: 交易日数量
            start_date: 开始日期
            regime: 市场状态
            include_financial: 是否包含财务数据
            include_factors: 是否包含因子数据

        Returns:
            Dict[str, pd.DataFrame]: 完整数据集
        """
        # 生成基础数据
        stock_codes = self.generate_stock_list(n_stocks)

        # 生成交易日历
        end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=int(n_days * 1.4))).strftime("%Y-%m-%d")
        all_trading_days = self.generate_trading_calendar(start_date, end_date)
        trading_days = all_trading_days[:n_days]

        # 生成各类数据
        dataset = {}

        # 股票日线数据
        dataset["stock_daily"] = self.generate_stock_daily_data(stock_codes, trading_days, regime)

        # 行业数据
        dataset["industry"] = self.generate_industry_data(stock_codes)

        # 收益率数据
        dataset["returns"] = self.generate_return_data(stock_codes, trading_days)

        # 基准数据
        dataset["benchmark"] = self.generate_benchmark_data(trading_days)

        # 财务数据（可选）
        if include_financial:
            dataset["financial"] = self.generate_financial_data(stock_codes)

        # 因子数据（可选）
        if include_factors:
            factor_names = [
                "sma_5",
                "sma_20",
                "rsi_14",
                "macd",
                "pe_ttm",
                "pb",
                "roe",
                "money_flow_5",
                "attention_score",
            ]
            dataset["factors"] = self.generate_factor_data(stock_codes, trading_days, factor_names)

        return dataset


# 便捷函数
def create_simple_test_data(n_stocks: int = 10, n_days: int = 50) -> Dict[str, pd.DataFrame]:
    """创建简单的测试数据"""
    generator = TestDataGenerator()
    return generator.create_test_dataset(
        n_stocks=n_stocks, n_days=n_days, include_financial=False, include_factors=False
    )


def create_full_test_data(n_stocks: int = 100, n_days: int = 252) -> Dict[str, pd.DataFrame]:
    """创建完整的测试数据"""
    generator = TestDataGenerator()
    return generator.create_test_dataset(n_stocks=n_stocks, n_days=n_days, include_financial=True, include_factors=True)


if __name__ == "__main__":
    # 示例用法
    generator = TestDataGenerator()

    # 创建测试数据集
    test_data = generator.create_test_dataset(n_stocks=20, n_days=100, regime=MarketRegime.BULL)

    print("生成的数据集包含:")
    for name, df in test_data.items():
        print(f"- {name}: {len(df)} 条记录")

    # 显示股票日线数据示例
    print("\n股票日线数据示例:")
    print(test_data["stock_daily"].head())

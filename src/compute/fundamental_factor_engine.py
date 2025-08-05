from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本面因子计算引擎
实现各类基本面指标因子的计算
"""


from .base_factor_engine import BaseFactorCalculator, BaseFactorEngine
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


class ValuationFactorCalculator(BaseFactorCalculator):
    """估值类因子计算器"""

    def calculate_pe_ratio(self, market_data: pd.DataFrame, financial_data: pd.DataFrame) -> pd.Series:
        """计算市盈率"""
        # 合并市场数据和财务数据
        merged = self._merge_market_financial_data(market_data, financial_data)

        # PE = 总市值 / 净利润TTM
        pe_ratio = merged["total_mv"] / merged["net_profit_ttm"]

        # 过滤负值和异常值
        pe_ratio = pe_ratio.where((pe_ratio > 0) & (pe_ratio < 1000))

        return pe_ratio

    def calculate_pb_ratio(self, market_data: pd.DataFrame, financial_data: pd.DataFrame) -> pd.Series:
        """计算市净率"""
        merged = self._merge_market_financial_data(market_data, financial_data)

        # PB = 总市值 / 净资产
        pb_ratio = merged["total_mv"] / merged["total_hldr_eqy_exc_min_int"]

        # 过滤负值和异常值
        pb_ratio = pb_ratio.where((pb_ratio > 0) & (pb_ratio < 100))

        return pb_ratio

    def calculate_ps_ratio(self, market_data: pd.DataFrame, financial_data: pd.DataFrame) -> pd.Series:
        """计算市销率"""
        merged = self._merge_market_financial_data(market_data, financial_data)

        # PS = 总市值 / 营业收入TTM
        ps_ratio = merged["total_mv"] / merged["revenue_ttm"]

        # 过滤负值和异常值
        ps_ratio = ps_ratio.where((ps_ratio > 0) & (ps_ratio < 100))

        return ps_ratio

    def calculate_pcf_ratio(self, market_data: pd.DataFrame, financial_data: pd.DataFrame) -> pd.Series:
        """计算市现率"""
        merged = self._merge_market_financial_data(market_data, financial_data)

        # PCF = 总市值 / 经营现金流TTM
        if "operating_cash_flow_ttm" in merged.columns:
            pcf_ratio = merged["total_mv"] / merged["operating_cash_flow_ttm"]
            pcf_ratio = pcf_ratio.where((pcf_ratio > 0) & (pcf_ratio < 1000))
            return pcf_ratio
        else:
            return pd.Series(dtype=float)

    def calculate_ev_ebitda(self, market_data: pd.DataFrame, financial_data: pd.DataFrame) -> pd.Series:
        """计算EV/EBITDA"""
        merged = self._merge_market_financial_data(market_data, financial_data)

        # EV = 总市值 + 净债务
        # 简化计算：EV ≈ 总市值（假设净债务为0）
        if "ebitda_ttm" in merged.columns:
            ev_ebitda = merged["total_mv"] / merged["ebitda_ttm"]
            ev_ebitda = ev_ebitda.where((ev_ebitda > 0) & (ev_ebitda < 1000))
            return ev_ebitda
        else:
            return pd.Series(dtype=float)

    def calculate_peg_ratio(self, market_data: pd.DataFrame, financial_data: pd.DataFrame) -> pd.Series:
        """计算PEG比率"""
        merged = self._merge_market_financial_data(market_data, financial_data)

        # PEG = PE / 净利润增长率
        if "net_profit_yoy" in merged.columns:
            pe_ratio = merged["total_mv"] / merged["net_profit_ttm"]
            growth_rate = merged["net_profit_yoy"] * 100  # 转换为百分比

            peg_ratio = pe_ratio / growth_rate
            peg_ratio = peg_ratio.where((peg_ratio > 0) & (peg_ratio < 10))
            return peg_ratio
        else:
            return pd.Series(dtype=float)

    def _merge_market_financial_data(self, market_data: pd.DataFrame, financial_data: pd.DataFrame) -> pd.DataFrame:
        """合并市场数据和财务数据"""
        if market_data.empty or financial_data.empty:
            return pd.DataFrame()

        # 按股票代码和日期合并
        merged = pd.merge(market_data, financial_data, on=["ts_code"], how="left", suffixes=("_market", "_financial"))

        return merged

    def calculate(self, market_data: pd.DataFrame, financial_data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """计算所有估值类因子"""
        results = {}

        try:
            # 基础估值指标
            results["pe_ttm"] = self.calculate_pe_ratio(market_data, financial_data)
            results["pb"] = self.calculate_pb_ratio(market_data, financial_data)
            results["ps_ttm"] = self.calculate_ps_ratio(market_data, financial_data)

            # 高级估值指标
            results["pcf_ttm"] = self.calculate_pcf_ratio(market_data, financial_data)
            results["ev_ebitda"] = self.calculate_ev_ebitda(market_data, financial_data)
            results["peg"] = self.calculate_peg_ratio(market_data, financial_data)

        except Exception as e:
            logger.error(f"计算估值类因子时出错: {e}")
            raise

        return results


class ProfitabilityFactorCalculator(BaseFactorCalculator):
    """盈利能力因子计算器"""

    def calculate_roe(self, financial_data: pd.DataFrame) -> pd.Series:
        """计算净资产收益率 (Return on Equity)

        ROE = 净利润 / 股东权益
        反映公司运用股东资本的效率，是衡量盈利能力的重要指标
        """
        # 数据验证
        required_cols = ["net_profit", "total_hldr_eqy_exc_min_int"]
        if not all(col in financial_data.columns for col in required_cols):
            logger.warning("缺少计算ROE所需的列")
            return pd.Series(dtype=float)

        # 计算ROE = 净利润 / 股东权益
        mask = (financial_data["total_hldr_eqy_exc_min_int"] != 0) & (
            financial_data["total_hldr_eqy_exc_min_int"].notna()
        )
        roe = pd.Series(dtype=float, index=financial_data.index)

        valid_data = financial_data[mask]
        if not valid_data.empty:
            roe_values = (valid_data["net_profit"] / valid_data["total_hldr_eqy_exc_min_int"]) * 100
            # 更精确的异常值过滤
            roe_values = roe_values.clip(lower=-100, upper=100)
            roe[valid_data.index] = roe_values

        return roe

    def calculate_roa(self, financial_data: pd.DataFrame) -> pd.Series:
        """计算总资产收益率 (Return on Assets)

        ROA = 净利润 / 总资产
        反映公司运用全部资产的盈利能力，是衡量资产运用效率的重要指标
        """
        # 数据验证
        required_cols = ["net_profit", "total_assets"]
        if not all(col in financial_data.columns for col in required_cols):
            logger.warning("缺少计算ROA所需的列")
            return pd.Series(dtype=float)

        # 计算ROA = 净利润 / 总资产
        mask = (financial_data["total_assets"] != 0) & (financial_data["total_assets"].notna())
        roa = pd.Series(dtype=float, index=financial_data.index)

        valid_data = financial_data[mask]
        if not valid_data.empty:
            roa_values = (valid_data["net_profit"] / valid_data["total_assets"]) * 100
            # 更精确的异常值过滤
            roa_values = roa_values.clip(lower=-100, upper=100)
            roa[valid_data.index] = roa_values

        return roa

    def calculate_roic(self, financial_data: pd.DataFrame) -> pd.Series:
        """计算投入资本回报率"""
        # ROIC = EBIT * (1 - 税率) / 投入资本
        # 简化计算：ROIC ≈ 营业利润 / (总资产 - 无息负债)
        if "operate_profit" in financial_data.columns:
            # 假设无息负债为流动负债的一部分
            invested_capital = financial_data["total_assets"] - financial_data.get("total_cur_liab", 0) * 0.5
            roic = financial_data["operate_profit"] / invested_capital

            roic = roic * 100
            roic = roic.where((roic > -50) & (roic < 50))
            return roic
        else:
            return pd.Series(dtype=float)

    def calculate_gross_margin(self, financial_data: pd.DataFrame) -> pd.Series:
        """计算毛利率"""
        # 毛利率 = (营业收入 - 营业成本) / 营业收入
        if "revenue" in financial_data.columns and "oper_cost" in financial_data.columns:
            gross_margin = (financial_data["revenue"] - financial_data["oper_cost"]) / financial_data["revenue"]
            gross_margin = gross_margin * 100
            gross_margin = gross_margin.where((gross_margin > -100) & (gross_margin < 100))
            return gross_margin
        else:
            return pd.Series(dtype=float)

    def calculate_net_margin(self, financial_data: pd.DataFrame) -> pd.Series:
        """计算净利率"""
        # 净利率 = 净利润 / 营业收入
        net_margin = financial_data["net_profit"] / financial_data["revenue"]

        net_margin = net_margin * 100
        net_margin = net_margin.where((net_margin > -100) & (net_margin < 100))

        return net_margin

    def calculate_operating_margin(self, financial_data: pd.DataFrame) -> pd.Series:
        """计算营业利润率"""
        if "operate_profit" in financial_data.columns:
            operating_margin = financial_data["operate_profit"] / financial_data["revenue"]
            operating_margin = operating_margin * 100
            operating_margin = operating_margin.where((operating_margin > -100) & (operating_margin < 100))
            return operating_margin
        else:
            return pd.Series(dtype=float)

    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """计算所有盈利能力因子"""
        results = {}

        try:
            # 基础盈利能力指标
            results["roe"] = self.calculate_roe(financial_data)
            results["roa"] = self.calculate_roa(financial_data)
            results["roic"] = self.calculate_roic(financial_data)

            # 利润率指标
            results["gross_margin"] = self.calculate_gross_margin(financial_data)
            results["net_margin"] = self.calculate_net_margin(financial_data)
            results["operating_margin"] = self.calculate_operating_margin(financial_data)

        except Exception as e:
            logger.error(f"计算盈利能力因子时出错: {e}")
            raise

        return results


class GrowthFactorCalculator(BaseFactorCalculator):
    """成长性因子计算器"""

    def calculate_revenue_growth(self, financial_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算营收增长率"""
        results = {}

        # 按股票代码分组计算增长率
        for ts_code in financial_data["ts_code"].unique():
            stock_data = financial_data[financial_data["ts_code"] == ts_code].sort_values("end_date")

            # 同比增长率（年度数据）
            revenue_yoy = stock_data["revenue"].pct_change(4) * 100  # 假设季度数据，4个季度为一年

            # 环比增长率
            revenue_qoq = stock_data["revenue"].pct_change() * 100

            # 3年复合增长率
            revenue_cagr_3y = self._calculate_cagr(stock_data["revenue"], periods=12)  # 3年=12个季度

            # 将结果添加到对应的索引位置
            for idx, (yoy, qoq, cagr) in zip(stock_data.index, zip(revenue_yoy, revenue_qoq, revenue_cagr_3y)):
                if ts_code not in results:
                    results[ts_code] = {"yoy": {}, "qoq": {}, "cagr_3y": {}}

                results[ts_code]["yoy"][idx] = yoy
                results[ts_code]["qoq"][idx] = qoq
                results[ts_code]["cagr_3y"][idx] = cagr

        # 转换为Series格式
        revenue_yoy_series = pd.Series(dtype=float, index=financial_data.index)
        revenue_qoq_series = pd.Series(dtype=float, index=financial_data.index)
        revenue_cagr_3y_series = pd.Series(dtype=float, index=financial_data.index)

        for ts_code, data in results.items():
            for idx, value in data["yoy"].items():
                revenue_yoy_series.loc[idx] = value
            for idx, value in data["qoq"].items():
                revenue_qoq_series.loc[idx] = value
            for idx, value in data["cagr_3y"].items():
                revenue_cagr_3y_series.loc[idx] = value

        return {
            "revenue_yoy": revenue_yoy_series.where((revenue_yoy_series > -100) & (revenue_yoy_series < 1000)),
            "revenue_qoq": revenue_qoq_series.where((revenue_qoq_series > -100) & (revenue_qoq_series < 1000)),
            "revenue_cagr_3y": revenue_cagr_3y_series.where(
                (revenue_cagr_3y_series > -100) & (revenue_cagr_3y_series < 1000)
            ),
        }

    def calculate_profit_growth(self, financial_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算净利润增长率"""
        results = {}

        # 按股票代码分组计算增长率
        for ts_code in financial_data["ts_code"].unique():
            stock_data = financial_data[financial_data["ts_code"] == ts_code].sort_values("end_date")

            # 同比增长率
            profit_yoy = stock_data["net_profit"].pct_change(4) * 100

            # 环比增长率
            profit_qoq = stock_data["net_profit"].pct_change() * 100

            # 3年复合增长率
            profit_cagr_3y = self._calculate_cagr(stock_data["net_profit"], periods=12)

            # 将结果添加到对应的索引位置
            for idx, (yoy, qoq, cagr) in zip(stock_data.index, zip(profit_yoy, profit_qoq, profit_cagr_3y)):
                if ts_code not in results:
                    results[ts_code] = {"yoy": {}, "qoq": {}, "cagr_3y": {}}

                results[ts_code]["yoy"][idx] = yoy
                results[ts_code]["qoq"][idx] = qoq
                results[ts_code]["cagr_3y"][idx] = cagr

        # 转换为Series格式
        profit_yoy_series = pd.Series(dtype=float, index=financial_data.index)
        profit_qoq_series = pd.Series(dtype=float, index=financial_data.index)
        profit_cagr_3y_series = pd.Series(dtype=float, index=financial_data.index)

        for ts_code, data in results.items():
            for idx, value in data["yoy"].items():
                profit_yoy_series.loc[idx] = value
            for idx, value in data["qoq"].items():
                profit_qoq_series.loc[idx] = value
            for idx, value in data["cagr_3y"].items():
                profit_cagr_3y_series.loc[idx] = value

        return {
            "net_profit_yoy": profit_yoy_series.where((profit_yoy_series > -1000) & (profit_yoy_series < 1000)),
            "net_profit_qoq": profit_qoq_series.where((profit_qoq_series > -1000) & (profit_qoq_series < 1000)),
            "net_profit_cagr_3y": profit_cagr_3y_series.where(
                (profit_cagr_3y_series > -1000) & (profit_cagr_3y_series < 1000)
            ),
        }

    def _calculate_cagr(self, series: pd.Series, periods: int) -> pd.Series:
        """计算复合年增长率"""
        if len(series) < periods:
            return pd.Series([np.nan] * len(series), index=series.index)

        cagr_values = []
        for i in range(len(series)):
            if i < periods:
                cagr_values.append(np.nan)
            else:
                start_value = series.iloc[i - periods]
                end_value = series.iloc[i]

                if pd.isna(start_value) or pd.isna(end_value) or start_value <= 0:
                    cagr_values.append(np.nan)
                else:
                    # CAGR = (结束值/开始值)^(1/年数) - 1
                    years = periods / 4  # 假设季度数据
                    cagr = (end_value / start_value) ** (1 / years) - 1
                    cagr_values.append(cagr * 100)  # 转换为百分比

        return pd.Series(cagr_values, index=series.index)

    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """计算所有成长性因子"""
        results = {}

        try:
            # 营收增长率
            revenue_growth = self.calculate_revenue_growth(financial_data)
            results.update(revenue_growth)

            # 净利润增长率
            profit_growth = self.calculate_profit_growth(financial_data)
            results.update(profit_growth)

        except Exception as e:
            logger.error(f"计算成长性因子时出错: {e}")
            raise

        return results


class QualityFactorCalculator(BaseFactorCalculator):
    """财务质量因子计算器"""

    def calculate_debt_ratios(self, financial_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算债务比率"""
        results = {}

        # 资产负债率
        debt_to_equity = financial_data["total_liab"] / financial_data["total_hldr_eqy_exc_min_int"]
        results["debt_to_equity"] = debt_to_equity.where((debt_to_equity >= 0) & (debt_to_equity < 10))

        return results

    def calculate_liquidity_ratios(self, financial_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算流动性比率"""
        results = {}

        # 流动比率
        if "total_cur_assets" in financial_data.columns and "total_cur_liab" in financial_data.columns:
            current_ratio = financial_data["total_cur_assets"] / financial_data["total_cur_liab"]
            results["current_ratio"] = current_ratio.where((current_ratio >= 0) & (current_ratio < 20))

        # 速动比率
        if all(col in financial_data.columns for col in ["total_cur_assets", "inventories", "total_cur_liab"]):
            quick_assets = financial_data["total_cur_assets"] - financial_data["inventories"].fillna(0)
            quick_ratio = quick_assets / financial_data["total_cur_liab"]
            results["quick_ratio"] = quick_ratio.where((quick_ratio >= 0) & (quick_ratio < 20))

        # 现金比率
        if "money_cap" in financial_data.columns and "total_cur_liab" in financial_data.columns:
            cash_ratio = financial_data["money_cap"] / financial_data["total_cur_liab"]
            results["cash_ratio"] = cash_ratio.where((cash_ratio >= 0) & (cash_ratio < 10))

        return results

    def calculate_efficiency_ratios(self, financial_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算效率比率"""
        results = {}

        # 资产周转率
        asset_turnover = financial_data["revenue"] / financial_data["total_assets"]
        results["asset_turnover"] = asset_turnover.where((asset_turnover >= 0) & (asset_turnover < 10))

        # 利息保障倍数
        if "operate_profit" in financial_data.columns and "fin_exp" in financial_data.columns:
            # 利息费用通常为负数，取绝对值
            interest_expense = financial_data["fin_exp"].abs()
            interest_coverage = financial_data["operate_profit"] / interest_expense
            results["interest_coverage"] = interest_coverage.where((interest_coverage > 0) & (interest_coverage < 1000))

        return results

    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """计算所有财务质量因子"""
        results = {}

        try:
            # 债务比率
            debt_ratios = self.calculate_debt_ratios(financial_data)
            results.update(debt_ratios)

            # 流动性比率
            liquidity_ratios = self.calculate_liquidity_ratios(financial_data)
            results.update(liquidity_ratios)

            # 效率比率
            efficiency_ratios = self.calculate_efficiency_ratios(financial_data)
            results.update(efficiency_ratios)

        except Exception as e:
            logger.error(f"计算财务质量因子时出错: {e}")
            raise

        return results


class FundamentalFactorEngine(BaseFactorEngine):
    """基本面因子计算引擎"""

    def __init__(self, engine):
        """初始化基本面因子引擎"""
        super().__init__(engine, FactorType.FUNDAMENTAL)

        # 初始化各类因子计算器
        self.valuation_calculator = ValuationFactorCalculator()
        self.profitability_calculator = ProfitabilityFactorCalculator()
        self.growth_calculator = GrowthFactorCalculator()
        self.quality_calculator = QualityFactorCalculator()

        logger.info("基本面因子引擎初始化完成")

    def _initialize_factors(self):
        """初始化因子配置和元数据"""
        # 估值类因子
        valuation_factors = [
            ("pe_ttm", "市盈率TTM", {}),
            ("pb", "市净率", {}),
            ("ps_ttm", "市销率TTM", {}),
            ("pcf_ttm", "市现率TTM", {}),
            ("ev_ebitda", "EV/EBITDA", {}),
            ("peg", "PEG比率", {}),
        ]

        for name, desc, params in valuation_factors:
            metadata = create_factor_metadata(
                name=name,
                description=desc,
                factor_type=FactorType.FUNDAMENTAL,
                category=FactorCategory.VALUATION,
                parameters=params,
                data_requirements=["market_data", "financial_data"],
                min_periods=1,
            )
            config = create_factor_config(name=name, parameters=params)
            self.register_factor(metadata, config)

        # 盈利能力类因子
        profitability_factors = [
            ("roe", "净资产收益率", {}),
            ("roa", "总资产收益率", {}),
            ("roic", "投入资本回报率", {}),
            ("gross_margin", "毛利率", {}),
            ("net_margin", "净利率", {}),
            ("operating_margin", "营业利润率", {}),
        ]

        for name, desc, params in profitability_factors:
            metadata = create_factor_metadata(
                name=name,
                description=desc,
                factor_type=FactorType.FUNDAMENTAL,
                category=FactorCategory.PROFITABILITY,
                parameters=params,
                data_requirements=["financial_data"],
                min_periods=1,
            )
            config = create_factor_config(name=name, parameters=params)
            self.register_factor(metadata, config)

        # 成长性类因子
        growth_factors = [
            ("revenue_yoy", "营收同比增长率", {}),
            ("revenue_qoq", "营收环比增长率", {}),
            ("revenue_cagr_3y", "营收3年复合增长率", {}),
            ("net_profit_yoy", "净利润同比增长率", {}),
            ("net_profit_qoq", "净利润环比增长率", {}),
            ("net_profit_cagr_3y", "净利润3年复合增长率", {}),
        ]

        for name, desc, params in growth_factors:
            metadata = create_factor_metadata(
                name=name,
                description=desc,
                factor_type=FactorType.FUNDAMENTAL,
                category=FactorCategory.GROWTH,
                parameters=params,
                data_requirements=["financial_data"],
                min_periods=4,  # 至少需要4个季度的数据
            )
            config = create_factor_config(name=name, parameters=params)
            self.register_factor(metadata, config)

        # 财务质量类因子
        quality_factors = [
            ("debt_to_equity", "资产负债率", {}),
            ("current_ratio", "流动比率", {}),
            ("quick_ratio", "速动比率", {}),
            ("cash_ratio", "现金比率", {}),
            ("asset_turnover", "资产周转率", {}),
            ("interest_coverage", "利息保障倍数", {}),
        ]

        for name, desc, params in quality_factors:
            metadata = create_factor_metadata(
                name=name,
                description=desc,
                factor_type=FactorType.FUNDAMENTAL,
                category=FactorCategory.QUALITY,
                parameters=params,
                data_requirements=["financial_data"],
                min_periods=1,
            )
            config = create_factor_config(name=name, parameters=params)
            self.register_factor(metadata, config)

    def get_required_data(
        self, ts_code: str, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """获取计算所需的财务数据和市场数据"""
        # 获取财务数据
        financial_query = """
            SELECT ts_code, end_date, ann_date, report_type,
                   total_revenue, revenue, net_profit, total_assets,
                   total_hldr_eqy_exc_min_int, total_liab, total_cur_assets,
                   total_cur_liab, inventories, money_cap, operate_profit,
                   oper_cost, fin_exp
            FROM income i
            LEFT JOIN balance_sheet b USING (ts_code, end_date, report_type)
            WHERE ts_code = :ts_code AND report_type = '1'
        """

        params = {"ts_code": ts_code}

        if start_date:
            financial_query += " AND end_date >= :start_date"
            params["start_date"] = start_date

        if end_date:
            financial_query += " AND end_date <= :end_date"
            params["end_date"] = end_date

        financial_query += " ORDER BY end_date DESC"

        # 获取市场数据
        market_query = """
            SELECT ts_code, trade_date, total_mv, pe_ttm, pb
            FROM daily_basic
            WHERE ts_code = :ts_code
        """

        if start_date:
            market_query += " AND trade_date >= :start_date"

        if end_date:
            market_query += " AND trade_date <= :end_date"

        market_query += " ORDER BY trade_date DESC"

        with self.engine.connect() as conn:
            financial_df = pd.read_sql_query(text(financial_query), conn, params=params)
            market_df = pd.read_sql_query(text(market_query), conn, params=params)

        # 转换日期格式
        if not financial_df.empty:
            financial_df["end_date"] = pd.to_datetime(financial_df["end_date"])

        if not market_df.empty:
            market_df["trade_date"] = pd.to_datetime(market_df["trade_date"])

        logger.info(f"获取股票 {ts_code} 财务数据 {len(financial_df)} 条，市场数据 {len(market_df)} 条")

        return financial_df, market_df

    def calculate_factors(
        self,
        ts_code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        factor_names: Optional[List[str]] = None,
    ) -> FactorResult:
        """计算基本面因子"""
        start_time = datetime.now()

        try:
            # 获取数据
            financial_data, market_data = self.get_required_data(ts_code, start_date, end_date)

            if financial_data.empty:
                return FactorResult(
                    ts_code=ts_code,
                    calculation_date=start_time,
                    factor_type=self.factor_type,
                    status=CalculationStatus.SKIPPED,
                    error_message="无可用财务数据",
                )

            # 计算各类因子
            all_factors = {}

            # 估值类因子（需要市场数据）
            if not market_data.empty:
                valuation_factors = self.valuation_calculator.calculate(market_data, financial_data)
                all_factors.update(valuation_factors)

            # 盈利能力类因子
            profitability_factors = self.profitability_calculator.calculate(financial_data)
            all_factors.update(profitability_factors)

            # 成长性类因子
            growth_factors = self.growth_calculator.calculate(financial_data)
            all_factors.update(growth_factors)

            # 财务质量类因子
            quality_factors = self.quality_calculator.calculate(financial_data)
            all_factors.update(quality_factors)

            # 创建因子结果
            result = FactorResult(
                ts_code=ts_code,
                calculation_date=start_time,
                factor_type=self.factor_type,
                status=CalculationStatus.SUCCESS,
                data_points=len(financial_data),
            )

            # 添加因子值
            for factor_name, factor_series in all_factors.items():
                if factor_names and factor_name not in factor_names:
                    continue

                for i, (report_date, value) in enumerate(zip(financial_data["end_date"], factor_series)):
                    if pd.notna(value):
                        factor_value = FactorValue(
                            ts_code=ts_code,
                            trade_date=report_date.date(),
                            factor_name=factor_name,
                            raw_value=float(value),
                        )
                        result.add_factor(factor_value)

            # 计算执行时间
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            logger.info(
                f"股票 {ts_code} 基本面因子计算完成，" f"计算了 {len(all_factors)} 个因子，耗时 {execution_time:.2f}秒"
            )

            return result

        except Exception as e:
            return self.handle_calculation_error(ts_code, e)


if __name__ == "__main__":
    # 测试代码
    print("基本面因子引擎测试")

    # 创建测试财务数据
    test_financial_data = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 8,
            "end_date": pd.date_range("2022-03-31", periods=8, freq="Q"),
            "revenue": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
            "net_profit": [100, 110, 120, 130, 140, 150, 160, 170],
            "total_assets": [5000, 5200, 5400, 5600, 5800, 6000, 6200, 6400],
            "total_hldr_eqy_exc_min_int": [2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700],
            "total_liab": [3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700],
            "total_cur_assets": [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200],
            "total_cur_liab": [800, 850, 900, 950, 1000, 1050, 1100, 1150],
            "oper_cost": [700, 770, 840, 910, 980, 1050, 1120, 1190],
            "operate_profit": [200, 220, 240, 260, 280, 300, 320, 340],
        }
    )

    # 创建测试市场数据
    test_market_data = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 8,
            "trade_date": pd.date_range("2022-03-31", periods=8, freq="Q"),
            "total_mv": [20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000],
        }
    )

    # 测试各类计算器
    valuation_calc = ValuationFactorCalculator()
    profitability_calc = ProfitabilityFactorCalculator()
    growth_calc = GrowthFactorCalculator()
    quality_calc = QualityFactorCalculator()

    print("测试估值类因子计算...")
    valuation_results = valuation_calc.calculate(test_market_data, test_financial_data)
    print(f"估值类因子数量: {len(valuation_results)}")

    print("测试盈利能力类因子计算...")
    profitability_results = profitability_calc.calculate(test_financial_data)
    print(f"盈利能力类因子数量: {len(profitability_results)}")

    print("测试成长性类因子计算...")
    growth_results = growth_calc.calculate(test_financial_data)
    print(f"成长性类因子数量: {len(growth_results)}")

    print("测试财务质量类因子计算...")
    quality_results = quality_calc.calculate(test_financial_data)
    print(f"财务质量类因子数量: {len(quality_results)}")

    print("基本面因子引擎测试完成")

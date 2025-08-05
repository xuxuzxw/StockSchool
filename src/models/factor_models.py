from datetime import date

from sqlalchemy import Column, Date, Float, ForeignKey, Index, Integer, PrimaryKeyConstraint, String, Text
from sqlalchemy.orm import relationship

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子相关数据模型
定义技术因子、基本面因子等ORM模型
"""

from .base import Base, TimestampMixin


class StockTechnicalFactors(Base, TimestampMixin):
    """股票技术因子表"""

    __tablename__ = "stock_technical_factors"

    ts_code = Column(String(20), ForeignKey("stock_basic.ts_code"), nullable=False, comment="TS代码")
    trade_date = Column(Date, nullable=False, comment="交易日期")

    # 动量因子
    momentum_5 = Column(Float, comment="5日动量")
    momentum_10 = Column(Float, comment="10日动量")
    momentum_20 = Column(Float, comment="20日动量")
    rsi_14 = Column(Float, comment="14日RSI")
    williams_r_14 = Column(Float, comment="14日威廉指标")

    # 趋势因子
    sma_5 = Column(Float, comment="5日简单移动平均")
    sma_10 = Column(Float, comment="10日简单移动平均")
    sma_20 = Column(Float, comment="20日简单移动平均")
    sma_60 = Column(Float, comment="60日简单移动平均")
    ema_12 = Column(Float, comment="12日指数移动平均")
    ema_26 = Column(Float, comment="26日指数移动平均")
    macd = Column(Float, comment="MACD")
    macd_signal = Column(Float, comment="MACD信号线")
    macd_histogram = Column(Float, comment="MACD柱状图")
    price_to_sma20 = Column(Float, comment="价格相对20日均线")
    price_to_sma60 = Column(Float, comment="价格相对60日均线")

    # 波动率因子
    volatility_5 = Column(Float, comment="5日波动率")
    volatility_20 = Column(Float, comment="20日波动率")
    volatility_60 = Column(Float, comment="60日波动率")
    atr_14 = Column(Float, comment="14日ATR")
    bb_upper = Column(Float, comment="布林带上轨")
    bb_middle = Column(Float, comment="布林带中轨")
    bb_lower = Column(Float, comment="布林带下轨")
    bb_width = Column(Float, comment="布林带宽度")
    bb_position = Column(Float, comment="布林带位置")

    # 成交量因子
    volume_sma_5 = Column(Float, comment="5日成交量均值")
    volume_sma_20 = Column(Float, comment="20日成交量均值")
    volume_ratio_5 = Column(Float, comment="5日成交量比率")
    volume_ratio_20 = Column(Float, comment="20日成交量比率")
    vpt = Column(Float, comment="量价趋势指标")
    mfi = Column(Float, comment="资金流量指标")

    # 建立与股票基础信息的关系
    stock_info = relationship("StockBasic", back_populates="technical_factors", foreign_keys=[ts_code])

    # 设置复合主键和索引
    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date"),
        Index("idx_technical_factors_code_date", "ts_code", "trade_date"),
        Index("idx_technical_factors_date", "trade_date"),
        {"extend_existing": True},
    )

    def __repr__(self):
        """方法描述"""

    @property
    def primary_key(self):
        """复合主键"""
        return (self.ts_code, self.trade_date)


class StockFundamentalFactors(Base, TimestampMixin):
    """股票基本面因子表"""

    __tablename__ = "stock_fundamental_factors"

    ts_code = Column(String(20), ForeignKey("stock_basic.ts_code"), nullable=False, comment="TS代码")
    report_date = Column(Date, nullable=False, comment="报告期")

    # 盈利能力因子
    roe = Column(Float, comment="净资产收益率")
    roa = Column(Float, comment="总资产收益率")
    roic = Column(Float, comment="投入资本回报率")
    gross_margin = Column(Float, comment="毛利率")
    net_margin = Column(Float, comment="净利率")

    # 估值因子
    pe_ttm = Column(Float, comment="市盈率TTM")
    pb = Column(Float, comment="市净率")
    ps_ttm = Column(Float, comment="市销率TTM")
    pcf_ttm = Column(Float, comment="市现率TTM")
    ev_ebitda = Column(Float, comment="EV/EBITDA")
    peg = Column(Float, comment="PEG比率")

    # 成长能力因子
    revenue_yoy = Column(Float, comment="营收同比增长率")
    net_profit_yoy = Column(Float, comment="净利润同比增长率")

    # 财务质量因子
    debt_to_equity = Column(Float, comment="资产负债率")
    current_ratio = Column(Float, comment="流动比率")
    quick_ratio = Column(Float, comment="速动比率")

    # 建立与股票基础信息的关系
    stock_info = relationship("StockBasic", back_populates="fundamental_factors", foreign_keys=[ts_code])

    # 设置复合主键和索引
    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "report_date"),
        Index("idx_fundamental_factors_code_date", "ts_code", "report_date"),
        Index("idx_fundamental_factors_date", "report_date"),
        {"extend_existing": True},
    )

    def __repr__(self):
        """方法描述"""

    @property
    def primary_key(self):
        """复合主键"""
        return (self.ts_code, self.report_date)


class FactorLibrary(Base, TimestampMixin):
    """因子库表 - 通用因子存储"""

    __tablename__ = "factor_library"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="主键ID")
    ts_code = Column(String(20), nullable=False, comment="TS代码")
    trade_date = Column(Date, nullable=False, comment="交易日期")
    factor_name = Column(String(100), nullable=False, comment="因子名称")
    factor_value = Column(Float, comment="因子值")
    factor_category = Column(String(50), comment="因子分类")
    factor_description = Column(Text, comment="因子描述")

    # 设置索引
    __table_args__ = (
        Index("idx_factor_library_code_date_name", "ts_code", "trade_date", "factor_name"),
        Index("idx_factor_library_name_date", "factor_name", "trade_date"),
        Index("idx_factor_library_category", "factor_category"),
        {"extend_existing": True},
    )

    def __repr__(self):
        """方法描述"""

from datetime import date

from sqlalchemy import BigInteger, Column, Date, Float, ForeignKey, Index, PrimaryKeyConstraint, String
from sqlalchemy.orm import relationship

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票相关数据模型
定义股票基础信息和交易数据的ORM模型
"""

from .base import Base, TimestampMixin


class StockBasic(Base, TimestampMixin):
    """股票基础信息表"""

    __tablename__ = "stock_basic"
    __table_args__ = ({"extend_existing": True},)

    ts_code = Column(String(20), primary_key=True, comment="TS代码")
    symbol = Column(String(20), nullable=False, comment="股票代码")
    name = Column(String(50), nullable=False, comment="股票名称")
    area = Column(String(20), comment="地域")
    industry = Column(String(50), comment="所属行业")
    market = Column(String(20), comment="市场类型")
    list_date = Column(Date, comment="上市日期")
    list_status = Column(String(10), comment="上市状态")
    delist_date = Column(Date, comment="退市日期")

    # 建立与日线数据的关系
    daily_data = relationship("StockDaily", back_populates="stock_info", foreign_keys="StockDaily.ts_code")
    technical_factors = relationship(
        "StockTechnicalFactors", back_populates="stock_info", foreign_keys="StockTechnicalFactors.ts_code"
    )
    fundamental_factors = relationship(
        "StockFundamentalFactors", back_populates="stock_info", foreign_keys="StockFundamentalFactors.ts_code"
    )

    def __repr__(self):
        """方法描述"""


class StockDaily(Base, TimestampMixin):
    """股票日线数据表"""

    __tablename__ = "stock_daily"
    __table_args__ = ({"extend_existing": True},)

    ts_code = Column(String(20), ForeignKey("stock_basic.ts_code"), primary_key=True, nullable=False, comment="TS代码")
    trade_date = Column(Date, primary_key=True, nullable=False, comment="交易日期")
    open = Column(Float, comment="开盘价")
    high = Column(Float, comment="最高价")
    low = Column(Float, comment="最低价")
    close = Column(Float, comment="收盘价")
    pre_close = Column(Float, comment="昨收价")
    change = Column(Float, comment="涨跌额")
    pct_chg = Column(Float, comment="涨跌幅")
    vol = Column(BigInteger, comment="成交量(手)")
    amount = Column(Float, comment="成交额(千元)")

    # 建立与股票基础信息的关系（添加外键约束）
    stock_info = relationship("StockBasic", back_populates="daily_data", foreign_keys=[ts_code])

    # 设置复合主键和索引
    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date"),
        Index("idx_stock_daily_ts_code", "ts_code"),
        Index("idx_stock_daily_trade_date", "trade_date"),
        Index("idx_stock_daily_ts_code_trade_date", "ts_code", "trade_date"),
        {"extend_existing": True},
    )

    def __repr__(self):
        """方法描述"""

    @property
    def primary_key(self):
        """复合主键"""
        return (self.ts_code, self.trade_date)

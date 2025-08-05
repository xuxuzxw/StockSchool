from datetime import date

from sqlalchemy import Column, Date, Float, Index, Integer, PrimaryKeyConstraint, String

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财务数据相关模型
定义财务指标、财务报表等ORM模型
"""

from .base import Base, TimestampMixin


class FinancialIndicator(Base, TimestampMixin):
    """财务指标表"""

    __tablename__ = "financial_indicator"

    ts_code = Column(String(20), nullable=False, comment="TS代码")
    ann_date = Column(Date, comment="公告日期")
    end_date = Column(Date, nullable=False, comment="报告期")

    # 基本每股收益
    eps = Column(Float, comment="基本每股收益")
    dt_eps = Column(Float, comment="稀释每股收益")

    # 净资产收益率
    roe = Column(Float, comment="净资产收益率")
    roe_waa = Column(Float, comment="加权平均净资产收益率")
    roe_dt = Column(Float, comment="净资产收益率(扣除非经常损益)")

    # 总资产收益率
    roa = Column(Float, comment="总资产收益率")

    # 营业收入
    revenue = Column(Float, comment="营业收入")
    revenue_yoy = Column(Float, comment="营业收入同比增长率")
    revenue_qoq = Column(Float, comment="营业收入环比增长率")

    # 净利润
    netprofit = Column(Float, comment="净利润")
    netprofit_yoy = Column(Float, comment="净利润同比增长率")
    netprofit_qoq = Column(Float, comment="净利润环比增长率")

    # 毛利率和净利率
    grossprofit_margin = Column(Float, comment="毛利率")
    netprofit_margin = Column(Float, comment="净利率")

    # 资产负债率
    debt_to_assets = Column(Float, comment="资产负债率")
    assets_to_eqt = Column(Float, comment="权益乘数")

    # 流动比率和速动比率
    current_ratio = Column(Float, comment="流动比率")
    quick_ratio = Column(Float, comment="速动比率")

    # 现金流相关
    ocf_to_revenue = Column(Float, comment="经营现金流/营业收入")
    ocf_to_netprofit = Column(Float, comment="经营现金流/净利润")

    # 周转率
    assets_turn = Column(Float, comment="总资产周转率")
    ca_turn = Column(Float, comment="流动资产周转率")
    fa_turn = Column(Float, comment="固定资产周转率")
    inv_turn = Column(Float, comment="存货周转率")
    ar_turn = Column(Float, comment="应收账款周转率")

    # 市值相关
    total_mv = Column(Float, comment="总市值")
    circ_mv = Column(Float, comment="流通市值")

    # 设置复合主键和索引
    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "end_date"),
        Index("idx_financial_indicator_ts_code", "ts_code"),
        Index("idx_financial_indicator_end_date", "end_date"),
        Index("idx_financial_indicator_ts_code_end_date", "ts_code", "end_date"),
        {"extend_existing": True},
    )

    def __repr__(self):
        """方法描述"""

    @property
    def primary_key(self):
        """复合主键"""
        return (self.ts_code, self.end_date)


class FinancialReports(Base, TimestampMixin):
    """财务报表数据表"""

    __tablename__ = "financial_reports"

    ts_code = Column(String(20), nullable=False, comment="TS代码")
    ann_date = Column(Date, comment="公告日期")
    f_ann_date = Column(Date, comment="实际公告日期")
    end_date = Column(Date, nullable=False, comment="报告期")
    report_type = Column(String(10), comment="报告类型")
    comp_type = Column(String(10), comment="公司类型")

    # 资产负债表主要科目
    total_assets = Column(Float, comment="资产总计")
    total_hldr_eqy_exc_min_int = Column(Float, comment="股东权益合计(不含少数股东权益)")
    total_liab = Column(Float, comment="负债合计")

    # 利润表主要科目
    total_revenue = Column(Float, comment="营业总收入")
    revenue = Column(Float, comment="营业收入")
    oper_profit = Column(Float, comment="营业利润")
    total_profit = Column(Float, comment="利润总额")
    n_income = Column(Float, comment="净利润")
    n_income_attr_p = Column(Float, comment="归属于母公司所有者的净利润")

    # 现金流量表主要科目
    n_cashflow_act = Column(Float, comment="经营活动产生的现金流量净额")
    n_cashflow_inv_act = Column(Float, comment="投资活动产生的现金流量净额")
    n_cashflow_fin_act = Column(Float, comment="筹资活动产生的现金流量净额")

    # 设置复合主键和索引
    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "end_date", "report_type"),
        Index("idx_financial_reports_ts_code", "ts_code"),
        Index("idx_financial_reports_end_date", "end_date"),
        Index("idx_financial_reports_report_type", "report_type"),
        Index("idx_financial_reports_ts_code_end_date", "ts_code", "end_date"),
        {"extend_existing": True},
    )

    def __repr__(self):
        """方法描述"""

    @property
    def primary_key(self):
        """复合主键"""
        return (self.ts_code, self.end_date)

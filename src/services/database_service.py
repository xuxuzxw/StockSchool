from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import and_, asc, desc, func, or_
from sqlalchemy.orm import Session

from ..models import (
    FactorLibrary,
    FinancialIndicator,
    FinancialReports,
    StockBasic,
    StockDaily,
    StockFundamentalFactors,
    StockTechnicalFactors,
)
from ..models.base import get_db_engine
from ..utils.db import init_db_session


class DatabaseService:
    """数据库服务类 - 提供统一的数据访问接口"""

    def __init__(self):
        """方法描述"""
        self.engine = get_db_engine()
        self.SessionLocal = init_db_session(self.engine)

    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()

    # ==================== 股票基础数据相关 ====================

    def get_all_stocks(self, list_status: str = "L") -> List[str]:
        """获取所有股票代码"""
        with self.get_session() as session:
            query = session.query(StockBasic.ts_code)
            if list_status:
                query = query.filter(StockBasic.list_status == list_status)
            return [row[0] for row in query.all()]

    def get_stock_info(self, ts_code: str) -> Optional[StockBasic]:
        """获取股票基础信息"""
        with self.get_session() as session:
            return session.query(StockBasic).filter(StockBasic.ts_code == ts_code).first()

    def get_stock_data(
        self,
        ts_code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """获取股票日线数据"""
        with self.get_session() as session:
            query = session.query(StockDaily).filter(StockDaily.ts_code == ts_code)

            if start_date:
                query = query.filter(StockDaily.trade_date >= start_date)
            if end_date:
                query = query.filter(StockDaily.trade_date <= end_date)

            query = query.order_by(desc(StockDaily.trade_date))

            if limit:
                query = query.limit(limit)

            results = query.all()

            # 转换为DataFrame
            data = [
                {
                    "ts_code": row.ts_code,
                    "trade_date": row.trade_date,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "pre_close": row.pre_close,
                    "change": row.change,
                    "pct_chg": row.pct_chg,
                    "vol": row.vol,
                    "amount": row.amount,
                }
                for row in results
            ]

            return pd.DataFrame(data)

    # ==================== 财务数据相关 ====================

    def get_financial_data(
        self, ts_code: str, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """获取财务指标数据"""
        with self.get_session() as session:
            query = session.query(FinancialIndicator).filter(FinancialIndicator.ts_code == ts_code)

            if start_date:
                query = query.filter(FinancialIndicator.end_date >= start_date)
            if end_date:
                query = query.filter(FinancialIndicator.end_date <= end_date)

            query = query.order_by(desc(FinancialIndicator.end_date))
            results = query.all()

            # 转换为DataFrame
            data = [row.to_dict() for row in results]
            return pd.DataFrame(data)

    def get_financial_reports(
        self, ts_code: str, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """获取财务报表数据"""
        with self.get_session() as session:
            query = session.query(FinancialReports).filter(FinancialReports.ts_code == ts_code)

            if start_date:
                query = query.filter(FinancialReports.end_date >= start_date)
            if end_date:
                query = query.filter(FinancialReports.end_date <= end_date)

            query = query.order_by(desc(FinancialReports.end_date))
            results = query.all()

            # 转换为DataFrame
            data = [row.to_dict() for row in results]
            return pd.DataFrame(data)

    # ==================== 因子数据相关 ====================

    def save_technical_factors(self, ts_code: str, trade_date: date, factors: Dict[str, float]):
        """保存技术因子数据"""
        with self.get_session() as session:
            # 检查是否已存在
            existing = (
                session.query(StockTechnicalFactors)
                .filter(and_(StockTechnicalFactors.ts_code == ts_code, StockTechnicalFactors.trade_date == trade_date))
                .first()
            )

            if existing:
                # 更新现有记录
                for key, value in factors.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                # 创建新记录
                factor_record = StockTechnicalFactors(ts_code=ts_code, trade_date=trade_date, **factors)
                session.add(factor_record)

            session.commit()

    def save_fundamental_factors(self, ts_code: str, report_date: date, factors: Dict[str, float]):
        """保存基本面因子数据"""
        with self.get_session() as session:
            # 检查是否已存在
            existing = (
                session.query(StockFundamentalFactors)
                .filter(
                    and_(StockFundamentalFactors.ts_code == ts_code, StockFundamentalFactors.report_date == report_date)
                )
                .first()
            )

            if existing:
                # 更新现有记录
                for key, value in factors.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                # 创建新记录
                factor_record = StockFundamentalFactors(ts_code=ts_code, report_date=report_date, **factors)
                session.add(factor_record)

            session.commit()

    def save_factor_to_library(
        self,
        ts_code: str,
        trade_date: date,
        factor_name: str,
        factor_value: float,
        factor_category: str = None,
        factor_description: str = None,
    ):
        """保存因子到因子库"""
        with self.get_session() as session:
            # 检查是否已存在
            existing = (
                session.query(FactorLibrary)
                .filter(
                    and_(
                        FactorLibrary.ts_code == ts_code,
                        FactorLibrary.trade_date == trade_date,
                        FactorLibrary.factor_name == factor_name,
                    )
                )
                .first()
            )

            if existing:
                # 更新现有记录
                existing.factor_value = factor_value
                if factor_category:
                    existing.factor_category = factor_category
                if factor_description:
                    existing.factor_description = factor_description
                existing.updated_at = datetime.utcnow()
            else:
                # 创建新记录
                factor_record = FactorLibrary(
                    ts_code=ts_code,
                    trade_date=trade_date,
                    factor_name=factor_name,
                    factor_value=factor_value,
                    factor_category=factor_category,
                    factor_description=factor_description,
                )
                session.add(factor_record)

            session.commit()

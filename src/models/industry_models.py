from datetime import datetime

from sqlalchemy import Boolean, Column, Date, Integer, String
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin


class IndustryClassification(Base, TimestampMixin):
    """行业分类模型"""

    __tablename__ = "industry_classification"

    industry_code = Column(String(50), primary_key=True, comment="行业代码")
    industry_name = Column(String(255), nullable=False, comment="行业名称")
    industry_level = Column(Integer, nullable=False, comment="行业级别")
    parent_code = Column(String(50), comment="父行业代码")
    source = Column(String(50), nullable=False, comment="数据来源")
    is_active = Column(Boolean, default=True, comment="是否活跃")

    # 定义与 StockIndustryMapping 的一对多关系
    members = relationship("StockIndustryMapping", back_populates="industry")

    def __repr__(self):
        """方法描述"""


class StockIndustryMapping(Base, TimestampMixin):
    """股票行业归属映射模型"""

    __tablename__ = "stock_industry_mapping"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="主键ID")
    ts_code = Column(String(50), nullable=False, comment="股票代码")
    industry_code = Column(String(50), nullable=False, comment="行业代码")
    in_date = Column(Date, nullable=False, comment="归属日期")
    out_date = Column(Date, comment="移出日期")
    is_current = Column(Boolean, default=True, comment="是否当前归属")
    source = Column(String(50), nullable=False, comment="数据来源")

    # 定义与 IndustryClassification 的多对一关系
    industry = relationship("IndustryClassification", back_populates="members")

    def __repr__(self):
        """方法描述"""

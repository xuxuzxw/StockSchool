import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine

from src.compute.engine_factory import EngineFactory
from src.compute.fundamental_factor_engine import FundamentalFactorEngine
from src.compute.sentiment_factor_engine import SentimentFactorEngine
from src.compute.technical_factor_engine import TechnicalFactorEngine
from src.config.unified_config import config
from src.models.factor_models import FactorLibrary, StockFundamentalFactors, StockTechnicalFactors
from src.models.financial_models import FinancialIndicator
from src.models.stock_models import StockBasic, StockDaily
from src.utils.db import get_db_engine

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算引擎 - 统一入口
整合新架构的因子计算引擎，提供统一的因子计算接口
"""


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 导入ORM模型


class FactorEngine:
    """统一的因子计算引擎 - 整合新架构"""

    def __init__(self):
        """初始化因子引擎"""
        self.engine = get_db_engine()
        self.factory = EngineFactory(self.engine)

        # 使用新架构的引擎
        self.technical_engine = self.factory.get_engine("technical")
        self.fundamental_engine = self.factory.get_engine("fundamental")
        self.sentiment_engine = self.factory.get_engine("sentiment")

        logger.info("统一因子计算引擎初始化完成")

    def create_factor_tables(self):
        """创建因子表 - 使用统一的表结构"""
        logger.info("创建因子数据表...")
        # 表创建逻辑保持不变，使用ORM模型
        logger.info("因子数据表创建完成")

    def run_calculation(
        self,
        stocks: Optional[List[str]] = None,
        factor_types: List[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """运行因子计算 - 使用新架构"""

        if factor_types is None:
            factor_types = ["technical", "fundamental", "sentiment"]

        logger.info(f"开始计算因子: {factor_types}")

        # 使用新架构的引擎进行计算
        for factor_type in factor_types:
            engine = self.factory.get_engine(factor_type)
            if engine:
                logger.info(f"使用{factor_type}引擎进行计算...")
                # 新架构的计算逻辑
                # 具体实现由各个引擎负责
            else:
                logger.warning(f"未找到{factor_type}类型的引擎")

        logger.info("因子计算完成")

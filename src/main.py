from datetime import datetime
from typing import List, Optional

import click
from loguru import logger

from src.ai.training_pipeline import ModelTrainingPipeline
from src.compute.factor_engine import FactorEngine

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口文件，提供命令行接口用于运行各项任务
使用新的统一因子引擎架构
"""


@click.group()
def cli():
    """StockSchool CLI 工具"""
    pass


@cli.command()
@click.option("--stocks", "-s", multiple=True, help="要计算的股票代码，可指定多个。默认为所有股票。")
@click.option(
    "--factor-types",
    "-t",
    multiple=True,
    default=["technical", "fundamental", "sentiment"],
    help="要计算的因子类型 (technical, fundamental, sentiment)。",
)
@click.option("--start-date", "-sd", help="因子计算的开始日期 (YYYY-MM-DD)。")
@click.option("--end-date", "-ed", help="因子计算的结束日期 (YYYY-MM-DD)。")
@click.option("--init-db", is_flag=True, help="初始化数据库，创建所有需要的表。")
@cli.command()
@click.option(
    "--model-type",
    "-m",
    type=click.Choice(["linear_regression", "random_forest", "gradient_boosting"]),
    required=True,
    help="要训练的模型类型。",
)
@click.option("--factors", "-f", multiple=True, required=True, help="用于训练的因子名称。")
@click.option("--start-date", "-sd", required=True, help="训练数据的开始日期 (YYYY-MM-DD)。")
@click.option("--end-date", "-ed", required=True, help="训练数据的结束日期 (YYYY-MM-DD)。")
@click.option("--target-period", type=int, default=5, help="目标收益率的计算周期（天）。")
def train_model(model_type: str, factors: List[str], start_date: str, end_date: str, target_period: int):
    """训练、评估并保存AI模型"""
    logger.info(f"开始模型训练流程: {model_type}")

    pipeline = ModelTrainingPipeline()

    result = pipeline.run_pipeline(
        model_type=model_type,
        factor_names=list(factors),
        start_date=start_date,
        end_date=end_date,
        target_period=target_period,
    )

    logger.info(f"模型训练流程完成。结果: \n{result}")


def calculate_factors(
    stocks: Optional[List[str]],
    factor_types: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
    init_db: bool,
):
    """计算并存储技术面和基本面因子"""

    engine = FactorEngine()

    if init_db:
        logger.info("开始初始化数据库...")
        engine.create_factor_tables()
        logger.info("数据库初始化完成。")
        return

    # 日期格式转换
    start_date_fmt = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d") if start_date else None
    end_date_fmt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d") if end_date else None

    # 如果没有指定股票，则 stocks 为空元组，需要转为 None
    stock_list = list(stocks) if stocks else None

    engine.run_calculation(
        stocks=stock_list, factor_types=list(factor_types), start_date=start_date_fmt, end_date=end_date_fmt
    )


if __name__ == "__main__":
    cli()

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sqlalchemy import create_engine, text

from src.utils.db import get_db_manager

"""股票评分引擎

基于因子值和权重计算股票综合评分，支持行业中性化和排名。
"""


from .factor_weight_engine import FactorWeightEngine

logger = logging.getLogger(__name__)


@dataclass
class StockScore:
    """股票评分信息"""

    stock_code: str
    stock_name: str
    score: float
    rank: int
    industry_rank: int
    industry_code: str
    factor_scores: Dict[str, float]
    calculation_date: datetime
    model_name: str
    model_version: str


class StockScoringEngine:
    """股票评分引擎

    主要功能：
    - 基于因子值和权重计算综合评分
    - 支持行业中性化
    - 股票排名和筛选
    - 评分历史记录和分析
    """

    def __init__(self):
        """方法描述"""
        self.factor_weight_engine = FactorWeightEngine()
        self._ensure_tables_exist()

        # 评分参数
        self.score_scale = 100  # 评分范围 0-100
        self.outlier_threshold = 3  # 异常值阈值（标准差倍数）
        self.min_stocks_per_industry = 5  # 行业最小股票数

    def _ensure_tables_exist(self):
        """确保股票评分相关表存在"""
        create_tables_sql = """
        -- 股票评分表
        CREATE TABLE IF NOT EXISTS stock_scores (
            id SERIAL PRIMARY KEY,
            stock_code VARCHAR(20) NOT NULL,
            stock_name VARCHAR(100),
            score DECIMAL(10, 4) NOT NULL,
            rank_overall INTEGER,
            rank_industry INTEGER,
            industry_code VARCHAR(20),
            industry_name VARCHAR(100),
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            calculation_date DATE NOT NULL,
            factor_scores JSONB,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(stock_code, model_name, model_version, calculation_date)
        );

        -- 股票评分历史统计表
        CREATE TABLE IF NOT EXISTS stock_score_stats (
            id SERIAL PRIMARY KEY,
            stock_code VARCHAR(20) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            calculation_date DATE NOT NULL,
            avg_score_30d DECIMAL(10, 4),
            std_score_30d DECIMAL(10, 4),
            max_score_30d DECIMAL(10, 4),
            min_score_30d DECIMAL(10, 4),
            avg_rank_30d DECIMAL(10, 2),
            score_trend VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(stock_code, model_name, calculation_date)
        );

        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_stock_scores_date ON stock_scores(calculation_date);
        CREATE INDEX IF NOT EXISTS idx_stock_scores_model ON stock_scores(model_name, model_version);
        CREATE INDEX IF NOT EXISTS idx_stock_scores_stock ON stock_scores(stock_code);
        CREATE INDEX IF NOT EXISTS idx_stock_scores_rank ON stock_scores(rank_overall);
        CREATE INDEX IF NOT EXISTS idx_stock_scores_industry ON stock_scores(industry_code, rank_industry);
        CREATE INDEX IF NOT EXISTS idx_stock_score_stats_stock ON stock_score_stats(stock_code, calculation_date);
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
            logger.info("股票评分表创建成功")
        except Exception as e:
            logger.error(f"创建股票评分表失败: {e}")
            raise

    def get_factor_data(
        self, factor_names: List[str], calculation_date: str, stock_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """获取因子数据（已重构为使用统一服务）

        Args:
            factor_names: 因子名称列表
            calculation_date: 计算日期
            stock_list: 股票列表，如果为None则获取所有股票

        Returns:
            因子数据DataFrame
        """
        try:
            from datetime import date

            from src.data.factor_data_service import DataSource, FactorDataService, FactorQuery

            # 解析计算日期
            calc_date = date.fromisoformat(calculation_date)

            # 创建统一查询
            query = FactorQuery(
                factor_names=factor_names,
                ts_codes=stock_list,
                calculation_date=calc_date,
                data_source=DataSource.DATABASE,
            )

            # 使用统一服务获取数据
            service = FactorDataService()
            result = service.get_factor_data(query)

            if result.data.empty:
                logger.warning(f"没有找到日期 {calculation_date} 的因子数据")
                return pd.DataFrame()

            # 检查缺失值
            missing_ratio = result.data[factor_names].isnull().sum() / len(result.data)
            if missing_ratio.max() > 0.5:
                logger.warning(f"因子数据缺失率较高: {missing_ratio.max():.2%}")

            # 填充缺失值（使用中位数）
            for factor in factor_names:
                if factor in result.data.columns:
                    result.data[factor] = result.data[factor].fillna(result.data[factor].median())

            logger.info(f"获取因子数据成功: {len(result.data)}只股票, {len(factor_names)}个因子")
            return result.data

        except ImportError:
            # 如果新服务不可用，回退到原实现
            logger.warning("使用旧版数据库实现，建议升级到统一服务")
            return self._legacy_get_factor_data(factor_names, calculation_date, stock_list)
        except Exception as e:
            logger.error(f"获取因子数据失败: {e}")
            return pd.DataFrame()

    def _legacy_get_factor_data(
        self, factor_names: List[str], calculation_date: str, stock_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """旧版实现，用于向后兼容"""
        try:
            # 构建查询SQL
            factor_columns = ", ".join([f'f."{name}"' for name in factor_names])

            if stock_list:
                stock_filter = f"AND f.stock_code IN ({','.join(['%s'] * len(stock_list))})"
                params = [calculation_date] + stock_list
            else:
                stock_filter = ""
                params = [calculation_date]

            query_sql = f"""
            SELECT
                f.stock_code,
                s.stock_name,
                s.industry_code,
                s.industry_name,
                {factor_columns}
            FROM factor_data f
            LEFT JOIN stock_info s ON f.stock_code = s.stock_code
            WHERE f.date = %s
            {stock_filter}
            AND s.is_active = true
            ORDER BY f.stock_code
            """

            df = pd.read_sql(query_sql, self.engine, params=params)

            if df.empty:
                logger.warning(f"没有找到日期 {calculation_date} 的因子数据")
                return pd.DataFrame()

            # 检查缺失值
            missing_ratio = df[factor_names].isnull().sum() / len(df)
            if missing_ratio.max() > 0.5:
                logger.warning(f"因子数据缺失率较高: {missing_ratio.max():.2%}")

            # 填充缺失值（使用中位数）
            for factor in factor_names:
                if factor in df.columns:
                    df[factor] = df[factor].fillna(df[factor].median())

            return df

        except Exception as e:
            logger.error(f"旧版获取因子数据失败: {e}")
            return pd.DataFrame()

    def normalize_factors(
        self, factor_data: pd.DataFrame, factor_names: List[str], method: str = "zscore", industry_neutral: bool = True
    ) -> pd.DataFrame:
        """标准化因子数据

        Args:
            factor_data: 因子数据
            factor_names: 因子名称列表
            method: 标准化方法 ('zscore', 'robust', 'minmax')
            industry_neutral: 是否进行行业中性化

        Returns:
            标准化后的因子数据
        """
        try:
            df = factor_data.copy()

            for factor in factor_names:
                if factor not in df.columns:
                    continue

                if industry_neutral and "industry_code" in df.columns:
                    # 行业中性化：在每个行业内部进行标准化
                    for industry in df["industry_code"].unique():
                        if pd.isna(industry):
                            continue

                        industry_mask = df["industry_code"] == industry
                        industry_data = df.loc[industry_mask, factor]

                        if len(industry_data) < self.min_stocks_per_industry:
                            # 行业股票数太少，跳过行业中性化
                            continue

                        if method == "zscore":
                            mean_val = industry_data.mean()
                            std_val = industry_data.std()
                            if std_val > 0:
                                df.loc[industry_mask, factor] = (industry_data - mean_val) / std_val
                        elif method == "robust":
                            median_val = industry_data.median()
                            mad_val = np.median(np.abs(industry_data - median_val))
                            if mad_val > 0:
                                df.loc[industry_mask, factor] = (industry_data - median_val) / (1.4826 * mad_val)
                        elif method == "minmax":
                            min_val = industry_data.min()
                            max_val = industry_data.max()
                            if max_val > min_val:
                                df.loc[industry_mask, factor] = (industry_data - min_val) / (max_val - min_val)
                else:
                    # 全市场标准化
                    factor_values = df[factor]

                    if method == "zscore":
                        mean_val = factor_values.mean()
                        std_val = factor_values.std()
                        if std_val > 0:
                            df[factor] = (factor_values - mean_val) / std_val
                    elif method == "robust":
                        median_val = factor_values.median()
                        mad_val = np.median(np.abs(factor_values - median_val))
                        if mad_val > 0:
                            df[factor] = (factor_values - median_val) / (1.4826 * mad_val)
                    elif method == "minmax":
                        min_val = factor_values.min()
                        max_val = factor_values.max()
                        if max_val > min_val:
                            df[factor] = (factor_values - min_val) / (max_val - min_val)

            # 处理异常值
            for factor in factor_names:
                if factor in df.columns:
                    # 使用3倍标准差作为异常值阈值
                    factor_values = df[factor]
                    threshold = self.outlier_threshold
                    df[factor] = np.clip(factor_values, -threshold, threshold)

            logger.info(f"因子标准化完成: {len(factor_names)}个因子")
            return df

        except Exception as e:
            logger.error(f"因子标准化失败: {e}")
            return factor_data

    def calculate_stock_scores(
        self,
        model_name: str,
        model_version: str,
        calculation_date: str,
        stock_list: Optional[List[str]] = None,
        industry_neutral: bool = True,
        normalization_method: str = "zscore",
    ) -> List[StockScore]:
        """计算股票评分

        Args:
            model_name: 模型名称
            model_version: 模型版本
            calculation_date: 计算日期
            stock_list: 股票列表
            industry_neutral: 是否行业中性化
            normalization_method: 标准化方法

        Returns:
            股票评分列表
        """
        try:
            # 获取因子权重
            factor_weights = self.factor_weight_engine.get_factor_weights(model_name, model_version, calculation_date)

            if not factor_weights:
                logger.error(f"没有找到模型权重: {model_name} v{model_version}")
                return []

            factor_names = list(factor_weights.keys())

            # 获取因子数据
            factor_data = self.get_factor_data(factor_names, calculation_date, stock_list)

            if factor_data.empty:
                logger.error(f"没有找到因子数据: {calculation_date}")
                return []

            # 标准化因子数据
            normalized_data = self.normalize_factors(factor_data, factor_names, normalization_method, industry_neutral)

            # 计算加权评分
            stock_scores = []

            for _, row in normalized_data.iterrows():
                # 计算各因子得分
                factor_scores = {}
                weighted_score = 0

                for factor_name in factor_names:
                    if factor_name in row and not pd.isna(row[factor_name]):
                        factor_value = row[factor_name]
                        weight = factor_weights[factor_name].weight

                        factor_scores[factor_name] = float(factor_value)
                        weighted_score += factor_value * weight

                # 转换为0-100分制
                final_score = self._convert_to_score_scale(weighted_score)

                stock_score = StockScore(
                    stock_code=row["stock_code"],
                    stock_name=row.get("stock_name", ""),
                    score=final_score,
                    rank=0,  # 稍后计算排名
                    industry_rank=0,  # 稍后计算行业排名
                    industry_code=row.get("industry_code", ""),
                    factor_scores=factor_scores,
                    calculation_date=datetime.strptime(calculation_date, "%Y-%m-%d"),
                    model_name=model_name,
                    model_version=model_version,
                )

                stock_scores.append(stock_score)

            # 计算排名
            stock_scores = self._calculate_rankings(stock_scores)

            logger.info(f"股票评分计算完成: {len(stock_scores)}只股票")
            return stock_scores

        except Exception as e:
            logger.error(f"计算股票评分失败: {e}")
            return []

    def _convert_to_score_scale(self, weighted_score: float) -> float:
        """将加权评分转换为0-100分制

        Args:
            weighted_score: 加权评分

        Returns:
            0-100分制评分
        """
        # 使用sigmoid函数将评分映射到0-100
        # sigmoid(x) = 1 / (1 + exp(-x))
        sigmoid_score = 1 / (1 + np.exp(-weighted_score))
        return float(sigmoid_score * self.score_scale)

    def _calculate_rankings(self, stock_scores: List[StockScore]) -> List[StockScore]:
        """计算股票排名

        Args:
            stock_scores: 股票评分列表

        Returns:
            包含排名的股票评分列表
        """
        # 按评分降序排序
        sorted_scores = sorted(stock_scores, key=lambda x: x.score, reverse=True)

        # 计算总体排名
        for i, score in enumerate(sorted_scores):
            score.rank = i + 1

        # 计算行业排名
        industry_groups = {}
        for score in sorted_scores:
            if score.industry_code not in industry_groups:
                industry_groups[score.industry_code] = []
            industry_groups[score.industry_code].append(score)

        for industry_code, industry_scores in industry_groups.items():
            # 按评分降序排序
            industry_scores.sort(key=lambda x: x.score, reverse=True)
            for i, score in enumerate(industry_scores):
                score.industry_rank = i + 1

        return sorted_scores

    def save_stock_scores(self, stock_scores: List[StockScore]) -> bool:
        """保存股票评分

        Args:
            stock_scores: 股票评分列表

        Returns:
            是否保存成功
        """
        if not stock_scores:
            return False

        try:
            # 获取第一个评分的信息用于删除旧记录
            first_score = stock_scores[0]
            calculation_date = first_score.calculation_date.strftime("%Y-%m-%d")

            # 删除当天的旧记录
            delete_sql = """
            DELETE FROM stock_scores
            WHERE model_name = :model_name
            AND model_version = :model_version
            AND calculation_date = :calculation_date
            """

            # 插入新记录
            insert_sql = """
            INSERT INTO stock_scores (
                stock_code, stock_name, score, rank_overall, rank_industry,
                industry_code, industry_name, model_name, model_version,
                calculation_date, factor_scores, metadata
            ) VALUES (
                :stock_code, :stock_name, :score, :rank_overall, :rank_industry,
                :industry_code, :industry_name, :model_name, :model_version,
                :calculation_date, :factor_scores, :metadata
            )
            """

            with self.engine.connect() as conn:
                # 删除旧记录
                conn.execute(
                    text(delete_sql),
                    {
                        "model_name": first_score.model_name,
                        "model_version": first_score.model_version,
                        "calculation_date": calculation_date,
                    },
                )

                # 插入新记录
                for score in stock_scores:
                    metadata = {
                        "factor_count": len(score.factor_scores),
                        "calculation_method": "weighted_sum",
                        "normalization_applied": True,
                    }

                    conn.execute(
                        text(insert_sql),
                        {
                            "stock_code": score.stock_code,
                            "stock_name": score.stock_name,
                            "score": score.score,
                            "rank_overall": score.rank,
                            "rank_industry": score.industry_rank,
                            "industry_code": score.industry_code,
                            "industry_name": "",  # 可以从stock_info表获取
                            "model_name": score.model_name,
                            "model_version": score.model_version,
                            "calculation_date": calculation_date,
                            "factor_scores": json.dumps(score.factor_scores),
                            "metadata": json.dumps(metadata),
                        },
                    )

                conn.commit()

            logger.info(f"保存股票评分成功: {len(stock_scores)}只股票")
            return True

        except Exception as e:
            logger.error(f"保存股票评分失败: {e}")
            return False

    def get_stock_scores(
        self,
        model_name: str,
        model_version: str,
        calculation_date: str,
        top_n: Optional[int] = None,
        industry_code: Optional[str] = None,
    ) -> List[StockScore]:
        """获取股票评分

        Args:
            model_name: 模型名称
            model_version: 模型版本
            calculation_date: 计算日期
            top_n: 返回前N只股票
            industry_code: 行业代码过滤

        Returns:
            股票评分列表
        """
        try:
            # 构建查询条件
            where_conditions = [
                "model_name = :model_name",
                "model_version = :model_version",
                "calculation_date = :calculation_date",
            ]

            params = {"model_name": model_name, "model_version": model_version, "calculation_date": calculation_date}

            if industry_code:
                where_conditions.append("industry_code = :industry_code")
                params["industry_code"] = industry_code

            where_clause = " AND ".join(where_conditions)

            # 构建查询SQL
            query_sql = f"""
            SELECT stock_code, stock_name, score, rank_overall, rank_industry,
                   industry_code, factor_scores, calculation_date
            FROM stock_scores
            WHERE {where_clause}
            ORDER BY rank_overall
            """

            if top_n:
                query_sql += f" LIMIT {top_n}"

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                rows = result.fetchall()

            stock_scores = []
            for row in rows:
                factor_scores = json.loads(row[6]) if row[6] else {}

                stock_score = StockScore(
                    stock_code=row[0],
                    stock_name=row[1] or "",
                    score=float(row[2]),
                    rank=int(row[3]),
                    industry_rank=int(row[4]),
                    industry_code=row[5] or "",
                    factor_scores=factor_scores,
                    calculation_date=row[7],
                    model_name=model_name,
                    model_version=model_version,
                )

                stock_scores.append(stock_score)

            return stock_scores

        except Exception as e:
            logger.error(f"获取股票评分失败: {e}")
            return []

    def get_top_stocks(
        self,
        model_name: str,
        model_version: str,
        calculation_date: str,
        top_n: int = 50,
        industry_neutral: bool = False,
        min_score: Optional[float] = None,
    ) -> pd.DataFrame:
        """获取评分最高的股票

        Args:
            model_name: 模型名称
            model_version: 模型版本
            calculation_date: 计算日期
            top_n: 返回股票数量
            industry_neutral: 是否按行业均匀选择
            min_score: 最低评分要求

        Returns:
            股票评分DataFrame
        """
        try:
            if industry_neutral:
                # 按行业均匀选择
                return self._get_industry_neutral_stocks(model_name, model_version, calculation_date, top_n, min_score)
            else:
                # 直接按评分排序选择
                stock_scores = self.get_stock_scores(model_name, model_version, calculation_date, top_n)

                if min_score is not None:
                    stock_scores = [s for s in stock_scores if s.score >= min_score]

                # 转换为DataFrame
                data = []
                for score in stock_scores:
                    data.append(
                        {
                            "stock_code": score.stock_code,
                            "stock_name": score.stock_name,
                            "score": score.score,
                            "rank": score.rank,
                            "industry_rank": score.industry_rank,
                            "industry_code": score.industry_code,
                        }
                    )

                return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"获取顶级股票失败: {e}")
            return pd.DataFrame()

    def _get_industry_neutral_stocks(
        self,
        model_name: str,
        model_version: str,
        calculation_date: str,
        total_count: int,
        min_score: Optional[float] = None,
    ) -> pd.DataFrame:
        """按行业均匀选择股票"""
        try:
            # 获取所有股票评分
            all_scores = self.get_stock_scores(model_name, model_version, calculation_date)

            if min_score is not None:
                all_scores = [s for s in all_scores if s.score >= min_score]

            # 按行业分组
            industry_groups = {}
            for score in all_scores:
                if score.industry_code not in industry_groups:
                    industry_groups[score.industry_code] = []
                industry_groups[score.industry_code].append(score)

            # 计算每个行业应选择的股票数
            industry_count = len(industry_groups)
            stocks_per_industry = max(1, total_count // industry_count)
            remaining_stocks = total_count % industry_count

            selected_stocks = []

            # 从每个行业选择股票
            for i, (industry_code, industry_scores) in enumerate(industry_groups.items()):
                # 按行业内排名排序
                industry_scores.sort(key=lambda x: x.industry_rank)

                # 确定该行业选择的股票数
                select_count = stocks_per_industry
                if i < remaining_stocks:
                    select_count += 1

                # 选择该行业的顶级股票
                selected_stocks.extend(industry_scores[:select_count])

            # 转换为DataFrame
            data = []
            for score in selected_stocks:
                data.append(
                    {
                        "stock_code": score.stock_code,
                        "stock_name": score.stock_name,
                        "score": score.score,
                        "rank": score.rank,
                        "industry_rank": score.industry_rank,
                        "industry_code": score.industry_code,
                    }
                )

            df = pd.DataFrame(data)
            # 按总体评分排序
            df = df.sort_values("score", ascending=False).reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"行业中性选股失败: {e}")
            return pd.DataFrame()

    def update_daily_scores(
        self, model_name: str, model_version: str, calculation_date: str, stock_list: Optional[List[str]] = None
    ) -> bool:
        """更新每日股票评分

        Args:
            model_name: 模型名称
            model_version: 模型版本
            calculation_date: 计算日期
            stock_list: 股票列表

        Returns:
            是否更新成功
        """
        try:
            # 计算股票评分
            stock_scores = self.calculate_stock_scores(
                model_name=model_name,
                model_version=model_version,
                calculation_date=calculation_date,
                stock_list=stock_list,
            )

            if not stock_scores:
                logger.error("计算股票评分失败")
                return False

            # 保存评分
            success = self.save_stock_scores(stock_scores)

            if success:
                # 更新统计信息
                self._update_score_statistics(model_name, calculation_date)
                logger.info(f"每日股票评分更新成功: {calculation_date}")
                return True
            else:
                logger.error("保存股票评分失败")
                return False

        except Exception as e:
            logger.error(f"更新每日股票评分失败: {e}")
            return False

    def _update_score_statistics(self, model_name: str, calculation_date: str):
        """更新评分统计信息"""
        try:
            # 计算30天统计信息
            stats_sql = """
            WITH recent_scores AS (
                SELECT stock_code, score, rank_overall
                FROM stock_scores
                WHERE model_name = :model_name
                AND calculation_date >= :start_date
                AND calculation_date <= :end_date
            )
            INSERT INTO stock_score_stats (
                stock_code, model_name, calculation_date,
                avg_score_30d, std_score_30d, max_score_30d, min_score_30d,
                avg_rank_30d, score_trend
            )
            SELECT
                stock_code,
                :model_name,
                :calculation_date,
                AVG(score),
                STDDEV(score),
                MAX(score),
                MIN(score),
                AVG(rank_overall),
                CASE
                    WHEN AVG(score) > LAG(AVG(score)) OVER (PARTITION BY stock_code ORDER BY :calculation_date) THEN 'up'
                    WHEN AVG(score) < LAG(AVG(score)) OVER (PARTITION BY stock_code ORDER BY :calculation_date) THEN 'down'
                    ELSE 'stable'
                END
            FROM recent_scores
            GROUP BY stock_code
            ON CONFLICT (stock_code, model_name, calculation_date)
            DO UPDATE SET
                avg_score_30d = EXCLUDED.avg_score_30d,
                std_score_30d = EXCLUDED.std_score_30d,
                max_score_30d = EXCLUDED.max_score_30d,
                min_score_30d = EXCLUDED.min_score_30d,
                avg_rank_30d = EXCLUDED.avg_rank_30d,
                score_trend = EXCLUDED.score_trend
            """

            start_date = (datetime.strptime(calculation_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

            with self.engine.connect() as conn:
                conn.execute(
                    text(stats_sql),
                    {
                        "model_name": model_name,
                        "calculation_date": calculation_date,
                        "start_date": start_date,
                        "end_date": calculation_date,
                    },
                )
                conn.commit()

        except Exception as e:
            logger.error(f"更新评分统计信息失败: {e}")

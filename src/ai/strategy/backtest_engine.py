import base64
import io
import json
import logging
import pickle
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sqlalchemy import create_engine, text

from src.utils.db import get_db_manager

"""回测引擎

提供AI策略的历史回测功能，评估策略表现。
"""

warnings.filterwarnings("ignore")


from .model_manager import AIModelManager
from .stock_scoring_engine import StockScoringEngine

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class BacktestConfig:
    """回测配置"""

    strategy_name: str
    model_name: str
    model_version: Optional[str] = None
    start_date: str = None
    end_date: str = None
    initial_capital: float = 1000000.0  # 初始资金100万
    position_size: float = 0.1  # 单只股票仓位10%
    max_positions: int = 10  # 最大持仓数量
    rebalance_frequency: str = "weekly"  # 调仓频率: daily, weekly, monthly
    selection_method: str = "top_score"  # 选股方法: top_score, score_threshold
    selection_threshold: float = 80.0  # 评分阈值
    transaction_cost: float = 0.003  # 交易成本0.3%
    benchmark: str = "000300.SH"  # 基准指数
    industry_neutral: bool = False  # 是否行业中性
    max_weight_per_industry: float = 0.3  # 单行业最大权重
    stop_loss: Optional[float] = None  # 止损比例
    take_profit: Optional[float] = None  # 止盈比例
    min_holding_days: int = 1  # 最小持仓天数


@dataclass
class Position:
    """持仓信息"""

    stock_code: str
    stock_name: str
    industry: str
    entry_date: str
    entry_price: float
    shares: int
    entry_value: float
    current_price: float
    current_value: float
    unrealized_pnl: float
    unrealized_return: float
    holding_days: int
    score: float
    weight: float


@dataclass
class Trade:
    """交易记录"""

    trade_id: str
    stock_code: str
    stock_name: str
    trade_type: str  # 'buy', 'sell'
    trade_date: str
    price: float
    shares: int
    amount: float
    commission: float
    reason: str  # 交易原因
    score: Optional[float] = None


@dataclass
class BacktestResult:
    """回测结果"""

    strategy_name: str
    model_name: str
    model_version: str
    config: BacktestConfig
    start_date: str
    end_date: str
    total_days: int
    trading_days: int

    # 收益指标
    total_return: float
    annualized_return: float
    benchmark_return: float
    excess_return: float

    # 风险指标
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    information_ratio: float
    calmar_ratio: float

    # 交易指标
    total_trades: int
    win_rate: float
    avg_holding_days: float
    turnover_rate: float

    # 详细数据
    daily_returns: pd.DataFrame
    positions_history: List[List[Position]]
    trades_history: List[Trade]
    portfolio_value_history: pd.DataFrame

    # 可视化
    performance_chart: Optional[str] = None
    drawdown_chart: Optional[str] = None
    monthly_returns_heatmap: Optional[str] = None

    created_at: Optional[datetime] = None


class BacktestEngine:
    """回测引擎

    主要功能：
    - 历史数据回测
    - 多种选股策略
    - 风险控制
    - 业绩评估
    - 交易成本计算
    - 基准比较
    - 可视化分析
    """

    def __init__(self):
        """方法描述"""
        self.model_manager = AIModelManager()
        self.scoring_engine = StockScoringEngine()

        self._ensure_tables_exist()

    def _ensure_tables_exist(self):
        """确保回测相关表存在"""
        create_tables_sql = """
        -- 回测结果表
        CREATE TABLE IF NOT EXISTS backtest_results (
            id SERIAL PRIMARY KEY,
            strategy_name VARCHAR(100) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            config JSONB NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            total_days INTEGER,
            trading_days INTEGER,

            -- 收益指标
            total_return DECIMAL(10, 6),
            annualized_return DECIMAL(10, 6),
            benchmark_return DECIMAL(10, 6),
            excess_return DECIMAL(10, 6),

            -- 风险指标
            volatility DECIMAL(10, 6),
            max_drawdown DECIMAL(10, 6),
            sharpe_ratio DECIMAL(10, 6),
            information_ratio DECIMAL(10, 6),
            calmar_ratio DECIMAL(10, 6),

            -- 交易指标
            total_trades INTEGER,
            win_rate DECIMAL(10, 6),
            avg_holding_days DECIMAL(10, 2),
            turnover_rate DECIMAL(10, 6),

            -- 详细数据
            daily_returns JSONB,
            trades_history JSONB,
            portfolio_value_history JSONB,

            -- 可视化
            performance_chart TEXT,
            drawdown_chart TEXT,
            monthly_returns_heatmap TEXT,

            status VARCHAR(20) DEFAULT 'completed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 回测任务表
        CREATE TABLE IF NOT EXISTS backtest_tasks (
            id SERIAL PRIMARY KEY,
            task_id VARCHAR(100) UNIQUE NOT NULL,
            strategy_name VARCHAR(100) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20),
            config JSONB NOT NULL,
            status VARCHAR(20) DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            current_date DATE,
            error_message TEXT,
            result_id INTEGER REFERENCES backtest_results(id),
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 回测持仓历史表
        CREATE TABLE IF NOT EXISTS backtest_positions (
            id SERIAL PRIMARY KEY,
            backtest_id INTEGER REFERENCES backtest_results(id),
            date DATE NOT NULL,
            stock_code VARCHAR(20) NOT NULL,
            stock_name VARCHAR(100),
            industry VARCHAR(50),
            shares INTEGER,
            price DECIMAL(10, 4),
            value DECIMAL(15, 2),
            weight DECIMAL(8, 6),
            score DECIMAL(8, 4),
            holding_days INTEGER,
            unrealized_pnl DECIMAL(15, 2),
            unrealized_return DECIMAL(10, 6)
        );

        -- 回测交易记录表
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id SERIAL PRIMARY KEY,
            backtest_id INTEGER REFERENCES backtest_results(id),
            trade_id VARCHAR(100) NOT NULL,
            stock_code VARCHAR(20) NOT NULL,
            stock_name VARCHAR(100),
            trade_type VARCHAR(10) NOT NULL,
            trade_date DATE NOT NULL,
            price DECIMAL(10, 4),
            shares INTEGER,
            amount DECIMAL(15, 2),
            commission DECIMAL(10, 2),
            reason VARCHAR(200),
            score DECIMAL(8, 4)
        );

        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy ON backtest_results(strategy_name, created_at);
        CREATE INDEX IF NOT EXISTS idx_backtest_results_model ON backtest_results(model_name, model_version);
        CREATE INDEX IF NOT EXISTS idx_backtest_tasks_status ON backtest_tasks(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_backtest_positions_date ON backtest_positions(backtest_id, date);
        CREATE INDEX IF NOT EXISTS idx_backtest_trades_date ON backtest_trades(backtest_id, trade_date);
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
            logger.info("回测表创建成功")
        except Exception as e:
            logger.error(f"创建回测表失败: {e}")
            raise

    def run_backtest(self, config: BacktestConfig) -> Optional[BacktestResult]:
        """运行回测

        Args:
            config: 回测配置

        Returns:
            回测结果
        """
        try:
            logger.info(f"开始回测: {config.strategy_name}")

            # 验证配置
            if not self._validate_config(config):
                return None

            # 获取回测日期范围
            start_date, end_date = self._get_backtest_dates(config)
            if not start_date or not end_date:
                return None

            # 获取交易日历
            trading_dates = self._get_trading_dates(start_date, end_date)
            if not trading_dates:
                return None

            # 获取基准数据
            benchmark_data = self._get_benchmark_data(config.benchmark, start_date, end_date)

            # 初始化回测状态
            portfolio = self._initialize_portfolio(config)
            positions = {}  # 当前持仓 {stock_code: Position}
            trades = []  # 交易记录
            daily_portfolio_values = []  # 每日组合价值
            positions_history = []  # 持仓历史

            # 逐日回测
            for i, current_date in enumerate(trading_dates):
                logger.info(f"回测进度: {current_date} ({i+1}/{len(trading_dates)})")

                # 更新持仓价格和价值
                self._update_positions_value(positions, current_date)

                # 检查调仓条件
                if self._should_rebalance(config, current_date, i):
                    # 获取当日股票评分
                    stock_scores = self._get_stock_scores(config, current_date)

                    if stock_scores is not None and not stock_scores.empty:
                        # 生成交易信号
                        buy_signals, sell_signals = self._generate_signals(
                            config, positions, stock_scores, current_date
                        )

                        # 执行卖出交易
                        for stock_code in sell_signals:
                            if stock_code in positions:
                                trade = self._execute_sell(
                                    config, positions[stock_code], current_date, sell_signals[stock_code]
                                )
                                if trade:
                                    trades.append(trade)
                                    portfolio["cash"] += trade.amount - trade.commission
                                    del positions[stock_code]

                        # 执行买入交易
                        for stock_code, signal_data in buy_signals.items():
                            if len(positions) < config.max_positions:
                                trade = self._execute_buy(config, portfolio, signal_data, current_date)
                                if trade:
                                    trades.append(trade)
                                    portfolio["cash"] -= trade.amount + trade.commission

                                    # 创建新持仓
                                    positions[stock_code] = Position(
                                        stock_code=stock_code,
                                        stock_name=signal_data["stock_name"],
                                        industry=signal_data["industry"],
                                        entry_date=current_date,
                                        entry_price=trade.price,
                                        shares=trade.shares,
                                        entry_value=trade.amount,
                                        current_price=trade.price,
                                        current_value=trade.amount,
                                        unrealized_pnl=0.0,
                                        unrealized_return=0.0,
                                        holding_days=0,
                                        score=signal_data["score"],
                                        weight=trade.amount / portfolio["total_value"],
                                    )

                # 更新持仓天数
                for position in positions.values():
                    position.holding_days += 1

                # 计算当日组合价值
                total_position_value = sum(pos.current_value for pos in positions.values())
                portfolio["total_value"] = portfolio["cash"] + total_position_value

                # 记录每日数据
                daily_portfolio_values.append(
                    {
                        "date": current_date,
                        "total_value": portfolio["total_value"],
                        "cash": portfolio["cash"],
                        "position_value": total_position_value,
                        "position_count": len(positions),
                    }
                )

                # 记录持仓历史
                positions_history.append(list(positions.values()).copy())

            # 计算回测结果
            result = self._calculate_backtest_result(
                config,
                start_date,
                end_date,
                trading_dates,
                daily_portfolio_values,
                positions_history,
                trades,
                benchmark_data,
            )

            # 保存回测结果
            if result:
                self._save_backtest_result(result)
                logger.info(f"回测完成: {config.strategy_name}")

            return result

        except Exception as e:
            logger.error(f"回测失败: {e}")
            return None

    def _validate_config(self, config: BacktestConfig) -> bool:
        """验证回测配置"""
        try:
            # 检查模型是否存在
            model_data = self.model_manager.load_model_for_prediction(config.model_name, config.model_version)
            if not model_data:
                logger.error(f"模型不存在: {config.model_name} v{config.model_version}")
                return False

            # 检查日期格式
            if config.start_date:
                datetime.strptime(config.start_date, "%Y-%m-%d")
            if config.end_date:
                datetime.strptime(config.end_date, "%Y-%m-%d")

            # 检查参数范围
            if config.initial_capital <= 0:
                logger.error("初始资金必须大于0")
                return False

            if not 0 < config.position_size <= 1:
                logger.error("单只股票仓位必须在0-1之间")
                return False

            if config.max_positions <= 0:
                logger.error("最大持仓数量必须大于0")
                return False

            return True

        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False

    def _get_backtest_dates(self, config: BacktestConfig) -> Tuple[Optional[str], Optional[str]]:
        """获取回测日期范围"""
        try:
            if config.start_date and config.end_date:
                return config.start_date, config.end_date

            # 默认回测最近1年
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

            return start_date, end_date

        except Exception as e:
            logger.error(f"获取回测日期失败: {e}")
            return None, None

    def _get_trading_dates(self, start_date: str, end_date: str) -> Optional[List[str]]:
        """获取交易日历"""
        try:
            query_sql = """
            SELECT DISTINCT date
            FROM stock_daily_data
            WHERE date BETWEEN :start_date AND :end_date
            ORDER BY date
            """

            df = pd.read_sql(query_sql, self.engine, params={"start_date": start_date, "end_date": end_date})

            if df.empty:
                logger.error("没有找到交易日数据")
                return None

            return df["date"].dt.strftime("%Y-%m-%d").tolist()

        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return None

    def _get_benchmark_data(self, benchmark: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取基准数据"""
        try:
            query_sql = """
            SELECT date, close_price
            FROM stock_daily_data
            WHERE stock_code = :benchmark
            AND date BETWEEN :start_date AND :end_date
            ORDER BY date
            """

            df = pd.read_sql(
                query_sql, self.engine, params={"benchmark": benchmark, "start_date": start_date, "end_date": end_date}
            )

            if df.empty:
                logger.warning(f"没有找到基准数据: {benchmark}")
                return None

            # 计算基准收益率
            df["return"] = df["close_price"].pct_change()
            return df

        except Exception as e:
            logger.error(f"获取基准数据失败: {e}")
            return None

    def _initialize_portfolio(self, config: BacktestConfig) -> Dict[str, float]:
        """初始化组合"""
        return {"cash": config.initial_capital, "total_value": config.initial_capital}

    def _update_positions_value(self, positions: Dict[str, Position], current_date: str):
        """更新持仓价值"""
        try:
            if not positions:
                return

            stock_codes = list(positions.keys())

            # 获取当日价格
            query_sql = """
            SELECT stock_code, close_price
            FROM stock_daily_data
            WHERE stock_code = ANY(:stock_codes)
            AND date = :current_date
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {"stock_codes": stock_codes, "current_date": current_date})
                price_data = {row[0]: float(row[1]) for row in result.fetchall()}

            # 更新持仓价值
            for stock_code, position in positions.items():
                if stock_code in price_data:
                    position.current_price = price_data[stock_code]
                    position.current_value = position.shares * position.current_price
                    position.unrealized_pnl = position.current_value - position.entry_value
                    position.unrealized_return = position.unrealized_pnl / position.entry_value

        except Exception as e:
            logger.error(f"更新持仓价值失败: {e}")

    def _should_rebalance(self, config: BacktestConfig, current_date: str, day_index: int) -> bool:
        """判断是否需要调仓"""
        try:
            if config.rebalance_frequency == "daily":
                return True
            elif config.rebalance_frequency == "weekly":
                # 每周一调仓
                date_obj = datetime.strptime(current_date, "%Y-%m-%d")
                return date_obj.weekday() == 0
            elif config.rebalance_frequency == "monthly":
                # 每月第一个交易日调仓
                if day_index == 0:
                    return True
                prev_date = datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=1)
                return prev_date.month != datetime.strptime(current_date, "%Y-%m-%d").month

            return False

        except Exception as e:
            logger.error(f"判断调仓条件失败: {e}")
            return False

    def _get_stock_scores(self, config: BacktestConfig, current_date: str) -> Optional[pd.DataFrame]:
        """获取股票评分"""
        try:
            # 尝试从数据库获取历史评分
            query_sql = """
            SELECT stock_code, stock_name, industry, score, rank, industry_rank
            FROM stock_scores
            WHERE date = :current_date
            AND score >= :min_score
            ORDER BY score DESC
            """

            df = pd.read_sql(
                query_sql,
                self.engine,
                params={
                    "current_date": current_date,
                    "min_score": config.selection_threshold if config.selection_method == "score_threshold" else 0,
                },
            )

            if df.empty:
                # 如果没有历史评分，尝试实时计算
                logger.warning(f"没有找到历史评分数据: {current_date}，尝试实时计算")
                return self._calculate_realtime_scores(config, current_date)

            return df

        except Exception as e:
            logger.error(f"获取股票评分失败: {e}")
            return None

    def _calculate_realtime_scores(self, config: BacktestConfig, current_date: str) -> Optional[pd.DataFrame]:
        """实时计算股票评分"""
        try:
            # 获取当日因子数据
            query_sql = """
            SELECT fd.*, si.stock_name, si.industry
            FROM factor_data fd
            JOIN stock_info si ON fd.stock_code = si.stock_code
            WHERE fd.date = :current_date
            AND si.is_active = true
            """

            df = pd.read_sql(query_sql, self.engine, params={"current_date": current_date})

            if df.empty:
                logger.warning(f"没有找到因子数据: {current_date}")
                return None

            # 使用评分引擎计算评分
            scores = self.scoring_engine.calculate_stock_scores(df, config.model_name, config.model_version)

            if scores is None or scores.empty:
                return None

            # 添加股票信息
            scores = scores.merge(
                df[["stock_code", "stock_name", "industry"]].drop_duplicates(), on="stock_code", how="left"
            )

            return scores

        except Exception as e:
            logger.error(f"实时计算评分失败: {e}")
            return None

    def _generate_signals(
        self, config: BacktestConfig, positions: Dict[str, Position], stock_scores: pd.DataFrame, current_date: str
    ) -> Tuple[Dict[str, Dict], Dict[str, str]]:
        """生成交易信号

        Returns:
            (买入信号, 卖出信号)
        """
        buy_signals = {}
        sell_signals = {}

        try:
            # 生成卖出信号
            for stock_code, position in positions.items():
                sell_reason = None

                # 检查止损
                if config.stop_loss and position.unrealized_return <= -config.stop_loss:
                    sell_reason = f"止损 ({position.unrealized_return:.2%})"

                # 检查止盈
                elif config.take_profit and position.unrealized_return >= config.take_profit:
                    sell_reason = f"止盈 ({position.unrealized_return:.2%})"

                # 检查最小持仓天数
                elif position.holding_days < config.min_holding_days:
                    continue

                # 检查评分下降
                elif stock_code in stock_scores["stock_code"].values:
                    current_score = stock_scores[stock_scores["stock_code"] == stock_code]["score"].iloc[0]
                    if current_score < config.selection_threshold:
                        sell_reason = f"评分下降 ({current_score:.1f})"

                # 不在当前评分列表中
                elif stock_code not in stock_scores["stock_code"].values:
                    sell_reason = "不在评分列表"

                if sell_reason:
                    sell_signals[stock_code] = sell_reason

            # 生成买入信号
            current_positions = set(positions.keys())

            if config.selection_method == "top_score":
                # 选择评分最高的股票
                candidates = stock_scores[~stock_scores["stock_code"].isin(current_positions)].head(
                    config.max_positions - len(positions) + len(sell_signals)
                )
            else:
                # 选择评分超过阈值的股票
                candidates = stock_scores[
                    (~stock_scores["stock_code"].isin(current_positions))
                    & (stock_scores["score"] >= config.selection_threshold)
                ]

            # 行业中性处理
            if config.industry_neutral:
                candidates = self._apply_industry_neutral(candidates, positions, config.max_weight_per_industry)

            for _, row in candidates.iterrows():
                buy_signals[row["stock_code"]] = {
                    "stock_name": row["stock_name"],
                    "industry": row["industry"],
                    "score": row["score"],
                }

            return buy_signals, sell_signals

        except Exception as e:
            logger.error(f"生成交易信号失败: {e}")
            return {}, {}

    def _apply_industry_neutral(
        self, candidates: pd.DataFrame, positions: Dict[str, Position], max_weight_per_industry: float
    ) -> pd.DataFrame:
        """应用行业中性约束"""
        try:
            # 计算当前行业权重
            industry_weights = {}
            total_value = sum(pos.current_value for pos in positions.values())

            if total_value > 0:
                for position in positions.values():
                    industry = position.industry
                    if industry not in industry_weights:
                        industry_weights[industry] = 0
                    industry_weights[industry] += position.current_value / total_value

            # 过滤候选股票
            filtered_candidates = []
            for _, row in candidates.iterrows():
                industry = row["industry"]
                current_weight = industry_weights.get(industry, 0)

                if current_weight < max_weight_per_industry:
                    filtered_candidates.append(row)

            return pd.DataFrame(filtered_candidates)

        except Exception as e:
            logger.error(f"应用行业中性约束失败: {e}")
            return candidates

    def _execute_sell(
        self, config: BacktestConfig, position: Position, trade_date: str, reason: str
    ) -> Optional[Trade]:
        """执行卖出交易"""
        try:
            # 获取当日价格
            price = self._get_stock_price(position.stock_code, trade_date)
            if price is None:
                return None

            amount = position.shares * price
            commission = amount * config.transaction_cost

            trade = Trade(
                trade_id=f"SELL_{position.stock_code}_{trade_date}",
                stock_code=position.stock_code,
                stock_name=position.stock_name,
                trade_type="sell",
                trade_date=trade_date,
                price=price,
                shares=position.shares,
                amount=amount,
                commission=commission,
                reason=reason,
            )

            return trade

        except Exception as e:
            logger.error(f"执行卖出交易失败: {e}")
            return None

    def _execute_buy(
        self, config: BacktestConfig, portfolio: Dict[str, float], signal_data: Dict[str, Any], trade_date: str
    ) -> Optional[Trade]:
        """执行买入交易"""
        try:
            stock_code = (
                list(signal_data.keys())[0]
                if isinstance(signal_data, dict) and len(signal_data) == 1
                else signal_data.get("stock_code")
            )
            if not stock_code:
                # signal_data 是包含股票信息的字典
                for key, value in signal_data.items():
                    if key == "stock_code" or (isinstance(value, str) and "." in value):
                        stock_code = value if key == "stock_code" else key
                        break

            if not stock_code:
                logger.error(f"无法确定股票代码: {signal_data}")
                return None

            # 获取当日价格
            price = self._get_stock_price(stock_code, trade_date)
            if price is None:
                return None

            # 计算买入金额
            target_amount = portfolio["total_value"] * config.position_size
            available_cash = portfolio["cash"]

            # 考虑交易成本
            max_amount = available_cash / (1 + config.transaction_cost)
            actual_amount = min(target_amount, max_amount)

            if actual_amount < price:  # 资金不足买入1手
                return None

            shares = int(actual_amount / price / 100) * 100  # 按手买入
            if shares == 0:
                return None

            actual_amount = shares * price
            commission = actual_amount * config.transaction_cost

            trade = Trade(
                trade_id=f"BUY_{stock_code}_{trade_date}",
                stock_code=stock_code,
                stock_name=signal_data.get("stock_name", ""),
                trade_type="buy",
                trade_date=trade_date,
                price=price,
                shares=shares,
                amount=actual_amount,
                commission=commission,
                reason="新建仓位",
                score=signal_data.get("score"),
            )

            return trade

        except Exception as e:
            logger.error(f"执行买入交易失败: {e}")
            return None

    def _get_stock_price(self, stock_code: str, trade_date: str) -> Optional[float]:
        """获取股票价格"""
        try:
            query_sql = """
            SELECT close_price
            FROM stock_daily_data
            WHERE stock_code = :stock_code
            AND date = :trade_date
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {"stock_code": stock_code, "trade_date": trade_date})
                row = result.fetchone()

            return float(row[0]) if row else None

        except Exception as e:
            logger.error(f"获取股票价格失败: {e}")
            return None

    def _calculate_backtest_result(
        self,
        config: BacktestConfig,
        start_date: str,
        end_date: str,
        trading_dates: List[str],
        daily_portfolio_values: List[Dict],
        positions_history: List[List[Position]],
        trades: List[Trade],
        benchmark_data: Optional[pd.DataFrame],
    ) -> Optional[BacktestResult]:
        """计算回测结果"""
        try:
            # 转换为DataFrame
            portfolio_df = pd.DataFrame(daily_portfolio_values)
            portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
            portfolio_df.set_index("date", inplace=True)

            # 计算收益率
            portfolio_df["return"] = portfolio_df["total_value"].pct_change()
            portfolio_df["cumulative_return"] = (1 + portfolio_df["return"]).cumprod() - 1

            # 基本指标
            total_return = (portfolio_df["total_value"].iloc[-1] / config.initial_capital) - 1
            trading_days_count = len(trading_dates)
            annualized_return = (1 + total_return) ** (252 / trading_days_count) - 1

            # 基准收益
            benchmark_return = 0.0
            if benchmark_data is not None and not benchmark_data.empty:
                benchmark_return = (benchmark_data["close_price"].iloc[-1] / benchmark_data["close_price"].iloc[0]) - 1

            excess_return = total_return - benchmark_return

            # 风险指标
            returns = portfolio_df["return"].dropna()
            volatility = returns.std() * np.sqrt(252)

            # 最大回撤
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # 夏普比率
            risk_free_rate = 0.03  # 假设无风险利率3%
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

            # 信息比率
            if benchmark_data is not None and not benchmark_data.empty:
                benchmark_returns = benchmark_data["return"].dropna()
                excess_returns = returns - benchmark_returns.reindex(returns.index, fill_value=0)
                information_ratio = (
                    excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
                )
            else:
                information_ratio = 0

            # 卡玛比率
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0

            # 交易指标
            total_trades = len(trades)

            # 胜率计算
            sell_trades = [t for t in trades if t.trade_type == "sell"]
            win_trades = 0
            total_holding_days = 0

            for sell_trade in sell_trades:
                # 找到对应的买入交易
                buy_trade = None
                for trade in trades:
                    if (
                        trade.trade_type == "buy"
                        and trade.stock_code == sell_trade.stock_code
                        and trade.trade_date < sell_trade.trade_date
                    ):
                        buy_trade = trade
                        break

                if buy_trade:
                    pnl = sell_trade.amount - buy_trade.amount - sell_trade.commission - buy_trade.commission
                    if pnl > 0:
                        win_trades += 1

                    # 计算持仓天数
                    holding_days = (
                        datetime.strptime(sell_trade.trade_date, "%Y-%m-%d")
                        - datetime.strptime(buy_trade.trade_date, "%Y-%m-%d")
                    ).days
                    total_holding_days += holding_days

            win_rate = win_trades / len(sell_trades) if sell_trades else 0
            avg_holding_days = total_holding_days / len(sell_trades) if sell_trades else 0

            # 换手率
            total_trade_amount = sum(t.amount for t in trades if t.trade_type == "buy")
            avg_portfolio_value = portfolio_df["total_value"].mean()
            turnover_rate = (
                total_trade_amount / avg_portfolio_value / (trading_days_count / 252) if avg_portfolio_value > 0 else 0
            )

            # 生成可视化
            performance_chart = self._create_performance_chart(portfolio_df, benchmark_data)
            drawdown_chart = self._create_drawdown_chart(portfolio_df)
            monthly_returns_heatmap = self._create_monthly_returns_heatmap(portfolio_df)

            # 创建结果对象
            result = BacktestResult(
                strategy_name=config.strategy_name,
                model_name=config.model_name,
                model_version=config.model_version or "latest",
                config=config,
                start_date=start_date,
                end_date=end_date,
                total_days=(datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days,
                trading_days=trading_days_count,
                # 收益指标
                total_return=total_return,
                annualized_return=annualized_return,
                benchmark_return=benchmark_return,
                excess_return=excess_return,
                # 风险指标
                volatility=volatility,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                information_ratio=information_ratio,
                calmar_ratio=calmar_ratio,
                # 交易指标
                total_trades=total_trades,
                win_rate=win_rate,
                avg_holding_days=avg_holding_days,
                turnover_rate=turnover_rate,
                # 详细数据
                daily_returns=portfolio_df,
                positions_history=positions_history,
                trades_history=trades,
                portfolio_value_history=portfolio_df,
                # 可视化
                performance_chart=performance_chart,
                drawdown_chart=drawdown_chart,
                monthly_returns_heatmap=monthly_returns_heatmap,
                created_at=datetime.now(),
            )

            return result

        except Exception as e:
            logger.error(f"计算回测结果失败: {e}")
            return None

    def _create_performance_chart(
        self, portfolio_df: pd.DataFrame, benchmark_data: Optional[pd.DataFrame]
    ) -> Optional[str]:
        """创建业绩图表"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            # 绘制组合净值曲线
            ax.plot(portfolio_df.index, portfolio_df["cumulative_return"], label="策略收益", linewidth=2, color="blue")

            # 绘制基准收益曲线
            if benchmark_data is not None and not benchmark_data.empty:
                benchmark_data["date"] = pd.to_datetime(benchmark_data["date"])
                benchmark_data.set_index("date", inplace=True)
                benchmark_cumret = (benchmark_data["close_price"] / benchmark_data["close_price"].iloc[0]) - 1
                ax.plot(benchmark_data.index, benchmark_cumret, label="基准收益", linewidth=2, color="red", alpha=0.7)

            ax.set_title("策略业绩表现", fontsize=14, fontweight="bold")
            ax.set_xlabel("日期")
            ax.set_ylabel("累计收益率")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 格式化y轴为百分比
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.1%}".format(y)))

            plt.tight_layout()

            # 转换为base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            logger.error(f"创建业绩图表失败: {e}")
            return None

    def _create_drawdown_chart(self, portfolio_df: pd.DataFrame) -> Optional[str]:
        """创建回撤图表"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            # 计算回撤
            returns = portfolio_df["return"].dropna()
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max

            # 绘制回撤曲线
            ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="red")
            ax.plot(drawdown.index, drawdown, color="red", linewidth=1)

            ax.set_title("策略回撤分析", fontsize=14, fontweight="bold")
            ax.set_xlabel("日期")
            ax.set_ylabel("回撤")
            ax.grid(True, alpha=0.3)

            # 格式化y轴为百分比
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.1%}".format(y)))

            plt.tight_layout()

            # 转换为base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            logger.error(f"创建回撤图表失败: {e}")
            return None

    def _create_monthly_returns_heatmap(self, portfolio_df: pd.DataFrame) -> Optional[str]:
        """创建月度收益热力图"""
        try:
            # 计算月度收益
            monthly_returns = portfolio_df["return"].resample("M").apply(lambda x: (1 + x).prod() - 1)

            if len(monthly_returns) < 2:
                return None

            # 创建年月矩阵
            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            monthly_data = monthly_returns.to_frame("return")
            monthly_data["year"] = monthly_data.index.year
            monthly_data["month"] = monthly_data.index.month

            # 透视表
            heatmap_data = monthly_data.pivot(index="year", columns="month", values="return")

            fig, ax = plt.subplots(figsize=(12, 6))

            # 创建热力图
            sns.heatmap(
                heatmap_data, annot=True, fmt=".2%", cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label": "月度收益率"}
            )

            ax.set_title("月度收益热力图", fontsize=14, fontweight="bold")
            ax.set_xlabel("月份")
            ax.set_ylabel("年份")

            plt.tight_layout()

            # 转换为base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            logger.error(f"创建月度收益热力图失败: {e}")
            return None

    def _save_backtest_result(self, result: BacktestResult):
        """保存回测结果"""
        try:
            # 保存主要结果
            insert_sql = """
            INSERT INTO backtest_results (
                strategy_name, model_name, model_version, config,
                start_date, end_date, total_days, trading_days,
                total_return, annualized_return, benchmark_return, excess_return,
                volatility, max_drawdown, sharpe_ratio, information_ratio, calmar_ratio,
                total_trades, win_rate, avg_holding_days, turnover_rate,
                daily_returns, trades_history, portfolio_value_history,
                performance_chart, drawdown_chart, monthly_returns_heatmap
            ) VALUES (
                :strategy_name, :model_name, :model_version, :config,
                :start_date, :end_date, :total_days, :trading_days,
                :total_return, :annualized_return, :benchmark_return, :excess_return,
                :volatility, :max_drawdown, :sharpe_ratio, :information_ratio, :calmar_ratio,
                :total_trades, :win_rate, :avg_holding_days, :turnover_rate,
                :daily_returns, :trades_history, :portfolio_value_history,
                :performance_chart, :drawdown_chart, :monthly_returns_heatmap
            ) RETURNING id
            """

            with self.engine.connect() as conn:
                result_row = conn.execute(
                    text(insert_sql),
                    {
                        "strategy_name": result.strategy_name,
                        "model_name": result.model_name,
                        "model_version": result.model_version,
                        "config": json.dumps(asdict(result.config), default=str),
                        "start_date": result.start_date,
                        "end_date": result.end_date,
                        "total_days": result.total_days,
                        "trading_days": result.trading_days,
                        "total_return": result.total_return,
                        "annualized_return": result.annualized_return,
                        "benchmark_return": result.benchmark_return,
                        "excess_return": result.excess_return,
                        "volatility": result.volatility,
                        "max_drawdown": result.max_drawdown,
                        "sharpe_ratio": result.sharpe_ratio,
                        "information_ratio": result.information_ratio,
                        "calmar_ratio": result.calmar_ratio,
                        "total_trades": result.total_trades,
                        "win_rate": result.win_rate,
                        "avg_holding_days": result.avg_holding_days,
                        "turnover_rate": result.turnover_rate,
                        "daily_returns": result.daily_returns.to_json(orient="records", date_format="iso"),
                        "trades_history": json.dumps([asdict(t) for t in result.trades_history], default=str),
                        "portfolio_value_history": result.portfolio_value_history.to_json(
                            orient="records", date_format="iso"
                        ),
                        "performance_chart": result.performance_chart,
                        "drawdown_chart": result.drawdown_chart,
                        "monthly_returns_heatmap": result.monthly_returns_heatmap,
                    },
                )

                backtest_id = result_row.fetchone()[0]
                conn.commit()

            logger.info(f"回测结果保存成功: {backtest_id}")

        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")

    def get_backtest_results(self, strategy_name: str = None, model_name: str = None, limit: int = 10) -> pd.DataFrame:
        """获取回测结果列表

        Args:
            strategy_name: 策略名称
            model_name: 模型名称
            limit: 返回数量限制

        Returns:
            回测结果DataFrame
        """
        try:
            where_clause = "WHERE 1=1"
            params = {"limit": limit}

            if strategy_name:
                where_clause += " AND strategy_name = :strategy_name"
                params["strategy_name"] = strategy_name

            if model_name:
                where_clause += " AND model_name = :model_name"
                params["model_name"] = model_name

            query_sql = f"""
            SELECT id, strategy_name, model_name, model_version,
                   start_date, end_date, total_return, annualized_return,
                   max_drawdown, sharpe_ratio, total_trades, win_rate,
                   created_at
            FROM backtest_results
            {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit
            """

            df = pd.read_sql(query_sql, self.engine, params=params)
            return df

        except Exception as e:
            logger.error(f"获取回测结果失败: {e}")
            return pd.DataFrame()

    def get_backtest_detail(self, backtest_id: int) -> Optional[BacktestResult]:
        """获取回测详细结果

        Args:
            backtest_id: 回测ID

        Returns:
            回测结果详情
        """
        try:
            query_sql = """
            SELECT *
            FROM backtest_results
            WHERE id = :backtest_id
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {"backtest_id": backtest_id})
                row = result.fetchone()

            if not row:
                return None

            # 重构BacktestResult对象
            config_data = json.loads(row[4])
            config = BacktestConfig(**config_data)

            daily_returns = pd.read_json(row[21], orient="records")
            daily_returns["date"] = pd.to_datetime(daily_returns["date"])
            daily_returns.set_index("date", inplace=True)

            trades_data = json.loads(row[22])
            trades = [Trade(**trade_data) for trade_data in trades_data]

            portfolio_value_history = pd.read_json(row[23], orient="records")
            portfolio_value_history["date"] = pd.to_datetime(portfolio_value_history["date"])
            portfolio_value_history.set_index("date", inplace=True)

            result = BacktestResult(
                strategy_name=row[1],
                model_name=row[2],
                model_version=row[3],
                config=config,
                start_date=row[5].strftime("%Y-%m-%d"),
                end_date=row[6].strftime("%Y-%m-%d"),
                total_days=row[7],
                trading_days=row[8],
                total_return=float(row[9]),
                annualized_return=float(row[10]),
                benchmark_return=float(row[11]),
                excess_return=float(row[12]),
                volatility=float(row[13]),
                max_drawdown=float(row[14]),
                sharpe_ratio=float(row[15]),
                information_ratio=float(row[16]),
                calmar_ratio=float(row[17]),
                total_trades=row[18],
                win_rate=float(row[19]),
                avg_holding_days=float(row[20]),
                turnover_rate=float(row[21]),
                daily_returns=daily_returns,
                positions_history=[],  # 简化处理
                trades_history=trades,
                portfolio_value_history=portfolio_value_history,
                performance_chart=row[24],
                drawdown_chart=row[25],
                monthly_returns_heatmap=row[26],
                created_at=row[28],
            )

            return result

        except Exception as e:
            logger.error(f"获取回测详细结果失败: {e}")
            return None

    def compare_strategies(self, backtest_ids: List[int]) -> Optional[Dict[str, Any]]:
        """比较多个策略的回测结果

        Args:
            backtest_ids: 回测ID列表

        Returns:
            比较结果
        """
        try:
            if len(backtest_ids) < 2:
                logger.error("至少需要2个回测结果进行比较")
                return None

            # 获取回测结果
            results = []
            for backtest_id in backtest_ids:
                result = self.get_backtest_detail(backtest_id)
                if result:
                    results.append(result)

            if len(results) < 2:
                logger.error("有效的回测结果不足2个")
                return None

            # 创建比较表
            comparison_data = []
            for result in results:
                comparison_data.append(
                    {
                        "strategy_name": result.strategy_name,
                        "model_name": result.model_name,
                        "total_return": result.total_return,
                        "annualized_return": result.annualized_return,
                        "volatility": result.volatility,
                        "max_drawdown": result.max_drawdown,
                        "sharpe_ratio": result.sharpe_ratio,
                        "information_ratio": result.information_ratio,
                        "calmar_ratio": result.calmar_ratio,
                        "win_rate": result.win_rate,
                        "total_trades": result.total_trades,
                        "turnover_rate": result.turnover_rate,
                    }
                )

            comparison_df = pd.DataFrame(comparison_data)

            # 生成比较图表
            comparison_chart = self._create_comparison_chart(results)

            return {
                "comparison_table": comparison_df,
                "comparison_chart": comparison_chart,
                "best_strategy": {
                    "by_return": comparison_df.loc[comparison_df["total_return"].idxmax()]["strategy_name"],
                    "by_sharpe": comparison_df.loc[comparison_df["sharpe_ratio"].idxmax()]["strategy_name"],
                    "by_calmar": comparison_df.loc[comparison_df["calmar_ratio"].idxmax()]["strategy_name"],
                },
            }

        except Exception as e:
            logger.error(f"策略比较失败: {e}")
            return None

    def _create_comparison_chart(self, results: List[BacktestResult]) -> Optional[str]:
        """创建策略比较图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # 净值曲线比较
            ax1 = axes[0, 0]
            for result in results:
                cumret = result.daily_returns["cumulative_return"]
                ax1.plot(cumret.index, cumret, label=result.strategy_name, linewidth=2)
            ax1.set_title("净值曲线比较")
            ax1.set_ylabel("累计收益率")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.1%}".format(y)))

            # 收益率分布
            ax2 = axes[0, 1]
            for result in results:
                returns = result.daily_returns["return"].dropna()
                ax2.hist(returns, alpha=0.6, label=result.strategy_name, bins=50)
            ax2.set_title("收益率分布")
            ax2.set_xlabel("日收益率")
            ax2.set_ylabel("频次")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 风险收益散点图
            ax3 = axes[1, 0]
            for result in results:
                ax3.scatter(result.volatility, result.annualized_return, s=100, label=result.strategy_name, alpha=0.7)
            ax3.set_title("风险收益散点图")
            ax3.set_xlabel("波动率")
            ax3.set_ylabel("年化收益率")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 关键指标雷达图
            ax4 = axes[1, 1]
            metrics = ["总收益", "夏普比率", "信息比率", "卡玛比率", "胜率"]

            # 标准化指标值
            normalized_data = []
            for result in results:
                values = [
                    result.total_return,
                    result.sharpe_ratio,
                    result.information_ratio,
                    result.calmar_ratio,
                    result.win_rate,
                ]
                # 简单标准化到0-1
                max_vals = [
                    max(r.total_return for r in results),
                    max(r.sharpe_ratio for r in results),
                    max(r.information_ratio for r in results),
                    max(r.calmar_ratio for r in results),
                    max(r.win_rate for r in results),
                ]

                normalized = [v / m if m > 0 else 0 for v, m in zip(values, max_vals)]
                normalized_data.append(normalized)

            # 绘制雷达图
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合

            for i, result in enumerate(results):
                values = normalized_data[i] + normalized_data[i][:1]  # 闭合
                ax4.plot(angles, values, "o-", linewidth=2, label=result.strategy_name)
                ax4.fill(angles, values, alpha=0.25)

            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics)
            ax4.set_title("关键指标比较")
            ax4.legend()
            ax4.grid(True)

            plt.tight_layout()

            # 转换为base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            logger.error(f"创建策略比较图表失败: {e}")
            return None

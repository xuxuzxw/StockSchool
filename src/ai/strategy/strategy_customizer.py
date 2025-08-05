import json
import logging
import pickle
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.utils.db import get_db_manager

"""策略定制器

提供AI策略的个性化定制功能，支持用户自定义策略参数。
"""

warnings.filterwarnings("ignore")


from .backtest_engine import BacktestConfig, BacktestEngine
from .factor_weight_engine import FactorWeightEngine
from .model_manager import AIModelManager
from .stock_scoring_engine import StockScoringEngine

logger = logging.getLogger(__name__)


@dataclass
class StrategyTemplate:
    """策略模板"""

    template_id: str
    template_name: str
    description: str
    category: str  # 'conservative', 'balanced', 'aggressive', 'custom'
    default_config: Dict[str, Any]
    parameter_constraints: Dict[str, Dict[str, Any]]
    risk_level: int  # 1-5, 1最保守，5最激进
    expected_return: float
    expected_volatility: float
    min_investment: float
    suitable_investors: List[str]
    created_at: Optional[datetime] = None


@dataclass
class UserStrategy:
    """用户策略"""

    strategy_id: str
    user_id: str
    strategy_name: str
    template_id: str
    custom_config: Dict[str, Any]
    risk_preference: str  # 'conservative', 'balanced', 'aggressive'
    investment_amount: float
    investment_horizon: int  # 投资期限（月）
    rebalance_frequency: str
    constraints: Dict[str, Any]  # 投资约束
    status: str  # 'active', 'paused', 'stopped'
    performance_target: Dict[str, float]  # 业绩目标
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class StrategyRecommendation:
    """策略推荐"""

    recommendation_id: str
    user_id: str
    recommended_strategies: List[Dict[str, Any]]
    recommendation_reason: str
    confidence_score: float
    risk_assessment: Dict[str, Any]
    expected_performance: Dict[str, float]
    created_at: Optional[datetime] = None


@dataclass
class RiskProfile:
    """风险画像"""

    user_id: str
    risk_tolerance: int  # 1-10
    investment_experience: str  # 'beginner', 'intermediate', 'advanced'
    age_group: str
    income_level: str
    investment_goals: List[str]
    time_horizon: int
    liquidity_needs: str
    loss_tolerance: float  # 最大可承受损失比例
    sector_preferences: List[str]
    esg_preference: bool  # 是否偏好ESG投资
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class StrategyCustomizer:
    """策略定制器

    主要功能：
    - 策略模板管理
    - 用户风险画像
    - 个性化推荐
    - 策略参数优化
    - 约束条件设置
    - 业绩目标设定
    - 策略回测验证
    """

    def __init__(self):
        """方法描述"""
        self.model_manager = AIModelManager()
        self.factor_engine = FactorWeightEngine()
        self.scoring_engine = StockScoringEngine()
        self.backtest_engine = BacktestEngine()

        self._ensure_tables_exist()
        self._initialize_default_templates()

    def _ensure_tables_exist(self):
        """确保策略定制相关表存在"""
        create_tables_sql = """
        -- 策略模板表
        CREATE TABLE IF NOT EXISTS strategy_templates (
            id SERIAL PRIMARY KEY,
            template_id VARCHAR(100) UNIQUE NOT NULL,
            template_name VARCHAR(200) NOT NULL,
            description TEXT,
            category VARCHAR(50) NOT NULL,
            default_config JSONB NOT NULL,
            parameter_constraints JSONB,
            risk_level INTEGER CHECK (risk_level BETWEEN 1 AND 5),
            expected_return DECIMAL(8, 6),
            expected_volatility DECIMAL(8, 6),
            min_investment DECIMAL(15, 2),
            suitable_investors JSONB,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 用户策略表
        CREATE TABLE IF NOT EXISTS user_strategies (
            id SERIAL PRIMARY KEY,
            strategy_id VARCHAR(100) UNIQUE NOT NULL,
            user_id VARCHAR(100) NOT NULL,
            strategy_name VARCHAR(200) NOT NULL,
            template_id VARCHAR(100) REFERENCES strategy_templates(template_id),
            custom_config JSONB NOT NULL,
            risk_preference VARCHAR(50),
            investment_amount DECIMAL(15, 2),
            investment_horizon INTEGER,
            rebalance_frequency VARCHAR(20),
            constraints JSONB,
            status VARCHAR(20) DEFAULT 'active',
            performance_target JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 用户风险画像表
        CREATE TABLE IF NOT EXISTS user_risk_profiles (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(100) UNIQUE NOT NULL,
            risk_tolerance INTEGER CHECK (risk_tolerance BETWEEN 1 AND 10),
            investment_experience VARCHAR(20),
            age_group VARCHAR(20),
            income_level VARCHAR(20),
            investment_goals JSONB,
            time_horizon INTEGER,
            liquidity_needs VARCHAR(20),
            loss_tolerance DECIMAL(6, 4),
            sector_preferences JSONB,
            esg_preference BOOLEAN DEFAULT false,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 策略推荐表
        CREATE TABLE IF NOT EXISTS strategy_recommendations (
            id SERIAL PRIMARY KEY,
            recommendation_id VARCHAR(100) UNIQUE NOT NULL,
            user_id VARCHAR(100) NOT NULL,
            recommended_strategies JSONB NOT NULL,
            recommendation_reason TEXT,
            confidence_score DECIMAL(4, 3),
            risk_assessment JSONB,
            expected_performance JSONB,
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 策略性能监控表
        CREATE TABLE IF NOT EXISTS strategy_performance (
            id SERIAL PRIMARY KEY,
            strategy_id VARCHAR(100) REFERENCES user_strategies(strategy_id),
            date DATE NOT NULL,
            portfolio_value DECIMAL(15, 2),
            daily_return DECIMAL(10, 6),
            cumulative_return DECIMAL(10, 6),
            volatility DECIMAL(10, 6),
            max_drawdown DECIMAL(10, 6),
            sharpe_ratio DECIMAL(10, 6),
            positions_count INTEGER,
            cash_ratio DECIMAL(6, 4),
            sector_allocation JSONB,
            top_holdings JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 策略调整记录表
        CREATE TABLE IF NOT EXISTS strategy_adjustments (
            id SERIAL PRIMARY KEY,
            strategy_id VARCHAR(100) REFERENCES user_strategies(strategy_id),
            adjustment_type VARCHAR(50) NOT NULL,
            old_config JSONB,
            new_config JSONB,
            reason TEXT,
            trigger_condition VARCHAR(100),
            performance_impact JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_strategy_templates_category ON strategy_templates(category, is_active);
        CREATE INDEX IF NOT EXISTS idx_user_strategies_user ON user_strategies(user_id, status);
        CREATE INDEX IF NOT EXISTS idx_user_strategies_template ON user_strategies(template_id);
        CREATE INDEX IF NOT EXISTS idx_strategy_recommendations_user ON strategy_recommendations(user_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy_date ON strategy_performance(strategy_id, date);
        CREATE INDEX IF NOT EXISTS idx_strategy_adjustments_strategy ON strategy_adjustments(strategy_id, created_at);
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
            logger.info("策略定制表创建成功")
        except Exception as e:
            logger.error(f"创建策略定制表失败: {e}")
            raise

    def _initialize_default_templates(self):
        """初始化默认策略模板"""
        try:
            # 检查是否已有模板
            query_sql = "SELECT COUNT(*) FROM strategy_templates"
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql))
                count = result.fetchone()[0]

            if count > 0:
                return

            # 创建默认模板
            default_templates = [
                {
                    "template_id": "conservative_value",
                    "template_name": "稳健价值策略",
                    "description": "注重价值投资，风险较低，适合保守型投资者",
                    "category": "conservative",
                    "default_config": {
                        "initial_capital": 1000000,
                        "position_size": 0.08,
                        "max_positions": 15,
                        "rebalance_frequency": "monthly",
                        "selection_method": "top_score",
                        "selection_threshold": 85,
                        "transaction_cost": 0.003,
                        "industry_neutral": True,
                        "max_weight_per_industry": 0.25,
                        "stop_loss": 0.15,
                        "take_profit": None,
                        "min_holding_days": 30,
                    },
                    "parameter_constraints": {
                        "position_size": {"min": 0.05, "max": 0.12},
                        "max_positions": {"min": 10, "max": 20},
                        "selection_threshold": {"min": 80, "max": 95},
                        "stop_loss": {"min": 0.10, "max": 0.20},
                    },
                    "risk_level": 2,
                    "expected_return": 0.12,
                    "expected_volatility": 0.15,
                    "min_investment": 100000,
                    "suitable_investors": ["保守型", "稳健型", "退休人员", "风险厌恶者"],
                },
                {
                    "template_id": "balanced_growth",
                    "template_name": "均衡成长策略",
                    "description": "平衡风险与收益，适合大多数投资者",
                    "category": "balanced",
                    "default_config": {
                        "initial_capital": 1000000,
                        "position_size": 0.10,
                        "max_positions": 12,
                        "rebalance_frequency": "weekly",
                        "selection_method": "top_score",
                        "selection_threshold": 80,
                        "transaction_cost": 0.003,
                        "industry_neutral": True,
                        "max_weight_per_industry": 0.30,
                        "stop_loss": 0.20,
                        "take_profit": 0.30,
                        "min_holding_days": 14,
                    },
                    "parameter_constraints": {
                        "position_size": {"min": 0.08, "max": 0.15},
                        "max_positions": {"min": 8, "max": 15},
                        "selection_threshold": {"min": 75, "max": 90},
                        "stop_loss": {"min": 0.15, "max": 0.25},
                    },
                    "risk_level": 3,
                    "expected_return": 0.18,
                    "expected_volatility": 0.22,
                    "min_investment": 50000,
                    "suitable_investors": ["均衡型", "成长型", "中等风险承受者"],
                },
                {
                    "template_id": "aggressive_momentum",
                    "template_name": "激进动量策略",
                    "description": "追求高收益，风险较高，适合激进型投资者",
                    "category": "aggressive",
                    "default_config": {
                        "initial_capital": 1000000,
                        "position_size": 0.15,
                        "max_positions": 8,
                        "rebalance_frequency": "daily",
                        "selection_method": "top_score",
                        "selection_threshold": 75,
                        "transaction_cost": 0.003,
                        "industry_neutral": False,
                        "max_weight_per_industry": 0.50,
                        "stop_loss": 0.25,
                        "take_profit": 0.50,
                        "min_holding_days": 5,
                    },
                    "parameter_constraints": {
                        "position_size": {"min": 0.12, "max": 0.20},
                        "max_positions": {"min": 5, "max": 10},
                        "selection_threshold": {"min": 70, "max": 85},
                        "stop_loss": {"min": 0.20, "max": 0.35},
                    },
                    "risk_level": 5,
                    "expected_return": 0.30,
                    "expected_volatility": 0.35,
                    "min_investment": 200000,
                    "suitable_investors": ["激进型", "高风险承受者", "专业投资者"],
                },
            ]

            # 插入模板
            for template_data in default_templates:
                self._save_strategy_template(StrategyTemplate(**template_data))

            logger.info("默认策略模板初始化完成")

        except Exception as e:
            logger.error(f"初始化默认策略模板失败: {e}")

    def create_user_risk_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Optional[RiskProfile]:
        """创建用户风险画像

        Args:
            user_id: 用户ID
            profile_data: 风险画像数据

        Returns:
            风险画像对象
        """
        try:
            # 验证数据
            required_fields = [
                "risk_tolerance",
                "investment_experience",
                "age_group",
                "income_level",
                "investment_goals",
                "time_horizon",
            ]

            for field in required_fields:
                if field not in profile_data:
                    logger.error(f"缺少必要字段: {field}")
                    return None

            # 创建风险画像
            risk_profile = RiskProfile(
                user_id=user_id,
                risk_tolerance=profile_data["risk_tolerance"],
                investment_experience=profile_data["investment_experience"],
                age_group=profile_data["age_group"],
                income_level=profile_data["income_level"],
                investment_goals=profile_data["investment_goals"],
                time_horizon=profile_data["time_horizon"],
                liquidity_needs=profile_data.get("liquidity_needs", "medium"),
                loss_tolerance=profile_data.get("loss_tolerance", 0.20),
                sector_preferences=profile_data.get("sector_preferences", []),
                esg_preference=profile_data.get("esg_preference", False),
                created_at=datetime.now(),
            )

            # 保存到数据库
            insert_sql = """
            INSERT INTO user_risk_profiles (
                user_id, risk_tolerance, investment_experience, age_group,
                income_level, investment_goals, time_horizon, liquidity_needs,
                loss_tolerance, sector_preferences, esg_preference
            ) VALUES (
                :user_id, :risk_tolerance, :investment_experience, :age_group,
                :income_level, :investment_goals, :time_horizon, :liquidity_needs,
                :loss_tolerance, :sector_preferences, :esg_preference
            )
            ON CONFLICT (user_id) DO UPDATE SET
                risk_tolerance = EXCLUDED.risk_tolerance,
                investment_experience = EXCLUDED.investment_experience,
                age_group = EXCLUDED.age_group,
                income_level = EXCLUDED.income_level,
                investment_goals = EXCLUDED.investment_goals,
                time_horizon = EXCLUDED.time_horizon,
                liquidity_needs = EXCLUDED.liquidity_needs,
                loss_tolerance = EXCLUDED.loss_tolerance,
                sector_preferences = EXCLUDED.sector_preferences,
                esg_preference = EXCLUDED.esg_preference,
                updated_at = CURRENT_TIMESTAMP
            """

            with self.engine.connect() as conn:
                conn.execute(
                    text(insert_sql),
                    {
                        "user_id": user_id,
                        "risk_tolerance": risk_profile.risk_tolerance,
                        "investment_experience": risk_profile.investment_experience,
                        "age_group": risk_profile.age_group,
                        "income_level": risk_profile.income_level,
                        "investment_goals": json.dumps(risk_profile.investment_goals),
                        "time_horizon": risk_profile.time_horizon,
                        "liquidity_needs": risk_profile.liquidity_needs,
                        "loss_tolerance": risk_profile.loss_tolerance,
                        "sector_preferences": json.dumps(risk_profile.sector_preferences),
                        "esg_preference": risk_profile.esg_preference,
                    },
                )
                conn.commit()

            logger.info(f"用户风险画像创建成功: {user_id}")
            return risk_profile

        except Exception as e:
            logger.error(f"创建用户风险画像失败: {e}")
            return None

    def get_user_risk_profile(self, user_id: str) -> Optional[RiskProfile]:
        """获取用户风险画像

        Args:
            user_id: 用户ID

        Returns:
            风险画像对象
        """
        try:
            query_sql = """
            SELECT *
            FROM user_risk_profiles
            WHERE user_id = :user_id
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {"user_id": user_id})
                row = result.fetchone()

            if not row:
                return None

            risk_profile = RiskProfile(
                user_id=row[1],
                risk_tolerance=row[2],
                investment_experience=row[3],
                age_group=row[4],
                income_level=row[5],
                investment_goals=json.loads(row[6]) if row[6] else [],
                time_horizon=row[7],
                liquidity_needs=row[8],
                loss_tolerance=float(row[9]),
                sector_preferences=json.loads(row[10]) if row[10] else [],
                esg_preference=row[11],
                created_at=row[12],
                updated_at=row[13],
            )

            return risk_profile

        except Exception as e:
            logger.error(f"获取用户风险画像失败: {e}")
            return None

    def recommend_strategies(self, user_id: str) -> Optional[StrategyRecommendation]:
        """为用户推荐策略

        Args:
            user_id: 用户ID

        Returns:
            策略推荐结果
        """
        try:
            # 获取用户风险画像
            risk_profile = self.get_user_risk_profile(user_id)
            if not risk_profile:
                logger.error(f"用户风险画像不存在: {user_id}")
                return None

            # 获取所有策略模板
            templates = self.get_strategy_templates()
            if not templates:
                logger.error("没有可用的策略模板")
                return None

            # 计算匹配度
            recommendations = []
            for template in templates:
                match_score = self._calculate_strategy_match_score(risk_profile, template)
                if match_score > 0.5:  # 匹配度阈值
                    recommendations.append(
                        {
                            "template_id": template.template_id,
                            "template_name": template.template_name,
                            "description": template.description,
                            "match_score": match_score,
                            "risk_level": template.risk_level,
                            "expected_return": template.expected_return,
                            "expected_volatility": template.expected_volatility,
                            "recommended_config": self._customize_config_for_user(
                                template.default_config, risk_profile
                            ),
                        }
                    )

            # 按匹配度排序
            recommendations.sort(key=lambda x: x["match_score"], reverse=True)

            # 生成推荐理由
            recommendation_reason = self._generate_recommendation_reason(risk_profile, recommendations)

            # 风险评估
            risk_assessment = self._assess_recommendation_risk(risk_profile, recommendations)

            # 预期表现
            expected_performance = self._estimate_expected_performance(recommendations)

            # 创建推荐结果
            recommendation = StrategyRecommendation(
                recommendation_id=f"REC_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                user_id=user_id,
                recommended_strategies=recommendations[:3],  # 推荐前3个
                recommendation_reason=recommendation_reason,
                confidence_score=min(recommendations[0]["match_score"] if recommendations else 0, 1.0),
                risk_assessment=risk_assessment,
                expected_performance=expected_performance,
                created_at=datetime.now(),
            )

            # 保存推荐结果
            self._save_strategy_recommendation(recommendation)

            return recommendation

        except Exception as e:
            logger.error(f"策略推荐失败: {e}")
            return None

    def _calculate_strategy_match_score(self, risk_profile: RiskProfile, template: StrategyTemplate) -> float:
        """计算策略匹配度"""
        try:
            score = 0.0

            # 风险承受能力匹配 (40%权重)
            risk_match = 1 - abs(risk_profile.risk_tolerance - template.risk_level * 2) / 10
            score += risk_match * 0.4

            # 投资经验匹配 (20%权重)
            experience_scores = {"beginner": 1, "intermediate": 3, "advanced": 5}
            exp_score = experience_scores.get(risk_profile.investment_experience, 3)
            exp_match = 1 - abs(exp_score - template.risk_level) / 5
            score += exp_match * 0.2

            # 投资期限匹配 (20%权重)
            if template.category == "conservative" and risk_profile.time_horizon >= 12:
                time_match = 1.0
            elif template.category == "balanced" and 6 <= risk_profile.time_horizon <= 36:
                time_match = 1.0
            elif template.category == "aggressive" and risk_profile.time_horizon <= 24:
                time_match = 1.0
            else:
                time_match = 0.5
            score += time_match * 0.2

            # 损失承受能力匹配 (20%权重)
            expected_max_loss = template.expected_volatility * 2  # 估算最大可能损失
            loss_match = 1.0 if expected_max_loss <= risk_profile.loss_tolerance else 0.5
            score += loss_match * 0.2

            return max(0, min(1, score))

        except Exception as e:
            logger.error(f"计算策略匹配度失败: {e}")
            return 0.0

    def _customize_config_for_user(self, default_config: Dict[str, Any], risk_profile: RiskProfile) -> Dict[str, Any]:
        """为用户定制配置"""
        try:
            custom_config = default_config.copy()

            # 根据风险承受能力调整仓位
            if risk_profile.risk_tolerance <= 3:
                custom_config["position_size"] *= 0.8
                custom_config["max_positions"] = min(custom_config["max_positions"] + 3, 20)
            elif risk_profile.risk_tolerance >= 8:
                custom_config["position_size"] *= 1.2
                custom_config["max_positions"] = max(custom_config["max_positions"] - 2, 5)

            # 根据投资期限调整调仓频率
            if risk_profile.time_horizon >= 24:
                if custom_config["rebalance_frequency"] == "daily":
                    custom_config["rebalance_frequency"] = "weekly"
                elif custom_config["rebalance_frequency"] == "weekly":
                    custom_config["rebalance_frequency"] = "monthly"

            # 根据损失承受能力调整止损
            if "stop_loss" in custom_config and custom_config["stop_loss"]:
                custom_config["stop_loss"] = min(custom_config["stop_loss"], risk_profile.loss_tolerance)

            # 行业偏好
            if risk_profile.sector_preferences:
                custom_config["preferred_sectors"] = risk_profile.sector_preferences

            # ESG偏好
            if risk_profile.esg_preference:
                custom_config["esg_filter"] = True

            return custom_config

        except Exception as e:
            logger.error(f"定制配置失败: {e}")
            return default_config

    def _generate_recommendation_reason(self, risk_profile: RiskProfile, recommendations: List[Dict]) -> str:
        """生成推荐理由"""
        try:
            if not recommendations:
                return "暂无合适的策略推荐"

            top_strategy = recommendations[0]

            reasons = []

            # 风险匹配
            if risk_profile.risk_tolerance <= 4:
                reasons.append("您的风险承受能力较为保守")
            elif risk_profile.risk_tolerance >= 7:
                reasons.append("您具有较强的风险承受能力")
            else:
                reasons.append("您的风险偏好相对均衡")

            # 经验匹配
            if risk_profile.investment_experience == "beginner":
                reasons.append("作为投资新手，建议从稳健策略开始")
            elif risk_profile.investment_experience == "advanced":
                reasons.append("凭借您的投资经验，可以考虑更多样化的策略")

            # 期限匹配
            if risk_profile.time_horizon >= 24:
                reasons.append("您的长期投资目标适合价值投资策略")
            elif risk_profile.time_horizon <= 6:
                reasons.append("短期投资目标建议选择灵活性较高的策略")

            reason_text = "，".join(reasons)
            reason_text += f"，因此推荐{top_strategy['template_name']}，该策略{top_strategy['description']}"

            return reason_text

        except Exception as e:
            logger.error(f"生成推荐理由失败: {e}")
            return "基于您的风险画像，为您推荐以下策略"

    def _assess_recommendation_risk(self, risk_profile: RiskProfile, recommendations: List[Dict]) -> Dict[str, Any]:
        """评估推荐风险"""
        try:
            if not recommendations:
                return {}

            top_strategy = recommendations[0]

            risk_assessment = {
                "overall_risk_level": top_strategy["risk_level"],
                "volatility_risk": (
                    "high"
                    if top_strategy["expected_volatility"] > 0.25
                    else "medium" if top_strategy["expected_volatility"] > 0.15 else "low"
                ),
                "liquidity_risk": "low",  # 股票策略流动性风险较低
                "concentration_risk": "medium",  # 取决于持仓集中度
                "market_risk": "high",  # 股票策略市场风险较高
                "risk_warnings": [],
            }

            # 风险警告
            if top_strategy["expected_volatility"] > risk_profile.loss_tolerance:
                risk_assessment["risk_warnings"].append(
                    f"策略预期波动率({top_strategy['expected_volatility']:.1%})可能超过您的损失承受能力"
                )

            if top_strategy["risk_level"] > risk_profile.risk_tolerance / 2:
                risk_assessment["risk_warnings"].append("策略风险等级可能高于您的风险偏好")

            return risk_assessment

        except Exception as e:
            logger.error(f"评估推荐风险失败: {e}")
            return {}

    def _estimate_expected_performance(self, recommendations: List[Dict]) -> Dict[str, float]:
        """估算预期表现"""
        try:
            if not recommendations:
                return {}

            top_strategy = recommendations[0]

            return {
                "expected_annual_return": top_strategy["expected_return"],
                "expected_volatility": top_strategy["expected_volatility"],
                "expected_sharpe_ratio": (top_strategy["expected_return"] - 0.03) / top_strategy["expected_volatility"],
                "expected_max_drawdown": top_strategy["expected_volatility"] * 1.5,
                "confidence_interval": {
                    "lower_bound": top_strategy["expected_return"] - top_strategy["expected_volatility"],
                    "upper_bound": top_strategy["expected_return"] + top_strategy["expected_volatility"],
                },
            }

        except Exception as e:
            logger.error(f"估算预期表现失败: {e}")
            return {}

    def create_user_strategy(
        self,
        user_id: str,
        strategy_name: str,
        template_id: str,
        custom_config: Dict[str, Any],
        investment_amount: float,
    ) -> Optional[UserStrategy]:
        """创建用户策略

        Args:
            user_id: 用户ID
            strategy_name: 策略名称
            template_id: 模板ID
            custom_config: 自定义配置
            investment_amount: 投资金额

        Returns:
            用户策略对象
        """
        try:
            # 验证模板存在
            template = self.get_strategy_template(template_id)
            if not template:
                logger.error(f"策略模板不存在: {template_id}")
                return None

            # 验证配置参数
            if not self._validate_custom_config(custom_config, template):
                logger.error("自定义配置验证失败")
                return None

            # 获取用户风险画像
            risk_profile = self.get_user_risk_profile(user_id)

            # 创建用户策略
            user_strategy = UserStrategy(
                strategy_id=f"STRATEGY_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                user_id=user_id,
                strategy_name=strategy_name,
                template_id=template_id,
                custom_config=custom_config,
                risk_preference=self._determine_risk_preference(risk_profile) if risk_profile else "balanced",
                investment_amount=investment_amount,
                investment_horizon=risk_profile.time_horizon if risk_profile else 12,
                rebalance_frequency=custom_config.get("rebalance_frequency", "weekly"),
                constraints=custom_config.get("constraints", {}),
                status="active",
                performance_target=custom_config.get("performance_target", {}),
                created_at=datetime.now(),
            )

            # 保存到数据库
            self._save_user_strategy(user_strategy)

            logger.info(f"用户策略创建成功: {user_strategy.strategy_id}")
            return user_strategy

        except Exception as e:
            logger.error(f"创建用户策略失败: {e}")
            return None

    def _validate_custom_config(self, custom_config: Dict[str, Any], template: StrategyTemplate) -> bool:
        """验证自定义配置"""
        try:
            constraints = template.parameter_constraints

            for param, value in custom_config.items():
                if param in constraints:
                    constraint = constraints[param]

                    # 检查最小值
                    if "min" in constraint and value < constraint["min"]:
                        logger.error(f"参数{param}值{value}小于最小值{constraint['min']}")
                        return False

                    # 检查最大值
                    if "max" in constraint and value > constraint["max"]:
                        logger.error(f"参数{param}值{value}大于最大值{constraint['max']}")
                        return False

                    # 检查枚举值
                    if "values" in constraint and value not in constraint["values"]:
                        logger.error(f"参数{param}值{value}不在允许值列表中")
                        return False

            return True

        except Exception as e:
            logger.error(f"验证自定义配置失败: {e}")
            return False

    def _determine_risk_preference(self, risk_profile: RiskProfile) -> str:
        """确定风险偏好"""
        if risk_profile.risk_tolerance <= 3:
            return "conservative"
        elif risk_profile.risk_tolerance <= 7:
            return "balanced"
        else:
            return "aggressive"

    def get_strategy_templates(self, category: str = None) -> List[StrategyTemplate]:
        """获取策略模板列表

        Args:
            category: 策略类别

        Returns:
            策略模板列表
        """
        try:
            where_clause = "WHERE is_active = true"
            params = {}

            if category:
                where_clause += " AND category = :category"
                params["category"] = category

            query_sql = f"""
            SELECT *
            FROM strategy_templates
            {where_clause}
            ORDER BY risk_level, created_at
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                rows = result.fetchall()

            templates = []
            for row in rows:
                template = StrategyTemplate(
                    template_id=row[1],
                    template_name=row[2],
                    description=row[3],
                    category=row[4],
                    default_config=json.loads(row[5]),
                    parameter_constraints=json.loads(row[6]) if row[6] else {},
                    risk_level=row[7],
                    expected_return=float(row[8]) if row[8] else 0,
                    expected_volatility=float(row[9]) if row[9] else 0,
                    min_investment=float(row[10]) if row[10] else 0,
                    suitable_investors=json.loads(row[11]) if row[11] else [],
                    created_at=row[13],
                )
                templates.append(template)

            return templates

        except Exception as e:
            logger.error(f"获取策略模板失败: {e}")
            return []

    def get_strategy_template(self, template_id: str) -> Optional[StrategyTemplate]:
        """获取单个策略模板

        Args:
            template_id: 模板ID

        Returns:
            策略模板对象
        """
        try:
            query_sql = """
            SELECT *
            FROM strategy_templates
            WHERE template_id = :template_id AND is_active = true
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {"template_id": template_id})
                row = result.fetchone()

            if not row:
                return None

            template = StrategyTemplate(
                template_id=row[1],
                template_name=row[2],
                description=row[3],
                category=row[4],
                default_config=json.loads(row[5]),
                parameter_constraints=json.loads(row[6]) if row[6] else {},
                risk_level=row[7],
                expected_return=float(row[8]) if row[8] else 0,
                expected_volatility=float(row[9]) if row[9] else 0,
                min_investment=float(row[10]) if row[10] else 0,
                suitable_investors=json.loads(row[11]) if row[11] else [],
                created_at=row[13],
            )

            return template

        except Exception as e:
            logger.error(f"获取策略模板失败: {e}")
            return None

    def _save_strategy_template(self, template: StrategyTemplate):
        """保存策略模板"""
        try:
            insert_sql = """
            INSERT INTO strategy_templates (
                template_id, template_name, description, category,
                default_config, parameter_constraints, risk_level,
                expected_return, expected_volatility, min_investment,
                suitable_investors
            ) VALUES (
                :template_id, :template_name, :description, :category,
                :default_config, :parameter_constraints, :risk_level,
                :expected_return, :expected_volatility, :min_investment,
                :suitable_investors
            )
            ON CONFLICT (template_id) DO UPDATE SET
                template_name = EXCLUDED.template_name,
                description = EXCLUDED.description,
                category = EXCLUDED.category,
                default_config = EXCLUDED.default_config,
                parameter_constraints = EXCLUDED.parameter_constraints,
                risk_level = EXCLUDED.risk_level,
                expected_return = EXCLUDED.expected_return,
                expected_volatility = EXCLUDED.expected_volatility,
                min_investment = EXCLUDED.min_investment,
                suitable_investors = EXCLUDED.suitable_investors
            """

            with self.engine.connect() as conn:
                conn.execute(
                    text(insert_sql),
                    {
                        "template_id": template.template_id,
                        "template_name": template.template_name,
                        "description": template.description,
                        "category": template.category,
                        "default_config": json.dumps(template.default_config),
                        "parameter_constraints": json.dumps(template.parameter_constraints),
                        "risk_level": template.risk_level,
                        "expected_return": template.expected_return,
                        "expected_volatility": template.expected_volatility,
                        "min_investment": template.min_investment,
                        "suitable_investors": json.dumps(template.suitable_investors),
                    },
                )
                conn.commit()

        except Exception as e:
            logger.error(f"保存策略模板失败: {e}")
            raise

    def _save_user_strategy(self, user_strategy: UserStrategy):
        """保存用户策略"""
        try:
            insert_sql = """
            INSERT INTO user_strategies (
                strategy_id, user_id, strategy_name, template_id,
                custom_config, risk_preference, investment_amount,
                investment_horizon, rebalance_frequency, constraints,
                status, performance_target
            ) VALUES (
                :strategy_id, :user_id, :strategy_name, :template_id,
                :custom_config, :risk_preference, :investment_amount,
                :investment_horizon, :rebalance_frequency, :constraints,
                :status, :performance_target
            )
            """

            with self.engine.connect() as conn:
                conn.execute(
                    text(insert_sql),
                    {
                        "strategy_id": user_strategy.strategy_id,
                        "user_id": user_strategy.user_id,
                        "strategy_name": user_strategy.strategy_name,
                        "template_id": user_strategy.template_id,
                        "custom_config": json.dumps(user_strategy.custom_config),
                        "risk_preference": user_strategy.risk_preference,
                        "investment_amount": user_strategy.investment_amount,
                        "investment_horizon": user_strategy.investment_horizon,
                        "rebalance_frequency": user_strategy.rebalance_frequency,
                        "constraints": json.dumps(user_strategy.constraints),
                        "status": user_strategy.status,
                        "performance_target": json.dumps(user_strategy.performance_target),
                    },
                )
                conn.commit()

        except Exception as e:
            logger.error(f"保存用户策略失败: {e}")
            raise

    def _save_strategy_recommendation(self, recommendation: StrategyRecommendation):
        """保存策略推荐"""
        try:
            insert_sql = """
            INSERT INTO strategy_recommendations (
                recommendation_id, user_id, recommended_strategies,
                recommendation_reason, confidence_score, risk_assessment,
                expected_performance
            ) VALUES (
                :recommendation_id, :user_id, :recommended_strategies,
                :recommendation_reason, :confidence_score, :risk_assessment,
                :expected_performance
            )
            """

            with self.engine.connect() as conn:
                conn.execute(
                    text(insert_sql),
                    {
                        "recommendation_id": recommendation.recommendation_id,
                        "user_id": recommendation.user_id,
                        "recommended_strategies": json.dumps(recommendation.recommended_strategies, default=str),
                        "recommendation_reason": recommendation.recommendation_reason,
                        "confidence_score": recommendation.confidence_score,
                        "risk_assessment": json.dumps(recommendation.risk_assessment),
                        "expected_performance": json.dumps(recommendation.expected_performance, default=str),
                    },
                )
                conn.commit()

        except Exception as e:
            logger.error(f"保存策略推荐失败: {e}")
            raise

    def get_user_strategies(self, user_id: str, status: str = None) -> List[UserStrategy]:
        """获取用户策略列表

        Args:
            user_id: 用户ID
            status: 策略状态

        Returns:
            用户策略列表
        """
        try:
            where_clause = "WHERE user_id = :user_id"
            params = {"user_id": user_id}

            if status:
                where_clause += " AND status = :status"
                params["status"] = status

            query_sql = f"""
            SELECT *
            FROM user_strategies
            {where_clause}
            ORDER BY created_at DESC
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), params)
                rows = result.fetchall()

            strategies = []
            for row in rows:
                strategy = UserStrategy(
                    strategy_id=row[1],
                    user_id=row[2],
                    strategy_name=row[3],
                    template_id=row[4],
                    custom_config=json.loads(row[5]),
                    risk_preference=row[6],
                    investment_amount=float(row[7]) if row[7] else 0,
                    investment_horizon=row[8],
                    rebalance_frequency=row[9],
                    constraints=json.loads(row[10]) if row[10] else {},
                    status=row[11],
                    performance_target=json.loads(row[12]) if row[12] else {},
                    created_at=row[13],
                    updated_at=row[14],
                )
                strategies.append(strategy)

            return strategies

        except Exception as e:
            logger.error(f"获取用户策略失败: {e}")
            return []

    def backtest_user_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """回测用户策略

        Args:
            strategy_id: 策略ID

        Returns:
            回测结果
        """
        try:
            # 获取用户策略
            user_strategy = self.get_user_strategy(strategy_id)
            if not user_strategy:
                logger.error(f"用户策略不存在: {strategy_id}")
                return None

            # 创建回测配置
            backtest_config = BacktestConfig(
                strategy_name=user_strategy.strategy_name,
                model_name=user_strategy.custom_config.get("model_name", "default_model"),
                model_version=user_strategy.custom_config.get("model_version"),
                initial_capital=user_strategy.investment_amount,
                **{k: v for k, v in user_strategy.custom_config.items() if k in BacktestConfig.__dataclass_fields__},
            )

            # 运行回测
            backtest_result = self.backtest_engine.run_backtest(backtest_config)

            if backtest_result:
                return {
                    "strategy_id": strategy_id,
                    "backtest_result": backtest_result,
                    "performance_summary": {
                        "total_return": backtest_result.total_return,
                        "annualized_return": backtest_result.annualized_return,
                        "volatility": backtest_result.volatility,
                        "max_drawdown": backtest_result.max_drawdown,
                        "sharpe_ratio": backtest_result.sharpe_ratio,
                        "win_rate": backtest_result.win_rate,
                    },
                }

            return None

        except Exception as e:
            logger.error(f"回测用户策略失败: {e}")
            return None

    def get_user_strategy(self, strategy_id: str) -> Optional[UserStrategy]:
        """获取单个用户策略

        Args:
            strategy_id: 策略ID

        Returns:
            用户策略对象
        """
        try:
            query_sql = """
            SELECT *
            FROM user_strategies
            WHERE strategy_id = :strategy_id
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {"strategy_id": strategy_id})
                row = result.fetchone()

            if not row:
                return None

            strategy = UserStrategy(
                strategy_id=row[1],
                user_id=row[2],
                strategy_name=row[3],
                template_id=row[4],
                custom_config=json.loads(row[5]),
                risk_preference=row[6],
                investment_amount=float(row[7]) if row[7] else 0,
                investment_horizon=row[8],
                rebalance_frequency=row[9],
                constraints=json.loads(row[10]) if row[10] else {},
                status=row[11],
                performance_target=json.loads(row[12]) if row[12] else {},
                created_at=row[13],
                updated_at=row[14],
            )

            return strategy

        except Exception as e:
            logger.error(f"获取用户策略失败: {e}")
            return None

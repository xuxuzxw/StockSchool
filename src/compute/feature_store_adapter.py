import hashlib
import inspect
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.compute.factor_engine import FactorEngine
from src.compute.fundamental_engine import FundamentalFactorEngine
from src.compute.sentiment_engine import SentimentFactorEngine
from src.compute.technical_engine import TechnicalFactorEngine
from src.config.unified_config import config
from src.features.factor_feature_store import FactorFeatureStore, FactorMetadata
from src.utils.db import get_db_engine

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征商店适配器
将因子计算引擎与特征商店集成，自动管理因子版本和数据存储
"""


# 配置日志
logger = logging.getLogger(__name__)


class FeatureStoreAdapter:
    """特征商店适配器"""

    def __init__(self, engine=None):
        """
        初始化适配器

        Args:
            engine: 数据库引擎
        """
        self.engine = engine or get_db_engine()
        self.feature_store = FactorFeatureStore(self.engine)
        self.factor_engine = FactorEngine(self.engine)
        self.technical_engine = TechnicalFactorEngine(self.engine)
        self.fundamental_engine = FundamentalFactorEngine(self.engine)
        self.sentiment_engine = SentimentFactorEngine(self.engine)

        # 因子引擎映射
        self.engine_mapping = {
            "technical": self.technical_engine,
            "fundamental": self.fundamental_engine,
            "sentiment": self.sentiment_engine,
        }

        # 自动注册所有因子
        self._auto_register_factors()

    def _auto_register_factors(self):
        """自动注册所有因子元数据"""
        try:
            # 获取所有因子定义
            factor_definitions = self._get_factor_definitions()

            for factor_def in factor_definitions:
                try:
                    metadata = FactorMetadata(
                        factor_name=factor_def["factor_name"],
                        factor_type=factor_def["factor_type"],
                        category=factor_def["category"],
                        description=factor_def["description"],
                        formula=factor_def["formula"],
                        parameters=factor_def["parameters"],
                        data_requirements=factor_def["data_requirements"],
                        update_frequency=factor_def["update_frequency"],
                        data_schema=self._infer_data_schema(factor_def["factor_name"]),
                        tags=self._generate_tags(factor_def),
                    )

                    self.feature_store.register_factor(metadata)

                except Exception as e:
                    logger.warning(f"注册因子 {factor_def['factor_name']} 失败: {e}")
                    continue

            logger.info("因子元数据自动注册完成")

        except Exception as e:
            logger.error(f"自动注册因子失败: {e}")

    def _get_factor_definitions(self) -> List[Dict[str, Any]]:
        """获取因子定义"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    """
                    SELECT factor_name, factor_type, category, description, formula,
                           parameters, data_requirements, update_frequency
                    FROM factor_definitions
                    WHERE is_active = true
                """
                )

                definitions = []
                for row in result:
                    definitions.append(
                        {
                            "factor_name": row.factor_name,
                            "factor_type": row.factor_type,
                            "category": row.category,
                            "description": row.description,
                            "formula": row.formula,
                            "parameters": row.parameters,
                            "data_requirements": row.data_requirements,
                            "update_frequency": row.update_frequency,
                        }
                    )

                return definitions

        except Exception as e:
            logger.error(f"获取因子定义失败: {e}")
            return []

    def _infer_data_schema(self, factor_name: str) -> Dict[str, str]:
        """推断因子数据结构"""
        base_schema = {"ts_code": "string", "factor_date": "date", factor_name: "float"}

        # 根据因子类型添加额外字段
        if "macd" in factor_name.lower():
            if "signal" in factor_name:
                base_schema[factor_name] = "float"
            elif "hist" in factor_name:
                base_schema[factor_name] = "float"
            else:
                base_schema[factor_name] = "float"

        return base_schema

    def _generate_tags(self, factor_def: Dict[str, Any]) -> List[str]:
        """生成因子标签"""
        tags = [factor_def["factor_type"], factor_def["category"]]

        # 根据因子名称添加特定标签
        factor_name = factor_def["factor_name"].lower()

        if "sma" in factor_name or "ema" in factor_name:
            tags.append("moving_average")
        if "rsi" in factor_name:
            tags.append("oscillator")
        if "macd" in factor_name:
            tags.append("momentum")
        if "bollinger" in factor_name:
            tags.append("volatility_band")
        if "volume" in factor_name:
            tags.append("volume_based")
        if "pe" in factor_name or "pb" in factor_name:
            tags.append("valuation")
        if "roe" in factor_name or "roa" in factor_name:
            tags.append("profitability")

        return list(set(tags))  # 去重

    def calculate_and_store_factor(
        self,
        factor_name: str,
        ts_codes: List[str] = None,
        start_date: date = None,
        end_date: date = None,
        force_new_version: bool = False,
    ) -> Tuple[bool, str]:
        """
        计算并存储因子数据

        Args:
            factor_name: 因子名称
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            force_new_version: 是否强制创建新版本

        Returns:
            Tuple[bool, str]: (是否成功, 版本ID)
        """
        try:
            # 获取因子元数据
            metadata = self.feature_store.get_factor_metadata(factor_name)
            if not metadata:
                raise ValueError(f"因子 {factor_name} 未注册")

            # 获取对应的计算引擎
            engine = self.engine_mapping.get(metadata.factor_type)
            if not engine:
                raise ValueError(f"不支持的因子类型: {metadata.factor_type}")

            # 获取计算方法
            calc_method = getattr(engine, f"calculate_{factor_name}", None)
            if not calc_method:
                # 尝试通用计算方法
                calc_method = getattr(engine, "calculate_factor", None)
                if not calc_method:
                    raise ValueError(f"找不到因子 {factor_name} 的计算方法")

            # 生成算法代码哈希
            algorithm_code = self._get_algorithm_code(calc_method, metadata.parameters)

            # 检查是否需要创建新版本
            version_id = None
            if not force_new_version:
                # 尝试找到现有版本
                versions = self.feature_store.get_factor_versions(factor_name)
                if versions:
                    # 检查算法是否有变化
                    latest_version = versions[0]
                    existing_hash = self._generate_algorithm_hash(algorithm_code, metadata.parameters)

                    # 这里简化处理，实际应该比较算法哈希
                    version_id = latest_version.version_id

            # 如果没有找到合适的版本，创建新版本
            if not version_id or force_new_version:
                version_id = self.feature_store.create_factor_version(
                    factor_name=factor_name,
                    algorithm_code=algorithm_code,
                    parameters=metadata.parameters,
                    metadata={
                        "calculation_date": datetime.now().isoformat(),
                        "data_range": {
                            "start_date": start_date.isoformat() if start_date else None,
                            "end_date": end_date.isoformat() if end_date else None,
                        },
                        "stock_count": len(ts_codes) if ts_codes else None,
                    },
                )

            # 执行因子计算
            if hasattr(calc_method, "__self__") and hasattr(calc_method.__self__, "calculate_factor"):
                # 使用通用计算方法
                factor_data = calc_method.__self__.calculate_factor(
                    factor_name=factor_name, ts_codes=ts_codes, start_date=start_date, end_date=end_date
                )
            else:
                # 使用特定计算方法
                factor_data = calc_method(ts_codes=ts_codes, start_date=start_date, end_date=end_date)

            if factor_data.empty:
                logger.warning(f"因子 {factor_name} 计算结果为空")
                return False, version_id

            # 存储到特征商店
            success = self.feature_store.store_factor_data(
                factor_name=factor_name,
                version_id=version_id,
                data=factor_data,
                partition_date=end_date or date.today(),
            )

            if success:
                logger.info(f"因子 {factor_name} 计算并存储成功，版本: {version_id}")
                return True, version_id
            else:
                logger.error(f"因子 {factor_name} 存储失败")
                return False, version_id

        except Exception as e:
            logger.error(f"计算并存储因子 {factor_name} 失败: {e}")
            return False, ""

    def batch_calculate_and_store(
        self, factor_names: List[str], ts_codes: List[str] = None, start_date: date = None, end_date: date = None
    ) -> Dict[str, Tuple[bool, str]]:
        """
        批量计算并存储因子

        Args:
            factor_names: 因子名称列表
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict[str, Tuple[bool, str]]: 因子名称 -> (是否成功, 版本ID)
        """
        results = {}

        for factor_name in factor_names:
            try:
                success, version_id = self.calculate_and_store_factor(
                    factor_name=factor_name, ts_codes=ts_codes, start_date=start_date, end_date=end_date
                )
                results[factor_name] = (success, version_id)

            except Exception as e:
                logger.error(f"批量计算因子 {factor_name} 失败: {e}")
                results[factor_name] = (False, "")

        return results

    def get_factor_data_with_version(
        self,
        factor_name: str,
        version_id: str = None,
        ts_codes: List[str] = None,
        start_date: date = None,
        end_date: date = None,
    ) -> pd.DataFrame:
        """
        获取带版本信息的因子数据

        Args:
            factor_name: 因子名称
            version_id: 版本ID
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: 因子数据
        """
        try:
            return self.feature_store.get_factor_data(
                factor_name=factor_name,
                version_id=version_id,
                ts_codes=ts_codes,
                start_date=start_date,
                end_date=end_date,
            )

        except Exception as e:
            logger.error(f"获取因子数据失败: {e}")
            return pd.DataFrame()

    def compare_factor_versions(
        self,
        factor_name: str,
        version_id1: str,
        version_id2: str,
        ts_codes: List[str] = None,
        start_date: date = None,
        end_date: date = None,
    ) -> Dict[str, Any]:
        """
        比较因子版本差异

        Args:
            factor_name: 因子名称
            version_id1: 版本1 ID
            version_id2: 版本2 ID
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict[str, Any]: 比较结果
        """
        try:
            # 获取两个版本的数据
            data1 = self.feature_store.get_factor_data(
                factor_name=factor_name,
                version_id=version_id1,
                ts_codes=ts_codes,
                start_date=start_date,
                end_date=end_date,
            )

            data2 = self.feature_store.get_factor_data(
                factor_name=factor_name,
                version_id=version_id2,
                ts_codes=ts_codes,
                start_date=start_date,
                end_date=end_date,
            )

            if data1.empty or data2.empty:
                return {"comparison_possible": False, "reason": "其中一个版本的数据为空"}

            # 合并数据进行比较
            merged = pd.merge(
                data1[["ts_code", "factor_date", factor_name]],
                data2[["ts_code", "factor_date", factor_name]],
                on=["ts_code", "factor_date"],
                suffixes=("_v1", "_v2"),
                how="inner",
            )

            if merged.empty:
                return {"comparison_possible": False, "reason": "两个版本没有共同的数据点"}

            # 计算统计指标
            factor_v1 = merged[f"{factor_name}_v1"]
            factor_v2 = merged[f"{factor_name}_v2"]

            # 相关性
            correlation = factor_v1.corr(factor_v2)

            # 差异统计
            diff = factor_v1 - factor_v2
            mean_diff = diff.mean()
            std_diff = diff.std()
            max_diff = diff.abs().max()

            # 相对差异
            relative_diff = diff / factor_v1.abs()
            mean_relative_diff = relative_diff.mean()

            return {
                "comparison_possible": True,
                "data_points": len(merged),
                "correlation": float(correlation) if not pd.isna(correlation) else None,
                "mean_difference": float(mean_diff) if not pd.isna(mean_diff) else None,
                "std_difference": float(std_diff) if not pd.isna(std_diff) else None,
                "max_absolute_difference": float(max_diff) if not pd.isna(max_diff) else None,
                "mean_relative_difference": float(mean_relative_diff) if not pd.isna(mean_relative_diff) else None,
                "version_1_stats": {
                    "mean": float(factor_v1.mean()),
                    "std": float(factor_v1.std()),
                    "min": float(factor_v1.min()),
                    "max": float(factor_v1.max()),
                },
                "version_2_stats": {
                    "mean": float(factor_v2.mean()),
                    "std": float(factor_v2.std()),
                    "min": float(factor_v2.min()),
                    "max": float(factor_v2.max()),
                },
            }

        except Exception as e:
            logger.error(f"比较因子版本失败: {e}")
            return {"comparison_possible": False, "reason": f"比较失败: {str(e)}"}

    def migrate_legacy_data(self, factor_name: str, source_table: str = None) -> bool:
        """
        迁移历史因子数据到特征商店

        Args:
            factor_name: 因子名称
            source_table: 源数据表名

        Returns:
            bool: 迁移是否成功
        """
        try:
            # 确定源表
            if not source_table:
                # 根据因子类型确定默认表
                metadata = self.feature_store.get_factor_metadata(factor_name)
                if not metadata:
                    raise ValueError(f"因子 {factor_name} 未注册")

                if metadata.factor_type == "technical":
                    source_table = "stock_factors_technical"
                elif metadata.factor_type == "fundamental":
                    source_table = "stock_factors_fundamental"
                elif metadata.factor_type == "sentiment":
                    source_table = "stock_factors_sentiment"
                else:
                    source_table = "stock_factors_wide"

            # 从源表读取数据
            with self.engine.connect() as conn:
                query = f"""
                    SELECT ts_code, factor_date, {factor_name}
                    FROM {source_table}
                    WHERE {factor_name} IS NOT NULL
                    ORDER BY factor_date, ts_code
                """

                legacy_data = pd.read_sql(query, conn)

            if legacy_data.empty:
                logger.warning(f"源表 {source_table} 中没有因子 {factor_name} 的数据")
                return True

            # 创建迁移版本
            version_id = self.feature_store.create_factor_version(
                factor_name=factor_name,
                algorithm_code="legacy_migration",
                parameters={"source_table": source_table},
                metadata={
                    "migration_date": datetime.now().isoformat(),
                    "source_table": source_table,
                    "data_count": len(legacy_data),
                    "date_range": {
                        "start_date": legacy_data["factor_date"].min().isoformat(),
                        "end_date": legacy_data["factor_date"].max().isoformat(),
                    },
                },
            )

            # 存储数据
            success = self.feature_store.store_factor_data(
                factor_name=factor_name, version_id=version_id, data=legacy_data
            )

            if success:
                logger.info(f"因子 {factor_name} 历史数据迁移成功，版本: {version_id}")
                return True
            else:
                logger.error(f"因子 {factor_name} 历史数据存储失败")
                return False

        except Exception as e:
            logger.error(f"迁移历史数据失败: {e}")
            return False

    def _get_algorithm_code(self, calc_method, parameters: Dict[str, Any]) -> str:
        """获取算法代码"""
        try:
            # 获取方法源代码
            source_code = inspect.getsource(calc_method)

            # 添加参数信息
            param_str = str(sorted(parameters.items()))

            return f"{source_code}\n# Parameters: {param_str}"

        except Exception as e:
            logger.warning(f"获取算法代码失败: {e}")
            return f"method_{calc_method.__name__}_params_{parameters}"

    def _generate_algorithm_hash(self, algorithm_code: str, parameters: Dict[str, Any]) -> str:
        """生成算法哈希"""
        content = f"{algorithm_code}_{str(sorted(parameters.items()))}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_adapter_statistics(self) -> Dict[str, Any]:
        """获取适配器统计信息"""
        try:
            # 获取特征商店统计
            store_stats = self.feature_store.get_storage_statistics()

            # 获取计算引擎统计
            engine_stats = {}
            for engine_type, engine in self.engine_mapping.items():
                if hasattr(engine, "get_statistics"):
                    engine_stats[engine_type] = engine.get_statistics()
                else:
                    engine_stats[engine_type] = {"status": "available"}

            return {
                "feature_store": store_stats,
                "calculation_engines": engine_stats,
                "adapter_info": {
                    "registered_engines": list(self.engine_mapping.keys()),
                    "last_updated": datetime.now().isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"获取适配器统计失败: {e}")
            return {}

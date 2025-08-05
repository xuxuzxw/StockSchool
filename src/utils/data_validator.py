import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证工具
提供因子计算过程中的数据质量检查和验证功能
"""


class ValidationLevel(Enum):
    """验证级别"""

    STRICT = "strict"  # 严格模式，任何问题都报错
    WARNING = "warning"  # 警告模式，记录问题但继续
    IGNORE = "ignore"  # 忽略模式，不进行验证


@dataclass
class ValidationResult:
    """验证结果"""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: Dict[str, Any]

    def add_error(self, message: str):
        """添加错误"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """添加警告"""
        self.warnings.append(message)

    def add_info(self, key: str, value: Any):
        """添加信息"""
        self.info[key] = value

    def summary(self) -> str:
        """获取验证结果摘要"""
        status = "通过" if self.is_valid else "失败"
        return f"验证{status} - 错误: {len(self.errors)}, 警告: {len(self.warnings)}"


class DataValidator:
    """数据验证器"""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.WARNING):
        """方法描述"""
        self.result = ValidationResult(True, [], [], {})

    def reset(self):
        """重置验证结果"""
        self.result = ValidationResult(True, [], [], {})

    def validate_dataframe(self, df: pd.DataFrame, name: str = "DataFrame") -> ValidationResult:
        """验证DataFrame基本属性"""
        self.reset()

        if self.validation_level == ValidationLevel.IGNORE:
            return self.result

        # 检查是否为空
        if df is None:
            self.result.add_error(f"{name}为None")
            return self.result

        if df.empty:
            self.result.add_warning(f"{name}为空")

        # 基本信息
        self.result.add_info("行数", len(df))
        self.result.add_info("列数", len(df.columns))
        self.result.add_info("内存使用", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB")

        # 检查重复行
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            message = f"{name}包含{duplicate_count}行重复数据"
            if self.validation_level == ValidationLevel.STRICT:
                self.result.add_error(message)
            else:
                self.result.add_warning(message)

        # 检查缺失值
        missing_info = df.isnull().sum()
        total_missing = missing_info.sum()

        if total_missing > 0:
            missing_ratio = total_missing / (len(df) * len(df.columns))
            message = f"{name}包含{total_missing}个缺失值 ({missing_ratio:.2%})"

            if missing_ratio > 0.5:  # 超过50%缺失
                if self.validation_level == ValidationLevel.STRICT:
                    self.result.add_error(message)
                else:
                    self.result.add_warning(message)
            elif missing_ratio > 0.1:  # 超过10%缺失
                self.result.add_warning(message)

            # 记录每列的缺失情况
            for col, count in missing_info.items():
                if count > 0:
                    self.result.add_info(f"缺失值_{col}", count)

        return self.result

    def validate_stock_data(self, df: pd.DataFrame) -> ValidationResult:
        """验证股票数据"""
        self.reset()

        if self.validation_level == ValidationLevel.IGNORE:
            return self.result

        # 基本验证
        basic_result = self.validate_dataframe(df, "股票数据")
        if not basic_result.is_valid and self.validation_level == ValidationLevel.STRICT:
            return basic_result

        # 检查必需列
        required_columns = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            message = f"缺少必需列: {missing_columns}"
            if self.validation_level == ValidationLevel.STRICT:
                self.result.add_error(message)
            else:
                self.result.add_warning(message)

        if df.empty:
            return self.result

        # 检查价格数据的合理性
        price_columns = ["open", "high", "low", "close"]
        available_price_cols = [col for col in price_columns if col in df.columns]

        for col in available_price_cols:
            if col in df.columns:
                # 检查负价格
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    message = f"{col}列包含{negative_count}个负值"
                    if self.validation_level == ValidationLevel.STRICT:
                        self.result.add_error(message)
                    else:
                        self.result.add_warning(message)

                # 检查异常大的价格变动
                if len(df) > 1:
                    price_change = df[col].pct_change().abs()
                    extreme_changes = (price_change > 0.5).sum()  # 超过50%的变动
                    if extreme_changes > 0:
                        self.result.add_warning(f"{col}列包含{extreme_changes}个极端价格变动(>50%)")

        # 检查高开低收的逻辑关系
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            # high应该是最高价
            high_logic_errors = (
                (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])
            ).sum()

            # low应该是最低价
            low_logic_errors = ((df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])).sum()

            if high_logic_errors > 0:
                message = f"发现{high_logic_errors}个最高价逻辑错误"
                if self.validation_level == ValidationLevel.STRICT:
                    self.result.add_error(message)
                else:
                    self.result.add_warning(message)

            if low_logic_errors > 0:
                message = f"发现{low_logic_errors}个最低价逻辑错误"
                if self.validation_level == ValidationLevel.STRICT:
                    self.result.add_error(message)
                else:
                    self.result.add_warning(message)

        # 检查成交量
        if "vol" in df.columns:
            negative_vol = (df["vol"] < 0).sum()
            if negative_vol > 0:
                message = f"成交量包含{negative_vol}个负值"
                if self.validation_level == ValidationLevel.STRICT:
                    self.result.add_error(message)
                else:
                    self.result.add_warning(message)

        # 检查日期格式和连续性
        if "trade_date" in df.columns:
            try:
                # 尝试转换日期
                dates = pd.to_datetime(df["trade_date"])
                self.result.add_info("日期范围", f"{dates.min()} 到 {dates.max()}")

                # 检查日期是否有序
                if not dates.is_monotonic_increasing:
                    self.result.add_warning("交易日期不是递增顺序")

                # 检查是否有重复日期（同一股票）
                if "ts_code" in df.columns:
                    duplicate_dates = df.groupby("ts_code")["trade_date"].apply(lambda x: x.duplicated().sum()).sum()

                    if duplicate_dates > 0:
                        message = f"发现{duplicate_dates}个重复的交易日期"
                        if self.validation_level == ValidationLevel.STRICT:
                            self.result.add_error(message)
                        else:
                            self.result.add_warning(message)

            except Exception as e:
                message = f"日期格式验证失败: {str(e)}"
                if self.validation_level == ValidationLevel.STRICT:
                    self.result.add_error(message)
                else:
                    self.result.add_warning(message)

        return self.result

    def validate_factor_data(self, df: pd.DataFrame, factor_type: str = "因子") -> ValidationResult:
        """验证因子数据"""
        self.reset()

        if self.validation_level == ValidationLevel.IGNORE:
            return self.result

        # 基本验证
        basic_result = self.validate_dataframe(df, f"{factor_type}数据")
        if not basic_result.is_valid and self.validation_level == ValidationLevel.STRICT:
            return basic_result

        if df.empty:
            return self.result

        # 检查因子值的分布
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col in ["ts_code", "trade_date"]:  # 跳过标识列
                continue

            series = df[col].dropna()
            if len(series) == 0:
                continue

            # 检查无穷值
            inf_count = np.isinf(series).sum()
            if inf_count > 0:
                message = f"{col}包含{inf_count}个无穷值"
                if self.validation_level == ValidationLevel.STRICT:
                    self.result.add_error(message)
                else:
                    self.result.add_warning(message)

            # 检查异常值（使用IQR方法）
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = ((series < lower_bound) | (series > upper_bound)).sum()

                if outliers > 0:
                    outlier_ratio = outliers / len(series)
                    if outlier_ratio > 0.1:  # 超过10%的异常值
                        self.result.add_warning(f"{col}包含{outliers}个异常值 ({outlier_ratio:.2%})")

            # 记录统计信息
            self.result.add_info(f"{col}_均值", round(series.mean(), 4))
            self.result.add_info(f"{col}_标准差", round(series.std(), 4))
            self.result.add_info(f"{col}_范围", f"[{series.min():.4f}, {series.max():.4f}]")

        return self.result

    def validate_financial_data(self, df: pd.DataFrame) -> ValidationResult:
        """验证财务数据"""
        self.reset()

        if self.validation_level == ValidationLevel.IGNORE:
            return self.result

        # 基本验证
        basic_result = self.validate_dataframe(df, "财务数据")
        if not basic_result.is_valid and self.validation_level == ValidationLevel.STRICT:
            return basic_result

        if df.empty:
            return self.result

        # 检查必需列
        required_columns = ["ts_code", "end_date"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            message = f"缺少必需列: {missing_columns}"
            if self.validation_level == ValidationLevel.STRICT:
                self.result.add_error(message)
            else:
                self.result.add_warning(message)

        # 检查财务指标的合理性
        financial_checks = {
            "roe": {"min": -1, "max": 1, "name": "ROE"},
            "roa": {"min": -1, "max": 1, "name": "ROA"},
            "debt_to_assets": {"min": 0, "max": 10, "name": "资产负债率"},
            "current_ratio": {"min": 0, "max": 100, "name": "流动比率"},
            "quick_ratio": {"min": 0, "max": 100, "name": "速动比率"},
        }

        for col, check in financial_checks.items():
            if col in df.columns:
                series = df[col].dropna()
                if len(series) == 0:
                    continue

                # 检查范围
                out_of_range = ((series < check["min"]) | (series > check["max"])).sum()
                if out_of_range > 0:
                    self.result.add_warning(f"{check['name']}包含{out_of_range}个超出合理范围的值")

        return self.result

    def validate_data_consistency(self, data_dict: Dict[str, pd.DataFrame]) -> ValidationResult:
        """验证多个数据集之间的一致性"""
        self.reset()

        if self.validation_level == ValidationLevel.IGNORE:
            return self.result

        if len(data_dict) < 2:
            self.result.add_warning("数据集少于2个，无法进行一致性检查")
            return self.result

        # 检查股票代码一致性
        stock_codes = {}
        for name, df in data_dict.items():
            if "ts_code" in df.columns:
                stock_codes[name] = set(df["ts_code"].unique())

        if len(stock_codes) > 1:
            all_codes = set.union(*stock_codes.values())
            common_codes = set.intersection(*stock_codes.values())

            self.result.add_info("总股票数", len(all_codes))
            self.result.add_info("共同股票数", len(common_codes))

            if len(common_codes) < len(all_codes) * 0.8:  # 共同股票少于80%
                self.result.add_warning("数据集之间的股票代码重叠度较低")

        # 检查日期范围一致性
        date_ranges = {}
        for name, df in data_dict.items():
            date_col = None
            for col in ["trade_date", "end_date", "ann_date"]:
                if col in df.columns:
                    date_col = col
                    break

            if date_col:
                try:
                    dates = pd.to_datetime(df[date_col])
                    date_ranges[name] = (dates.min(), dates.max())
                except:
                    continue

        if len(date_ranges) > 1:
            all_starts = [r[0] for r in date_ranges.values()]
            all_ends = [r[1] for r in date_ranges.values()]

            earliest_start = min(all_starts)
            latest_start = max(all_starts)
            earliest_end = min(all_ends)
            latest_end = max(all_ends)

            self.result.add_info("最早开始日期", earliest_start)
            self.result.add_info("最晚开始日期", latest_start)
            self.result.add_info("最早结束日期", earliest_end)
            self.result.add_info("最晚结束日期", latest_end)

            # 检查日期范围差异
            start_diff = (latest_start - earliest_start).days
            end_diff = (latest_end - earliest_end).days

            if start_diff > 365:  # 开始日期相差超过1年
                self.result.add_warning(f"数据集开始日期相差{start_diff}天")

            if end_diff > 90:  # 结束日期相差超过3个月
                self.result.add_warning(f"数据集结束日期相差{end_diff}天")

        return self.result


class FactorValidator(DataValidator):
    """因子专用验证器"""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.WARNING):
        """方法描述"""
        self.factor_rules = self._load_factor_rules()

    def _load_factor_rules(self) -> Dict[str, Dict[str, Any]]:
        """加载因子验证规则"""
        return {
            # 技术因子规则
            "rsi": {"min": 0, "max": 100, "type": "momentum"},
            "macd": {"min": -np.inf, "max": np.inf, "type": "momentum"},
            "bollinger_position": {"min": 0, "max": 1, "type": "trend"},
            "volatility": {"min": 0, "max": np.inf, "type": "volatility"},
            "volume_ratio": {"min": 0, "max": np.inf, "type": "volume"},
            # 基本面因子规则
            "pe_ratio": {"min": 0, "max": 1000, "type": "valuation"},
            "pb_ratio": {"min": 0, "max": 100, "type": "valuation"},
            "roe": {"min": -1, "max": 1, "type": "profitability"},
            "debt_ratio": {"min": 0, "max": 10, "type": "leverage"},
            "revenue_growth": {"min": -1, "max": 10, "type": "growth"},
        }

    def validate_factor_values(self, df: pd.DataFrame, factor_name: str) -> ValidationResult:
        """验证特定因子的值"""
        self.reset()

        if self.validation_level == ValidationLevel.IGNORE:
            return self.result

        if factor_name not in df.columns:
            self.result.add_error(f"因子{factor_name}不存在于数据中")
            return self.result

        series = df[factor_name].dropna()
        if len(series) == 0:
            self.result.add_warning(f"因子{factor_name}没有有效值")
            return self.result

        # 应用因子特定规则
        if factor_name in self.factor_rules:
            rule = self.factor_rules[factor_name]

            # 检查范围
            if "min" in rule and rule["min"] != -np.inf:
                below_min = (series < rule["min"]).sum()
                if below_min > 0:
                    message = f"因子{factor_name}有{below_min}个值低于最小值{rule['min']}"
                    if self.validation_level == ValidationLevel.STRICT:
                        self.result.add_error(message)
                    else:
                        self.result.add_warning(message)

            if "max" in rule and rule["max"] != np.inf:
                above_max = (series > rule["max"]).sum()
                if above_max > 0:
                    message = f"因子{factor_name}有{above_max}个值高于最大值{rule['max']}"
                    if self.validation_level == ValidationLevel.STRICT:
                        self.result.add_error(message)
                    else:
                        self.result.add_warning(message)

        # 通用检查
        # 检查因子分布
        unique_ratio = len(series.unique()) / len(series)
        if unique_ratio < 0.1:  # 唯一值比例低于10%
            self.result.add_warning(f"因子{factor_name}的唯一值比例较低({unique_ratio:.2%})")

        # 检查极端值集中度
        extreme_threshold = 3  # 3倍标准差
        mean_val = series.mean()
        std_val = series.std()

        if std_val > 0:
            extreme_count = (np.abs(series - mean_val) > extreme_threshold * std_val).sum()
            if extreme_count > len(series) * 0.05:  # 超过5%的极端值
                self.result.add_warning(
                    f"因子{factor_name}包含{extreme_count}个极端值({extreme_count/len(series):.2%})"
                )

        return self.result


# 便捷函数
def validate_stock_data(
    df: pd.DataFrame, validation_level: ValidationLevel = ValidationLevel.WARNING
) -> ValidationResult:
    """验证股票数据的便捷函数"""
    validator = DataValidator(validation_level)
    return validator.validate_stock_data(df)


def validate_factor_data(
    df: pd.DataFrame, factor_type: str = "因子", validation_level: ValidationLevel = ValidationLevel.WARNING
) -> ValidationResult:
    """验证因子数据的便捷函数"""
    validator = DataValidator(validation_level)
    return validator.validate_factor_data(df, factor_type)


def validate_financial_data(
    df: pd.DataFrame, validation_level: ValidationLevel = ValidationLevel.WARNING
) -> ValidationResult:
    """验证财务数据的便捷函数"""
    validator = DataValidator(validation_level)
    return validator.validate_financial_data(df)


# 示例用法
if __name__ == "__main__":
    # 创建测试数据
    test_data = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 100,
            "trade_date": pd.date_range("2023-01-01", periods=100),
            "open": np.random.uniform(10, 20, 100),
            "high": np.random.uniform(15, 25, 100),
            "low": np.random.uniform(5, 15, 100),
            "close": np.random.uniform(10, 20, 100),
            "vol": np.random.uniform(1000, 10000, 100),
        }
    )

    # 验证股票数据
    result = validate_stock_data(test_data)
    print(result.summary())

    if result.errors:
        print("错误:")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print("警告:")
        for warning in result.warnings:
            print(f"  - {warning}")

    print("\n信息:")
    for key, value in result.info.items():
        print(f"  {key}: {value}")

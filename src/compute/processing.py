#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子预处理模块

实现因子标准化、去极值、中性化等预处理功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sqlalchemy import create_engine, text
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

from src.utils.config_loader import config
from src.utils.db import get_db_engine
from src.utils.retry import idempotent_retry

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorProcessor:
    """因子预处理器
    
    提供因子标准化、去极值、中性化等预处理功能
    """
    
    def __init__(self):
        """初始化因子预处理器"""
        self.engine = get_db_engine()
        self.processing_config = config.get('factor_processing', {})
        logger.info("因子预处理器初始化完成")
    
    def load_factor_data(self, 
                        factor_names: List[str],
                        start_date: str,
                        end_date: str,
                        stock_list: Optional[List[str]] = None) -> pd.DataFrame:
        """加载因子数据
        
        Args:
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            stock_list: 股票列表，为空则加载所有股票
            
        Returns:
            因子数据DataFrame
        """
        logger.info(f"加载因子数据: {factor_names}, {start_date} 到 {end_date}")
        
        # 构建查询SQL
        query = """
            SELECT ts_code, trade_date, factor_name, factor_value
            FROM factor_library
            WHERE factor_name = ANY(:factor_names)
            AND trade_date BETWEEN :start_date AND :end_date
        """
        
        params = {
            'factor_names': factor_names,
            'start_date': start_date,
            'end_date': end_date
        }
        
        if stock_list:
            query += " AND ts_code = ANY(:stock_list)"
            params['stock_list'] = stock_list
            
        query += " ORDER BY trade_date, ts_code, factor_name"
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn, params=params)
                
            if df.empty:
                logger.warning("未找到因子数据")
                return pd.DataFrame()
                
            # 透视表转换
            factor_df = df.pivot_table(
                index=['ts_code', 'trade_date'],
                columns='factor_name',
                values='factor_value',
                aggfunc='first'
            ).reset_index()
            
            # 重命名列
            factor_df.columns.name = None
            
            logger.info(f"加载完成，共 {len(factor_df)} 条记录，{len(factor_names)} 个因子")
            return factor_df
            
        except Exception as e:
            logger.error(f"加载因子数据失败: {str(e)}")
            raise
    
    def load_stock_basic_info(self, stock_list: Optional[List[str]] = None) -> pd.DataFrame:
        """加载股票基础信息
        
        Args:
            stock_list: 股票列表
            
        Returns:
            股票基础信息DataFrame
        """
        query = """
            SELECT ts_code, industry, market, list_date
            FROM stock_basic
            WHERE list_status = 'L'
        """
        
        params = {}
        if stock_list:
            query += " AND ts_code = ANY(:stock_list)"
            params['stock_list'] = stock_list
            
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn, params=params)
                
            return df
            
        except Exception as e:
            logger.error(f"加载股票基础信息失败: {str(e)}")
            raise
    
    def winsorize_factors(self, 
                         factor_df: pd.DataFrame,
                         factor_columns: List[str],
                         method: str = 'quantile',
                         lower_pct: float = 0.01,
                         upper_pct: float = 0.99,
                         mad_threshold: float = 3.0) -> pd.DataFrame:
        """因子去极值处理
        
        Args:
            factor_df: 因子数据DataFrame
            factor_columns: 需要处理的因子列
            method: 去极值方法 ('quantile', 'mad', 'zscore')
            lower_pct: 下分位数阈值
            upper_pct: 上分位数阈值
            mad_threshold: MAD方法的阈值
            
        Returns:
            处理后的因子数据
        """
        logger.info(f"开始因子去极值处理，方法: {method}")
        
        df = factor_df.copy()
        
        for factor in factor_columns:
            if factor not in df.columns:
                logger.warning(f"因子 {factor} 不存在，跳过")
                continue
                
            # 按日期分组处理
            for date in df['trade_date'].unique():
                date_mask = df['trade_date'] == date
                factor_values = df.loc[date_mask, factor].dropna()
                
                if len(factor_values) == 0:
                    continue
                    
                if method == 'quantile':
                    # 分位数去极值
                    lower_bound = factor_values.quantile(lower_pct)
                    upper_bound = factor_values.quantile(upper_pct)
                    
                elif method == 'mad':
                    # MAD去极值
                    median = factor_values.median()
                    mad = np.median(np.abs(factor_values - median))
                    lower_bound = median - mad_threshold * mad
                    upper_bound = median + mad_threshold * mad
                    
                elif method == 'zscore':
                    # Z-score去极值
                    mean = factor_values.mean()
                    std = factor_values.std()
                    lower_bound = mean - mad_threshold * std
                    upper_bound = mean + mad_threshold * std
                    
                else:
                    logger.warning(f"未知的去极值方法: {method}")
                    continue
                
                # 应用去极值
                df.loc[date_mask, factor] = np.clip(
                    df.loc[date_mask, factor],
                    lower_bound,
                    upper_bound
                )
        
        logger.info("因子去极值处理完成")
        return df
    
    def standardize_factors(self, 
                           factor_df: pd.DataFrame,
                           factor_columns: List[str],
                           method: str = 'zscore') -> pd.DataFrame:
        """因子标准化处理
        
        Args:
            factor_df: 因子数据DataFrame
            factor_columns: 需要处理的因子列
            method: 标准化方法 ('zscore', 'robust', 'rank')
            
        Returns:
            标准化后的因子数据
        """
        logger.info(f"开始因子标准化处理，方法: {method}")
        
        df = factor_df.copy()
        
        for factor in factor_columns:
            if factor not in df.columns:
                logger.warning(f"因子 {factor} 不存在，跳过")
                continue
                
            # 按日期分组处理
            for date in df['trade_date'].unique():
                date_mask = df['trade_date'] == date
                factor_values = df.loc[date_mask, factor].dropna()
                
                if len(factor_values) <= 1:
                    continue
                    
                if method == 'zscore':
                    # Z-score标准化
                    mean = factor_values.mean()
                    std = factor_values.std()
                    if std > 0:
                        df.loc[date_mask, factor] = (df.loc[date_mask, factor] - mean) / std
                        
                elif method == 'robust':
                    # 鲁棒标准化
                    median = factor_values.median()
                    mad = np.median(np.abs(factor_values - median))
                    if mad > 0:
                        df.loc[date_mask, factor] = (df.loc[date_mask, factor] - median) / mad
                        
                elif method == 'rank':
                    # 排序标准化
                    ranks = factor_values.rank(method='average')
                    n = len(ranks)
                    # 转换为标准正态分布
                    normalized_ranks = (ranks - 0.5) / n
                    df.loc[date_mask, factor] = stats.norm.ppf(normalized_ranks.clip(0.001, 0.999))
                    
                else:
                    logger.warning(f"未知的标准化方法: {method}")
                    continue
        
        logger.info("因子标准化处理完成")
        return df
    
    def neutralize_factors(self, 
                          factor_df: pd.DataFrame,
                          factor_columns: List[str],
                          neutralize_factors: List[str] = None) -> pd.DataFrame:
        """因子中性化处理
        
        Args:
            factor_df: 因子数据DataFrame
            factor_columns: 需要处理的因子列
            neutralize_factors: 中性化因子列表，默认为行业和市值
            
        Returns:
            中性化后的因子数据
        """
        if neutralize_factors is None:
            neutralize_factors = ['industry', 'market_cap']
            
        logger.info(f"开始因子中性化处理，中性化因子: {neutralize_factors}")
        
        df = factor_df.copy()
        
        # 加载股票基础信息
        stock_info = self.load_stock_basic_info(df['ts_code'].unique().tolist())
        
        # 加载市值数据（如果需要）
        if 'market_cap' in neutralize_factors:
            market_cap_data = self._load_market_cap_data(
                df['ts_code'].unique().tolist(),
                df['trade_date'].min(),
                df['trade_date'].max()
            )
            df = df.merge(market_cap_data, on=['ts_code', 'trade_date'], how='left')
        
        # 合并行业信息
        if 'industry' in neutralize_factors:
            df = df.merge(stock_info[['ts_code', 'industry']], on='ts_code', how='left')
            # 行业哑变量
            industry_dummies = pd.get_dummies(df['industry'], prefix='industry')
            df = pd.concat([df, industry_dummies], axis=1)
        
        # 按日期分组进行中性化
        for date in df['trade_date'].unique():
            date_mask = df['trade_date'] == date
            date_df = df[date_mask].copy()
            
            if len(date_df) <= 10:  # 样本太少跳过
                continue
                
            # 构建中性化变量
            neutralize_vars = []
            
            if 'market_cap' in neutralize_factors and 'market_cap' in date_df.columns:
                # 对数市值
                date_df['log_market_cap'] = np.log(date_df['market_cap'].replace(0, np.nan))
                neutralize_vars.append('log_market_cap')
            
            if 'industry' in neutralize_factors:
                industry_cols = [col for col in date_df.columns if col.startswith('industry_')]
                neutralize_vars.extend(industry_cols)
            
            if not neutralize_vars:
                continue
                
            # 对每个因子进行中性化
            for factor in factor_columns:
                if factor not in date_df.columns:
                    continue
                    
                # 准备数据
                y = date_df[factor].dropna()
                X_cols = neutralize_vars + [factor]
                valid_data = date_df[X_cols].dropna()
                
                if len(valid_data) <= len(neutralize_vars) + 1:
                    continue
                    
                y_valid = valid_data[factor]
                X_valid = valid_data[neutralize_vars]
                
                try:
                    # 线性回归
                    reg = LinearRegression()
                    reg.fit(X_valid, y_valid)
                    
                    # 计算残差
                    residuals = y_valid - reg.predict(X_valid)
                    
                    # 更新因子值
                    df.loc[date_mask & df.index.isin(valid_data.index), factor] = residuals
                    
                except Exception as e:
                    logger.warning(f"因子 {factor} 在日期 {date} 中性化失败: {str(e)}")
                    continue
        
        # 清理临时列
        cols_to_drop = [col for col in df.columns if col.startswith('industry_') or col in ['industry', 'market_cap', 'log_market_cap']]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        logger.info("因子中性化处理完成")
        return df
    
    def _load_market_cap_data(self, 
                             stock_list: List[str],
                             start_date: str,
                             end_date: str) -> pd.DataFrame:
        """加载市值数据
        
        Args:
            stock_list: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            市值数据DataFrame
        """
        query = """
            SELECT ts_code, trade_date, total_mv as market_cap
            FROM daily_basic
            WHERE ts_code = ANY(:stock_list)
            AND trade_date BETWEEN :start_date AND :end_date
            AND total_mv IS NOT NULL
        """
        
        params = {
            'stock_list': stock_list,
            'start_date': start_date,
            'end_date': end_date
        }
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn, params=params)
                
            return df
            
        except Exception as e:
            logger.error(f"加载市值数据失败: {str(e)}")
            return pd.DataFrame()
    
    def calculate_factor_correlation(self, 
                                   factor_df: pd.DataFrame,
                                   factor_columns: List[str],
                                   method: str = 'pearson') -> pd.DataFrame:
        """计算因子相关性矩阵
        
        Args:
            factor_df: 因子数据DataFrame
            factor_columns: 因子列表
            method: 相关性计算方法
            
        Returns:
            相关性矩阵
        """
        logger.info("计算因子相关性矩阵")
        
        # 选择有效因子列
        valid_factors = [f for f in factor_columns if f in factor_df.columns]
        
        if not valid_factors:
            logger.warning("没有有效的因子列")
            return pd.DataFrame()
        
        # 计算相关性
        corr_matrix = factor_df[valid_factors].corr(method=method)
        
        logger.info(f"因子相关性计算完成，矩阵大小: {corr_matrix.shape}")
        return corr_matrix
    
    def process_factors_pipeline(self, 
                               factor_names: List[str],
                               start_date: str,
                               end_date: str,
                               stock_list: Optional[List[str]] = None,
                               winsorize: bool = True,
                               standardize: bool = True,
                               neutralize: bool = True,
                               save_processed: bool = True) -> pd.DataFrame:
        """因子预处理流水线
        
        Args:
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            stock_list: 股票列表
            winsorize: 是否去极值
            standardize: 是否标准化
            neutralize: 是否中性化
            save_processed: 是否保存处理后的数据
            
        Returns:
            处理后的因子数据
        """
        logger.info(f"开始因子预处理流水线: {factor_names}")
        
        # 1. 加载原始因子数据
        factor_df = self.load_factor_data(factor_names, start_date, end_date, stock_list)
        
        if factor_df.empty:
            logger.warning("没有加载到因子数据")
            return pd.DataFrame()
        
        # 2. 去极值处理
        if winsorize:
            winsorize_config = self.processing_config.get('winsorize', {})
            factor_df = self.winsorize_factors(
                factor_df,
                factor_names,
                method=winsorize_config.get('method', 'quantile'),
                lower_pct=winsorize_config.get('lower_pct', 0.01),
                upper_pct=winsorize_config.get('upper_pct', 0.99),
                mad_threshold=winsorize_config.get('mad_threshold', 3.0)
            )
        
        # 3. 标准化处理
        if standardize:
            standardize_config = self.processing_config.get('standardize', {})
            factor_df = self.standardize_factors(
                factor_df,
                factor_names,
                method=standardize_config.get('method', 'zscore')
            )
        
        # 4. 中性化处理
        if neutralize:
            neutralize_config = self.processing_config.get('neutralize', {})
            factor_df = self.neutralize_factors(
                factor_df,
                factor_names,
                neutralize_factors=neutralize_config.get('factors', ['industry', 'market_cap'])
            )
        
        # 5. 保存处理后的数据
        if save_processed:
            self._save_processed_factors(factor_df, factor_names)
        
        logger.info("因子预处理流水线完成")
        return factor_df
    
    def _save_processed_factors(self, 
                              factor_df: pd.DataFrame,
                              factor_names: List[str]) -> None:
        """保存处理后的因子数据
        
        Args:
            factor_df: 处理后的因子数据
            factor_names: 因子名称列表
        """
        logger.info("保存处理后的因子数据")
        
        # 转换为长格式
        factor_records = []
        for _, row in factor_df.iterrows():
            for factor_name in factor_names:
                if factor_name in factor_df.columns and not pd.isna(row[factor_name]):
                    factor_records.append({
                        'ts_code': row['ts_code'],
                        'trade_date': row['trade_date'],
                        'factor_name': f"{factor_name}_processed",
                        'factor_value': row[factor_name],
                        'factor_category': 'processed',
                        'created_at': pd.Timestamp.now()
                    })
        
        if not factor_records:
            logger.warning("没有处理后的因子数据需要保存")
            return
        
        processed_df = pd.DataFrame(factor_records)
        
        try:
            with self.engine.connect() as conn:
                # 删除已存在的处理后数据
                processed_factor_names = [f"{name}_processed" for name in factor_names]
                delete_sql = """
                    DELETE FROM factor_library 
                    WHERE factor_name = ANY(:factor_names)
                    AND trade_date BETWEEN :start_date AND :end_date
                """
                
                conn.execute(text(delete_sql), {
                    'factor_names': processed_factor_names,
                    'start_date': factor_df['trade_date'].min(),
                    'end_date': factor_df['trade_date'].max()
                })
                
                # 插入新数据
                processed_df.to_sql(
                    'factor_library',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                conn.commit()
                
            logger.info(f"保存完成，共 {len(processed_df)} 条记录")
            
        except Exception as e:
            logger.error(f"保存处理后的因子数据失败: {str(e)}")
            raise
    
    def get_factor_statistics(self, 
                            factor_df: pd.DataFrame,
                            factor_columns: List[str]) -> Dict:
        """获取因子统计信息
        
        Args:
            factor_df: 因子数据DataFrame
            factor_columns: 因子列表
            
        Returns:
            因子统计信息字典
        """
        logger.info("计算因子统计信息")
        
        stats_dict = {}
        
        for factor in factor_columns:
            if factor not in factor_df.columns:
                continue
                
            factor_values = factor_df[factor].dropna()
            
            if len(factor_values) == 0:
                continue
                
            stats_dict[factor] = {
                'count': len(factor_values),
                'mean': factor_values.mean(),
                'std': factor_values.std(),
                'min': factor_values.min(),
                'max': factor_values.max(),
                'q25': factor_values.quantile(0.25),
                'q50': factor_values.quantile(0.50),
                'q75': factor_values.quantile(0.75),
                'skewness': factor_values.skew(),
                'kurtosis': factor_values.kurtosis(),
                'missing_rate': (len(factor_df) - len(factor_values)) / len(factor_df)
            }
        
        return stats_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='因子预处理工具')
    parser.add_argument('--action', choices=['process', 'correlation', 'statistics'], 
                       default='process', help='执行动作')
    parser.add_argument('--factors', nargs='+', required=True, help='因子名称列表')
    parser.add_argument('--start-date', required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--stocks', nargs='*', help='股票代码列表')
    parser.add_argument('--no-winsorize', action='store_true', help='跳过去极值处理')
    parser.add_argument('--no-standardize', action='store_true', help='跳过标准化处理')
    parser.add_argument('--no-neutralize', action='store_true', help='跳过中性化处理')
    parser.add_argument('--no-save', action='store_true', help='不保存处理后的数据')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = FactorProcessor()
    
    if args.action == 'process':
        # 因子预处理
        result_df = processor.process_factors_pipeline(
            factor_names=args.factors,
            start_date=args.start_date,
            end_date=args.end_date,
            stock_list=args.stocks,
            winsorize=not args.no_winsorize,
            standardize=not args.no_standardize,
            neutralize=not args.no_neutralize,
            save_processed=not args.no_save
        )
        print(f"因子预处理完成，处理 {len(result_df)} 条记录")
        
    elif args.action == 'correlation':
        # 计算因子相关性
        factor_df = processor.load_factor_data(
            args.factors, args.start_date, args.end_date, args.stocks
        )
        corr_matrix = processor.calculate_factor_correlation(factor_df, args.factors)
        print("因子相关性矩阵:")
        print(corr_matrix)
        
    elif args.action == 'statistics':
        # 计算因子统计信息
        factor_df = processor.load_factor_data(
            args.factors, args.start_date, args.end_date, args.stocks
        )
        stats = processor.get_factor_statistics(factor_df, args.factors)
        print("因子统计信息:")
        for factor, stat in stats.items():
            print(f"\n{factor}:")
            for key, value in stat.items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
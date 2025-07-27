import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from loguru import logger
from sqlalchemy import text, MetaData, Table, Column, String, Float, DateTime, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID
import uuid
import json
import sys
import os
from src.utils.config_loader import config

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class FeatureStore:
    """
    特征商店 - 管理和存储量化因子特征
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.metadata = MetaData()
        self._init_feature_tables()
    
    def _init_feature_tables(self):
        """
        初始化特征存储表
        """
        # 特征定义表
        self.feature_definitions = Table(
            'feature_definitions', self.metadata,
            Column('feature_id', String(50), primary_key=True),
            Column('feature_name', String(100), nullable=False),
            Column('feature_type', String(20), nullable=False),  # technical, fundamental, market
            Column('category', String(50), nullable=False),  # momentum, trend, volatility, etc.
            Column('description', String(500)),
            Column('calculation_method', String(1000)),
            Column('data_type', String(20), nullable=False),  # float, int, bool
            Column('update_frequency', String(20), nullable=False),  # daily, weekly, monthly
            Column('lookback_period', Integer),  # 回看期间
            Column('is_active', Boolean, default=True),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        )
        
        # 特征值存储表
        self.feature_values = Table(
            'feature_values', self.metadata,
            Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
            Column('feature_id', String(50), nullable=False),
            Column('ts_code', String(20), nullable=False),
            Column('trade_date', DateTime, nullable=False),
            Column('value', Float),
            Column('raw_value', Float),  # 原始值
            Column('normalized_value', Float),  # 标准化值
            Column('percentile_rank', Float),  # 百分位排名
            Column('z_score', Float),  # Z分数
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # 特征组合表
        self.feature_groups = Table(
            'feature_groups', self.metadata,
            Column('group_id', String(50), primary_key=True),
            Column('group_name', String(100), nullable=False),
            Column('description', String(500)),
            Column('feature_ids', String(1000)),  # JSON格式存储特征ID列表
            Column('weights', String(1000)),  # JSON格式存储权重
            Column('is_active', Boolean, default=True),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # 特征统计表
        self.feature_statistics = Table(
            'feature_statistics', self.metadata,
            Column('feature_id', String(50), primary_key=True),
            Column('stat_date', DateTime, primary_key=True),
            Column('mean_value', Float),
            Column('std_value', Float),
            Column('min_value', Float),
            Column('max_value', Float),
            Column('median_value', Float),
            Column('q25_value', Float),
            Column('q75_value', Float),
            Column('skewness', Float),
            Column('kurtosis', Float),
            Column('sample_count', Integer),
            Column('null_count', Integer),
            Column('updated_at', DateTime, default=datetime.utcnow)
        )
    
    def create_tables(self):
        """
        创建特征存储表
        """
        try:
            self.metadata.create_all(self.engine)
            
            # 创建索引
            with self.engine.connect() as conn:
                # 特征值表索引
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_feature_values_feature_date 
                    ON feature_values(feature_id, trade_date DESC)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_feature_values_stock_date 
                    ON feature_values(ts_code, trade_date DESC)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_feature_values_date 
                    ON feature_values(trade_date DESC)
                """))
                
                conn.commit()
            
            logger.info("特征存储表创建完成")
            
        except Exception as e:
            logger.error(f"创建特征存储表失败: {e}")
            raise
    
    def register_feature(self, feature_id: str, feature_name: str, 
                        feature_type: str, category: str, 
                        description: str = None, 
                        calculation_method: str = None,
                        data_type: str = 'float',
                        update_frequency: str = 'daily',
                        lookback_period: int = None) -> bool:
        """
        注册新特征
        
        Args:
            feature_id: 特征ID
            feature_name: 特征名称
            feature_type: 特征类型 (technical, fundamental, market)
            category: 特征分类 (momentum, trend, volatility, etc.)
            description: 特征描述
            calculation_method: 计算方法
            data_type: 数据类型
            update_frequency: 更新频率
            lookback_period: 回看期间
        
        Returns:
            是否注册成功
        """
        try:
            with self.engine.connect() as conn:
                # 检查特征是否已存在
                result = conn.execute(
                    text("SELECT feature_id FROM feature_definitions WHERE feature_id = :feature_id"),
                    {'feature_id': feature_id}
                ).fetchone()
                
                if result:
                    logger.warning(f"特征已存在: {feature_id}")
                    return False
                
                # 插入新特征定义
                conn.execute(
                    text("""
                        INSERT INTO feature_definitions 
                        (feature_id, feature_name, feature_type, category, description, 
                         calculation_method, data_type, update_frequency, lookback_period)
                        VALUES (:feature_id, :feature_name, :feature_type, :category, 
                                :description, :calculation_method, :data_type, 
                                :update_frequency, :lookback_period)
                    """),
                    {
                        'feature_id': feature_id,
                        'feature_name': feature_name,
                        'feature_type': feature_type,
                        'category': category,
                        'description': description,
                        'calculation_method': calculation_method,
                        'data_type': data_type,
                        'update_frequency': update_frequency,
                        'lookback_period': lookback_period
                    }
                )
                conn.commit()
                
            logger.info(f"特征注册成功: {feature_id}")
            return True
            
        except Exception as e:
            logger.error(f"特征注册失败: {feature_id} - {e}")
            return False
    
    def store_feature_values(self, feature_id: str, data: pd.DataFrame) -> bool:
        """
        存储特征值
        
        Args:
            feature_id: 特征ID
            data: 特征数据，包含ts_code, trade_date, value列
        
        Returns:
            是否存储成功
        """
        try:
            if data.empty:
                logger.warning(f"特征数据为空: {feature_id}")
                return False
            
            # 验证必要列
            required_columns = ['ts_code', 'trade_date', 'value']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"特征数据缺少必要列: {required_columns}")
                return False
            
            # 准备数据
            feature_data = data.copy()
            feature_data['feature_id'] = feature_id
            feature_data['raw_value'] = feature_data['value']
            
            # 计算标准化值和统计指标
            feature_data = self._calculate_feature_statistics(feature_data)
            
            # 删除已存在的数据
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        DELETE FROM feature_values 
                        WHERE feature_id = :feature_id 
                        AND ts_code = ANY(:ts_codes)
                        AND trade_date = ANY(:trade_dates)
                    """),
                    {
                        'feature_id': feature_id,
                        'ts_codes': feature_data['ts_code'].unique().tolist(),
                        'trade_dates': feature_data['trade_date'].unique().tolist()
                    }
                )
                
                # 插入新数据
                feature_data.to_sql(
                    'feature_values', 
                    conn, 
                    if_exists='append', 
                    index=False,
                    method='multi'
                )
                
                conn.commit()
            
            logger.info(f"特征值存储成功: {feature_id}, {len(feature_data)} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"特征值存储失败: {feature_id} - {e}")
            return False
    
    def _calculate_feature_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算特征统计指标
        
        Args:
            data: 特征数据
        
        Returns:
            包含统计指标的数据
        """
        result = data.copy()
        
        # 按日期分组计算统计指标
        for trade_date in data['trade_date'].unique():
            date_mask = data['trade_date'] == trade_date
            date_values = data.loc[date_mask, 'value']
            
            if len(date_values) > 0:
                # 标准化值 (Z-score)
                mean_val = date_values.mean()
                std_val = date_values.std()
                
                if std_val > 0:
                    result.loc[date_mask, 'z_score'] = (date_values - mean_val) / std_val
                    result.loc[date_mask, 'normalized_value'] = result.loc[date_mask, 'z_score']
                else:
                    result.loc[date_mask, 'z_score'] = 0
                    result.loc[date_mask, 'normalized_value'] = 0
                
                # 百分位排名
                result.loc[date_mask, 'percentile_rank'] = date_values.rank(pct=True)
        
        return result
    
    def get_feature_values(self, feature_id: str, 
                          ts_codes: List[str] = None,
                          start_date: str = None, 
                          end_date: str = None,
                          normalized: bool = False) -> pd.DataFrame:
        """
        获取特征值
        
        Args:
            feature_id: 特征ID
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            normalized: 是否返回标准化值
        
        Returns:
            特征值DataFrame
        """
        try:
            query = """
                SELECT ts_code, trade_date, 
                       CASE WHEN :normalized THEN normalized_value ELSE value END as value,
                       percentile_rank, z_score
                FROM feature_values 
                WHERE feature_id = :feature_id
            """
            
            params = {
                'feature_id': feature_id,
                'normalized': normalized
            }
            
            conditions = []
            
            if ts_codes:
                conditions.append("ts_code = ANY(:ts_codes)")
                params['ts_codes'] = ts_codes
            
            if start_date:
                conditions.append("trade_date >= :start_date")
                params['start_date'] = start_date
            
            if end_date:
                conditions.append("trade_date <= :end_date")
                params['end_date'] = end_date
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            query += " ORDER BY trade_date DESC, ts_code"
            
            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            return df
            
        except Exception as e:
            logger.error(f"获取特征值失败: {feature_id} - {e}")
            return pd.DataFrame()
    
    def get_feature_matrix(self, feature_ids: List[str],
                          ts_codes: List[str] = None,
                          trade_date: str = None,
                          normalized: bool = True) -> pd.DataFrame:
        """
        获取特征矩阵
        
        Args:
            feature_ids: 特征ID列表
            ts_codes: 股票代码列表
            trade_date: 交易日期
            normalized: 是否使用标准化值
        
        Returns:
            特征矩阵DataFrame，行为股票，列为特征
        """
        try:
            if not feature_ids:
                return pd.DataFrame()
            
            query = """
                SELECT ts_code, feature_id,
                       CASE WHEN :normalized THEN normalized_value ELSE value END as value
                FROM feature_values 
                WHERE feature_id = ANY(:feature_ids)
            """
            
            params = {
                'feature_ids': feature_ids,
                'normalized': normalized
            }
            
            conditions = []
            
            if ts_codes:
                conditions.append("ts_code = ANY(:ts_codes)")
                params['ts_codes'] = ts_codes
            
            if trade_date:
                conditions.append("trade_date = :trade_date")
                params['trade_date'] = trade_date
            else:
                # 获取最新日期的数据
                conditions.append("""
                    trade_date = (
                        SELECT MAX(trade_date) FROM feature_values 
                        WHERE feature_id = ANY(:feature_ids)
                    )
                """)
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            if df.empty:
                return pd.DataFrame()
            
            # 透视为特征矩阵
            matrix = df.pivot(index='ts_code', columns='feature_id', values='value')
            
            return matrix
            
        except Exception as e:
            logger.error(f"获取特征矩阵失败: {e}")
            return pd.DataFrame()
    
    def update_feature_statistics(self, feature_id: str, trade_date: str = None):
        """
        更新特征统计信息
        
        Args:
            feature_id: 特征ID
            trade_date: 交易日期，如果为None则更新最新日期
        """
        try:
            if trade_date is None:
                # 获取最新交易日期
                with self.engine.connect() as conn:
                    result = conn.execute(
                        text("""
                            SELECT MAX(trade_date) as max_date 
                            FROM feature_values 
                            WHERE feature_id = :feature_id
                        """),
                        {'feature_id': feature_id}
                    ).fetchone()
                    
                    if not result or not result[0]:
                        logger.warning(f"未找到特征数据: {feature_id}")
                        return
                    
                    trade_date = result[0]
            
            # 计算统计指标
            with self.engine.connect() as conn:
                stats_result = conn.execute(
                    text("""
                        SELECT 
                            AVG(value) as mean_value,
                            STDDEV(value) as std_value,
                            MIN(value) as min_value,
                            MAX(value) as max_value,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median_value,
                            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) as q25_value,
                            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) as q75_value,
                            COUNT(*) as sample_count,
                            COUNT(*) - COUNT(value) as null_count
                        FROM feature_values 
                        WHERE feature_id = :feature_id AND trade_date = :trade_date
                    """),
                    {'feature_id': feature_id, 'trade_date': trade_date}
                ).fetchone()
                
                if stats_result:
                    # 删除已存在的统计数据
                    conn.execute(
                        text("""
                            DELETE FROM feature_statistics 
                            WHERE feature_id = :feature_id AND stat_date = :stat_date
                        """),
                        {'feature_id': feature_id, 'stat_date': trade_date}
                    )
                    
                    # 插入新的统计数据
                    conn.execute(
                        text("""
                            INSERT INTO feature_statistics 
                            (feature_id, stat_date, mean_value, std_value, min_value, 
                             max_value, median_value, q25_value, q75_value, 
                             sample_count, null_count)
                            VALUES (:feature_id, :stat_date, :mean_value, :std_value, 
                                    :min_value, :max_value, :median_value, :q25_value, 
                                    :q75_value, :sample_count, :null_count)
                        """),
                        {
                            'feature_id': feature_id,
                            'stat_date': trade_date,
                            'mean_value': stats_result[0],
                            'std_value': stats_result[1],
                            'min_value': stats_result[2],
                            'max_value': stats_result[3],
                            'median_value': stats_result[4],
                            'q25_value': stats_result[5],
                            'q75_value': stats_result[6],
                            'sample_count': stats_result[7],
                            'null_count': stats_result[8]
                        }
                    )
                    
                    conn.commit()
                    
                    logger.info(f"特征统计更新完成: {feature_id} - {trade_date}")
                
        except Exception as e:
            logger.error(f"更新特征统计失败: {feature_id} - {e}")
    
    def list_features(self, feature_type: str = None, category: str = None) -> pd.DataFrame:
        """
        列出特征定义
        
        Args:
            feature_type: 特征类型过滤
            category: 特征分类过滤
        
        Returns:
            特征定义DataFrame
        """
        try:
            query = "SELECT * FROM feature_definitions WHERE is_active = true"
            params = {}
            
            conditions = []
            
            if feature_type:
                conditions.append("feature_type = :feature_type")
                params['feature_type'] = feature_type
            
            if category:
                conditions.append("category = :category")
                params['category'] = category
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            query += " ORDER BY feature_type, category, feature_name"
            
            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            return df
            
        except Exception as e:
            logger.error(f"列出特征失败: {e}")
            return pd.DataFrame()

if __name__ == '__main__':
    # 测试代码
    from src.utils.db import get_db_engine
    
    print("测试特征商店模块...")
    
    # 初始化数据库连接
    engine = get_db_engine()
    
    # 创建特征商店
    feature_store = FeatureStore(engine)
    
    # 创建表
    feature_store.create_tables()
    
    # 注册测试特征
    success = feature_store.register_feature(
        feature_id='rsi_14',
        feature_name='14日RSI',
        feature_type='technical',
        category='momentum',
        description='14日相对强弱指标',
        calculation_method='RSI计算公式',
        lookback_period=config.get('factor_params.rsi.window', 14)
    )
    
    print(f"特征注册结果: {success}")
    
    # 列出特征
    features = feature_store.list_features()
    print(f"\n已注册特征数量: {len(features)}")
    
    if not features.empty:
        print("特征列表:")
        for _, feature in features.iterrows():
            print(f"  {feature['feature_id']}: {feature['feature_name']} ({feature['category']})")
    
    print("\n特征商店模块测试完成!")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量计算优化器
实现只计算新增日期的因子值和增量存储
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Any, Set, Tuple
from loguru import logger
from sqlalchemy import text
import json
from enum import Enum

from .factor_models import FactorType, FactorResult, FactorValue, CalculationStatus


class DependencyType(Enum):
    """依赖类型枚举"""
    DATA_DEPENDENCY = "data"  # 数据依赖
    CALCULATION_DEPENDENCY = "calculation"  # 计算依赖
    TIME_DEPENDENCY = "time"  # 时间依赖


class FactorDependency:
    """因子依赖关系"""
    
    def __init__(self, factor_name: str, dependency_type: DependencyType,
                 dependencies: List[str], lookback_days: int = 0):
        """
        初始化因子依赖
        
        Args:
            factor_name: 因子名称
            dependency_type: 依赖类型
            dependencies: 依赖项列表
            lookback_days: 回看天数
        """
        self.factor_name = factor_name
        self.dependency_type = dependency_type
        self.dependencies = dependencies
        self.lookback_days = lookback_days


class DependencyManager:
    """依赖关系管理器"""
    
    def __init__(self):
        """初始化依赖管理器"""
        self.dependencies = {}  # factor_name -> FactorDependency
        self.dependency_graph = {}  # factor_name -> set of dependent factors
        self._build_default_dependencies()
    
    def _build_default_dependencies(self):
        """构建默认的因子依赖关系"""
        # 技术面因子依赖
        technical_dependencies = [
            FactorDependency('sma_5', DependencyType.DATA_DEPENDENCY, ['close'], 5),
            FactorDependency('sma_20', DependencyType.DATA_DEPENDENCY, ['close'], 20),
            FactorDependency('ema_12', DependencyType.DATA_DEPENDENCY, ['close'], 12),
            FactorDependency('ema_26', DependencyType.DATA_DEPENDENCY, ['close'], 26),
            FactorDependency('macd', DependencyType.CALCULATION_DEPENDENCY, ['ema_12', 'ema_26'], 26),
            FactorDependency('rsi_14', DependencyType.DATA_DEPENDENCY, ['close'], 14),
            FactorDependency('bollinger_upper', DependencyType.CALCULATION_DEPENDENCY, ['sma_20'], 20),
            FactorDependency('bollinger_lower', DependencyType.CALCULATION_DEPENDENCY, ['sma_20'], 20),
            FactorDependency('atr_14', DependencyType.DATA_DEPENDENCY, ['high', 'low', 'close'], 14),
        ]
        
        # 基本面因子依赖
        fundamental_dependencies = [
            FactorDependency('pe_ttm', DependencyType.DATA_DEPENDENCY, ['market_data', 'financial_data'], 0),
            FactorDependency('pb', DependencyType.DATA_DEPENDENCY, ['market_data', 'financial_data'], 0),
            FactorDependency('roe', DependencyType.DATA_DEPENDENCY, ['financial_data'], 0),
            FactorDependency('roa', DependencyType.DATA_DEPENDENCY, ['financial_data'], 0),
            FactorDependency('revenue_yoy', DependencyType.DATA_DEPENDENCY, ['financial_data'], 365),
        ]
        
        # 情绪面因子依赖
        sentiment_dependencies = [
            FactorDependency('money_flow_5', DependencyType.DATA_DEPENDENCY, ['amount', 'vol'], 5),
            FactorDependency('attention_score', DependencyType.CALCULATION_DEPENDENCY, 
                           ['turnover_attention', 'volume_attention', 'price_attention'], 20),
            FactorDependency('sentiment_strength', DependencyType.CALCULATION_DEPENDENCY,
                           ['price_momentum_sentiment', 'volatility_sentiment'], 20),
        ]
        
        # 注册所有依赖
        all_dependencies = technical_dependencies + fundamental_dependencies + sentiment_dependencies
        
        for dep in all_dependencies:
            self.register_dependency(dep)
    
    def register_dependency(self, dependency: FactorDependency):
        """注册因子依赖"""
        self.dependencies[dependency.factor_name] = dependency
        
        # 构建依赖图
        for dep_name in dependency.dependencies:
            if dep_name not in self.dependency_graph:
                self.dependency_graph[dep_name] = set()
            self.dependency_graph[dep_name].add(dependency.factor_name)
    
    def get_dependency(self, factor_name: str) -> Optional[FactorDependency]:
        """获取因子依赖"""
        return self.dependencies.get(factor_name)
    
    def get_dependent_factors(self, factor_name: str) -> Set[str]:
        """获取依赖于指定因子的因子列表"""
        return self.dependency_graph.get(factor_name, set())
    
    def get_calculation_order(self, factor_names: List[str]) -> List[str]:
        """获取因子计算顺序（拓扑排序）"""
        # 构建子图
        subgraph = {}
        in_degree = {}
        
        # 初始化
        for factor in factor_names:
            subgraph[factor] = []
            in_degree[factor] = 0
        
        # 构建子图的边和入度
        for factor in factor_names:
            dependency = self.get_dependency(factor)
            if dependency and dependency.dependency_type == DependencyType.CALCULATION_DEPENDENCY:
                for dep in dependency.dependencies:
                    if dep in factor_names:
                        subgraph[dep].append(factor)
                        in_degree[factor] += 1
        
        # 拓扑排序
        result = []
        queue = [factor for factor in factor_names if in_degree[factor] == 0]
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in subgraph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否有循环依赖
        if len(result) != len(factor_names):
            logger.warning("检测到循环依赖，使用原始顺序")
            return factor_names
        
        return result
    
    def get_required_lookback_days(self, factor_names: List[str]) -> int:
        """获取所需的最大回看天数"""
        max_lookback = 0
        
        for factor_name in factor_names:
            dependency = self.get_dependency(factor_name)
            if dependency:
                max_lookback = max(max_lookback, dependency.lookback_days)
        
        return max_lookback


class IncrementalDataManager:
    """增量数据管理器"""
    
    def __init__(self, engine):
        """初始化增量数据管理器"""
        self.engine = engine
    
    def get_last_calculation_date(self, ts_code: str, factor_name: str) -> Optional[date]:
        """获取因子的最后计算日期"""
        try:
            query = text("""
                SELECT MAX(factor_date) as last_date
                FROM factor_values 
                WHERE ts_code = :ts_code 
                AND factor_name = :factor_name
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    'ts_code': ts_code,
                    'factor_name': factor_name
                })
                
                row = result.fetchone()
                if row and row.last_date:
                    return row.last_date
                
            return None
            
        except Exception as e:
            logger.error(f"获取最后计算日期失败: {e}")
            return None
    
    def get_missing_dates(self, ts_code: str, factor_name: str,
                         end_date: Optional[date] = None) -> List[date]:
        """获取需要计算的缺失日期"""
        if end_date is None:
            end_date = date.today()
        
        # 获取最后计算日期
        last_date = self.get_last_calculation_date(ts_code, factor_name)
        
        if last_date is None:
            # 如果从未计算过，需要获取股票的上市日期
            start_date = self.get_stock_list_date(ts_code)
            if start_date is None:
                start_date = end_date - timedelta(days=365)  # 默认一年
        else:
            start_date = last_date + timedelta(days=1)
        
        # 获取交易日历
        trading_dates = self.get_trading_dates(start_date, end_date)
        
        return trading_dates
    
    def get_stock_list_date(self, ts_code: str) -> Optional[date]:
        """获取股票上市日期"""
        try:
            query = text("""
                SELECT list_date 
                FROM stock_basic 
                WHERE ts_code = :ts_code
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'ts_code': ts_code})
                row = result.fetchone()
                
                if row and row.list_date:
                    return pd.to_datetime(row.list_date).date()
                
            return None
            
        except Exception as e:
            logger.error(f"获取股票上市日期失败: {e}")
            return None
    
    def get_trading_dates(self, start_date: date, end_date: date) -> List[date]:
        """获取交易日期列表"""
        try:
            query = text("""
                SELECT DISTINCT trade_date 
                FROM stock_daily 
                WHERE trade_date BETWEEN :start_date AND :end_date
                ORDER BY trade_date
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                dates = [row.trade_date for row in result.fetchall()]
                return [pd.to_datetime(d).date() for d in dates]
                
        except Exception as e:
            logger.error(f"获取交易日期失败: {e}")
            return []
    
    def get_incremental_data(self, ts_code: str, missing_dates: List[date],
                           lookback_days: int = 0) -> pd.DataFrame:
        """获取增量计算所需的数据"""
        if not missing_dates:
            return pd.DataFrame()
        
        # 计算实际需要的开始日期（考虑回看期）
        actual_start_date = min(missing_dates) - timedelta(days=lookback_days + 10)  # 额外缓冲
        end_date = max(missing_dates)
        
        try:
            query = text("""
                SELECT ts_code, trade_date, open, high, low, close, pre_close,
                       vol, amount, turnover_rate, circ_mv, total_mv
                FROM stock_daily 
                WHERE ts_code = :ts_code
                AND trade_date BETWEEN :start_date AND :end_date
                ORDER BY trade_date
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    'ts_code': ts_code,
                    'start_date': actual_start_date,
                    'end_date': end_date
                })
                
                data = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not data.empty:
                data['trade_date'] = pd.to_datetime(data['trade_date'])
                # 转换数值列
                numeric_columns = ['open', 'high', 'low', 'close', 'pre_close', 
                                 'vol', 'amount', 'turnover_rate', 'circ_mv', 'total_mv']
                for col in numeric_columns:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
            
            return data
            
        except Exception as e:
            logger.error(f"获取增量数据失败: {e}")
            return pd.DataFrame()


class IncrementalStorage:
    """增量存储管理器"""
    
    def __init__(self, engine):
        """初始化增量存储管理器"""
        self.engine = engine
    
    def save_incremental_factors(self, ts_code: str, factor_name: str,
                                factor_values: List[FactorValue],
                                replace_existing: bool = False) -> bool:
        """保存增量因子数据"""
        if not factor_values:
            return True
        
        try:
            # 准备数据
            data_to_save = []
            for fv in factor_values:
                data_to_save.append({
                    'ts_code': ts_code,
                    'factor_name': factor_name,
                    'factor_date': fv.date,
                    'factor_value': fv.value,
                    'created_time': datetime.now()
                })
            
            df = pd.DataFrame(data_to_save)
            
            # 如果需要替换现有数据，先删除
            if replace_existing:
                self._delete_existing_factors(ts_code, factor_name, 
                                            [fv.date for fv in factor_values])
            
            # 保存数据
            df.to_sql(
                'factor_values',
                self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            logger.info(f"成功保存 {len(factor_values)} 个 {factor_name} 因子值")
            return True
            
        except Exception as e:
            logger.error(f"保存增量因子数据失败: {e}")
            return False
    
    def _delete_existing_factors(self, ts_code: str, factor_name: str, 
                               dates: List[date]):
        """删除现有的因子数据"""
        try:
            date_list = "','".join([d.isoformat() for d in dates])
            
            query = text(f"""
                DELETE FROM factor_values 
                WHERE ts_code = :ts_code 
                AND factor_name = :factor_name
                AND factor_date IN ('{date_list}')
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    'ts_code': ts_code,
                    'factor_name': factor_name
                })
                
                logger.info(f"删除了 {result.rowcount} 个现有因子值")
                
        except Exception as e:
            logger.error(f"删除现有因子数据失败: {e}")
    
    def batch_save_factors(self, factor_results: Dict[str, Dict[str, List[FactorValue]]],
                          batch_size: int = 1000) -> bool:
        """批量保存因子数据"""
        try:
            all_data = []
            
            # 准备所有数据
            for ts_code, factors in factor_results.items():
                for factor_name, factor_values in factors.items():
                    for fv in factor_values:
                        all_data.append({
                            'ts_code': ts_code,
                            'factor_name': factor_name,
                            'factor_date': fv.date,
                            'factor_value': fv.value,
                            'created_time': datetime.now()
                        })
            
            if not all_data:
                return True
            
            # 分批保存
            df = pd.DataFrame(all_data)
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                
                batch_df.to_sql(
                    'factor_values',
                    self.engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                logger.info(f"保存批次 {i // batch_size + 1}/{total_batches}，"
                          f"包含 {len(batch_df)} 条记录")
            
            logger.info(f"批量保存完成，总计 {len(all_data)} 条因子数据")
            return True
            
        except Exception as e:
            logger.error(f"批量保存因子数据失败: {e}")
            return False
    
    def update_calculation_log(self, ts_code: str, factor_names: List[str],
                             calculation_date: date, status: str,
                             execution_time: timedelta, error_message: str = None):
        """更新计算日志"""
        try:
            log_data = {
                'ts_code': ts_code,
                'factor_names': json.dumps(factor_names),
                'calculation_date': calculation_date,
                'status': status,
                'execution_time_seconds': execution_time.total_seconds(),
                'error_message': error_message,
                'created_time': datetime.now()
            }
            
            df = pd.DataFrame([log_data])
            df.to_sql(
                'factor_calculation_log',
                self.engine,
                if_exists='append',
                index=False
            )
            
        except Exception as e:
            logger.error(f"更新计算日志失败: {e}")


class IncrementalFactorCalculator:
    """增量因子计算器"""
    
    def __init__(self, engine):
        """初始化增量因子计算器"""
        self.engine = engine
        self.dependency_manager = DependencyManager()
        self.data_manager = IncrementalDataManager(engine)
        self.storage_manager = IncrementalStorage(engine)
    
    def calculate_incremental_factors(self, ts_codes: List[str],
                                    factor_names: List[str],
                                    end_date: Optional[date] = None,
                                    force_recalculate: bool = False) -> Dict[str, Any]:
        """
        增量计算因子
        
        Args:
            ts_codes: 股票代码列表
            factor_names: 因子名称列表
            end_date: 结束日期
            force_recalculate: 是否强制重新计算
            
        Returns:
            计算结果统计
        """
        start_time = datetime.now()
        
        if end_date is None:
            end_date = date.today()
        
        logger.info(f"开始增量计算因子: {len(ts_codes)}只股票, {len(factor_names)}个因子")
        
        # 获取因子计算顺序
        ordered_factors = self.dependency_manager.get_calculation_order(factor_names)
        logger.info(f"因子计算顺序: {ordered_factors}")
        
        # 获取所需的最大回看天数
        max_lookback = self.dependency_manager.get_required_lookback_days(factor_names)
        logger.info(f"最大回看天数: {max_lookback}")
        
        calculation_stats = {
            'total_stocks': len(ts_codes),
            'total_factors': len(factor_names),
            'processed_stocks': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'skipped_calculations': 0,
            'total_factor_values': 0
        }
        
        all_results = {}
        
        try:
            for ts_code in ts_codes:
                logger.info(f"处理股票: {ts_code}")
                stock_results = {}
                
                for factor_name in ordered_factors:
                    try:
                        # 获取需要计算的日期
                        if force_recalculate:
                            # 强制重新计算最近30天
                            missing_dates = self.data_manager.get_trading_dates(
                                end_date - timedelta(days=30), end_date
                            )
                        else:
                            missing_dates = self.data_manager.get_missing_dates(
                                ts_code, factor_name, end_date
                            )
                        
                        if not missing_dates:
                            logger.debug(f"因子 {factor_name} 无需计算")
                            calculation_stats['skipped_calculations'] += 1
                            continue
                        
                        logger.info(f"需要计算 {len(missing_dates)} 个日期的 {factor_name} 因子")
                        
                        # 获取增量数据
                        incremental_data = self.data_manager.get_incremental_data(
                            ts_code, missing_dates, max_lookback
                        )
                        
                        if incremental_data.empty:
                            logger.warning(f"无法获取股票 {ts_code} 的数据")
                            calculation_stats['failed_calculations'] += 1
                            continue
                        
                        # 执行因子计算（这里需要调用具体的因子计算逻辑）
                        factor_values = self._calculate_single_factor(
                            factor_name, incremental_data, missing_dates
                        )
                        
                        if factor_values:
                            # 保存增量因子数据
                            success = self.storage_manager.save_incremental_factors(
                                ts_code, factor_name, factor_values, force_recalculate
                            )
                            
                            if success:
                                stock_results[factor_name] = factor_values
                                calculation_stats['successful_calculations'] += 1
                                calculation_stats['total_factor_values'] += len(factor_values)
                            else:
                                calculation_stats['failed_calculations'] += 1
                        else:
                            calculation_stats['failed_calculations'] += 1
                            
                    except Exception as e:
                        logger.error(f"计算因子 {factor_name} 失败: {e}")
                        calculation_stats['failed_calculations'] += 1
                
                if stock_results:
                    all_results[ts_code] = stock_results
                
                calculation_stats['processed_stocks'] += 1
        
        except Exception as e:
            logger.error(f"增量计算失败: {e}")
            raise
        
        finally:
            # 记录执行时间
            execution_time = datetime.now() - start_time
            calculation_stats['execution_time'] = execution_time
            
            logger.info("=== 增量计算统计 ===")
            logger.info(f"处理股票数: {calculation_stats['processed_stocks']}")
            logger.info(f"成功计算: {calculation_stats['successful_calculations']}")
            logger.info(f"失败计算: {calculation_stats['failed_calculations']}")
            logger.info(f"跳过计算: {calculation_stats['skipped_calculations']}")
            logger.info(f"总因子值: {calculation_stats['total_factor_values']}")
            logger.info(f"执行时间: {execution_time}")
            logger.info("==================")
        
        return {
            'results': all_results,
            'statistics': calculation_stats
        }
    
    def _calculate_single_factor(self, factor_name: str, data: pd.DataFrame,
                               target_dates: List[date]) -> List[FactorValue]:
        """
        计算单个因子（简化实现，实际需要调用具体的因子计算逻辑）
        
        Args:
            factor_name: 因子名称
            data: 计算数据
            target_dates: 目标日期列表
            
        Returns:
            因子值列表
        """
        # 这里是简化实现，实际应该调用具体的因子计算引擎
        try:
            factor_values = []
            
            # 根据因子名称选择计算方法
            if factor_name == 'sma_5':
                # 5日简单移动平均
                data['sma_5'] = data['close'].rolling(window=5).mean()
                
                for target_date in target_dates:
                    date_data = data[data['trade_date'].dt.date == target_date]
                    if not date_data.empty and pd.notna(date_data['sma_5'].iloc[0]):
                        factor_values.append(FactorValue(
                            date=target_date,
                            value=float(date_data['sma_5'].iloc[0])
                        ))
            
            elif factor_name == 'rsi_14':
                # 14日RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                for target_date in target_dates:
                    date_data = data[data['trade_date'].dt.date == target_date]
                    if not date_data.empty:
                        rsi_value = rsi.loc[date_data.index[0]]
                        if pd.notna(rsi_value):
                            factor_values.append(FactorValue(
                                date=target_date,
                                value=float(rsi_value)
                            ))
            
            # 可以添加更多因子的计算逻辑
            
            return factor_values
            
        except Exception as e:
            logger.error(f"计算因子 {factor_name} 失败: {e}")
            return []
    
    def get_calculation_progress(self, ts_codes: List[str], 
                               factor_names: List[str]) -> Dict[str, Any]:
        """获取计算进度"""
        progress_info = {
            'total_tasks': len(ts_codes) * len(factor_names),
            'completed_tasks': 0,
            'progress_by_stock': {},
            'overall_progress': 0.0
        }
        
        try:
            for ts_code in ts_codes:
                stock_progress = {
                    'total_factors': len(factor_names),
                    'completed_factors': 0,
                    'factor_status': {}
                }
                
                for factor_name in factor_names:
                    last_date = self.data_manager.get_last_calculation_date(ts_code, factor_name)
                    
                    if last_date and last_date >= date.today() - timedelta(days=1):
                        stock_progress['completed_factors'] += 1
                        stock_progress['factor_status'][factor_name] = 'completed'
                        progress_info['completed_tasks'] += 1
                    else:
                        stock_progress['factor_status'][factor_name] = 'pending'
                
                stock_progress['progress_rate'] = (
                    stock_progress['completed_factors'] / stock_progress['total_factors']
                )
                
                progress_info['progress_by_stock'][ts_code] = stock_progress
            
            # 计算总体进度
            progress_info['overall_progress'] = (
                progress_info['completed_tasks'] / progress_info['total_tasks']
            )
            
        except Exception as e:
            logger.error(f"获取计算进度失败: {e}")
        
        return progress_info
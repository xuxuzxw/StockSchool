#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日预测脚本
实现基于训练好的模型进行每日股票预测的功能
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.db import DatabaseManager
from utils.config_loader import config
from utils.retry import idempotent_retry
from compute.factor_engine import FactorEngine
from compute.processing import FactorProcessor
from monitoring.logger import setup_logger

logger = setup_logger(__name__)

class StockPredictor:
    """
    股票预测器
    基于训练好的模型进行每日股票预测
    """
    
    def __init__(self, model_path: str = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径，如果为None则使用默认路径
        """
        self.db_manager = DatabaseManager()
        self.factor_engine = FactorEngine()
        self.factor_processor = FactorProcessor()
        
        # 设置模型路径
        if model_path is None:
            self.model_path = config.get('model', {}).get('save_path', 'models/stock_prediction_model.pkl')
        else:
            self.model_path = model_path
            
        self.model = None
        self.feature_columns = None
        self.scaler = None
        
        # 预测配置
        self.prediction_config = config.get('prediction', {
            'prediction_days': 5,  # 预测未来几天
            'min_data_days': 60,   # 最少需要的历史数据天数
            'confidence_threshold': 0.6,  # 置信度阈值
            'top_n_stocks': 50     # 输出前N只股票
        })
        
    def load_model(self) -> bool:
        """
        加载训练好的模型
        
        Returns:
            bool: 是否成功加载模型
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                return False
                
            # 加载模型和相关组件
            model_data = joblib.load(self.model_path)
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_columns = model_data.get('feature_columns')
                self.scaler = model_data.get('scaler')
            else:
                # 兼容旧版本只保存模型的情况
                self.model = model_data
                
            logger.info(f"成功加载模型: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
            
    @idempotent_retry(max_retries=3)
    def get_latest_factors(self, trade_date: str = None) -> pd.DataFrame:
        """
        获取最新的因子数据
        
        Args:
            trade_date: 交易日期，如果为None则使用最新交易日
            
        Returns:
            pd.DataFrame: 因子数据
        """
        try:
            if trade_date is None:
                # 获取最新交易日
                query = """
                SELECT MAX(trade_date) as latest_date 
                FROM factor_library 
                WHERE factor_value IS NOT NULL
                """
                result = self.db_manager.execute_query(query)
                if result.empty:
                    logger.error("未找到因子数据")
                    return pd.DataFrame()
                trade_date = result.iloc[0]['latest_date']
                
            # 获取指定日期的因子数据
            query = """
            SELECT 
                ts_code,
                factor_name,
                factor_value,
                factor_category
            FROM factor_library 
            WHERE trade_date = %s
            AND factor_value IS NOT NULL
            ORDER BY ts_code, factor_name
            """
            
            factors_df = self.db_manager.execute_query(query, (trade_date,))
            
            if factors_df.empty:
                logger.warning(f"日期 {trade_date} 没有因子数据")
                return pd.DataFrame()
                
            # 透视表转换
            pivot_df = factors_df.pivot_table(
                index='ts_code',
                columns='factor_name',
                values='factor_value',
                aggfunc='first'
            ).reset_index()
            
            logger.info(f"获取到 {len(pivot_df)} 只股票的因子数据")
            return pivot_df
            
        except Exception as e:
            logger.error(f"获取因子数据失败: {e}")
            return pd.DataFrame()
            
    def prepare_prediction_data(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        准备预测数据
        
        Args:
            factors_df: 原始因子数据
            
        Returns:
            pd.DataFrame: 处理后的预测数据
        """
        try:
            if factors_df.empty:
                return pd.DataFrame()
                
            # 如果有特征列信息，只保留训练时使用的特征
            if self.feature_columns is not None:
                # 确保所有需要的特征列都存在
                missing_cols = set(self.feature_columns) - set(factors_df.columns)
                if missing_cols:
                    logger.warning(f"缺少特征列: {missing_cols}")
                    # 用0填充缺失的特征列
                    for col in missing_cols:
                        factors_df[col] = 0
                        
                # 只保留训练时使用的特征列（除了ts_code）
                feature_cols = ['ts_code'] + [col for col in self.feature_columns if col in factors_df.columns]
                factors_df = factors_df[feature_cols]
                
            # 数据清洗
            # 1. 处理无穷大值
            factors_df = factors_df.replace([np.inf, -np.inf], np.nan)
            
            # 2. 填充缺失值
            numeric_cols = factors_df.select_dtypes(include=[np.number]).columns
            factors_df[numeric_cols] = factors_df[numeric_cols].fillna(0)
            
            # 3. 如果有scaler，进行标准化
            if self.scaler is not None:
                feature_cols = [col for col in factors_df.columns if col != 'ts_code']
                factors_df[feature_cols] = self.scaler.transform(factors_df[feature_cols])
                
            logger.info(f"预测数据准备完成，共 {len(factors_df)} 只股票")
            return factors_df
            
        except Exception as e:
            logger.error(f"准备预测数据失败: {e}")
            return pd.DataFrame()
            
    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        进行预测
        
        Args:
            data: 预测数据
            
        Returns:
            pd.DataFrame: 预测结果
        """
        try:
            if data.empty or self.model is None:
                return pd.DataFrame()
                
            # 准备特征数据
            feature_cols = [col for col in data.columns if col != 'ts_code']
            X = data[feature_cols].values
            
            # 进行预测
            if hasattr(self.model, 'predict_proba'):
                # 分类模型，获取概率
                predictions = self.model.predict_proba(X)[:, 1]  # 获取正类概率
                pred_labels = self.model.predict(X)
            else:
                # 回归模型
                predictions = self.model.predict(X)
                pred_labels = (predictions > 0).astype(int)  # 转换为二分类标签
                
            # 创建结果DataFrame
            results = pd.DataFrame({
                'ts_code': data['ts_code'],
                'prediction_score': predictions,
                'prediction_label': pred_labels,
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'model_version': os.path.basename(self.model_path)
            })
            
            # 按预测分数排序
            results = results.sort_values('prediction_score', ascending=False)
            
            logger.info(f"完成 {len(results)} 只股票的预测")
            return results
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return pd.DataFrame()
            
    @idempotent_retry(max_retries=3)
    def save_predictions(self, predictions: pd.DataFrame) -> bool:
        """
        保存预测结果到数据库
        
        Args:
            predictions: 预测结果
            
        Returns:
            bool: 是否成功保存
        """
        try:
            if predictions.empty:
                return False
                
            # 创建预测结果表（如果不存在）
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS prediction_results (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                ts_code VARCHAR(20) NOT NULL,
                prediction_score DECIMAL(10, 6),
                prediction_label INT,
                prediction_date DATE NOT NULL,
                model_version VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_prediction_date (prediction_date),
                INDEX idx_ts_code (ts_code),
                INDEX idx_score (prediction_score DESC)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
            
            self.db_manager.execute_query(create_table_sql)
            
            # 删除当天的旧预测结果
            today = datetime.now().strftime('%Y-%m-%d')
            delete_sql = "DELETE FROM prediction_results WHERE prediction_date = %s"
            self.db_manager.execute_query(delete_sql, (today,))
            
            # 插入新的预测结果
            insert_sql = """
            INSERT INTO prediction_results 
            (ts_code, prediction_score, prediction_label, prediction_date, model_version)
            VALUES (%s, %s, %s, %s, %s)
            """
            
            data_to_insert = [
                (
                    row['ts_code'],
                    float(row['prediction_score']),
                    int(row['prediction_label']),
                    row['prediction_date'],
                    row['model_version']
                )
                for _, row in predictions.iterrows()
            ]
            
            self.db_manager.execute_many(insert_sql, data_to_insert)
            
            logger.info(f"成功保存 {len(predictions)} 条预测结果")
            return True
            
        except Exception as e:
            logger.error(f"保存预测结果失败: {e}")
            return False
            
    def get_stock_info(self, ts_codes: List[str]) -> pd.DataFrame:
        """
        获取股票基本信息
        
        Args:
            ts_codes: 股票代码列表
            
        Returns:
            pd.DataFrame: 股票信息
        """
        try:
            if not ts_codes:
                return pd.DataFrame()
                
            placeholders = ','.join(['%s'] * len(ts_codes))
            query = f"""
            SELECT ts_code, symbol, name, area, industry, market
            FROM stock_basic 
            WHERE ts_code IN ({placeholders})
            AND list_status = 'L'
            """
            
            return self.db_manager.execute_query(query, ts_codes)
            
        except Exception as e:
            logger.error(f"获取股票信息失败: {e}")
            return pd.DataFrame()
            
    def generate_prediction_report(self, predictions: pd.DataFrame) -> Dict:
        """
        生成预测报告
        
        Args:
            predictions: 预测结果
            
        Returns:
            Dict: 预测报告
        """
        try:
            if predictions.empty:
                return {}
                
            # 获取股票信息
            top_stocks = predictions.head(self.prediction_config['top_n_stocks'])
            stock_info = self.get_stock_info(top_stocks['ts_code'].tolist())
            
            # 合并信息
            report_data = top_stocks.merge(stock_info, on='ts_code', how='left')
            
            # 统计信息
            stats = {
                'total_predictions': len(predictions),
                'positive_predictions': len(predictions[predictions['prediction_label'] == 1]),
                'negative_predictions': len(predictions[predictions['prediction_label'] == 0]),
                'avg_score': float(predictions['prediction_score'].mean()),
                'max_score': float(predictions['prediction_score'].max()),
                'min_score': float(predictions['prediction_score'].min()),
                'prediction_date': predictions['prediction_date'].iloc[0],
                'model_version': predictions['model_version'].iloc[0]
            }
            
            # 按行业分组统计
            if not stock_info.empty:
                industry_stats = report_data.groupby('industry').agg({
                    'prediction_score': ['count', 'mean'],
                    'prediction_label': 'sum'
                }).round(4)
                
                industry_stats.columns = ['stock_count', 'avg_score', 'positive_count']
                industry_stats = industry_stats.reset_index()
                stats['industry_stats'] = industry_stats.to_dict('records')
                
            return {
                'statistics': stats,
                'top_predictions': report_data.to_dict('records'),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"生成预测报告失败: {e}")
            return {}
            
    def run_daily_prediction(self, trade_date: str = None) -> Dict:
        """
        运行每日预测流程
        
        Args:
            trade_date: 交易日期，如果为None则使用最新日期
            
        Returns:
            Dict: 预测结果和报告
        """
        try:
            logger.info("开始每日预测流程")
            
            # 1. 加载模型
            if not self.load_model():
                return {'error': '模型加载失败'}
                
            # 2. 获取最新因子数据
            factors_df = self.get_latest_factors(trade_date)
            if factors_df.empty:
                return {'error': '无法获取因子数据'}
                
            # 3. 准备预测数据
            prediction_data = self.prepare_prediction_data(factors_df)
            if prediction_data.empty:
                return {'error': '预测数据准备失败'}
                
            # 4. 进行预测
            predictions = self.make_predictions(prediction_data)
            if predictions.empty:
                return {'error': '预测失败'}
                
            # 5. 保存预测结果
            if not self.save_predictions(predictions):
                logger.warning("保存预测结果失败，但预测已完成")
                
            # 6. 生成预测报告
            report = self.generate_prediction_report(predictions)
            
            logger.info("每日预测流程完成")
            return {
                'success': True,
                'predictions': predictions.to_dict('records'),
                'report': report
            }
            
        except Exception as e:
            logger.error(f"每日预测流程失败: {e}")
            return {'error': str(e)}
            
def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='股票每日预测脚本')
    parser.add_argument('--model-path', type=str, help='模型文件路径')
    parser.add_argument('--trade-date', type=str, help='交易日期 (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='输出文件路径')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = StockPredictor(model_path=args.model_path)
    
    # 运行预测
    result = predictor.run_daily_prediction(trade_date=args.trade_date)
    
    if 'error' in result:
        logger.error(f"预测失败: {result['error']}")
        sys.exit(1)
        
    # 输出结果
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"预测结果已保存到: {args.output}")
    else:
        # 打印前10个预测结果
        print("\n=== 预测结果 ===")
        if 'report' in result and 'top_predictions' in result['report']:
            for i, pred in enumerate(result['report']['top_predictions'][:10], 1):
                print(f"{i:2d}. {pred.get('name', pred['ts_code'])} ({pred['ts_code']}) - "
                      f"分数: {pred['prediction_score']:.4f}, "
                      f"标签: {pred['prediction_label']}, "
                      f"行业: {pred.get('industry', 'N/A')}")
                      
        # 打印统计信息
        if 'report' in result and 'statistics' in result['report']:
            stats = result['report']['statistics']
            print(f"\n=== 统计信息 ===")
            print(f"总预测数量: {stats['total_predictions']}")
            print(f"看涨预测: {stats['positive_predictions']}")
            print(f"看跌预测: {stats['negative_predictions']}")
            print(f"平均分数: {stats['avg_score']:.4f}")
            print(f"预测日期: {stats['prediction_date']}")
            
if __name__ == '__main__':
    main()
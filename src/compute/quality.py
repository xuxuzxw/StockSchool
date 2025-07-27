import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from sqlalchemy import text
import sys
import os
from utils.config_loader import config

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def detect_outliers(data: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
    """
    检测异常值
    
    Args:
        data: 数据DataFrame
        column: 要检测的列名
        method: 检测方法 ('iqr', 'zscore', '3sigma')
    
    Returns:
        布尔序列，True表示异常值
    """
    if method == 'iqr':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        outliers = z_scores > 3
        
    elif method == '3sigma':
        mean = data[column].mean()
        std_dev = data[column].std()
        outliers = (data[column] - mean).abs() > 3 * std_dev
        
    else:
        raise ValueError(f"不支持的检测方法: {method}")
    
    return outliers

def fill_missing_values(data: pd.DataFrame, column: str, method: str = 'forward') -> pd.Series:
    """
    填充缺失值
    
    Args:
        data: 数据DataFrame
        column: 要填充的列名
        method: 填充方法 ('forward', 'backward', 'mean', 'median', 'industry_mean')
    
    Returns:
        填充后的序列
    """
    filled = data[column].copy()
    
    if method == 'forward':
        filled = filled.ffill()
        
    elif method == 'backward':
        filled = filled.bfill()
        
    elif method == 'mean':
        filled = filled.fillna(data[column].mean())
        
    elif method == 'median':
        filled = filled.fillna(data[column].median())
        
    elif method == 'industry_mean':
        # 优先使用前向填充
        filled = filled.ffill()
        
        # 如果还有缺失，使用行业均值填充 (假设data中已有'industry'列)
        if filled.isnull().any() and 'industry' in data.columns:
            industry_mean = data.groupby('industry')[column].transform('mean')
            filled.fillna(industry_mean, inplace=True)
        
        # 如果还有缺失，使用全局均值
        if filled.isnull().any():
            filled.fillna(data[column].mean(), inplace=True)
    
    return filled

def validate_price_data(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    验证价格数据的合理性
    
    Args:
        df: 包含OHLC数据的DataFrame
    
    Returns:
        包含各种错误类型的字典
    """
    errors = {
        'negative_prices': [],
        'invalid_ohlc': [],
        'extreme_changes': [],
        'zero_volume': []
    }
    
    # 检查负价格
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in df.columns:
            negative_mask = df[col] < 0
            if negative_mask.any():
                errors['negative_prices'].extend(
                    df[negative_mask].index.tolist()
                )
    
    # 检查OHLC逻辑关系
    if all(col in df.columns for col in price_columns):
        # high应该是最高价
        invalid_high = (df['high'] < df['open']) | (df['high'] < df['close']) | (df['high'] < df['low'])
        # low应该是最低价
        invalid_low = (df['low'] > df['open']) | (df['low'] > df['close']) | (df['low'] > df['high'])
        
        invalid_ohlc = invalid_high | invalid_low
        if invalid_ohlc.any():
            errors['invalid_ohlc'].extend(
                df[invalid_ohlc].index.tolist()
            )
    
    # 检查极端价格变化 (单日涨跌幅超过20%)
    if 'close' in df.columns and 'pre_close' in df.columns:
        pct_change = ((df['close'] - df['pre_close']) / df['pre_close']).abs()
        extreme_change = pct_change > 0.2
        if extreme_change.any():
            errors['extreme_changes'].extend(
                df[extreme_change].index.tolist()
            )
    
    # 检查零成交量
    if 'vol' in df.columns:
        zero_vol = df['vol'] == 0
        if zero_vol.any():
            errors['zero_volume'].extend(
                df[zero_vol].index.tolist()
            )
    
    return errors

def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗价格数据
    
    Args:
        df: 原始价格数据DataFrame
    
    Returns:
        清洗后的DataFrame
    """
    cleaned_df = df.copy()
    
    # 填充缺失值
    price_columns = ['open', 'high', 'low', 'close', 'pre_close']
    for col in price_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = fill_missing_values(cleaned_df, col, 'forward')
    
    # 处理成交量缺失值
    if 'vol' in cleaned_df.columns:
        cleaned_df['vol'] = fill_missing_values(cleaned_df, 'vol', 'median')
    
    # 处理成交额缺失值
    if 'amount' in cleaned_df.columns:
        cleaned_df['amount'] = fill_missing_values(cleaned_df, 'amount', 'median')
    
    # 检测并处理异常值
    for col in price_columns:
        if col in cleaned_df.columns:
            outliers = detect_outliers(cleaned_df, col, 'iqr')
            if outliers.any():
                logger.warning(f"检测到 {outliers.sum()} 个 {col} 异常值")
                # 用前一个有效值替换异常值
                cleaned_df.loc[outliers, col] = cleaned_df[col].ffill()
    
    return cleaned_df

def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算数据质量评分
    
    Args:
        df: 数据DataFrame
    
    Returns:
        包含各项质量指标的字典
    """
    total_rows = len(df)
    if total_rows == 0:
        return {'overall_score': 0.0}
    
    scores = {}
    
    # 完整性评分 (缺失值比例)
    missing_ratio = df.isnull().sum().sum() / (total_rows * len(df.columns))
    scores['completeness'] = max(0, 1 - missing_ratio)
    
    # 一致性评分 (OHLC逻辑关系)
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        errors = validate_price_data(df)
        error_count = sum(len(error_list) for error_list in errors.values())
        scores['consistency'] = max(0, 1 - error_count / total_rows)
    else:
        scores['consistency'] = 1.0
    
    # 准确性评分 (异常值比例)
    outlier_count = 0
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        outliers = detect_outliers(df, col, 'iqr')
        outlier_count += outliers.sum()
    
    if len(numeric_columns) > 0:
        scores['accuracy'] = max(0, 1 - outlier_count / (total_rows * len(numeric_columns)))
    else:
        scores['accuracy'] = 1.0
    
    # 及时性评分 (数据更新频率)
    if 'trade_date' in df.columns:
        df_sorted = df.sort_values('trade_date')
        date_diffs = df_sorted['trade_date'].diff().dt.days
        # 假设正常交易日间隔为1-3天
        normal_intervals = (date_diffs >= 1) & (date_diffs <= 3)
        scores['timeliness'] = normal_intervals.sum() / len(date_diffs.dropna()) if len(date_diffs.dropna()) > 0 else 1.0
    else:
        scores['timeliness'] = 1.0
    
    # 综合评分
    scores['overall_score'] = np.mean(list(scores.values()))
    
    return scores

def generate_quality_report(df: pd.DataFrame, stock_code: str = None) -> str:
    """
    生成数据质量报告
    
    Args:
        df: 数据DataFrame
        stock_code: 股票代码
    
    Returns:
        质量报告字符串
    """
    report = []
    report.append("=" * 50)
    report.append(f"数据质量报告 - {stock_code or '未知股票'}")
    report.append("=" * 50)
    
    # 基本统计信息
    report.append(f"数据行数: {len(df)}")
    report.append(f"数据列数: {len(df.columns)}")
    
    if not df.empty:
        if 'trade_date' in df.columns:
            report.append(f"数据时间范围: {df['trade_date'].min()} 至 {df['trade_date'].max()}")
    
    # 缺失值统计
    missing_stats = df.isnull().sum()
    if missing_stats.sum() > 0:
        report.append("\n缺失值统计:")
        for col, count in missing_stats.items():
            if count > 0:
                percentage = (count / len(df)) * 100
                report.append(f"  {col}: {count} ({percentage:.2f}%)")
    else:
        report.append("\n✓ 无缺失值")
    
    # 数据验证结果
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        errors = validate_price_data(df)
        report.append("\n数据验证结果:")
        
        for error_type, error_indices in errors.items():
            if error_indices:
                report.append(f"  {error_type}: {len(error_indices)} 个错误")
            else:
                report.append(f"  ✓ {error_type}: 无错误")
    
    # 质量评分
    scores = calculate_data_quality_score(df)
    report.append("\n质量评分:")
    for metric, score in scores.items():
        report.append(f"  {metric}: {score:.3f}")
    
    report.append("=" * 50)
    
    return "\n".join(report)

class DataQualityMonitor:
    """
    数据质量监控器
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.quality_thresholds = {
            'completeness': 0.95,
            'consistency': 0.98,
            'accuracy': 0.90,
            'timeliness': 0.85,
            'overall_score': 0.90
        }
    
    def check_stock_data_quality(self, ts_code: str, days: int = 30) -> Dict:
        """
        检查单只股票的数据质量
        
        Args:
            ts_code: 股票代码
            days: 检查最近多少天的数据
        
        Returns:
            质量检查结果
        """
        query = """
        SELECT * FROM stock_daily 
        WHERE ts_code = :ts_code 
        AND trade_date >= CURRENT_DATE - INTERVAL ':days days'
        ORDER BY trade_date DESC
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(
                text(query), 
                conn, 
                params={'ts_code': ts_code, 'days': days}
            )
        
        if df.empty:
            return {
                'ts_code': ts_code,
                'status': 'NO_DATA',
                'message': '无数据',
                'scores': {}
            }
        
        # 计算质量评分
        scores = calculate_data_quality_score(df)
        
        # 检查是否达到阈值
        alerts = []
        for metric, score in scores.items():
            if metric in self.quality_thresholds:
                threshold = self.quality_thresholds[metric]
                if score < threshold:
                    alerts.append(f"{metric}: {score:.3f} < {threshold}")
        
        status = 'PASS' if not alerts else 'ALERT'
        
        return {
            'ts_code': ts_code,
            'status': status,
            'scores': scores,
            'alerts': alerts,
            'data_count': len(df),
            'report': generate_quality_report(df, ts_code)
        }
    
    def batch_quality_check(self, stock_codes: List[str] = None, sample_size: int = None) -> List[Dict]:
        """
        批量质量检查
        
        Args:
            stock_codes: 股票代码列表，如果为None则随机抽样
            sample_size: 抽样大小
        
        Returns:
            质量检查结果列表
        """
        if stock_codes is None:
            # 从配置获取样本大小
            if sample_size is None:
                sample_size = config.get('quality_params.sample_size', 100)
            
            # 随机抽样股票
            query = """
            SELECT DISTINCT ts_code FROM stock_daily 
            ORDER BY RANDOM() 
            LIMIT :sample_size
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'sample_size': sample_size})
                stock_codes = [row[0] for row in result]
        
        results = []
        for ts_code in stock_codes:
            try:
                result = self.check_stock_data_quality(ts_code)
                results.append(result)
                logger.info(f"质量检查完成: {ts_code} - {result['status']}")
            except Exception as e:
                logger.error(f"质量检查失败: {ts_code} - {e}")
                results.append({
                    'ts_code': ts_code,
                    'status': 'ERROR',
                    'message': str(e),
                    'scores': {}
                })
        
        return results

if __name__ == '__main__':
    # 测试代码
    from src.utils.db import get_db_engine
    
    print("测试数据质量控制模块...")
    
    # 初始化数据库连接
    engine = get_db_engine()
    
    # 创建质量监控器
    monitor = DataQualityMonitor(engine)
    
    # 测试单只股票质量检查
    result = monitor.check_stock_data_quality('000001.SZ', days=30)
    print(f"\n质量检查结果: {result['status']}")
    print(f"数据行数: {result.get('data_count', 0)}")
    
    if result['scores']:
        print("质量评分:")
        for metric, score in result['scores'].items():
            print(f"  {metric}: {score:.3f}")
    
    if result.get('alerts'):
        print("\n质量告警:")
        for alert in result['alerts']:
            print(f"  - {alert}")
    
    print("\n数据质量控制模块测试完成!")
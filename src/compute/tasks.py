from celery import Celery
from celery.schedules import crontab
import os
import sys
from typing import List, Optional
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.db import get_db_engine
from data.tushare_sync import TushareDataSync
from compute.factor_engine import FactorEngine
from compute.quality import DataQualityMonitor

# 创建Celery应用
app = Celery('stockschool')

# Celery配置
app.conf.update(
    broker_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30分钟超时
    task_soft_time_limit=25 * 60,  # 25分钟软超时
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_compression='gzip',
    result_compression='gzip',
    task_routes={
        'stockschool.data_sync.*': {'queue': 'data_sync'},
        'stockschool.factor_calc.*': {'queue': 'factor_calc'},
        'stockschool.quality_check.*': {'queue': 'quality_check'},
    },
    beat_schedule={
        # 每日数据同步任务
        'daily-data-sync': {
            'task': 'stockschool.sync_daily_data',
            'schedule': crontab(hour=18, minute=30),  # 每天18:30执行
        },
        # 每日因子计算任务
        'daily-factor-calculation': {
            'task': 'stockschool.calculate_daily_factors',
            'schedule': crontab(hour=19, minute=30),  # 每天19:30执行
        },
        # 每周数据质量检查
        'weekly-quality-check': {
            'task': 'stockschool.weekly_quality_check',
            'schedule': crontab(hour=20, minute=0, day_of_week=0),  # 每周日20:00执行
        },
        # 每月全量因子重算
        'monthly-factor-recalc': {
            'task': 'stockschool.monthly_factor_recalculation',
            'schedule': crontab(hour=2, minute=0, day_of_month=1),  # 每月1号02:00执行
        },
    }
)

@app.task(bind=True, name='stockschool.sync_daily_data')
def sync_daily_data(self, trade_date: Optional[str] = None):
    """
    同步每日数据任务
    
    Args:
        trade_date: 交易日期，格式YYYYMMDD，如果为None则同步最新交易日
    """
    try:
        logger.info(f"开始同步每日数据任务: {trade_date or '最新交易日'}")
        
        engine = get_db_engine()
        sync = TushareDataSync(engine)
        
        if trade_date is None:
            # 获取最新交易日
            trade_date = sync.get_latest_trade_date()
        
        # 同步基础数据
        sync.sync_stock_basic()
        sync.sync_trade_calendar()
        
        # 同步交易数据
        sync.sync_daily_data(trade_date)
        sync.sync_daily_basic(trade_date)
        sync.sync_adj_factor(trade_date)
        
        logger.info(f"每日数据同步完成: {trade_date}")
        return {'status': 'success', 'trade_date': trade_date}
        
    except Exception as e:
        logger.error(f"每日数据同步失败: {e}")
        self.retry(countdown=300, max_retries=3)  # 5分钟后重试，最多3次

@app.task(bind=True, name='stockschool.sync_stock_data')
def sync_stock_data(self, ts_code: str, start_date: str, end_date: str):
    """
    同步单只股票数据任务
    
    Args:
        ts_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    """
    try:
        logger.info(f"开始同步股票数据: {ts_code} ({start_date} - {end_date})")
        
        engine = get_db_engine()
        sync = TushareDataSync(engine)
        
        # 同步股票日线数据
        sync.sync_stock_daily(ts_code, start_date, end_date)
        
        # 同步每日基本面数据
        sync.sync_stock_daily_basic(ts_code, start_date, end_date)
        
        logger.info(f"股票数据同步完成: {ts_code}")
        return {'status': 'success', 'ts_code': ts_code}
        
    except Exception as e:
        logger.error(f"股票数据同步失败: {ts_code} - {e}")
        self.retry(countdown=60, max_retries=3)

@app.task(bind=True, name='stockschool.calculate_daily_factors')
def calculate_daily_factors(self, trade_date: Optional[str] = None):
    """
    计算每日因子任务
    
    Args:
        trade_date: 交易日期，如果为None则计算最新交易日
    """
    try:
        logger.info(f"开始计算每日因子: {trade_date or '最新交易日'}")
        
        engine = get_db_engine()
        factor_engine = FactorEngine(engine)
        
        if trade_date is None:
            # 获取最新交易日
            from data.tushare_sync import TushareDataSync
            sync = TushareDataSync(engine)
            trade_date = sync.get_latest_trade_date()
        
        # 获取所有活跃股票
        active_stocks = factor_engine.get_active_stocks(trade_date)
        
        success_count = 0
        error_count = 0
        
        for ts_code in active_stocks:
            try:
                factor_engine.calculate_stock_factors(ts_code, days=60)
                success_count += 1
                
                if success_count % 100 == 0:
                    logger.info(f"已完成 {success_count} 只股票的因子计算")
                    
            except Exception as e:
                logger.error(f"计算股票因子失败: {ts_code} - {e}")
                error_count += 1
        
        logger.info(f"每日因子计算完成: 成功 {success_count}, 失败 {error_count}")
        return {
            'status': 'success', 
            'trade_date': trade_date,
            'success_count': success_count,
            'error_count': error_count
        }
        
    except Exception as e:
        logger.error(f"每日因子计算失败: {e}")
        self.retry(countdown=600, max_retries=2)  # 10分钟后重试，最多2次

@app.task(bind=True, name='stockschool.calculate_stock_factors')
def calculate_stock_factors(self, ts_code: str, days: int = 60):
    """
    计算单只股票因子任务
    
    Args:
        ts_code: 股票代码
        days: 计算天数
    """
    try:
        logger.info(f"开始计算股票因子: {ts_code}")
        
        engine = get_db_engine()
        factor_engine = FactorEngine(engine)
        
        # 计算因子
        factors = factor_engine.calculate_stock_factors(ts_code, days)
        
        if factors is not None and not factors.empty:
            logger.info(f"股票因子计算完成: {ts_code}, 计算了 {len(factors)} 条记录")
            return {'status': 'success', 'ts_code': ts_code, 'factor_count': len(factors)}
        else:
            logger.warning(f"股票因子计算无结果: {ts_code}")
            return {'status': 'no_data', 'ts_code': ts_code}
        
    except Exception as e:
        logger.error(f"股票因子计算失败: {ts_code} - {e}")
        self.retry(countdown=30, max_retries=3)

@app.task(bind=True, name='stockschool.weekly_quality_check')
def weekly_quality_check(self, sample_size: int = 200):
    """
    每周数据质量检查任务
    
    Args:
        sample_size: 抽样检查的股票数量
    """
    try:
        logger.info(f"开始每周数据质量检查，抽样数量: {sample_size}")
        
        engine = get_db_engine()
        monitor = DataQualityMonitor(engine)
        
        # 批量质量检查
        results = monitor.batch_quality_check(sample_size=sample_size)
        
        # 统计结果
        total_count = len(results)
        pass_count = sum(1 for r in results if r['status'] == 'PASS')
        alert_count = sum(1 for r in results if r['status'] == 'ALERT')
        error_count = sum(1 for r in results if r['status'] == 'ERROR')
        
        # 计算平均质量评分
        valid_scores = [r['scores'].get('overall_score', 0) for r in results if r['scores']]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
        logger.info(f"数据质量检查完成: 总数 {total_count}, 通过 {pass_count}, 告警 {alert_count}, 错误 {error_count}")
        logger.info(f"平均质量评分: {avg_score:.3f}")
        
        return {
            'status': 'success',
            'total_count': total_count,
            'pass_count': pass_count,
            'alert_count': alert_count,
            'error_count': error_count,
            'avg_score': avg_score
        }
        
    except Exception as e:
        logger.error(f"数据质量检查失败: {e}")
        self.retry(countdown=300, max_retries=2)

@app.task(bind=True, name='stockschool.monthly_factor_recalculation')
def monthly_factor_recalculation(self):
    """
    每月全量因子重算任务
    """
    try:
        logger.info("开始每月全量因子重算")
        
        engine = get_db_engine()
        factor_engine = FactorEngine(engine)
        
        # 获取所有股票
        all_stocks = factor_engine.get_all_stocks()
        
        success_count = 0
        error_count = 0
        
        for ts_code in all_stocks:
            try:
                # 重算最近250天的因子
                factor_engine.calculate_stock_factors(ts_code, days=250)
                success_count += 1
                
                if success_count % 50 == 0:
                    logger.info(f"已完成 {success_count}/{len(all_stocks)} 只股票的因子重算")
                    
            except Exception as e:
                logger.error(f"重算股票因子失败: {ts_code} - {e}")
                error_count += 1
        
        logger.info(f"每月因子重算完成: 成功 {success_count}, 失败 {error_count}")
        return {
            'status': 'success',
            'success_count': success_count,
            'error_count': error_count
        }
        
    except Exception as e:
        logger.error(f"每月因子重算失败: {e}")
        self.retry(countdown=1800, max_retries=1)  # 30分钟后重试，最多1次

@app.task(bind=True, name='stockschool.data_quality_check')
def data_quality_check(self, ts_code: str, days: int = 30):
    """
    单只股票数据质量检查任务
    
    Args:
        ts_code: 股票代码
        days: 检查天数
    """
    try:
        logger.info(f"开始数据质量检查: {ts_code}")
        
        engine = get_db_engine()
        monitor = DataQualityMonitor(engine)
        
        result = monitor.check_stock_data_quality(ts_code, days)
        
        logger.info(f"数据质量检查完成: {ts_code} - {result['status']}")
        return result
        
    except Exception as e:
        logger.error(f"数据质量检查失败: {ts_code} - {e}")
        self.retry(countdown=60, max_retries=2)

# 批量任务组合
@app.task(bind=True, name='stockschool.batch_sync_stocks')
def batch_sync_stocks(self, stock_codes: List[str], start_date: str, end_date: str):
    """
    批量同步股票数据
    
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
    """
    try:
        logger.info(f"开始批量同步 {len(stock_codes)} 只股票数据")
        
        # 创建子任务
        job = app.group([
            sync_stock_data.s(ts_code, start_date, end_date) 
            for ts_code in stock_codes
        ])
        
        # 执行批量任务
        result = job.apply_async()
        
        # 等待所有任务完成
        results = result.get()
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        
        logger.info(f"批量同步完成: 成功 {success_count}/{len(stock_codes)}")
        return {
            'status': 'success',
            'total_count': len(stock_codes),
            'success_count': success_count
        }
        
    except Exception as e:
        logger.error(f"批量同步失败: {e}")
        self.retry(countdown=300, max_retries=2)

@app.task(bind=True, name='stockschool.batch_calculate_factors')
def batch_calculate_factors(self, stock_codes: List[str], days: int = 60):
    """
    批量计算股票因子
    
    Args:
        stock_codes: 股票代码列表
        days: 计算天数
    """
    try:
        logger.info(f"开始批量计算 {len(stock_codes)} 只股票因子")
        
        # 创建子任务
        job = app.group([
            calculate_stock_factors.s(ts_code, days) 
            for ts_code in stock_codes
        ])
        
        # 执行批量任务
        result = job.apply_async()
        
        # 等待所有任务完成
        results = result.get()
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        
        logger.info(f"批量因子计算完成: 成功 {success_count}/{len(stock_codes)}")
        return {
            'status': 'success',
            'total_count': len(stock_codes),
            'success_count': success_count
        }
        
    except Exception as e:
        logger.error(f"批量因子计算失败: {e}")
        self.retry(countdown=300, max_retries=2)

if __name__ == '__main__':
    # 启动Celery Worker
    app.start()
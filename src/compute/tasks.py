import eventlet
eventlet.monkey_patch()
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

from src.utils.config_loader import config
from src.utils.db import get_db_engine
from src.data.tushare_sync import TushareSynchronizer as TushareDataSync
from src.compute.factor_engine import FactorEngine
from src.compute.quality import DataQualityMonitor

# 创建Celery应用
app = Celery('stockschool')

# Celery配置
app.conf.update(
        broker_url=f"redis://:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/{os.getenv('REDIS_DB')}",
    result_backend=f"redis://:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/{os.getenv('REDIS_DB')}",
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=config.get('task_params.timeout_minutes', 30) * 60,  # N分钟超时
    task_soft_time_limit=config.get('task_params.soft_timeout_minutes', 25) * 60,  # N分钟软超时
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
            'schedule': crontab(hour=config.get('task_params.schedule_hour', 20), minute=0, day_of_week=0),  # 每周日N:00执行
        },
        # 每月全量因子重算
        'monthly-factor-recalc': {
            'task': 'stockschool.monthly_factor_recalculation',
            'schedule': crontab(hour=2, minute=0, day_of_month=1),  # 每月1号02:00执行
        },
    }
)

@app.task(bind=True, name='stockschool.sync_daily_data')
def sync_daily_data(self, trade_date: Optional[str] = None, test_mode: bool = False):
    """
    同步每日数据任务
    
    Args:
        trade_date: 交易日期，格式YYYYMMDD，如果为None则同步最新交易日
    """
    try:
        logger.info(f"开始同步每日数据任务: {trade_date or '最新交易日'}")
        
        sync = TushareDataSync()
        
        if trade_date is None:
            # 获取最新交易日
            trade_date = sync.get_last_trade_date()
        
        # 同步基础数据
        self.update_state(state='PROGRESS', meta={'current_step': '同步基础数据', 'progress': '20%'}) # 添加进度更新
        logger.info("开始同步基础数据...")
        sync.sync_stock_basic()
        sync.sync_trade_calendar()
        logger.info("基础数据同步完成。")
        
        # 同步交易数据
        self.update_state(state='PROGRESS', meta={'current_step': '同步交易数据', 'progress': '50%'}) # 添加进度更新
        logger.info("开始同步交易数据...")
        if test_mode:
            logger.info("测试模式下，每日数据同步仅同步最近一天数据。")
            sync.update_daily_data(max_days=1)
            sync.sync_indicator_data(start_date=trade_date, end_date=trade_date)
        else:
            sync.update_daily_data(max_days=100)
            sync.sync_indicator_data(start_date=trade_date)
        logger.info("交易数据同步完成。")
        
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
        
        sync = TushareDataSync()
        
        # 同步股票日线数据（使用update_daily_data方法）
        sync.update_daily_data(max_days=30)
        
        # 同步每日基本面数据（使用sync_indicator_data方法）
        sync.sync_indicator_data(start_date=start_date)
        
        logger.info(f"股票数据同步完成: {ts_code}")
        return {'status': 'success', 'ts_code': ts_code}
        
    except Exception as e:
        logger.error(f"股票数据同步失败: {ts_code} - {e}")
        retry_countdown = config.get('task_params.retry_countdown', 60)
        max_retries = config.get('task_params.max_retries', 3)
        self.retry(countdown=retry_countdown, max_retries=max_retries)

@app.task(bind=True, name='stockschool.calculate_daily_factors')
def calculate_daily_factors(self, trade_date: Optional[str] = None, test_mode: bool = False):
    """
    计算每日因子任务
    
    Args:
        trade_date: 交易日期，如果为None则计算最新交易日
    """
    try:
        logger.info(f"开始计算每日因子: {trade_date or '最新交易日'}")
        
        factor_engine = FactorEngine()
        
        if trade_date is None:
            # 获取最新交易日
            sync = TushareDataSync()
            trade_date = sync.get_last_trade_date()
        
        # 获取所有活跃股票
        if test_mode:
            logger.info("测试模式下，每日因子计算仅计算少量股票。")
            active_stocks = factor_engine.get_all_stocks()[:10] # 只取前10只股票进行测试
        else:
            active_stocks = factor_engine.get_all_stocks()
        
        success_count = 0
        error_count = 0
        total_stocks = len(active_stocks)
        
        for i, ts_code in enumerate(active_stocks):
            try:
                min_data_days = config.get('factor_params.min_data_days', 60)
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=min_data_days)).strftime('%Y-%m-%d')
                
                success = factor_engine.calculate_stock_factors(ts_code, start_date, end_date)
                if success:
                    success_count += 1
                else:
                    error_count += 1
                
                # 更新进度
                progress_percentage = int(((i + 1) / total_stocks) * 100)
                self.update_state(state='PROGRESS', meta={'current_stock': ts_code, 'processed_count': i + 1, 'total_count': total_stocks, 'progress': f'{progress_percentage}%'})
                
                progress_interval = config.get('data_sync_params.progress_interval', 100)
                if (i + 1) % progress_interval == 0 or (i + 1) == total_stocks:
                    logger.info(f"已完成 {i + 1}/{total_stocks} 只股票的因子计算 ({progress_percentage}%) - 当前股票: {ts_code}")
                    
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
def calculate_stock_factors(self, ts_code: str, days: int = None):
    if days is None:
        days = config.get('factor_params.min_data_days', 60)
    """
    计算单只股票因子任务
    
    Args:
        ts_code: 股票代码
        days: 计算天数
    """
    try:
        logger.info(f"开始计算股票因子: {ts_code}")
        
        factor_engine = FactorEngine()
        
        # 计算因子 - 将days转换为日期范围
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        success = factor_engine.calculate_stock_factors(ts_code, start_date, end_date)
        
        if success:
            logger.info(f"股票因子计算完成: {ts_code}")
            return {'status': 'success', 'ts_code': ts_code}
        else:
            logger.warning(f"股票因子计算失败: {ts_code}")
            return {'status': 'failed', 'ts_code': ts_code}
        
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
        
        factor_engine = FactorEngine()
        
        # 获取所有股票
        all_stocks = factor_engine.get_all_stocks()
        
        success_count = 0
        error_count = 0
        
        for ts_code in all_stocks:
            try:
                # 重算最近250天的因子
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
                
                success = factor_engine.calculate_stock_factors(ts_code, start_date, end_date)
                if success:
                    success_count += 1
                else:
                    error_count += 1
                
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
def batch_calculate_factors(self, stock_codes: List[str], days: int = None):
    """
    批量计算股票因子
    
    Args:
        stock_codes: 股票代码列表
        days: 计算天数
    """
    if days is None:
        days = config.get('factor_params.min_data_days', 60)
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

# AI模型相关任务
@app.task(bind=True, name='stockschool.train_ai_model')
def train_ai_model(self, start_date: str, end_date: str, stock_pool: List[str] = None, 
                  model_type: str = 'lightgbm'):
    """
    AI模型训练任务
    
    Args:
        start_date: 训练数据开始日期
        end_date: 训练数据结束日期
        stock_pool: 股票池
        model_type: 模型类型
    """
    try:
        logger.info(f"开始训练AI模型: {model_type}, 数据期间: {start_date} 到 {end_date}")
        
        from ..strategy.ai_model import AIModelTrainer
        
        # 创建训练器
        trainer = AIModelTrainer()
        
        # 准备训练数据
        training_data = trainer.prepare_training_data(start_date, end_date, stock_pool)
        
        if training_data.empty:
            logger.warning("训练数据为空")
            return {'status': 'no_data', 'message': '训练数据为空'}
        
        # 训练模型
        result = trainer.train_model(training_data, model_type)
        
        # 保存模型
        model_path = trainer.save_model(model_type)
        
        logger.info(f"AI模型训练完成: {model_type}, R²: {result['metrics']['r2_score']:.4f}")
        
        return {
            'status': 'success',
            'model_type': model_type,
            'model_path': model_path,
            'metrics': result['metrics'],
            'training_samples': result['training_samples'],
            'test_samples': result['test_samples']
        }
        
    except Exception as e:
        logger.error(f"AI模型训练失败: {e}")
        self.retry(countdown=600, max_retries=2)  # 10分钟后重试

@app.task(bind=True, name='stockschool.run_prediction')
def run_prediction(self, stock_codes: List[str], trade_date: str, model_path: str = None):
    """
    AI模型预测任务
    
    Args:
        stock_codes: 股票代码列表
        trade_date: 预测日期
        model_path: 模型文件路径
    """
    try:
        logger.info(f"开始AI预测: {len(stock_codes)} 只股票, 日期: {trade_date}")
        
        from ..strategy.ai_model import AIModelPredictor
        
        # 创建预测器
        predictor = AIModelPredictor(model_path)
        
        # 执行预测
        predictions = predictor.predict(stock_codes, trade_date)
        
        if not predictions:
            logger.warning("预测结果为空")
            return {'status': 'no_data', 'message': '预测结果为空'}
        
        # 保存预测结果到数据库
        engine = get_db_engine()
        
        prediction_records = []
        for ts_code, predicted_return in predictions.items():
            prediction_records.append({
                'ts_code': ts_code,
                'trade_date': trade_date,
                'predicted_return': predicted_return,
                'prediction_time': datetime.now(),
                'model_path': model_path
            })
        
        # 插入预测结果
        if prediction_records:
            df = pd.DataFrame(prediction_records)
            df.to_sql('ai_predictions', engine, if_exists='append', index=False)
        
        logger.info(f"AI预测完成: {len(predictions)} 只股票")
        
        return {
            'status': 'success',
            'prediction_count': len(predictions),
            'trade_date': trade_date,
            'predictions': predictions
        }
        
    except Exception as e:
        logger.error(f"AI预测失败: {e}")
        self.retry(countdown=300, max_retries=3)  # 5分钟后重试

@app.task(bind=True, name='stockschool.batch_prediction')
def batch_prediction(self, stock_codes: List[str], start_date: str, end_date: str, 
                    model_path: str = None):
    """
    批量AI预测任务
    
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        model_path: 模型文件路径
    """
    try:
        logger.info(f"开始批量AI预测: {len(stock_codes)} 只股票, {start_date} 到 {end_date}")
        
        from ..strategy.ai_model import AIModelPredictor
        
        # 创建预测器
        predictor = AIModelPredictor(model_path)
        
        # 批量预测
        predictions_df = predictor.batch_predict(stock_codes, start_date, end_date)
        
        if predictions_df.empty:
            logger.warning("批量预测结果为空")
            return {'status': 'no_data', 'message': '批量预测结果为空'}
        
        # 保存预测结果到数据库
        engine = get_db_engine()
        predictions_df.to_sql('ai_predictions', engine, if_exists='append', index=False)
        
        logger.info(f"批量AI预测完成: {len(predictions_df)} 条预测记录")
        
        return {
            'status': 'success',
            'prediction_count': len(predictions_df),
            'start_date': start_date,
            'end_date': end_date
        }
        
    except Exception as e:
        logger.error(f"批量AI预测失败: {e}")
        self.retry(countdown=600, max_retries=2)  # 10分钟后重试

if __name__ == '__main__':
    # 启动Celery Worker
    app.start()
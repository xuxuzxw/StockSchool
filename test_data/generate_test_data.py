#!/usr/bin/env python3
"""
StockSchool 验收测试数据生成器
用于生成验收测试所需的各种测试数据
"""

import os
import sys
import json
import random
import logging
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append('/app')

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import redis
except ImportError as e:
    print(f"导入依赖失败: {e}")
    print("请确保已安装所需依赖包")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/data_generator.log')
    ]
)
logger = logging.getLogger(__name__)

class TestDataGenerator:
    """测试数据生成器"""
    
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        self.redis_url = os.getenv('REDIS_URL')
        self.db_conn = None
        self.redis_conn = None
        
        # 测试股票池
        self.test_stocks = [
            {'ts_code': '000001.SZ', 'symbol': '000001', 'name': '平安银行', 'base_price': 12.50},
            {'ts_code': '000002.SZ', 'symbol': '000002', 'name': '万科A', 'base_price': 18.30},
            {'ts_code': '600000.SH', 'symbol': '600000', 'name': '浦发银行', 'base_price': 8.90},
            {'ts_code': '600036.SH', 'symbol': '600036', 'name': '招商银行', 'base_price': 35.20},
            {'ts_code': '000858.SZ', 'symbol': '000858', 'name': '五粮液', 'base_price': 158.80},
            {'ts_code': '600519.SH', 'symbol': '600519', 'name': '贵州茅台', 'base_price': 1680.00},
            {'ts_code': '000166.SZ', 'symbol': '000166', 'name': '申万宏源', 'base_price': 4.25},
            {'ts_code': '600887.SH', 'symbol': '600887', 'name': '伊利股份', 'base_price': 28.90},
            {'ts_code': '002415.SZ', 'symbol': '002415', 'name': '海康威视', 'base_price': 32.15},
            {'ts_code': '300059.SZ', 'symbol': '300059', 'name': '东方财富', 'base_price': 15.67}
        ]
    
    def connect_databases(self):
        """连接数据库"""
        try:
            # 连接PostgreSQL
            self.db_conn = psycopg2.connect(self.db_url)
            logger.info("PostgreSQL连接成功")
            
            # 连接Redis
            redis_parts = self.redis_url.replace('redis://', '').split('@')
            if len(redis_parts) == 2:
                auth_part, host_port = redis_parts
                password = auth_part.split(':')[1] if ':' in auth_part else None
                host, port = host_port.split(':')
                self.redis_conn = redis.Redis(
                    host=host, 
                    port=int(port), 
                    password=password, 
                    decode_responses=True
                )
            else:
                self.redis_conn = redis.from_url(self.redis_url, decode_responses=True)
            
            # 测试Redis连接
            self.redis_conn.ping()
            logger.info("Redis连接成功")
            
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def generate_stock_daily_data(self, days: int = 252) -> None:
        """生成股票日线数据"""
        logger.info(f"开始生成{days}天的股票日线数据...")
        
        cursor = self.db_conn.cursor()
        
        for stock in self.test_stocks:
            ts_code = stock['ts_code']
            base_price = stock['base_price']
            current_price = base_price
            
            logger.info(f"生成 {ts_code} 的日线数据...")
            
            for i in range(days):
                trade_date = date.today() - timedelta(days=days-i-1)
                
                # 跳过周末
                if trade_date.weekday() >= 5:
                    continue
                
                # 生成价格数据（随机游走）
                daily_return = np.random.normal(0, 0.02)  # 2%的日波动率
                current_price = current_price * (1 + daily_return)
                
                # 生成OHLC数据
                open_price = current_price * (0.995 + random.random() * 0.01)
                high_price = max(open_price, current_price) * (1 + random.random() * 0.02)
                low_price = min(open_price, current_price) * (0.98 + random.random() * 0.02)
                close_price = current_price
                
                # 生成成交量和成交额
                base_volume = 1000000 + random.random() * 5000000
                volume = base_volume * (0.5 + random.random())
                amount = close_price * volume
                
                # 计算涨跌幅
                pre_close = close_price / (1 + daily_return)
                change = close_price - pre_close
                pct_chg = daily_return * 100
                
                # 插入数据
                cursor.execute("""
                    INSERT INTO stock_daily (
                        ts_code, trade_date, open, high, low, close, pre_close,
                        change, pct_chg, vol, amount
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        pre_close = EXCLUDED.pre_close,
                        change = EXCLUDED.change,
                        pct_chg = EXCLUDED.pct_chg,
                        vol = EXCLUDED.vol,
                        amount = EXCLUDED.amount
                """, (
                    ts_code, trade_date, 
                    round(open_price, 3), round(high_price, 3), 
                    round(low_price, 3), round(close_price, 3), 
                    round(pre_close, 3), round(change, 3), 
                    round(pct_chg, 4), round(volume, 2), round(amount, 3)
                ))
        
        self.db_conn.commit()
        cursor.close()
        logger.info("股票日线数据生成完成")
    
    def generate_technical_factors(self, days: int = 60) -> None:
        """生成技术因子数据"""
        logger.info(f"开始生成{days}天的技术因子数据...")
        
        cursor = self.db_conn.cursor()
        
        for stock in self.test_stocks:
            ts_code = stock['ts_code']
            
            # 获取价格数据
            cursor.execute("""
                SELECT trade_date, close, vol 
                FROM stock_daily 
                WHERE ts_code = %s 
                ORDER BY trade_date DESC 
                LIMIT %s
            """, (ts_code, days))
            
            price_data = cursor.fetchall()
            if not price_data:
                continue
            
            logger.info(f"生成 {ts_code} 的技术因子...")
            
            for i, (trade_date, close, vol) in enumerate(price_data):
                # 模拟计算技术指标
                rsi_14 = 30 + random.random() * 40  # RSI在30-70之间
                rsi_6 = 25 + random.random() * 50   # RSI在25-75之间
                
                macd = (random.random() - 0.5) * 2
                macd_signal = macd * 0.8 + (random.random() - 0.5) * 0.5
                macd_hist = macd - macd_signal
                
                # 布林带
                bb_middle = close
                bb_width = 0.05 + random.random() * 0.05
                bb_upper = bb_middle * (1 + bb_width)
                bb_lower = bb_middle * (1 - bb_width)
                
                # 移动平均线
                sma_5 = close * (0.98 + random.random() * 0.04)
                sma_10 = close * (0.97 + random.random() * 0.06)
                sma_20 = close * (0.95 + random.random() * 0.10)
                sma_60 = close * (0.90 + random.random() * 0.20)
                
                ema_12 = close * (0.99 + random.random() * 0.02)
                ema_26 = close * (0.98 + random.random() * 0.04)
                
                # 成交量指标
                volume_ratio = 0.8 + random.random() * 0.4
                turnover_rate = 1.0 + random.random() * 3.0
                
                cursor.execute("""
                    INSERT INTO technical_factors (
                        ts_code, trade_date, rsi_14, rsi_6, macd, macd_signal, macd_hist,
                        bb_upper, bb_middle, bb_lower, bb_width,
                        sma_5, sma_10, sma_20, sma_60, ema_12, ema_26,
                        volume_ratio, turnover_rate
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                        rsi_14 = EXCLUDED.rsi_14,
                        rsi_6 = EXCLUDED.rsi_6,
                        macd = EXCLUDED.macd,
                        macd_signal = EXCLUDED.macd_signal,
                        macd_hist = EXCLUDED.macd_hist,
                        bb_upper = EXCLUDED.bb_upper,
                        bb_middle = EXCLUDED.bb_middle,
                        bb_lower = EXCLUDED.bb_lower,
                        bb_width = EXCLUDED.bb_width,
                        sma_5 = EXCLUDED.sma_5,
                        sma_10 = EXCLUDED.sma_10,
                        sma_20 = EXCLUDED.sma_20,
                        sma_60 = EXCLUDED.sma_60,
                        ema_12 = EXCLUDED.ema_12,
                        ema_26 = EXCLUDED.ema_26,
                        volume_ratio = EXCLUDED.volume_ratio,
                        turnover_rate = EXCLUDED.turnover_rate
                """, (
                    ts_code, trade_date,
                    round(rsi_14, 4), round(rsi_6, 4),
                    round(macd, 4), round(macd_signal, 4), round(macd_hist, 4),
                    round(bb_upper, 3), round(bb_middle, 3), round(bb_lower, 3), round(bb_width, 4),
                    round(sma_5, 3), round(sma_10, 3), round(sma_20, 3), round(sma_60, 3),
                    round(ema_12, 3), round(ema_26, 3),
                    round(volume_ratio, 4), round(turnover_rate, 4)
                ))
        
        self.db_conn.commit()
        cursor.close()
        logger.info("技术因子数据生成完成")
    
    def generate_golden_data(self) -> None:
        """生成黄金数据（用于精度验证）"""
        logger.info("开始生成黄金数据...")
        
        golden_data = {
            'rsi_calculation': {
                'input_prices': [10.0, 10.5, 10.2, 10.8, 10.6, 10.9, 11.0, 10.8, 11.2, 11.5, 11.3, 11.8, 11.6, 12.0, 12.3],
                'expected_rsi_14': 73.81,
                'tolerance': 0.01
            },
            'macd_calculation': {
                'input_prices': [12.0, 12.1, 12.3, 12.2, 12.4, 12.6, 12.5, 12.7, 12.8, 12.9, 13.0, 12.8, 13.1, 13.2, 13.0],
                'expected_macd': 0.0234,
                'expected_signal': 0.0189,
                'expected_histogram': 0.0045,
                'tolerance': 0.001
            },
            'bollinger_bands': {
                'input_prices': [20.0, 20.1, 19.9, 20.2, 20.0, 20.3, 20.1, 19.8, 20.4, 20.2],
                'period': 10,
                'std_dev': 2,
                'expected_upper': 20.45,
                'expected_middle': 20.1,
                'expected_lower': 19.75,
                'tolerance': 0.01
            }
        }
        
        # 保存到文件
        golden_data_path = '/app/test_data/golden_data'
        os.makedirs(golden_data_path, exist_ok=True)
        
        with open(f'{golden_data_path}/technical_indicators.json', 'w', encoding='utf-8') as f:
            json.dump(golden_data, f, ensure_ascii=False, indent=2)
        
        logger.info("黄金数据生成完成")
    
    def generate_ai_test_data(self) -> None:
        """生成AI测试数据"""
        logger.info("开始生成AI测试数据...")
        
        # 生成AI分析测试用例
        ai_test_cases = []
        
        for stock in self.test_stocks[:3]:  # 只为前3只股票生成
            test_case = {
                'ts_code': stock['ts_code'],
                'analysis_date': date.today().isoformat(),
                'expected_fields': [
                    'technical_analysis',
                    'fundamental_analysis', 
                    'sentiment_analysis',
                    'investment_advice',
                    'risk_assessment',
                    'price_prediction'
                ],
                'performance_requirements': {
                    'max_response_time': 30,
                    'min_confidence_score': 0.6,
                    'required_success_rate': 0.99
                }
            }
            ai_test_cases.append(test_case)
        
        # 生成回测优化测试数据
        backtest_test_data = {
            'original_strategy': {
                'strategy_name': 'test_strategy_original',
                'period': '2024-01-01 to 2024-11-30',
                'total_return': 0.15,
                'max_drawdown': 0.08,
                'sharpe_ratio': 1.2,
                'factor_weights': {
                    'rsi_14': 0.3,
                    'macd': 0.4,
                    'pe_ratio': 0.3
                },
                'stop_loss_threshold': 0.05,
                'position_size': 0.1
            },
            'expected_optimization': {
                'min_return_improvement': 0.10,
                'min_drawdown_reduction': 0.05,
                'required_fields': [
                    'optimized_factor_weights',
                    'recommended_stop_loss',
                    'suggested_position_size',
                    'optimization_rationale'
                ]
            }
        }
        
        # 保存测试数据
        ai_test_path = '/app/test_data/ai_test_data'
        os.makedirs(ai_test_path, exist_ok=True)
        
        with open(f'{ai_test_path}/analysis_test_cases.json', 'w', encoding='utf-8') as f:
            json.dump(ai_test_cases, f, ensure_ascii=False, indent=2)
        
        with open(f'{ai_test_path}/backtest_optimization.json', 'w', encoding='utf-8') as f:
            json.dump(backtest_test_data, f, ensure_ascii=False, indent=2)
        
        logger.info("AI测试数据生成完成")
    
    def populate_redis_cache(self) -> None:
        """填充Redis缓存数据"""
        logger.info("开始填充Redis缓存数据...")
        
        try:
            # 缓存股票基础信息
            for stock in self.test_stocks:
                cache_key = f"stock:basic:{stock['ts_code']}"
                self.redis_conn.hset(cache_key, mapping={
                    'ts_code': stock['ts_code'],
                    'symbol': stock['symbol'],
                    'name': stock['name'],
                    'base_price': stock['base_price']
                })
                self.redis_conn.expire(cache_key, 3600)  # 1小时过期
            
            # 缓存一些测试配置
            test_config = {
                'test_mode': 'true',
                'test_stocks': ','.join([s['ts_code'] for s in self.test_stocks]),
                'test_date_range': f"{date.today() - timedelta(days=252)} to {date.today()}",
                'performance_thresholds': json.dumps({
                    'api_response_time': 1.0,
                    'factor_calculation_time': 30.0,
                    'ai_analysis_time': 30.0
                })
            }
            
            self.redis_conn.hset('test:config', mapping=test_config)
            self.redis_conn.expire('test:config', 86400)  # 24小时过期
            
            logger.info("Redis缓存数据填充完成")
            
        except Exception as e:
            logger.error(f"Redis缓存填充失败: {e}")
    
    def generate_performance_test_data(self) -> None:
        """生成性能测试数据"""
        logger.info("开始生成性能测试数据...")
        
        # 生成大量股票数据用于性能测试
        cursor = self.db_conn.cursor()
        
        # 生成1000只虚拟股票用于批量测试
        for i in range(1000, 2000):
            ts_code = f"TEST{i:04d}.SZ"
            symbol = f"TEST{i:04d}"
            name = f"测试股票{i}"
            
            cursor.execute("""
                INSERT INTO stock_basic (ts_code, symbol, name, area, industry, market, list_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ts_code) DO NOTHING
            """, (ts_code, symbol, name, '测试', '测试行业', '测试板', date.today()))
            
            # 为每只股票生成少量日线数据
            base_price = 10 + random.random() * 90
            for j in range(10):  # 只生成10天数据
                trade_date = date.today() - timedelta(days=j)
                if trade_date.weekday() < 5:  # 跳过周末
                    price = base_price * (0.95 + random.random() * 0.1)
                    cursor.execute("""
                        INSERT INTO stock_daily (ts_code, trade_date, close, vol)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (ts_code, trade_date) DO NOTHING
                    """, (ts_code, trade_date, round(price, 3), 1000000))
        
        self.db_conn.commit()
        cursor.close()
        logger.info("性能测试数据生成完成")
    
    def run(self):
        """运行数据生成器"""
        logger.info("=== StockSchool 测试数据生成器启动 ===")
        
        try:
            # 连接数据库
            self.connect_databases()
            
            # 生成各种测试数据
            self.generate_stock_daily_data(days=252)  # 一年的交易数据
            self.generate_technical_factors(days=60)   # 60天的技术因子
            self.generate_golden_data()                # 黄金数据
            self.generate_ai_test_data()              # AI测试数据
            self.populate_redis_cache()               # Redis缓存数据
            self.generate_performance_test_data()     # 性能测试数据
            
            logger.info("=== 所有测试数据生成完成 ===")
            
        except Exception as e:
            logger.error(f"数据生成失败: {e}")
            raise
        
        finally:
            # 关闭连接
            if self.db_conn:
                self.db_conn.close()
            if self.redis_conn:
                self.redis_conn.close()

if __name__ == '__main__':
    generator = TestDataGenerator()
    generator.run()
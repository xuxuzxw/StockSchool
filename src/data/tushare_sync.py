import os
import sys
import pandas as pd
import tushare as ts
from sqlalchemy import text
from datetime import datetime, timedelta
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.db import get_db_engine
from src.utils.retry import idempotent_retry

class TushareSynchronizer:
    def __init__(self):
        """初始化Tushare同步器"""
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            raise ValueError("TUSHARE_TOKEN环境变量未设置")
        
        self.pro = ts.pro_api(token)
        self.engine = get_db_engine()
        print("✅ Tushare同步器初始化成功")
    
    @idempotent_retry()
    def get_last_trade_date(self):
        """获取数据库中最后一个交易日期"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT max(trade_date) FROM stock_daily")).scalar()
            return result.strftime('%Y%m%d') if result else '20100101'
        except Exception as e:
            print(f"获取最后交易日期失败: {e}")
            return '20100101'
    
    @idempotent_retry()
    def sync_stock_basic(self):
        """同步股票基本信息"""
        print("开始同步股票基本信息...")
        try:
            # 获取股票基本信息
            df = self.pro.stock_basic(
                exchange='', 
                list_status='L', 
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )
            
            if df.empty:
                print("⚠️ 未获取到股票基本信息")
                return
            
            # 检查表是否存在，如果不存在则创建
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS stock_basic (
                        ts_code VARCHAR(10) PRIMARY KEY,
                        symbol VARCHAR(10),
                        name VARCHAR(50),
                        area VARCHAR(20),
                        industry VARCHAR(50),
                        market VARCHAR(10),
                        list_date DATE
                    )
                """))
                conn.commit()
            
            # 清空表并插入新数据
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM stock_basic"))
                conn.commit()
            
            df.to_sql('stock_basic', self.engine, if_exists='append', index=False)
            print(f"✅ 成功同步 {len(df)} 只股票的基本信息")
            
        except Exception as e:
            print(f"❌ 同步股票基本信息失败: {e}")
            raise
    
    @idempotent_retry()
    def sync_trade_calendar(self, start_year=2020):
        """同步交易日历"""
        print("开始同步交易日历...")
        try:
            current_year = datetime.now().year
            start_date = f'{start_year}0101'
            end_date = f'{current_year + 1}1231'
            
            df = self.pro.trade_cal(
                exchange='', 
                start_date=start_date, 
                end_date=end_date
            )
            
            if df.empty:
                print("⚠️ 未获取到交易日历数据")
                return
            
            # 检查表是否存在，如果不存在则创建
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS trade_calendar (
                        exchange VARCHAR(10),
                        cal_date DATE,
                        is_open INTEGER,
                        pretrade_date DATE,
                        PRIMARY KEY (exchange, cal_date)
                    )
                """))
                conn.commit()
            
            # 逐条插入数据库，避免参数过多错误
            total_records = len(df)
            
            with self.engine.connect() as conn:
                # 先清空表
                conn.execute(text("DELETE FROM trade_calendar"))
                conn.commit()
                
                # 逐条插入（只插入表中存在的列）
                for i, row in df.iterrows():
                    conn.execute(text("""
                        INSERT INTO trade_calendar (cal_date, is_open, pretrade_date)
                        VALUES (:cal_date, :is_open, :pretrade_date)
                    """), {
                        'cal_date': row['cal_date'],
                        'is_open': row['is_open'],
                        'pretrade_date': row['pretrade_date']
                    })
                    if (i + 1) % 100 == 0:
                        print(f"已插入 {i + 1}/{total_records} 条记录")
                
                conn.commit()
            
            print(f"✅ 成功同步 {total_records} 条交易日历记录")
            
        except Exception as e:
            print(f"❌ 同步交易日历失败: {e}")
            raise
    
    @idempotent_retry()
    def update_daily_data(self, max_days=30):
        """更新日线数据"""
        print("开始更新日线数据...")
        try:
            # 获取最后交易日期
            last_date = self.get_last_trade_date()
            print(f"数据库中最后交易日期: {last_date}")
            
            # 获取交易日历
            end_date = datetime.now().strftime('%Y%m%d')
            trade_cal = self.pro.trade_cal(exchange='', start_date=last_date, end_date=end_date)
            trade_dates = trade_cal[trade_cal.is_open == 1]['cal_date'].tolist()
            
            # 限制更新天数，避免一次性更新太多数据
            if len(trade_dates) > max_days:
                trade_dates = trade_dates[:max_days]
                print(f"⚠️ 限制更新天数为 {max_days} 天")
            
            if not trade_dates:
                print("✅ 数据已是最新，无需更新")
                return
            
            print(f"需要更新 {len(trade_dates)} 个交易日的数据")
            
            success_count = 0
            for i, date in enumerate(trade_dates, 1):
                try:
                    print(f"[{i}/{len(trade_dates)}] 正在同步 {date} 的数据...")
                    
                    # 获取日线数据
                    df = self.pro.daily(trade_date=date)
                    
                    if df.empty:
                        print(f"⚠️ {date} 无数据")
                        continue
                    
                    # 逐条插入数据库，避免参数过多错误
                    total_records = len(df)
                    
                    with self.engine.connect() as conn:
                        for i, row in df.iterrows():
                             conn.execute(text("""
                                 INSERT INTO stock_daily (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                                 VALUES (:ts_code, :trade_date, :open, :high, :low, :close, :pre_close, :change, :pct_chg, :vol, :amount)
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
                             """), {
                                'ts_code': row['ts_code'],
                                'trade_date': row['trade_date'],
                                'open': row['open'],
                                'high': row['high'],
                                'low': row['low'],
                                'close': row['close'],
                                'pre_close': row['pre_close'],
                                'change': row['change'],
                                'pct_chg': row['pct_chg'],
                                'vol': row['vol'],
                                'amount': row['amount']
                            })
                        conn.commit()
                    
                    success_count += 1
                    print(f"✅ {date} 数据同步成功，共 {len(df)} 条记录")
                    
                    # API限制：每分钟最多200次调用，添加延时
                    time.sleep(0.3)
                    
                except Exception as e:
                    print(f"❌ {date} 数据同步失败: {e}")
                    continue
            
            print(f"\n📊 日线数据更新完成，成功更新 {success_count}/{len(trade_dates)} 个交易日")
            
        except Exception as e:
            print(f"❌ 更新日线数据失败: {e}")
            raise
    
    def full_sync(self):
        """完整同步：基本信息 + 交易日历 + 日线数据"""
        print("🚀 开始完整数据同步...")
        try:
            # 1. 同步股票基本信息
            self.sync_stock_basic()
            
            # 2. 同步交易日历
            self.sync_trade_calendar()
            
            # 3. 更新日线数据
            self.update_daily_data()
            
            print("🎉 完整数据同步完成！")
            
        except Exception as e:
            print(f"💥 完整数据同步失败: {e}")
            raise

if __name__ == '__main__':
    try:
        synchronizer = TushareSynchronizer()
        
        # 可以选择不同的同步模式
        import sys
        if len(sys.argv) > 1:
            mode = sys.argv[1]
            if mode == 'full':
                synchronizer.full_sync()
            elif mode == 'daily':
                synchronizer.update_daily_data()
            elif mode == 'basic':
                synchronizer.sync_stock_basic()
            elif mode == 'calendar':
                synchronizer.sync_trade_calendar()
            else:
                print("使用方法: python tushare_sync.py [full|daily|basic|calendar]")
        else:
            # 默认只更新日线数据
            synchronizer.update_daily_data()
            
    except Exception as e:
        print(f"💥 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
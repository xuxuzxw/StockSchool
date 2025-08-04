import os
import sys
import pandas as pd
import tushare as ts
from loguru import logger
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.config.unified_config import config
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
        self.config = config
        print("✅ Tushare同步器初始化成功")
    
    @idempotent_retry()
    def get_last_trade_date(self):
        """获取数据库中最后一个交易日期"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT max(trade_date) FROM stock_daily")).scalar()
            if result:
                return result.strftime('%Y%m%d')
            else:
                # 如果表中没有数据，返回一年前的今天
                return (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        except Exception as e:
            print(f"获取最后交易日期失败: {e}")
            return '20100101'
    
    @idempotent_retry()
    def sync_stock_basic(self):
        """同步股票基本信息"""
        logger.info("开始执行 sync_stock_basic 函数")
        print("开始同步股票基本信息...")
        try:
            # 获取股票基本信息
            df = self.pro.stock_basic(
                exchange='', 
                list_status='L', 
                fields='ts_code,symbol,name,area,industry,market,list_status,list_date'
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
                        list_status VARCHAR(2),
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
            logger.info("sync_stock_basic 函数执行成功")
            
        except Exception as e:
            print(f"❌ 同步股票基本信息失败: {e}")
            raise
    
    @idempotent_retry()
    def sync_trade_calendar(self, start_year=None):
        """同步交易日历"""
        print("开始同步交易日历...")
        try:
            if start_year is None:
                start_year = self.config.get('data_sync_params.start_year', 2020)
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

            # 过滤掉未来日期
            today = datetime.now().strftime('%Y%m%d')
            df = df[df.cal_date <= today]
            
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
                    progress_interval = config.get('data_sync_params.progress_interval', 100)
                    if (i + 1) % progress_interval == 0:
                        print(f"已插入 {i + 1}/{total_records} 条记录")
                
                conn.commit()
            
            print(f"✅ 成功同步 {total_records} 条交易日历记录")
            
        except Exception as e:
            print(f"❌ 同步交易日历失败: {e}")
            raise
    
    @idempotent_retry()
    def update_daily_data(self, max_days=None):
        """更新日线数据"""
        print("开始更新日线数据...")
        try:
            # 获取最后交易日期
            last_date = self.get_last_trade_date()
            print(f"数据库中最后交易日期: {last_date}")
            
            # 获取交易日历
            end_date = datetime.now().strftime('%Y%m%d')
            trade_cal = self.pro.trade_cal(exchange='', start_date=last_date, end_date=end_date)
            # 过滤掉未来日期
            trade_cal = trade_cal[trade_cal.cal_date <= end_date]
            trade_dates = trade_cal[trade_cal.is_open == 1]['cal_date'].tolist()
            
            # 限制更新天数，避免一次性更新太多数据
            if max_days is None:
                max_days = self.config.get('data_sync_params.max_days', 30)
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
                    time.sleep(self.config.get('data_sync_params.sleep_interval', 0.3))
                    
                except Exception as e:
                    print(f"❌ {date} 数据同步失败: {e}")
                    continue
            
            print(f"\n📊 日线数据更新完成，成功更新 {success_count}/{len(trade_dates)} 个交易日")
            
        except Exception as e:
            print(f"❌ 更新日线数据失败: {e}")
            raise
    
    @idempotent_retry()
    def sync_financial_data(self, start_date=None):
        """同步财务数据（利润表、资产负债表、现金流量表）"""
        print("开始同步财务数据...")
        try:
            if start_date is None:
                start_date = self.config.get('data_sync_params.financial_data.start_date', '20200101')
            
            batch_size = self.config.get('data_sync_params.financial_data.batch_size', 100)
            sleep_interval = self.config.get('data_sync_params.financial_data.sleep_interval', 0.5)
            
            # 获取所有股票代码
            stocks_df = pd.read_sql(
                "SELECT ts_code FROM stock_basic WHERE list_status = 'L'",
                self.engine
            )
            
            total_stocks = len(stocks_df)
            print(f"需要同步 {total_stocks} 只股票的财务数据")
            
            # 分批处理股票
            for i in range(0, total_stocks, batch_size):
                batch_stocks = stocks_df.iloc[i:i+batch_size]
                print(f"正在处理第 {i//batch_size + 1} 批，共 {len(batch_stocks)} 只股票")
                
                for _, stock in batch_stocks.iterrows():
                    ts_code = stock['ts_code']
                    try:
                        # 同步利润表数据
                        income_df = self.pro.income(ts_code=ts_code, start_date=start_date)
                        if not income_df.empty:
                            # 同步资产负债表数据
                            balance_df = self.pro.balancesheet(ts_code=ts_code, start_date=start_date)
                            # 同步现金流量表数据
                            cashflow_df = self.pro.cashflow(ts_code=ts_code, start_date=start_date)
                            
                            # 合并三大财务报表数据
                            self._merge_financial_reports(income_df, balance_df, cashflow_df)
                            
                        time.sleep(sleep_interval)
                        
                    except Exception as e:
                        print(f"❌ 同步 {ts_code} 财务数据失败: {e}")
                        continue
                
                print(f"第 {i//batch_size + 1} 批处理完成")
            
            print("✅ 财务数据同步完成")
            
        except Exception as e:
            print(f"❌ 财务数据同步失败: {e}")
            raise
    
    def _merge_financial_reports(self, income_df, balance_df, cashflow_df):
        """合并三大财务报表数据到financial_reports表"""
        try:
            # 收集所有合并后的数据用于批量插入
            merged_data_list = []
            
            # 以利润表为基础，合并其他报表数据
            for _, income_row in income_df.iterrows():
                # 查找对应的资产负债表和现金流量表数据
                balance_row = balance_df[
                    (balance_df['ts_code'] == income_row['ts_code']) & 
                    (balance_df['end_date'] == income_row['end_date'])
                ]
                cashflow_row = cashflow_df[
                    (cashflow_df['ts_code'] == income_row['ts_code']) & 
                    (cashflow_df['end_date'] == income_row['end_date'])
                ]
                
                # 构建合并数据
                merged_data = {
                    'ts_code': income_row['ts_code'],
                    'ann_date': income_row.get('ann_date'),
                    'f_ann_date': income_row.get('f_ann_date'),
                    'end_date': income_row['end_date'],
                    'report_type': income_row.get('report_type'),
                    'comp_type': income_row.get('comp_type'),
                    # 利润表字段
                    'total_revenue': income_row.get('total_revenue'),
                    'revenue': income_row.get('revenue'),
                    'operate_profit': income_row.get('operate_profit'),
                    'total_profit': income_row.get('total_profit'),
                    'n_income': income_row.get('n_income'),
                    'n_income_attr_p': income_row.get('n_income_attr_p'),
                    'basic_eps': income_row.get('basic_eps'),
                    'diluted_eps': income_row.get('diluted_eps'),
                }
                
                # 添加资产负债表字段
                if not balance_row.empty:
                    balance_data = balance_row.iloc[0]
                    merged_data.update({
                        'total_assets': balance_data.get('total_assets'),
                        'total_liab': balance_data.get('total_liab'),
                        'total_hldr_eqy_inc_min_int': balance_data.get('total_hldr_eqy_inc_min_int'),
                        'total_share': balance_data.get('total_share'),
                    })
                
                # 添加现金流量表字段
                if not cashflow_row.empty:
                    cashflow_data = cashflow_row.iloc[0]
                    merged_data.update({
                        'c_fr_sale_sg': cashflow_data.get('c_fr_sale_sg'),
                        'c_paid_goods_s': cashflow_data.get('c_paid_goods_s'),
                        'n_cashflow_act': cashflow_data.get('n_cashflow_act'),
                        'n_cashflow_inv_act': cashflow_data.get('n_cashflow_inv_act'),
                        'n_cashflow_fin_act': cashflow_data.get('n_cashflow_fin_act'),
                        'c_cash_equ_end_period': cashflow_data.get('c_cash_equ_end_period'),
                    })
                
                # 添加到批量处理列表
                merged_data_list.append(merged_data)
            
            # 批量插入或更新数据
            self._batch_upsert_financial_reports(merged_data_list)
                
        except Exception as e:
            print(f"❌ 合并财务报表数据失败: {e}")
            raise
    
    def _batch_upsert_financial_reports(self, data_list):
        """批量插入或更新财务报表数据，支持失败降级到逐条插入"""
        if not data_list:
            return

        start_time = time.perf_counter()
        try:
            # 构建批量UPSERT SQL语句
            columns = list(data_list[0].keys())
            placeholders = [f":{col}" for col in columns]

            # 排除主键和updated_at字段，这些字段不应在DO UPDATE SET中被EXCLUDED
            update_cols = [f"{col} = EXCLUDED.{col}" for col in columns if col not in ['ts_code', 'end_date', 'report_type']]
            update_set_clause = ', '.join(update_cols)

            sql = f"""
            INSERT INTO financial_reports ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT (ts_code, end_date, report_type)
            DO UPDATE SET
                {update_set_clause},
                updated_at = CURRENT_TIMESTAMP
            """

            with self.engine.connect() as conn:
                with conn.begin(): # 开启事务
                    conn.execute(text(sql), data_list)
            end_time = time.perf_counter()
            print(f"✅ 批量插入 {len(data_list)} 条财务报表数据成功，耗时: {end_time - start_time:.4f} 秒")

        except Exception as e:
            print(f"❌ 批量插入财务报表数据失败: {e}，尝试回退到逐条插入模式...")
            # 降级到逐条插入
            self._fallback_upsert_financial_reports(data_list)

    def _fallback_upsert_financial_reports(self, data_list):
        """批量插入失败时的降级方案：逐条插入财务报表数据"""
        print(f"开始逐条插入 {len(data_list)} 条财务报表数据...")
        start_time = time.perf_counter()
        success_count = 0
        for i, data in enumerate(data_list):
            try:
                columns = list(data.keys())
                placeholders = [f":{col}" for col in columns]
                update_cols = [f"{col} = EXCLUDED.{col}" for col in columns if col not in ['ts_code', 'end_date', 'report_type']]
                update_set_clause = ', '.join(update_cols)

                sql = f"""
                INSERT INTO financial_reports ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT (ts_code, end_date, report_type)
                DO UPDATE SET
                    {update_set_clause},
                    updated_at = CURRENT_TIMESTAMP
                """
                with self.engine.connect() as conn:
                    with conn.begin(): # 开启事务
                        conn.execute(text(sql), data)
                success_count += 1
            except Exception as e:
                print(f"❌ 逐条插入第 {i+1} 条数据失败: {e}")
        end_time = time.perf_counter()
        print(f"✅ 逐条插入完成，成功 {success_count}/{len(data_list)} 条，耗时: {end_time - start_time:.4f} 秒")
    

    
    @idempotent_retry()
    def sync_indicator_data(self, start_date=None):
        """同步指标数据（每日基本面指标）"""
        print("开始同步指标数据...")
        try:
            if start_date is None:
                start_date = self.config.get('data_sync_params.indicator_data.start_date', '20200101')
            
            batch_size = self.config.get('data_sync_params.indicator_data.batch_size', 500)
            sleep_interval = self.config.get('data_sync_params.indicator_data.sleep_interval', 0.2)
            
            # 获取交易日历
            trade_dates = pd.read_sql(
                f"SELECT cal_date FROM trade_calendar WHERE cal_date >= '{start_date}' AND is_open = 1 ORDER BY cal_date",
                self.engine
            )['cal_date'].tolist()
            
            print(f"需要同步 {len(trade_dates)} 个交易日的指标数据")
            
            # 分批处理交易日
            for i, trade_date in enumerate(trade_dates):
                try:
                    trade_date_str = trade_date.strftime('%Y%m%d')
                    print(f"正在同步 {trade_date_str} 的指标数据 ({i+1}/{len(trade_dates)})")
                    
                    # 获取每日基本面指标
                    daily_basic_df = self.pro.daily_basic(
                        trade_date=trade_date_str,
                        fields='ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv'
                    )
                    
                    if not daily_basic_df.empty:
                        # 将数据插入到数据库
                        daily_basic_df.to_sql(
                            'daily_basic',
                            self.engine,
                            if_exists='append',
                            index=False,
                            method='multi'
                        )
                        print(f"✅ {trade_date_str} 指标数据同步完成，共 {len(daily_basic_df)} 条记录")
                    
                    time.sleep(sleep_interval)
                    
                except Exception as e:
                    print(f"❌ 同步 {trade_date_str} 指标数据失败: {e}")
                    continue
            
            print("✅ 指标数据同步完成")
            
        except Exception as e:
            print(f"❌ 指标数据同步失败: {e}")
            raise
    
    @idempotent_retry()
    def sync_sw_industry(self, level='L1'):
        """
        同步申万行业分类数据
        
        Args:
            level (str): 行业层级，可选 'L1', 'L2', 'L3'
        """
        logger.info(f"开始同步申万{level}行业分类数据...")
        try:
            # 验证参数
            if level not in ['L1', 'L2', 'L3']:
                raise ValueError(f"无效的行业层级: {level}")
            
            sleep_interval = self.config.get('data_sync_params.sleep_interval', 0.3)
            if not isinstance(sleep_interval, (int, float)) or sleep_interval < 0:
                logger.warning("配置的sleep_interval无效，使用默认值0.3秒")
                sleep_interval = 0.3

            # 获取行业列表
            sw_df = self.pro.index_classify(level=level, src='SW')
            if sw_df.empty:
                logger.warning(f"未获取到申万{level}行业列表")
                return

            total_records = 0
            error_count = 0
            
            # 批量获取所有行业的成分股
            index_codes = ','.join(sw_df['index_code'].tolist())
            all_members_df = self.pro.index_member(index_code=index_codes)
            
            if all_members_df.empty:
                logger.warning("批量获取行业成分股失败")
                return
            
            # 按行业处理数据
            for _, sw_row in sw_df.iterrows():
                try:
                    index_code = sw_row['index_code']
                    industry_name = sw_row['industry_name']
                    
                    # 从批量数据中筛选当前行业
                    members_df = all_members_df[all_members_df['index_code'] == index_code]
                    
                    if members_df.empty:
                        logger.warning(f"行业 {industry_name} 无成分股数据")
                        continue

                    # 数据验证
                    if 'ts_code' not in members_df.columns or members_df['ts_code'].isnull().all():
                        logger.warning(f"行业 {industry_name} 数据无效，缺少ts_code字段")
                        continue
                    
                    # 添加行业层级信息
                    if level == 'L1':
                        members_df['sw_l1'] = industry_name
                    elif level == 'L2':
                        members_df['sw_l2'] = industry_name
                    elif level == 'L3':
                        members_df['sw_l3'] = industry_name

                    # 确保所有层级字段存在
                    for col in ['sw_l1', 'sw_l2', 'sw_l3']:
                        if col not in members_df.columns:
                            members_df[col] = None

                    # 数据去重
                    members_df = members_df.drop_duplicates(
                        subset=['ts_code', 'trade_date', 'sw_l1', 'sw_l2', 'sw_l3']
                    )

                    # 批量UPSERT
                    with self.engine.connect() as conn:
                        for _, row in members_df.iterrows():
                            conn.execute(text("""
                                INSERT INTO sw_industry_history 
                                (ts_code, trade_date, sw_l1, sw_l2, sw_l3, in_date, out_date)
                                VALUES 
                                (:ts_code, :trade_date, :sw_l1, :sw_l2, :sw_l3, :in_date, :out_date)
                                ON CONFLICT (ts_code, trade_date, sw_l1, sw_l2, sw_l3)
                                DO UPDATE SET
                                    in_date = EXCLUDED.in_date,
                                    out_date = EXCLUDED.out_date
                            """), {
                                'ts_code': row['ts_code'],
                                'trade_date': row.get('trade_date', datetime.now().date()),
                                'sw_l1': row.get('sw_l1'),
                                'sw_l2': row.get('sw_l2'),
                                'sw_l3': row.get('sw_l3'),
                                'in_date': row.get('in_date'),
                                'out_date': row.get('out_date')
                            })
                        conn.commit()
                        
                        total_records += len(members_df)
                        logger.info(f"同步 {industry_name}: {len(members_df)} 只股票")
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"处理行业 {industry_name} 时出错: {e}")
                    continue

                # 频率控制
                time.sleep(sleep_interval)

            logger.info(f"申万{level}行业分类数据同步完成，共处理 {total_records} 条记录，{error_count} 个行业出错")
            
        except Exception as e:
            logger.error(f"申万{level}行业分类数据同步失败: {e}", exc_info=True)
            raise

    def sync_sw_industry_full(self):
        """
        同步完整的申万行业分类数据（L1, L2, L3）
        """
        logger.info("开始同步完整的申万行业分类数据...")
        try:
            # 按层级同步
            for level in ['L1', 'L2', 'L3']:
                try:
                    self.sync_sw_industry(level=level)
                except Exception as e:
                    logger.error(f"同步申万{level}行业数据失败: {e}")
                    continue
                    
            logger.info("完整的申万行业分类数据同步完成")
            
        except Exception as e:
            logger.error(f"完整的申万行业分类数据同步失败: {e}", exc_info=True)
            raise

    def update_sw_industry_for_stocks(self, stock_list=None):
        """
        更新指定股票列表的申万行业分类历史数据
        
        Args:
            stock_list (list): 股票代码列表，如果为None则更新所有股票
        """
        logger.info("开始更新股票申万行业分类历史数据...")
        try:
            # 如果没有指定股票列表，则获取所有股票
            if stock_list is None:
                stocks_df = pd.read_sql("SELECT ts_code FROM stock_basic", self.engine)
                stock_list = stocks_df['ts_code'].tolist()
            
            if not stock_list:
                logger.warning("股票列表为空，无法更新行业分类")
                return
            
            total_stocks = len(stock_list)
            processed_count = 0
            error_count = 0
            
            # 分批处理
            batch_size = self.config.get('data_sync_params.batch_size', 1000)
            
            for i in range(0, total_stocks, batch_size):
                batch_stocks = stock_list[i:i+batch_size]
                logger.info(f"处理股票批次 {i//batch_size + 1}/{(total_stocks-1)//batch_size + 1}")
                
                try:
                    # 批量获取行业数据
                    batch_str = ','.join(batch_stocks)
                    industry_df = self.pro.index_member(ts_code=batch_str, src='SW')
                    
                    if not industry_df.empty:
                        # 批量插入/更新
                        with self.engine.connect() as conn:
                            for _, row in industry_df.iterrows():
                                conn.execute(text("""
                                    INSERT INTO sw_industry_history 
                                    (ts_code, trade_date, sw_l1, sw_l2, sw_l3, in_date, out_date)
                                    VALUES 
                                    (:ts_code, :trade_date, :sw_l1, :sw_l2, :sw_l3, :in_date, :out_date)
                                    ON CONFLICT (ts_code, trade_date, sw_l1, sw_l2, sw_l3)
                                    DO UPDATE SET
                                        in_date = EXCLUDED.in_date,
                                        out_date = EXCLUDED.out_date
                                """), {
                                    'ts_code': row['ts_code'],
                                    'trade_date': row.get('trade_date', datetime.now().date()),
                                    'sw_l1': row.get('sw_l1'),
                                    'sw_l2': row.get('sw_l2'),
                                    'sw_l3': row.get('sw_l3'),
                                    'in_date': row.get('in_date'),
                                    'out_date': row.get('out_date')
                                })
                            conn.commit()
                        
                            processed_count += len(industry_df)
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"处理股票批次 {i//batch_size + 1} 时出错: {e}")
                    continue
                
                # 控制API调用频率
                time.sleep(self.config.get('data_sync_params.sleep_interval', 0.3))
            
            logger.info(f"股票申万行业分类历史数据更新完成，共处理 {processed_count} 条记录，{error_count} 个批次出错")
            
        except Exception as e:
            logger.error(f"股票申万行业分类历史数据更新失败: {e}", exc_info=True)
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
            
            # 4. 同步财务数据
            self.sync_financial_data()
            
            # 5. 同步指标数据
            self.sync_indicator_data()
            
            # 6. 同步申万行业分类数据
            self.sync_sw_industry_full()
            
            print("🎉 完整数据同步完成！")
            
        except Exception as e:
            print(f"💥 完整数据同步失败: {e}")
            raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Tushare数据同步工具')
    parser.add_argument('--mode', choices=['basic', 'calendar', 'daily', 'financial', 'indicator', 'industry', 'industry_full', 'industry_update', 'full'], 
                       default='daily', help='同步模式')
    parser.add_argument('--days', type=int, 
                       help='日线数据更新天数（仅在daily模式下有效）')
    parser.add_argument('--start-date', type=str,
                       help='开始日期（格式：YYYYMMDD，适用于financial和indicator模式）')
    parser.add_argument('--level', choices=['L1', 'L2', 'L3'], default='L1',
                       help='行业层级（仅在industry模式下有效）')
    parser.add_argument('--stock-file', type=str,
                       help='股票代码文件路径（用于industry_update模式）')
    
    args = parser.parse_args()
    
    try:
        # 创建同步器实例
        synchronizer = TushareSynchronizer()
        
        # 根据模式执行不同的同步任务
        if args.mode == 'basic':
            synchronizer.sync_stock_basic()
        elif args.mode == 'calendar':
            synchronizer.sync_trade_calendar()
        elif args.mode == 'daily':
            synchronizer.update_daily_data(max_days=args.days)
        elif args.mode == 'financial':
            synchronizer.sync_financial_data(start_date=args.start_date)
        elif args.mode == 'indicator':
            synchronizer.sync_indicator_data(start_date=args.start_date)
        elif args.mode == 'industry':
            synchronizer.sync_sw_industry(level=args.level)
        elif args.mode == 'industry_full':
            synchronizer.sync_sw_industry_full()
        elif args.mode == 'industry_update':
            stock_list = None
            if args.stock_file:
                try:
                    with open(args.stock_file, 'r') as f:
                        stock_list = [line.strip() for line in f if line.strip()]
                    logger.info(f"从文件加载了 {len(stock_list)} 个股票代码")
                except Exception as e:
                    logger.error(f"读取股票文件失败: {e}")
            
            synchronizer.update_sw_industry_for_stocks(stock_list=stock_list)
        elif args.mode == 'full':
            synchronizer.full_sync()
            
    except Exception as e:
        print(f"❌ 数据同步失败: {e}")
        raise

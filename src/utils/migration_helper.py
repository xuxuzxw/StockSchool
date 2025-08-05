from datetime import date, datetime

from models import (
    用于将原有的直接SQL查询迁移到ORM模型,
    Any,
    DatabaseManager,
    DataValidator,
    Dict,
    List,
    Optional,
    Path,
    PerformanceMonitor,
    Tuple,
    ValidationLevel,
    """,
    .parent.parent,
    __file__,
    from,
    import,
    logging,
)
from models import numpy as np  # !/usr/bin/env python3; -*- coding: utf-8 -*-; 添加项目根目录到路径
from models import pandas as pd
from models import (
    pathlib,
    setup_database_logger,
    str,
    sys,
    sys.path.append,
    typing,
    utils.data_validator,
    utils.db,
    utils.logger,
    utils.performance_monitor,
    数据库迁移助手,
)

    Base, StockBasic, StockDaily, StockTechnicalFactors,
    StockFundamentalFactors, FinancialIndicator, FinancialReports,
    get_db_session
)

class MigrationHelper:
    """数据库迁移助手类"""

    def __init__(self, validation_level: str = 'warning'):
        """初始化迁移助手"""
        self.logger = setup_database_logger()
        self.db_manager = DatabaseManager()
        self.validator = DataValidator(ValidationLevel(validation_level))
        self.performance_monitor = PerformanceMonitor()

        self.logger.info("数据库迁移助手初始化完成")

    def migrate_stock_basic_data(self, batch_size: int = 1000) -> bool:
        """迁移股票基础数据"""
        try:
            with self.performance_monitor.monitor_operation("迁移股票基础数据") as session_id:
                # 从原始表获取数据
                query = """
                SELECT ts_code, symbol, name, area, industry, market,
                       list_date, delist_date, is_hs
                FROM stock_basic
                ORDER BY ts_code
                """

                df = self.db_manager.execute_query(query)

                if df.empty:
                    self.logger.warning("未找到股票基础数据")
                    return False

                # 验证数据
                validation_result = self.validator.validate_dataframe(
                    df, "股票基础数据", required_columns=['ts_code', 'name']
                )

                if not validation_result.is_valid:
                    self.logger.error(f"股票基础数据验证失败: {validation_result.errors}")
                    return False

                # 分批迁移
                session = get_db_session()
                try:
                    migrated_count = 0
                    total_count = len(df)

                    for i in range(0, total_count, batch_size):
                        batch_df = df.iloc[i:i + batch_size]

                        for _, row in batch_df.iterrows():
                            # 检查是否已存在
                            existing = session.query(StockBasic).filter(
                                StockBasic.ts_code == row['ts_code']
                            ).first()

                            if existing:
                                # 更新现有记录
                                existing.symbol = row['symbol']
                                existing.name = row['name']
                                existing.area = row['area']
                                existing.industry = row['industry']
                                existing.market = row['market']
                                existing.list_date = pd.to_datetime(row['list_date']).date() if pd.notna(row['list_date']) else None
                                existing.delist_date = pd.to_datetime(row['delist_date']).date() if pd.notna(row['delist_date']) else None
                                existing.is_hs = row['is_hs']
                            else:
                                # 创建新记录
                                stock = StockBasic(
                                    ts_code=row['ts_code'],
                                    symbol=row['symbol'],
                                    name=row['name'],
                                    area=row['area'],
                                    industry=row['industry'],
                                    market=row['market'],
                                    list_date=pd.to_datetime(row['list_date']).date() if pd.notna(row['list_date']) else None,
                                    delist_date=pd.to_datetime(row['delist_date']).date() if pd.notna(row['delist_date']) else None,
                                    is_hs=row['is_hs']
                                )
                                session.add(stock)

                            migrated_count += 1

                        # 提交批次
                        session.commit()
                        self.logger.info(f"已迁移 {migrated_count}/{total_count} 条股票基础数据")

                    self.performance_monitor.add_custom_metric(session_id, "migrated_count", migrated_count)
                    self.logger.info(f"股票基础数据迁移完成，共迁移 {migrated_count} 条记录")
                    return True

                except Exception as e:
                    session.rollback()
                    self.logger.error(f"股票基础数据迁移失败: {e}")
                    return False
                finally:
                    session.close()

        except Exception as e:
            self.logger.error(f"股票基础数据迁移过程失败: {e}")
            return False

    def migrate_stock_daily_data(self, start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               batch_size: int = 5000) -> bool:
        """迁移股票日线数据"""
        try:
            with self.performance_monitor.monitor_operation("迁移股票日线数据") as session_id:
                # 构建查询条件
                where_clause = ""
                if start_date:
                    where_clause += f" AND trade_date >= '{start_date}'"
                if end_date:
                    where_clause += f" AND trade_date <= '{end_date}'"

                query = f"""
                SELECT ts_code, trade_date, open, high, low, close,
                       pre_close, change, pct_chg, vol, amount
                FROM stock_daily
                WHERE 1=1 {where_clause}
                ORDER BY ts_code, trade_date
                """

                df = self.db_manager.execute_query(query)

                if df.empty:
                    self.logger.warning("未找到股票日线数据")
                    return False

                # 验证数据
                validation_result = self.validator.validate_stock_data(df)
                if not validation_result.is_valid:
                    self.logger.error(f"股票日线数据验证失败: {validation_result.errors}")
                    if self.validator.validation_level == ValidationLevel.STRICT:
                        return False

                # 分批迁移
                session = get_db_session()
                try:
                    migrated_count = 0
                    total_count = len(df)

                    for i in range(0, total_count, batch_size):
                        batch_df = df.iloc[i:i + batch_size]

                        for _, row in batch_df.iterrows():
                            # 检查是否已存在
                            existing = session.query(StockDaily).filter(
                                StockDaily.ts_code == row['ts_code'],
                                StockDaily.trade_date == pd.to_datetime(row['trade_date']).date()
                            ).first()

                            if existing:
                                # 更新现有记录
                                existing.open = float(row['open']) if pd.notna(row['open']) else None
                                existing.high = float(row['high']) if pd.notna(row['high']) else None
                                existing.low = float(row['low']) if pd.notna(row['low']) else None
                                existing.close = float(row['close']) if pd.notna(row['close']) else None
                                existing.pre_close = float(row['pre_close']) if pd.notna(row['pre_close']) else None
                                existing.change = float(row['change']) if pd.notna(row['change']) else None
                                existing.pct_chg = float(row['pct_chg']) if pd.notna(row['pct_chg']) else None
                                existing.vol = float(row['vol']) if pd.notna(row['vol']) else None
                                existing.amount = float(row['amount']) if pd.notna(row['amount']) else None
                            else:
                                # 创建新记录
                                stock_daily = StockDaily(
                                    ts_code=row['ts_code'],
                                    trade_date=pd.to_datetime(row['trade_date']).date(),
                                    open=float(row['open']) if pd.notna(row['open']) else None,
                                    high=float(row['high']) if pd.notna(row['high']) else None,
                                    low=float(row['low']) if pd.notna(row['low']) else None,
                                    close=float(row['close']) if pd.notna(row['close']) else None,
                                    pre_close=float(row['pre_close']) if pd.notna(row['pre_close']) else None,
                                    change=float(row['change']) if pd.notna(row['change']) else None,
                                    pct_chg=float(row['pct_chg']) if pd.notna(row['pct_chg']) else None,
                                    vol=float(row['vol']) if pd.notna(row['vol']) else None,
                                    amount=float(row['amount']) if pd.notna(row['amount']) else None
                                )
                                session.add(stock_daily)

                            migrated_count += 1

                        # 提交批次
                        session.commit()

                        if i % (batch_size * 10) == 0:  # 每10批次记录一次日志
                            self.logger.info(f"已迁移 {migrated_count}/{total_count} 条股票日线数据")

                    self.performance_monitor.add_custom_metric(session_id, "migrated_count", migrated_count)
                    self.logger.info(f"股票日线数据迁移完成，共迁移 {migrated_count} 条记录")
                    return True

                except Exception as e:
                    session.rollback()
                    self.logger.error(f"股票日线数据迁移失败: {e}")
                    return False
                finally:
                    session.close()

        except Exception as e:
            self.logger.error(f"股票日线数据迁移过程失败: {e}")
            return False

    def migrate_financial_data(self, batch_size: int = 1000) -> bool:
        """迁移财务数据"""
        try:
            with self.performance_monitor.monitor_operation("迁移财务数据") as session_id:
                # 迁移财务指标数据
                query = """
                SELECT ts_code, ann_date, end_date, eps, dt_eps, total_revenue_ps,
                       revenue_ps, capital_rese_ps, surplus_rese_ps, undist_profit_ps,
                       extra_item, profit_dedt, gross_margin, current_ratio, quick_ratio,
                       cash_ratio, invturn_days, arturn_days, inv_turn, ar_turn,
                       ca_turn, fa_turn, assets_turn, op_income, valuechange_income,
                       interst_income, daa, ebit, ebitda, fcff, fcfe, current_exint,
                       noncurrent_exint, interestdebt, netdebt, tangible_asset, working_capital,
                       networking_capital, invest_capital, retained_earnings, diluted2_eps,
                       bps, ocfps, retainedps, cfps, ebit_ps, fcff_ps, fcfe_ps, netprofit_margin,
                       grossprofit_margin, cogs_of_sales, expense_of_sales, profit_to_gr,
                       saleexp_to_gr, adminexp_of_gr, finaexp_of_gr, impai_ttm, gc_of_gr,
                       op_of_gr, ebit_of_gr, roe, roe_waa, roe_dt, roa, npta, roic,
                       roe_yearly, roa_yearly, roe_avg, opincome_of_ebt, investincome_of_ebt,
                       n_op_profit_of_ebt, tax_to_ebt, dtprofit_to_profit, salescash_to_or,
                       ocf_to_or, ocf_to_opincome, capitalized_to_da, debt_to_assets,
                       assets_to_eqt, dp_assets_to_eqt, ca_to_assets, nca_to_assets,
                       tbassets_to_totalassets, int_to_talcap, eqt_to_talcapital, currentdebt_to_debt,
                       longdeb_to_debt, ocf_to_shortdebt, debt_to_eqt, eqt_to_debt,
                       eqt_to_interestdebt, tangibleasset_to_debt, tangasset_to_intdebt,
                       tangibleasset_to_netdebt, ocf_to_debt, ocf_to_interestdebt, ocf_to_netdebt,
                       ebit_to_interest, longdebt_to_workingcapital, ebitda_to_debt,
                       turn_days, roa_yearly_2, roa_dp, fixed_assets, profit_prefin_exp,
                       non_op_profit, op_to_ebt, nop_to_ebt, ocf_to_profit, cash_to_liqdebt,
                       cash_to_liqdebt_withinterest, op_to_liqdebt, op_to_debt, roic_yearly,
                       total_fa_trun, profit_to_op, q_opincome, q_investincome, q_dtprofit,
                       q_eps, q_netprofit_margin, q_gsprofit_margin, q_exp_to_sales,
                       q_profit_to_gr, q_saleexp_to_gr, q_adminexp_to_gr, q_finaexp_to_gr,
                       q_impair_to_gr_ttm, q_gc_to_gr, q_op_to_gr, q_roe, q_dt_roe,
                       q_npta, q_opincome_to_ebt, q_investincome_to_ebt, q_dtprofit_to_profit,
                       q_salescash_to_or, q_ocf_to_sales, q_ocf_to_or, basic_eps_yoy,
                       dt_eps_yoy, cfps_yoy, op_yoy, ebt_yoy, netprofit_yoy, dt_netprofit_yoy,
                       ocf_yoy, roe_yoy, bps_yoy, assets_yoy, eqt_yoy, tr_yoy, or_yoy,
                       q_gr_yoy, q_gr_qoq, q_sales_yoy, q_sales_qoq, q_op_yoy, q_op_qoq,
                       q_profit_yoy, q_profit_qoq, q_netprofit_yoy, q_netprofit_qoq,
                       equity_yoy, rd_exp, update_flag
                FROM fina_indicator
                ORDER BY ts_code, end_date
                """

                df = self.db_manager.execute_query(query)

                if df.empty:
                    self.logger.warning("未找到财务指标数据")
                    return False

                # 验证数据
                validation_result = self.validator.validate_financial_data(df)
                if not validation_result.is_valid:
                    self.logger.warning(f"财务数据验证警告: {validation_result.errors}")

                # 分批迁移
                session = get_db_session()
                try:
                    migrated_count = 0
                    total_count = len(df)

                    for i in range(0, total_count, batch_size):
                        batch_df = df.iloc[i:i + batch_size]

                        for _, row in batch_df.iterrows():
                            # 检查是否已存在
                            existing = session.query(FinancialIndicator).filter(
                                FinancialIndicator.ts_code == row['ts_code'],
                                FinancialIndicator.end_date == pd.to_datetime(row['end_date']).date()
                            ).first()

                            if not existing:
                                # 创建新记录（只创建不存在的记录）
                                financial_indicator = FinancialIndicator(
                                    ts_code=row['ts_code'],
                                    ann_date=pd.to_datetime(row['ann_date']).date() if pd.notna(row['ann_date']) else None,
                                    end_date=pd.to_datetime(row['end_date']).date(),
                                    eps=float(row['eps']) if pd.notna(row['eps']) else None,
                                    dt_eps=float(row['dt_eps']) if pd.notna(row['dt_eps']) else None,
                                    total_revenue_ps=float(row['total_revenue_ps']) if pd.notna(row['total_revenue_ps']) else None,
                                    revenue_ps=float(row['revenue_ps']) if pd.notna(row['revenue_ps']) else None,
                                    capital_rese_ps=float(row['capital_rese_ps']) if pd.notna(row['capital_rese_ps']) else None,
                                    surplus_rese_ps=float(row['surplus_rese_ps']) if pd.notna(row['surplus_rese_ps']) else None,
                                    undist_profit_ps=float(row['undist_profit_ps']) if pd.notna(row['undist_profit_ps']) else None,
                                    extra_item=float(row['extra_item']) if pd.notna(row['extra_item']) else None,
                                    profit_dedt=float(row['profit_dedt']) if pd.notna(row['profit_dedt']) else None,
                                    gross_margin=float(row['gross_margin']) if pd.notna(row['gross_margin']) else None,
                                    current_ratio=float(row['current_ratio']) if pd.notna(row['current_ratio']) else None,
                                    quick_ratio=float(row['quick_ratio']) if pd.notna(row['quick_ratio']) else None,
                                    cash_ratio=float(row['cash_ratio']) if pd.notna(row['cash_ratio']) else None,
                                    # 添加更多字段...
                                    roe=float(row['roe']) if pd.notna(row['roe']) else None,
                                    roa=float(row['roa']) if pd.notna(row['roa']) else None,
                                    debt_to_assets=float(row['debt_to_assets']) if pd.notna(row['debt_to_assets']) else None,
                                    netprofit_margin=float(row['netprofit_margin']) if pd.notna(row['netprofit_margin']) else None
                                )
                                session.add(financial_indicator)
                                migrated_count += 1

                        # 提交批次
                        session.commit()

                        if i % (batch_size * 5) == 0:  # 每5批次记录一次日志
                            self.logger.info(f"已迁移 {migrated_count}/{total_count} 条财务指标数据")

                    self.performance_monitor.add_custom_metric(session_id, "migrated_count", migrated_count)
                    self.logger.info(f"财务指标数据迁移完成，共迁移 {migrated_count} 条记录")
                    return True

                except Exception as e:
                    session.rollback()
                    self.logger.error(f"财务指标数据迁移失败: {e}")
                    return False
                finally:
                    session.close()

        except Exception as e:
            self.logger.error(f"财务数据迁移过程失败: {e}")
            return False

    def verify_migration(self) -> Dict[str, Any]:
        """验证迁移结果"""
        try:
            with self.performance_monitor.monitor_operation("验证迁移结果") as session_id:
                session = get_db_session()

                try:
                    # 统计各表记录数
                    stock_basic_count = session.query(StockBasic).count()
                    stock_daily_count = session.query(StockDaily).count()
                    financial_indicator_count = session.query(FinancialIndicator).count()

                    # 检查数据完整性
                    latest_trade_date = session.query(StockDaily.trade_date).order_by(
                        StockDaily.trade_date.desc()
                    ).first()

                    earliest_trade_date = session.query(StockDaily.trade_date).order_by(
                        StockDaily.trade_date.asc()
                    ).first()

                    # 检查是否有空值过多的记录
                    null_check_query = """
                    SELECT COUNT(*) as null_count
                    FROM stock_daily
                    WHERE close IS NULL OR open IS NULL
                    """

                    null_count_result = self.db_manager.execute_query(null_check_query)
                    null_count = null_count_result.iloc[0]['null_count'] if not null_count_result.empty else 0

                    verification_result = {
                        'stock_basic_count': stock_basic_count,
                        'stock_daily_count': stock_daily_count,
                        'financial_indicator_count': financial_indicator_count,
                        'latest_trade_date': latest_trade_date[0] if latest_trade_date else None,
                        'earliest_trade_date': earliest_trade_date[0] if earliest_trade_date else None,
                        'null_price_count': null_count,
                        'data_quality_score': max(0, 100 - (null_count / max(stock_daily_count, 1)) * 100)
                    }

                    self.performance_monitor.add_custom_metric(session_id, "total_records",
                                                             stock_basic_count + stock_daily_count + financial_indicator_count)

                    self.logger.info(f"迁移验证完成: {verification_result}")
                    return verification_result

                except Exception as e:
                    self.logger.error(f"迁移验证失败: {e}")
                    return {'error': str(e)}
                finally:
                    session.close()

        except Exception as e:
            self.logger.error(f"迁移验证过程失败: {e}")
            return {'error': str(e)}

    def export_migration_report(self) -> Dict[str, Any]:
        """导出迁移报告"""
        try:
            performance_stats = self.performance_monitor.export_statistics()
            verification_result = self.verify_migration()

            migration_report = {
                'migration_time': datetime.now().isoformat(),
                'performance_stats': performance_stats,
                'verification_result': verification_result,
                'migration_status': 'completed' if 'error' not in verification_result else 'failed'
            }

            self.logger.info("迁移报告导出完成")
            return migration_report

        except Exception as e:
            self.logger.error(f"导出迁移报告失败: {e}")
            return {'error': str(e)}

    def cleanup(self):
        """清理资源"""
        try:
            self.performance_monitor.cleanup()
            self.logger.info("迁移助手资源清理完成")
        except Exception as e:
            self.logger.error(f"迁移助手资源清理失败: {e}")

def main():
    """主函数 - 执行完整的数据迁移流程"""
    migration_helper = MigrationHelper()

    try:
        print("开始数据库迁移...")

        # 1. 迁移股票基础数据
        print("\n1. 迁移股票基础数据...")
        if migration_helper.migrate_stock_basic_data():
            print("✓ 股票基础数据迁移完成")
        else:
            print("✗ 股票基础数据迁移失败")
            return

        # 2. 迁移股票日线数据（最近1年）
        print("\n2. 迁移股票日线数据...")
        start_date = (datetime.now() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        if migration_helper.migrate_stock_daily_data(start_date=start_date):
            print("✓ 股票日线数据迁移完成")
        else:
            print("✗ 股票日线数据迁移失败")

        # 3. 迁移财务数据
        print("\n3. 迁移财务数据...")
        if migration_helper.migrate_financial_data():
            print("✓ 财务数据迁移完成")
        else:
            print("✗ 财务数据迁移失败")

        # 4. 验证迁移结果
        print("\n4. 验证迁移结果...")
        verification_result = migration_helper.verify_migration()
        print(f"验证结果: {verification_result}")

        # 5. 导出迁移报告
        print("\n5. 导出迁移报告...")
        migration_report = migration_helper.export_migration_report()
        print(f"迁移报告: {migration_report['migration_status']}")

        print("\n数据库迁移流程完成！")

    except Exception as e:
        print(f"迁移过程中发生错误: {e}")
    finally:
        migration_helper.cleanup()

if __name__ == "__main__":
    main()
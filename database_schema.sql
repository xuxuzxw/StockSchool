-- StockSchool 数据库Schema设计
-- 基于PostgreSQL + TimescaleDB

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 创建数据库（如果不存在）
-- CREATE DATABASE stockschool;

-- 使用数据库
-- \c stockschool;

-- ================================
-- 1. 基础信息表
-- ================================

-- 股票基础信息表
CREATE TABLE IF NOT EXISTS stock_basic (
    ts_code VARCHAR(20) PRIMARY KEY,  -- 股票代码
    symbol VARCHAR(10) NOT NULL,      -- 股票代码（不带后缀）
    name VARCHAR(50) NOT NULL,        -- 股票名称
    area VARCHAR(20),                 -- 地域
    industry VARCHAR(50),             -- 所属行业
    market VARCHAR(10),               -- 市场类型（主板/创业板等）
    exchange VARCHAR(10),             -- 交易所代码
    curr_type VARCHAR(10),            -- 交易货币
    list_status VARCHAR(1),           -- 上市状态 L上市 D退市 P暂停上市
    list_date DATE,                   -- 上市日期
    delist_date DATE,                 -- 退市日期
    is_hs VARCHAR(1),                 -- 是否沪深港通标的
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 交易日历表
CREATE TABLE IF NOT EXISTS trade_calendar (
    cal_date DATE PRIMARY KEY,        -- 日历日期
    is_open INTEGER NOT NULL,         -- 是否交易 0休市 1交易
    pretrade_date DATE,               -- 上一交易日
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- 2. 行情数据表（时间序列）
-- ================================

-- A股日线行情表
CREATE TABLE IF NOT EXISTS stock_daily (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    trade_date DATE NOT NULL,         -- 交易日期
    open DECIMAL(10,3),               -- 开盘价
    high DECIMAL(10,3),               -- 最高价
    low DECIMAL(10,3),                -- 最低价
    close DECIMAL(10,3),              -- 收盘价
    pre_close DECIMAL(10,3),          -- 昨收价
    change DECIMAL(10,3),             -- 涨跌额
    pct_chg DECIMAL(8,4),             -- 涨跌幅 (%)
    vol DECIMAL(15,2),                -- 成交量 (手)
    amount DECIMAL(20,4),             -- 成交额 (千元)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

-- 创建时间序列表
SELECT create_hypertable('stock_daily', 'trade_date', if_not_exists => TRUE);

-- 每日指标表
CREATE TABLE IF NOT EXISTS daily_basic (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    trade_date DATE NOT NULL,         -- 交易日期
    close DECIMAL(10,3),              -- 当日收盘价
    turnover_rate DECIMAL(8,4),       -- 换手率（%）
    turnover_rate_f DECIMAL(8,4),     -- 换手率（自由流通股）
    volume_ratio DECIMAL(8,4),        -- 量比
    pe DECIMAL(10,4),                 -- 市盈率（总市值/净利润， 亏损的PE为空）
    pe_ttm DECIMAL(10,4),             -- 市盈率（TTM，亏损的PE为空）
    pb DECIMAL(10,4),                 -- 市净率（总市值/净资产）
    ps DECIMAL(10,4),                 -- 市销率
    ps_ttm DECIMAL(10,4),             -- 市销率（TTM）
    dv_ratio DECIMAL(8,4),            -- 股息率 （%）
    dv_ttm DECIMAL(8,4),              -- 股息率（TTM）（%）
    total_share DECIMAL(20,4),        -- 总股本 （万股）
    float_share DECIMAL(20,4),        -- 流通股本 （万股）
    free_share DECIMAL(20,4),         -- 自由流通股本 （万）
    total_mv DECIMAL(20,4),           -- 总市值 （万元）
    circ_mv DECIMAL(20,4),            -- 流通市值（万元）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

SELECT create_hypertable('daily_basic', 'trade_date', if_not_exists => TRUE);

-- 复权因子表
CREATE TABLE IF NOT EXISTS adj_factor (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    trade_date DATE NOT NULL,         -- 交易日期
    adj_factor DECIMAL(12,8),         -- 复权因子
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

SELECT create_hypertable('adj_factor', 'trade_date', if_not_exists => TRUE);

-- ================================
-- 3. 财务数据表
-- ================================

-- 利润表
CREATE TABLE IF NOT EXISTS income (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    ann_date DATE,                    -- 公告日期
    f_ann_date DATE,                  -- 实际公告日期
    end_date DATE NOT NULL,           -- 报告期
    report_type VARCHAR(10),          -- 报告类型
    comp_type VARCHAR(10),            -- 公司类型
    basic_eps DECIMAL(10,4),          -- 基本每股收益
    diluted_eps DECIMAL(10,4),        -- 稀释每股收益
    total_revenue DECIMAL(20,4),      -- 营业总收入
    revenue DECIMAL(20,4),            -- 营业收入
    int_income DECIMAL(20,4),         -- 利息收入
    prem_earned DECIMAL(20,4),        -- 已赚保费
    comm_income DECIMAL(20,4),        -- 手续费及佣金收入
    n_commis_income DECIMAL(20,4),    -- 手续费及佣金净收入
    n_oth_income DECIMAL(20,4),       -- 其他经营净收益
    n_oth_b_income DECIMAL(20,4),     -- 加:其他业务净收益
    prem_income DECIMAL(20,4),        -- 保险业务收入
    out_prem DECIMAL(20,4),           -- 减:分出保费
    une_prem_reser DECIMAL(20,4),     -- 提取未到期责任准备金
    reins_income DECIMAL(20,4),       -- 其中:分保费收入
    n_sec_tb_income DECIMAL(20,4),    -- 代理买卖证券业务净收入
    n_sec_uw_income DECIMAL(20,4),    -- 证券承销业务净收入
    n_asset_mg_income DECIMAL(20,4),  -- 受托客户资产管理业务净收入
    oth_b_income DECIMAL(20,4),       -- 其他业务收入
    fv_value_chg_gain DECIMAL(20,4),  -- 加:公允价值变动净收益
    invest_income DECIMAL(20,4),      -- 加:投资净收益
    ass_invest_income DECIMAL(20,4),  -- 其中:对联营企业和合营企业的投资收益
    forex_gain DECIMAL(20,4),         -- 加:汇兑净收益
    total_cogs DECIMAL(20,4),         -- 营业总成本
    oper_cost DECIMAL(20,4),          -- 减:营业成本
    int_exp DECIMAL(20,4),            -- 减:利息支出
    comm_exp DECIMAL(20,4),           -- 减:手续费及佣金支出
    biz_tax_surchg DECIMAL(20,4),     -- 减:营业税金及附加
    sell_exp DECIMAL(20,4),           -- 减:销售费用
    admin_exp DECIMAL(20,4),          -- 减:管理费用
    fin_exp DECIMAL(20,4),            -- 减:财务费用
    assets_impair_loss DECIMAL(20,4), -- 减:资产减值损失
    prem_refund DECIMAL(20,4),        -- 退保金
    compens_payout DECIMAL(20,4),     -- 赔付总支出
    reser_insur_liab DECIMAL(20,4),   -- 提取保险责任准备金
    div_payt DECIMAL(20,4),           -- 保户红利支出
    reins_exp DECIMAL(20,4),          -- 分保费用
    oper_exp DECIMAL(20,4),           -- 营业支出
    compens_payout_refu DECIMAL(20,4), -- 减:摊回赔付支出
    insur_reser_refu DECIMAL(20,4),   -- 减:摊回保险责任准备金
    reins_cost_refund DECIMAL(20,4),  -- 减:摊回分保费用
    other_bus_cost DECIMAL(20,4),     -- 其他业务成本
    operate_profit DECIMAL(20,4),     -- 营业利润
    non_oper_income DECIMAL(20,4),    -- 加:营业外收入
    non_oper_exp DECIMAL(20,4),       -- 减:营业外支出
    nca_disploss DECIMAL(20,4),       -- 其中:减:非流动资产处置净损失
    total_profit DECIMAL(20,4),       -- 利润总额
    income_tax DECIMAL(20,4),         -- 减:所得税费用
    n_income DECIMAL(20,4),           -- 净利润(含少数股东损益)
    n_income_attr_p DECIMAL(20,4),    -- 净利润(不含少数股东损益)
    minority_gain DECIMAL(20,4),      -- 少数股东损益
    oth_compr_income DECIMAL(20,4),   -- 其他综合收益
    t_compr_income DECIMAL(20,4),     -- 综合收益总额
    compr_inc_attr_p DECIMAL(20,4),   -- 归属于母公司(或股东)的综合收益总额
    compr_inc_attr_m_s DECIMAL(20,4), -- 归属于少数股东的综合收益总额
    ebit DECIMAL(20,4),               -- 息税前利润
    ebitda DECIMAL(20,4),             -- 息税折旧摊销前利润
    insurance_exp DECIMAL(20,4),      -- 保险业务支出
    undist_profit DECIMAL(20,4),      -- 年初未分配利润
    distable_profit DECIMAL(20,4),    -- 可分配利润
    rd_exp DECIMAL(20,4),             -- 研发费用
    fin_exp_int_exp DECIMAL(20,4),    -- 财务费用:利息费用
    fin_exp_int_inc DECIMAL(20,4),    -- 财务费用:利息收入
    transfer_surplus_rese DECIMAL(20,4), -- 盈余公积转入
    transfer_housing_imprest DECIMAL(20,4), -- 住房周转金转入
    transfer_oth DECIMAL(20,4),       -- 其他转入
    adj_lossgain DECIMAL(20,4),       -- 调整以前年度损益
    withdra_legal_surplus DECIMAL(20,4), -- 提取法定盈余公积
    withdra_legal_pubfund DECIMAL(20,4), -- 提取法定公益金
    withdra_biz_devfund DECIMAL(20,4), -- 提取企业发展基金
    withdra_rese_fund DECIMAL(20,4),  -- 提取储备基金
    withdra_oth_ersu DECIMAL(20,4),   -- 提取任意盈余公积金
    workers_welfare DECIMAL(20,4),    -- 职工奖金福利
    distr_profit_shrhder DECIMAL(20,4), -- 可供股东分配的利润
    prfshare_payable_dvd DECIMAL(20,4), -- 应付优先股股利
    comshare_payable_dvd DECIMAL(20,4), -- 应付普通股股利
    capit_comstock_div DECIMAL(20,4), -- 转作股本的普通股股利
    net_after_nr_lp_correct DECIMAL(20,4), -- 扣除非经常性损益后的净利润
    credit_impa_loss DECIMAL(20,4),   -- 信用减值损失
    net_expo_hedging_benefits DECIMAL(20,4), -- 净敞口套期收益
    oth_impair_loss_assets DECIMAL(20,4), -- 其他资产减值损失
    total_opcost DECIMAL(20,4),       -- 营业总成本2
    amodcost_fin_assets DECIMAL(20,4), -- 以摊余成本计量的金融资产终止确认收益
    oth_income DECIMAL(20,4),         -- 其他收益
    asset_disp_income DECIMAL(20,4),  -- 资产处置收益
    continued_net_profit DECIMAL(20,4), -- 持续经营净利润
    end_net_profit DECIMAL(20,4),     -- 终止经营净利润
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, end_date, report_type)
);

-- 资产负债表
CREATE TABLE IF NOT EXISTS balance_sheet (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    ann_date DATE,                    -- 公告日期
    f_ann_date DATE,                  -- 实际公告日期
    end_date DATE NOT NULL,           -- 报告期
    report_type VARCHAR(10),          -- 报告类型
    comp_type VARCHAR(10),            -- 公司类型
    total_share DECIMAL(20,4),        -- 期末总股本
    cap_rese DECIMAL(20,4),           -- 资本公积金
    undistr_porfit DECIMAL(20,4),     -- 未分配利润
    surplus_rese DECIMAL(20,4),       -- 盈余公积金
    special_rese DECIMAL(20,4),       -- 专项储备
    money_cap DECIMAL(20,4),          -- 货币资金
    trad_asset DECIMAL(20,4),         -- 交易性金融资产
    notes_receiv DECIMAL(20,4),       -- 应收票据
    accounts_receiv DECIMAL(20,4),    -- 应收账款
    oth_receiv DECIMAL(20,4),         -- 其他应收款
    prepayment DECIMAL(20,4),         -- 预付款项
    div_receiv DECIMAL(20,4),         -- 应收股利
    int_receiv DECIMAL(20,4),         -- 应收利息
    inventories DECIMAL(20,4),        -- 存货
    amor_exp DECIMAL(20,4),           -- 长期待摊费用
    nca_within_1y DECIMAL(20,4),      -- 一年内到期的非流动资产
    sett_rsrv DECIMAL(20,4),          -- 结算备付金
    loanto_oth_bank_fi DECIMAL(20,4), -- 拆出资金
    premium_receiv DECIMAL(20,4),     -- 应收保费
    reinsur_receiv DECIMAL(20,4),     -- 应收分保账款
    reinsur_res_receiv DECIMAL(20,4), -- 应收分保合同准备金
    pur_resale_fa DECIMAL(20,4),      -- 买入返售金融资产
    oth_cur_assets DECIMAL(20,4),     -- 其他流动资产
    total_cur_assets DECIMAL(20,4),   -- 流动资产合计
    fa_avail_for_sale DECIMAL(20,4),  -- 可供出售金融资产
    htm_invest DECIMAL(20,4),         -- 持有至到期投资
    lt_eqt_invest DECIMAL(20,4),      -- 长期股权投资
    invest_real_estate DECIMAL(20,4), -- 投资性房地产
    time_deposits DECIMAL(20,4),      -- 定期存款
    oth_assets DECIMAL(20,4),         -- 其他资产
    lt_rec DECIMAL(20,4),             -- 长期应收款
    fix_assets DECIMAL(20,4),         -- 固定资产
    cip DECIMAL(20,4),                -- 在建工程
    const_materials DECIMAL(20,4),    -- 工程物资
    fixed_assets_disp DECIMAL(20,4),  -- 固定资产清理
    produc_bio_assets DECIMAL(20,4),  -- 生产性生物资产
    oil_and_gas_assets DECIMAL(20,4), -- 油气资产
    intan_assets DECIMAL(20,4),       -- 无形资产
    r_and_d DECIMAL(20,4),            -- 开发支出
    goodwill DECIMAL(20,4),           -- 商誉
    lt_amor_exp DECIMAL(20,4),        -- 长期待摊费用
    defer_tax_assets DECIMAL(20,4),   -- 递延所得税资产
    decr_in_disbur DECIMAL(20,4),     -- 发放贷款及垫款
    oth_nca DECIMAL(20,4),            -- 其他非流动资产
    total_nca DECIMAL(20,4),          -- 非流动资产合计
    cash_reser_cb DECIMAL(20,4),      -- 现金及存放中央银行款项
    depos_in_oth_bfi DECIMAL(20,4),   -- 存放同业和其它金融机构款项
    prec_metals DECIMAL(20,4),        -- 贵金属
    deriv_assets DECIMAL(20,4),       -- 衍生金融资产
    rr_reins_une_prem DECIMAL(20,4),  -- 应收分保未到期责任准备金
    rr_reins_outstd_cla DECIMAL(20,4), -- 应收分保未决赔款准备金
    rr_reins_lins_liab DECIMAL(20,4), -- 应收分保寿险责任准备金
    rr_reins_lthins_liab DECIMAL(20,4), -- 应收分保长期健康险责任准备金
    refund_depos DECIMAL(20,4),       -- 存出保证金
    ph_pledge_loans DECIMAL(20,4),    -- 保户质押贷款
    refund_cap_depos DECIMAL(20,4),   -- 存出资本保证金
    indep_acct_assets DECIMAL(20,4),  -- 独立账户资产
    client_depos DECIMAL(20,4),       -- 其中：客户资金存款
    client_prov DECIMAL(20,4),        -- 其中：客户备付金
    transac_seat_fee DECIMAL(20,4),   -- 其中:交易席位费
    invest_as_receiv DECIMAL(20,4),   -- 应收款项类投资
    total_assets DECIMAL(20,4),       -- 资产总计
    lt_borr DECIMAL(20,4),            -- 长期借款
    st_borr DECIMAL(20,4),            -- 短期借款
    cb_borr DECIMAL(20,4),            -- 向中央银行借款
    depos_ib_deposits DECIMAL(20,4),  -- 吸收存款及同业存放
    loan_oth_bank DECIMAL(20,4),      -- 拆入资金
    trading_fl DECIMAL(20,4),         -- 交易性金融负债
    notes_payable DECIMAL(20,4),      -- 应付票据
    acct_payable DECIMAL(20,4),       -- 应付账款
    adv_receipts DECIMAL(20,4),       -- 预收款项
    sold_for_repur_fa DECIMAL(20,4),  -- 卖出回购金融资产款
    comm_payable DECIMAL(20,4),       -- 应付手续费及佣金
    payroll_payable DECIMAL(20,4),    -- 应付职工薪酬
    taxes_payable DECIMAL(20,4),      -- 应交税费
    int_payable DECIMAL(20,4),        -- 应付利息
    div_payable DECIMAL(20,4),        -- 应付股利
    oth_payable DECIMAL(20,4),        -- 其他应付款
    acc_exp DECIMAL(20,4),            -- 预提费用
    deferred_inc DECIMAL(20,4),       -- 递延收益
    st_bonds_payable DECIMAL(20,4),   -- 应付短期债券
    payable_to_reinsurer DECIMAL(20,4), -- 应付分保账款
    rsrv_insur_cont DECIMAL(20,4),    -- 保险合同准备金
    acting_trading_sec DECIMAL(20,4), -- 代理买卖证券款
    acting_uw_sec DECIMAL(20,4),      -- 代理承销证券款
    non_cur_liab_due_1y DECIMAL(20,4), -- 一年内到期的非流动负债
    oth_cur_liab DECIMAL(20,4),       -- 其他流动负债
    total_cur_liab DECIMAL(20,4),     -- 流动负债合计
    bond_payable DECIMAL(20,4),       -- 应付债券
    lt_payable DECIMAL(20,4),         -- 长期应付款
    specific_payables DECIMAL(20,4),  -- 专项应付款
    estimated_liab DECIMAL(20,4),     -- 预计负债
    defer_tax_liab DECIMAL(20,4),     -- 递延所得税负债
    defer_inc_non_cur_liab DECIMAL(20,4), -- 递延收益-非流动负债
    oth_ncl DECIMAL(20,4),            -- 其他非流动负债
    total_ncl DECIMAL(20,4),          -- 非流动负债合计
    depos_oth_bfi DECIMAL(20,4),      -- 同业和其它金融机构存放款项
    deriv_liab DECIMAL(20,4),         -- 衍生金融负债
    depos DECIMAL(20,4),              -- 吸收存款
    agency_bus_liab DECIMAL(20,4),    -- 代理业务负债
    oth_liab DECIMAL(20,4),           -- 其他负债
    prem_receiv_adva DECIMAL(20,4),   -- 预收保费
    depos_received DECIMAL(20,4),     -- 存入保证金
    ph_invest DECIMAL(20,4),          -- 保户储金及投资款
    reser_une_prem DECIMAL(20,4),     -- 未到期责任准备金
    reser_outstd_claims DECIMAL(20,4), -- 未决赔款准备金
    reser_lins_liab DECIMAL(20,4),    -- 寿险责任准备金
    reser_lthins_liab DECIMAL(20,4),  -- 长期健康险责任准备金
    indept_acc_liab DECIMAL(20,4),    -- 独立账户负债
    pledge_borr DECIMAL(20,4),        -- 其中:质押借款
    indem_payable DECIMAL(20,4),      -- 应付赔付款
    policy_div_payable DECIMAL(20,4), -- 应付保单红利
    total_liab DECIMAL(20,4),         -- 负债合计
    treasury_share DECIMAL(20,4),     -- 减:库存股
    ordin_risk_reser DECIMAL(20,4),   -- 一般风险准备
    forex_differ DECIMAL(20,4),       -- 外币报表折算差额
    invest_loss_unconf DECIMAL(20,4), -- 未确认的投资损失
    minority_int DECIMAL(20,4),       -- 少数股东权益
    total_hldr_eqy_exc_min_int DECIMAL(20,4), -- 股东权益合计(不含少数股东权益)
    total_hldr_eqy_inc_min_int DECIMAL(20,4), -- 股东权益合计(含少数股东权益)
    total_liab_hldr_eqy DECIMAL(20,4), -- 负债及股东权益总计
    lt_payroll_payable DECIMAL(20,4), -- 长期应付职工薪酬
    oth_comp_income DECIMAL(20,4),    -- 其他综合收益
    oth_eqt_tools DECIMAL(20,4),      -- 其他权益工具
    oth_eqt_tools_p_shr DECIMAL(20,4), -- 其他权益工具(优先股)
    lending_funds DECIMAL(20,4),      -- 融出资金
    acc_receivable DECIMAL(20,4),     -- 应收款项
    st_fin_payable DECIMAL(20,4),     -- 应付短期融资款
    payables DECIMAL(20,4),           -- 应付款项
    hfs_assets DECIMAL(20,4),         -- 持有待售资产
    hfs_sales DECIMAL(20,4),          -- 持有待售负债
    cost_fin_assets DECIMAL(20,4),    -- 以摊余成本计量的金融资产
    fair_value_fin_assets DECIMAL(20,4), -- 以公允价值计量且其变动计入其他综合收益的金融资产
    cip_total DECIMAL(20,4),          -- 在建工程(合计)
    oth_pay_total DECIMAL(20,4),      -- 其他应付款(合计)
    long_pay_total DECIMAL(20,4),     -- 长期应付款(合计)
    debt_invest DECIMAL(20,4),        -- 债权投资
    oth_debt_invest DECIMAL(20,4),    -- 其他债权投资
    oth_eq_invest DECIMAL(20,4),      -- 其他权益工具投资
    oth_illiq_fin_assets DECIMAL(20,4), -- 其他非流动金融资产
    oth_eq_ppbond DECIMAL(20,4),      -- 其他权益工具:永续债
    receiv_financing DECIMAL(20,4),   -- 应收款项融资
    use_right_assets DECIMAL(20,4),   -- 使用权资产
    lease_liab DECIMAL(20,4),         -- 租赁负债
    contract_assets DECIMAL(20,4),    -- 合同资产
    contract_liab DECIMAL(20,4),      -- 合同负债
    accounts_receiv_bill DECIMAL(20,4), -- 应收票据及应收账款
    accounts_pay DECIMAL(20,4),       -- 应付票据及应付账款
    oth_rcv_total DECIMAL(20,4),      -- 其他应收款(合计)
    fix_assets_total DECIMAL(20,4),   -- 固定资产(合计)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, end_date, report_type)
);

-- 现金流量表
CREATE TABLE IF NOT EXISTS cash_flow (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    ann_date DATE,                    -- 公告日期
    f_ann_date DATE,                  -- 实际公告日期
    end_date DATE NOT NULL,           -- 报告期
    report_type VARCHAR(10),          -- 报告类型
    comp_type VARCHAR(10),            -- 公司类型
    net_profit DECIMAL(20,4),         -- 净利润
    finan_exp DECIMAL(20,4),          -- 财务费用
    c_fr_sale_sg DECIMAL(20,4),       -- 销售商品、提供劳务收到的现金
    recp_tax_rends DECIMAL(20,4),     -- 收到的税费返还
    n_depos_incr_fi DECIMAL(20,4),    -- 客户存款和同业存放款项净增加额
    n_incr_loans_cb DECIMAL(20,4),    -- 向央行借款净增加额
    n_inc_borr_oth_fi DECIMAL(20,4),  -- 向其他金融机构拆入资金净增加额
    prem_fr_orig_contr DECIMAL(20,4), -- 收到原保险合同保费取得的现金
    n_incr_insured_dep DECIMAL(20,4), -- 保户储金净增加额
    n_reinsur_prem DECIMAL(20,4),     -- 收到再保业务现金净额
    n_incr_disp_tfa DECIMAL(20,4),    -- 处置交易性金融资产净增加额
    ifc_cash_incr DECIMAL(20,4),      -- 收取利息和手续费净增加额
    n_incr_disp_faas DECIMAL(20,4),   -- 处置可供出售金融资产净增加额
    n_incr_loans_oth_bank DECIMAL(20,4), -- 拆入资金净增加额
    n_cap_incr_repur DECIMAL(20,4),   -- 回购业务资金净增加额
    c_fr_oth_operate_a DECIMAL(20,4), -- 收到其他与经营活动有关的现金
    c_inf_fr_operate_a DECIMAL(20,4), -- 经营活动现金流入小计
    c_paid_goods_s DECIMAL(20,4),     -- 购买商品、接受劳务支付的现金
    c_paid_to_for_empl DECIMAL(20,4), -- 支付给职工以及为职工支付的现金
    c_paid_for_taxes DECIMAL(20,4),   -- 支付的各项税费
    n_incr_clt_loan_adv DECIMAL(20,4), -- 客户贷款及垫款净增加额
    n_incr_dep_cbob DECIMAL(20,4),    -- 存放央行和同业款项净增加额
    c_pay_claims_orig_inco DECIMAL(20,4), -- 支付原保险合同赔付款项的现金
    pay_handling_chrg DECIMAL(20,4),  -- 支付手续费的现金
    pay_comm_insur_plcy DECIMAL(20,4), -- 支付保单红利的现金
    c_paid_oth_operate_a DECIMAL(20,4), -- 支付其他与经营活动有关的现金
    c_outf_fr_operate_a DECIMAL(20,4), -- 经营活动现金流出小计
    n_cashflow_operate_a DECIMAL(20,4), -- 经营活动产生的现金流量净额
    c_disp_withdrwl_invest DECIMAL(20,4), -- 收回投资收到的现金
    c_recp_return_invest DECIMAL(20,4), -- 取得投资收益收到的现金
    n_recp_disp_fiolta DECIMAL(20,4), -- 处置固定资产无形资产和其他长期资产收回的现金净额
    n_recp_disp_sobu DECIMAL(20,4),   -- 处置子公司及其他营业单位收到的现金净额
    c_recp_oth_invest_a DECIMAL(20,4), -- 收到其他与投资活动有关的现金
    c_inf_fr_invest_a DECIMAL(20,4),  -- 投资活动现金流入小计
    c_paid_acq_const_fiolta DECIMAL(20,4), -- 购建固定资产无形资产和其他长期资产支付的现金
    c_paid_invest DECIMAL(20,4),      -- 投资支付的现金
    n_disp_subs_oth_biz DECIMAL(20,4), -- 质押贷款净增加额
    n_paid_acq_sobu DECIMAL(20,4),    -- 取得子公司及其他营业单位支付的现金净额
    c_paid_oth_invest_a DECIMAL(20,4), -- 支付其他与投资活动有关的现金
    c_outf_fr_invest_a DECIMAL(20,4), -- 投资活动现金流出小计
    n_cashflow_invest_a DECIMAL(20,4), -- 投资活动产生的现金流量净额
    c_recp_borrow DECIMAL(20,4),      -- 取得借款收到的现金
    proc_issue_bonds DECIMAL(20,4),   -- 发行债券收到的现金
    c_recp_oth_fin_a DECIMAL(20,4),   -- 收到其他与筹资活动有关的现金
    c_inf_fr_fin_a DECIMAL(20,4),     -- 筹资活动现金流入小计
    c_prepay_amt_borr DECIMAL(20,4),  -- 偿还债务支付的现金
    c_pay_dist_dpcp_int_exp DECIMAL(20,4), -- 分配股利利润或偿付利息支付的现金
    incl_dvd_profit_paid_sc DECIMAL(20,4), -- 其中:子公司支付给少数股东的股利利润
    c_paid_oth_fin_a DECIMAL(20,4),   -- 支付其他与筹资活动有关的现金
    c_outf_fr_fin_a DECIMAL(20,4),    -- 筹资活动现金流出小计
    n_cash_flows_fnc_act DECIMAL(20,4), -- 筹资活动产生的现金流量净额
    eff_fx_flu_cash DECIMAL(20,4),    -- 汇率变动对现金的影响
    n_incr_cash_cash_equ DECIMAL(20,4), -- 现金及现金等价物净增加额
    c_cash_equ_beg_period DECIMAL(20,4), -- 期初现金及现金等价物余额
    c_cash_equ_end_period DECIMAL(20,4), -- 期末现金及现金等价物余额
    c_recp_cap_contrib DECIMAL(20,4), -- 吸收投资收到的现金
    incl_cash_rec_saims DECIMAL(20,4), -- 其中:子公司吸收少数股东投资收到的现金
    uncon_invest_loss DECIMAL(20,4),  -- 未确认投资损失
    prov_depr_assets DECIMAL(20,4),   -- 加:资产减值准备
    depr_fa_coga_dpba DECIMAL(20,4),  -- 固定资产折旧油气资产折耗生产性生物资产折旧
    amort_intang_assets DECIMAL(20,4), -- 无形资产摊销
    lt_amort_deferred_exp DECIMAL(20,4), -- 长期待摊费用摊销
    decr_deferred_exp DECIMAL(20,4),  -- 待摊费用减少
    incr_acc_exp DECIMAL(20,4),       -- 预提费用增加
    loss_disp_fiolta DECIMAL(20,4),   -- 处置固定、无形资产和其他长期资产的损失
    loss_scr_fa DECIMAL(20,4),        -- 固定资产报废损失
    loss_fv_chg DECIMAL(20,4),        -- 公允价值变动损失
    invest_loss DECIMAL(20,4),        -- 投资损失
    decr_def_inc_tax_assets DECIMAL(20,4), -- 递延所得税资产减少
    incr_def_inc_tax_liab DECIMAL(20,4), -- 递延所得税负债增加
    decr_inventories DECIMAL(20,4),   -- 存货的减少
    decr_oper_payable DECIMAL(20,4),  -- 经营性应收项目的减少
    incr_oper_payable DECIMAL(20,4),  -- 经营性应付项目的增加
    others DECIMAL(20,4),             -- 其他
    im_net_cashflow_oper_act DECIMAL(20,4), -- 经营活动产生的现金流量净额(间接法)
    conv_debt_into_cap DECIMAL(20,4), -- 债务转为资本
    conv_copbonds_due_within_1y DECIMAL(20,4), -- 一年内到期的可转换公司债券
    fa_fnc_leases DECIMAL(20,4),      -- 融资租入固定资产
    end_bal_cash DECIMAL(20,4),       -- 现金的期末余额
    beg_bal_cash DECIMAL(20,4),       -- 减:现金的期初余额
    end_bal_cash_equ DECIMAL(20,4),   -- 加:现金等价物的期末余额
    beg_bal_cash_equ DECIMAL(20,4),   -- 减:现金等价物的期初余额
    im_n_incr_cash_equ DECIMAL(20,4), -- 现金及现金等价物净增加额(间接法)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, end_date, report_type)
);

-- 财务指标数据表
CREATE TABLE IF NOT EXISTS financial_indicator (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    ann_date DATE,                    -- 公告日期
    end_date DATE NOT NULL,           -- 报告期
    eps DECIMAL(10,4),                -- 基本每股收益
    dt_eps DECIMAL(10,4),             -- 稀释每股收益
    total_revenue_ps DECIMAL(10,4),   -- 每股营业总收入
    revenue_ps DECIMAL(10,4),         -- 每股营业收入
    capital_rese_ps DECIMAL(10,4),    -- 每股资本公积
    surplus_rese_ps DECIMAL(10,4),    -- 每股盈余公积
    undist_profit_ps DECIMAL(10,4),   -- 每股未分配利润
    extra_item DECIMAL(20,4),         -- 非经常性损益
    profit_dedt DECIMAL(20,4),        -- 扣除非经常性损益后的净利润
    gross_margin DECIMAL(8,4),        -- 毛利
    current_ratio DECIMAL(8,4),       -- 流动比率
    quick_ratio DECIMAL(8,4),         -- 速动比率
    cash_ratio DECIMAL(8,4),          -- 保守速动比率
    invturn_days DECIMAL(8,4),        -- 存货周转天数
    arturn_days DECIMAL(8,4),         -- 应收账款周转天数
    inv_turn DECIMAL(8,4),            -- 存货周转率
    ar_turn DECIMAL(8,4),             -- 应收账款周转率
    ca_turn DECIMAL(8,4),             -- 流动资产周转率
    fa_turn DECIMAL(8,4),             -- 固定资产周转率
    assets_turn DECIMAL(8,4),         -- 总资产周转率
    op_income DECIMAL(20,4),          -- 经营活动净收益
    valuechange_income DECIMAL(20,4), -- 价值变动净收益
    interst_income DECIMAL(20,4),     -- 利息费用
    daa DECIMAL(20,4),                -- 折旧与摊销
    ebit DECIMAL(20,4),               -- 息税前利润
    ebitda DECIMAL(20,4),             -- 息税折旧摊销前利润
    fcff DECIMAL(20,4),               -- 企业自由现金流量
    fcfe DECIMAL(20,4),               -- 股权自由现金流量
    current_exint DECIMAL(20,4),      -- 无息流动负债
    noncurrent_exint DECIMAL(20,4),   -- 无息非流动负债
    interestdebt DECIMAL(20,4),       -- 带息债务
    netdebt DECIMAL(20,4),            -- 净债务
    tangible_asset DECIMAL(20,4),     -- 有形资产
    working_capital DECIMAL(20,4),    -- 营运资金
    networking_capital DECIMAL(20,4), -- 营运流动资金
    invest_capital DECIMAL(20,4),     -- 全部投入资本
    retained_earnings DECIMAL(20,4),  -- 留存收益
    diluted2_eps DECIMAL(10,4),       -- 期末摊薄每股收益
    bps DECIMAL(10,4),                -- 每股净资产
    ocfps DECIMAL(10,4),              -- 每股经营活动产生的现金流量净额
    retainedps DECIMAL(10,4),         -- 每股留存收益
    cfps DECIMAL(10,4),               -- 每股现金流量净额
    ebit_ps DECIMAL(10,4),            -- 每股息税前利润
    fcff_ps DECIMAL(10,4),            -- 每股企业自由现金流量
    fcfe_ps DECIMAL(10,4),            -- 每股股东自由现金流量
    netprofit_margin DECIMAL(8,4),    -- 销售净利率
    grossprofit_margin DECIMAL(8,4),  -- 销售毛利率
    cogs_of_sales DECIMAL(8,4),       -- 销售成本率
    expense_of_sales DECIMAL(8,4),    -- 销售期间费用率
    profit_to_gr DECIMAL(8,4),        -- 净利润/营业总收入
    saleexp_to_gr DECIMAL(8,4),       -- 销售费用/营业总收入
    adminexp_to_gr DECIMAL(8,4),      -- 管理费用/营业总收入
    finaexp_to_gr DECIMAL(8,4),       -- 财务费用/营业总收入
    impai_ttm DECIMAL(8,4),           -- 资产减值损失/营业总收入
    gc_of_gr DECIMAL(8,4),            -- 营业总成本/营业总收入
    op_of_gr DECIMAL(8,4),            -- 营业利润/营业总收入
    ebit_of_gr DECIMAL(8,4),          -- 息税前利润/营业总收入
    roe DECIMAL(8,4),                 -- 净资产收益率
    roe_waa DECIMAL(8,4),             -- 加权平均净资产收益率
    roe_dt DECIMAL(8,4),              -- 净资产收益率(扣除非经常损益)
    roa DECIMAL(8,4),                 -- 总资产报酬率
    npta DECIMAL(8,4),                -- 总资产净利润率
    roic DECIMAL(8,4),                -- 投入资本回报率
    roe_yearly DECIMAL(8,4),          -- 年化净资产收益率
    roa2_yearly DECIMAL(8,4),         -- 年化总资产报酬率
    roe_avg DECIMAL(8,4),             -- 平均净资产收益率(增发条件)
    opincome_of_ebt DECIMAL(8,4),     -- 经营活动净收益/利润总额
    investincome_of_ebt DECIMAL(8,4), -- 价值变动净收益/利润总额
    n_op_profit_of_ebt DECIMAL(8,4),  -- 营业外收支净额/利润总额
    tax_to_ebt DECIMAL(8,4),          -- 所得税/利润总额
    dtprofit_to_profit DECIMAL(8,4),  -- 扣除非经常损益后的净利润/净利润
    salescash_to_or DECIMAL(8,4),     -- 销售商品提供劳务收到的现金/营业收入
    ocf_to_or DECIMAL(8,4),           -- 经营活动产生的现金流量净额/营业收入
    ocf_to_opincome DECIMAL(8,4),     -- 经营活动产生的现金流量净额/经营活动净收益
    capitalized_to_da DECIMAL(8,4),   -- 资本支出/折旧和摊销
    debt_to_assets DECIMAL(8,4),      -- 资产负债率
    assets_to_eqt DECIMAL(8,4),       -- 权益乘数
    dp_assets_to_eqt DECIMAL(8,4),    -- 权益乘数(杜邦分析)
    ca_to_assets DECIMAL(8,4),        -- 流动资产/总资产
    nca_to_assets DECIMAL(8,4),       -- 非流动资产/总资产
    tbassets_to_totalassets DECIMAL(8,4), -- 有形资产/总资产
    int_to_talcap DECIMAL(8,4),       -- 带息债务/全部投入资本
    eqt_to_talcapital DECIMAL(8,4),   -- 归属于母公司的股东权益/全部投入资本
    currentdebt_to_debt DECIMAL(8,4), -- 流动负债/负债合计
    longdeb_to_debt DECIMAL(8,4),     -- 非流动负债/负债合计
    ocf_to_shortdebt DECIMAL(8,4),    -- 经营活动产生的现金流量净额/流动负债
    debt_to_eqt DECIMAL(8,4),         -- 产权比率
    eqt_to_debt DECIMAL(8,4),         -- 归属于母公司的股东权益/负债合计
    eqt_to_interestdebt DECIMAL(8,4), -- 归属于母公司的股东权益/带息债务
    tangibleasset_to_debt DECIMAL(8,4), -- 有形资产/负债合计
    tangasset_to_intdebt DECIMAL(8,4), -- 有形资产/带息债务
    tangibleasset_to_netdebt DECIMAL(8,4), -- 有形资产/净债务
    ocf_to_debt DECIMAL(8,4),         -- 经营活动产生现金流量净额/负债合计
    ocf_to_interestdebt DECIMAL(8,4), -- 经营活动产生现金流量净额/带息债务
    ocf_to_netdebt DECIMAL(8,4),      -- 经营活动产生现金流量净额/净债务
    ebit_to_interest DECIMAL(8,4),    -- 已获利息倍数(EBIT/利息费用)
    longdebt_to_workingcapital DECIMAL(8,4), -- 长期债务与营运资金比率
    ebitda_to_debt DECIMAL(8,4),      -- 息税折旧摊销前利润/负债合计
    turn_days DECIMAL(8,4),           -- 营业周期
    roa_yearly DECIMAL(8,4),          -- 年化总资产净利率
    roa_dp DECIMAL(8,4),              -- 总资产净利率(杜邦分析)
    fixed_assets DECIMAL(20,4),       -- 固定资产合计
    profit_prefin_exp DECIMAL(20,4),  -- 扣除财务费用前营业利润
    non_op_profit DECIMAL(20,4),      -- 非营业利润
    op_to_ebt DECIMAL(8,4),           -- 营业利润／利润总额
    nop_to_ebt DECIMAL(8,4),          -- 非营业利润／利润总额
    ocf_to_profit DECIMAL(8,4),       -- 经营活动产生的现金流量净额／营业利润
    cash_to_liqdebt DECIMAL(8,4),     -- 货币资金／流动负债
    cash_to_liqdebt_withinterest DECIMAL(8,4), -- 货币资金／带息流动负债
    op_to_liqdebt DECIMAL(8,4),       -- 营业利润／流动负债
    op_to_debt DECIMAL(8,4),          -- 营业利润／负债合计
    roic_yearly DECIMAL(8,4),         -- 年化投入资本回报率
    total_fa_trun DECIMAL(8,4),       -- 固定资产合计周转率
    profit_to_op DECIMAL(8,4),        -- 利润总额／营业收入
    q_opincome DECIMAL(20,4),         -- 经营活动单季度净收益
    q_investincome DECIMAL(20,4),     -- 价值变动单季度净收益
    q_dtprofit DECIMAL(20,4),         -- 扣除非经常损益后的单季度净利润
    q_eps DECIMAL(10,4),              -- 每股收益(单季度)
    q_netprofit_margin DECIMAL(8,4),  -- 销售净利率(单季度)
    q_gsprofit_margin DECIMAL(8,4),   -- 销售毛利率(单季度)
    q_exp_to_sales DECIMAL(8,4),      -- 销售期间费用率(单季度)
    q_profit_to_gr DECIMAL(8,4),      -- 净利润／营业总收入(单季度)
    q_saleexp_to_gr DECIMAL(8,4),     -- 销售费用／营业总收入 (单季度)
    q_adminexp_to_gr DECIMAL(8,4),    -- 管理费用／营业总收入 (单季度)
    q_finaexp_to_gr DECIMAL(8,4),     -- 财务费用／营业总收入 (单季度)
    q_impair_to_gr_ttm DECIMAL(8,4),  -- 资产减值损失／营业总收入(单季度)
    q_gc_to_gr DECIMAL(8,4),          -- 营业总成本／营业总收入 (单季度)
    q_op_to_gr DECIMAL(8,4),          -- 营业利润／营业总收入(单季度)
    q_roe DECIMAL(8,4),               -- 净资产收益率(单季度)
    q_dt_roe DECIMAL(8,4),            -- 净资产单季度收益率(扣除非经常损益)
    q_npta DECIMAL(8,4),              -- 总资产净利润率(单季度)
    q_opincome_to_ebt DECIMAL(8,4),   -- 经营活动净收益／利润总额(单季度)
    q_investincome_to_ebt DECIMAL(8,4), -- 价值变动净收益／利润总额(单季度)
    q_dtprofit_to_profit DECIMAL(8,4), -- 扣除非经常损益后的净利润／净利润(单季度)
    q_salescash_to_or DECIMAL(8,4),   -- 销售商品提供劳务收到的现金／营业收入(单季度)
    q_ocf_to_sales DECIMAL(8,4),      -- 经营活动产生的现金流量净额／营业收入(单季度)
    q_ocf_to_or DECIMAL(8,4),         -- 经营活动产生的现金流量净额／经营活动净收益(单季度)
    basic_eps_yoy DECIMAL(8,4),       -- 基本每股收益同比增长率(%)
    dt_eps_yoy DECIMAL(8,4),          -- 稀释每股收益同比增长率(%)
    cfps_yoy DECIMAL(8,4),            -- 每股经营活动产生的现金流量净额同比增长率(%)
    op_yoy DECIMAL(8,4),              -- 营业利润同比增长率(%)
    ebt_yoy DECIMAL(8,4),             -- 利润总额同比增长率(%)
    netprofit_yoy DECIMAL(8,4),       -- 归属母公司股东的净利润同比增长率(%)
    dt_netprofit_yoy DECIMAL(8,4),    -- 归属母公司股东的净利润-扣除非经常损益同比增长率(%)
    ocf_yoy DECIMAL(8,4),             -- 经营活动产生的现金流量净额同比增长率(%)
    roe_yoy DECIMAL(8,4),             -- 净资产收益率(摊薄)同比增长率(%)
    bps_yoy DECIMAL(8,4),             -- 每股净资产相对年初增长率(%)
    assets_yoy DECIMAL(8,4),          -- 资产总计相对年初增长率(%)
    eqt_yoy DECIMAL(8,4),             -- 归属母公司的股东权益相对年初增长率(%)
    tr_yoy DECIMAL(8,4),              -- 营业总收入同比增长率(%)
    or_yoy DECIMAL(8,4),              -- 营业收入同比增长率(%)
    q_gr_yoy DECIMAL(8,4),            -- 营业总收入同比增长率(%)(单季度)
    q_gr_qoq DECIMAL(8,4),            -- 营业总收入环比增长率(%)(单季度)
    q_sales_yoy DECIMAL(8,4),         -- 营业收入同比增长率(%)(单季度)
    q_sales_qoq DECIMAL(8,4),         -- 营业收入环比增长率(%)(单季度)
    q_op_yoy DECIMAL(8,4),            -- 营业利润同比增长率(%)(单季度)
    q_op_qoq DECIMAL(8,4),            -- 营业利润环比增长率(%)(单季度)
    q_profit_yoy DECIMAL(8,4),        -- 净利润同比增长率(%)(单季度)
    q_profit_qoq DECIMAL(8,4),        -- 净利润环比增长率(%)(单季度)
    q_netprofit_yoy DECIMAL(8,4),     -- 归属母公司股东的净利润同比增长率(%)(单季度)
    q_netprofit_qoq DECIMAL(8,4),     -- 归属母公司股东的净利润环比增长率(%)(单季度)
    equity_yoy DECIMAL(8,4),          -- 净资产同比增长率
    rd_exp DECIMAL(20,4),             -- 研发费用
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, end_date)
);

-- ================================
-- 4. 行业数据表
-- ================================

-- 申万行业历史数据表
CREATE TABLE IF NOT EXISTS sw_industry_history (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    trade_date DATE NOT NULL,         -- 交易日期
    sw_l1 VARCHAR(50),                -- 申万一级行业
    sw_l2 VARCHAR(50),                -- 申万二级行业
    sw_l3 VARCHAR(50),                -- 申万三级行业
    in_date DATE,                     -- 纳入日期
    out_date DATE,                    -- 剔除日期
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date, sw_l1, sw_l2, sw_l3)
);

SELECT create_hypertable('sw_industry_history', 'trade_date', if_not_exists => TRUE);

-- ================================
-- 5. 资金流向数据表
-- ================================

-- 个股资金流向表
CREATE TABLE IF NOT EXISTS money_flow (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    trade_date DATE NOT NULL,         -- 交易日期
    buy_sm_vol DECIMAL(15,2),         -- 小单买入量(手)
    buy_sm_amount DECIMAL(20,4),      -- 小单买入金额(万元)
    sell_sm_vol DECIMAL(15,2),        -- 小单卖出量(手)
    sell_sm_amount DECIMAL(20,4),     -- 小单卖出金额(万元)
    buy_md_vol DECIMAL(15,2),         -- 中单买入量(手)
    buy_md_amount DECIMAL(20,4),      -- 中单买入金额(万元)
    sell_md_vol DECIMAL(15,2),        -- 中单卖出量(手)
    sell_md_amount DECIMAL(20,4),     -- 中单卖出金额(万元)
    buy_lg_vol DECIMAL(15,2),         -- 大单买入量(手)
    buy_lg_amount DECIMAL(20,4),      -- 大单买入金额(万元)
    sell_lg_vol DECIMAL(15,2),        -- 大单卖出量(手)
    sell_lg_amount DECIMAL(20,4),     -- 大单卖出金额(万元)
    buy_elg_vol DECIMAL(15,2),        -- 特大单买入量(手)
    buy_elg_amount DECIMAL(20,4),     -- 特大单买入金额(万元)
    sell_elg_vol DECIMAL(15,2),       -- 特大单卖出量(手)
    sell_elg_amount DECIMAL(20,4),    -- 特大单卖出金额(万元)
    net_mf_vol DECIMAL(15,2),         -- 净流入量(手)
    net_mf_amount DECIMAL(20,4),      -- 净流入额(万元)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

SELECT create_hypertable('money_flow', 'trade_date', if_not_exists => TRUE);

-- ================================
-- 6. 指数数据表
-- ================================

-- 指数日线行情表
CREATE TABLE IF NOT EXISTS index_daily (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 指数代码
    trade_date DATE NOT NULL,         -- 交易日期
    open DECIMAL(10,3),               -- 开盘点位
    high DECIMAL(10,3),               -- 最高点位
    low DECIMAL(10,3),                -- 最低点位
    close DECIMAL(10,3),              -- 收盘点位
    pre_close DECIMAL(10,3),          -- 昨收盘点位
    change DECIMAL(10,3),             -- 涨跌点
    pct_chg DECIMAL(8,4),             -- 涨跌幅 (%)
    vol DECIMAL(15,2),                -- 成交量 (手)
    amount DECIMAL(20,4),             -- 成交额 (千元)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

SELECT create_hypertable('index_daily', 'trade_date', if_not_exists => TRUE);

-- 大盘指数每日指标表
CREATE TABLE IF NOT EXISTS index_daily_basic (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 指数代码
    trade_date DATE NOT NULL,         -- 交易日期
    total_mv DECIMAL(20,4),           -- 当日总市值（元）
    float_mv DECIMAL(20,4),           -- 当日流通市值（元）
    total_share DECIMAL(20,4),        -- 当日总股本（股）
    float_share DECIMAL(20,4),        -- 当日流通股本（股）
    free_share DECIMAL(20,4),         -- 当日自由流通股本（股）
    turnover_rate DECIMAL(8,4),       -- 换手率
    turnover_rate_f DECIMAL(8,4),     -- 换手率（基于自由流通股本）
    pe DECIMAL(10,4),                 -- 市盈率
    pe_ttm DECIMAL(10,4),             -- 市盈率TTM
    pb DECIMAL(10,4),                 -- 市净率
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

SELECT create_hypertable('index_daily_basic', 'trade_date', if_not_exists => TRUE);

-- ================================
-- 7. 宏观经济数据表
-- ================================

-- GDP数据表
CREATE TABLE IF NOT EXISTS macro_gdp (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    quarter VARCHAR(10) NOT NULL,     -- 季度
    gdp DECIMAL(15,2),                -- 国内生产总值(亿元)
    gdp_yoy DECIMAL(8,4),             -- GDP同比增长率
    pi DECIMAL(15,2),                 -- 第一产业(亿元)
    pi_yoy DECIMAL(8,4),              -- 第一产业同比增长率
    si DECIMAL(15,2),                 -- 第二产业(亿元)
    si_yoy DECIMAL(8,4),              -- 第二产业同比增长率
    ti DECIMAL(15,2),                 -- 第三产业(亿元)
    ti_yoy DECIMAL(8,4),              -- 第三产业同比增长率
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(quarter)
);

-- CPI数据表
CREATE TABLE IF NOT EXISTS macro_cpi (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    month VARCHAR(10) NOT NULL,       -- 月份
    nt_val DECIMAL(8,4),              -- 全国当月值
    nt_yoy DECIMAL(8,4),              -- 全国同比增长
    nt_mom DECIMAL(8,4),              -- 全国环比增长
    nt_accu DECIMAL(8,4),             -- 全国累计值
    town_val DECIMAL(8,4),            -- 城市当月值
    town_yoy DECIMAL(8,4),            -- 城市同比增长
    town_mom DECIMAL(8,4),            -- 城市环比增长
    town_accu DECIMAL(8,4),           -- 城市累计值
    cnt_val DECIMAL(8,4),             -- 农村当月值
    cnt_yoy DECIMAL(8,4),             -- 农村同比增长
    cnt_mom DECIMAL(8,4),             -- 农村环比增长
    cnt_accu DECIMAL(8,4),            -- 农村累计值
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(month)
);

-- PMI数据表
CREATE TABLE IF NOT EXISTS macro_pmi (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    month VARCHAR(10) NOT NULL,       -- 月份
    pmi DECIMAL(8,4),                 -- PMI
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(month)
);

-- 货币供应量数据表
CREATE TABLE IF NOT EXISTS macro_money_supply (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    month VARCHAR(10) NOT NULL,       -- 月份
    m0 DECIMAL(15,2),                 -- 货币和准货币(M0)期末余额(亿元)
    m0_yoy DECIMAL(8,4),              -- M0同比增长率
    m1 DECIMAL(15,2),                 -- 货币和准货币(M1)期末余额(亿元)
    m1_yoy DECIMAL(8,4),              -- M1同比增长率
    m2 DECIMAL(15,2),                 -- 货币和准货币(M2)期末余额(亿元)
    m2_yoy DECIMAL(8,4),              -- M2同比增长率
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(month)
);

-- ================================
-- 8. 因子数据表
-- ================================

-- 因子数据表
CREATE TABLE IF NOT EXISTS factor_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    trade_date DATE NOT NULL,         -- 交易日期
    factor_name VARCHAR(50) NOT NULL, -- 因子名称
    factor_value DECIMAL(15,8),       -- 因子值
    factor_category VARCHAR(20),      -- 因子分类（技术、基本面、情绪等）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date, factor_name)
);

SELECT create_hypertable('factor_data', 'trade_date', if_not_exists => TRUE);

-- 因子评分表
CREATE TABLE IF NOT EXISTS factor_scores (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    trade_date DATE NOT NULL,         -- 交易日期
    technical_score DECIMAL(8,4),     -- 技术面评分
    fundamental_score DECIMAL(8,4),   -- 基本面评分
    sentiment_score DECIMAL(8,4),     -- 情绪面评分
    composite_score DECIMAL(8,4),     -- 综合评分
    rank_percentile DECIMAL(8,4),     -- 排名百分位
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

SELECT create_hypertable('factor_scores', 'trade_date', if_not_exists => TRUE);

-- ================================
-- 9. 数据质量监控表
-- ================================

-- 数据质量监控表
CREATE TABLE IF NOT EXISTS data_quality_monitor (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,  -- 表名
    check_date DATE NOT NULL,         -- 检查日期
    total_records BIGINT,             -- 总记录数
    null_records BIGINT,              -- 空值记录数
    duplicate_records BIGINT,         -- 重复记录数
    outlier_records BIGINT,           -- 异常值记录数
    quality_score DECIMAL(8,4),       -- 质量评分
    status VARCHAR(20),               -- 状态（PASS/WARN/FAIL）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- 10. 任务调度表
-- ================================

-- 任务调度表
CREATE TABLE IF NOT EXISTS task_schedule (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    task_name VARCHAR(100) NOT NULL,  -- 任务名称
    task_type VARCHAR(50),            -- 任务类型
    cron_expression VARCHAR(100),     -- Cron表达式
    last_run_time TIMESTAMP,          -- 上次运行时间
    next_run_time TIMESTAMP,          -- 下次运行时间
    status VARCHAR(20),               -- 状态（ACTIVE/INACTIVE/RUNNING/FAILED）
    retry_count INTEGER DEFAULT 0,    -- 重试次数
    max_retries INTEGER DEFAULT 3,    -- 最大重试次数
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 任务执行日志表
CREATE TABLE IF NOT EXISTS task_execution_log (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    task_id UUID REFERENCES task_schedule(id),
    start_time TIMESTAMP,             -- 开始时间
    end_time TIMESTAMP,               -- 结束时间
    status VARCHAR(20),               -- 执行状态
    error_message TEXT,               -- 错误信息
    records_processed BIGINT,         -- 处理记录数
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- 11. 索引创建
-- ================================

-- 股票基础信息表索引
CREATE INDEX IF NOT EXISTS idx_stock_basic_industry ON stock_basic(industry);
CREATE INDEX IF NOT EXISTS idx_stock_basic_market ON stock_basic(market);
CREATE INDEX IF NOT EXISTS idx_stock_basic_list_status ON stock_basic(list_status);

-- 日线行情表索引
CREATE INDEX IF NOT EXISTS idx_stock_daily_ts_code ON stock_daily(ts_code);
CREATE INDEX IF NOT EXISTS idx_stock_daily_trade_date ON stock_daily(trade_date);
CREATE INDEX IF NOT EXISTS idx_stock_daily_composite ON stock_daily(ts_code, trade_date);

-- 每日指标表索引
CREATE INDEX IF NOT EXISTS idx_daily_basic_ts_code ON daily_basic(ts_code);
CREATE INDEX IF NOT EXISTS idx_daily_basic_trade_date ON daily_basic(trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_basic_composite ON daily_basic(ts_code, trade_date);

-- 财务数据表索引
CREATE INDEX IF NOT EXISTS idx_income_ts_code ON income(ts_code);
CREATE INDEX IF NOT EXISTS idx_income_end_date ON income(end_date);
CREATE INDEX IF NOT EXISTS idx_income_ann_date ON income(ann_date);

CREATE INDEX IF NOT EXISTS idx_balance_sheet_ts_code ON balance_sheet(ts_code);
CREATE INDEX IF NOT EXISTS idx_balance_sheet_end_date ON balance_sheet(end_date);
CREATE INDEX IF NOT EXISTS idx_balance_sheet_ann_date ON balance_sheet(ann_date);

CREATE INDEX IF NOT EXISTS idx_cash_flow_ts_code ON cash_flow(ts_code);
CREATE INDEX IF NOT EXISTS idx_cash_flow_end_date ON cash_flow(end_date);
CREATE INDEX IF NOT EXISTS idx_cash_flow_ann_date ON cash_flow(ann_date);

CREATE INDEX IF NOT EXISTS idx_financial_indicator_ts_code ON financial_indicator(ts_code);
CREATE INDEX IF NOT EXISTS idx_financial_indicator_end_date ON financial_indicator(end_date);
CREATE INDEX IF NOT EXISTS idx_financial_indicator_ann_date ON financial_indicator(ann_date);

-- 因子数据表索引
CREATE INDEX IF NOT EXISTS idx_factor_data_ts_code ON factor_data(ts_code);
CREATE INDEX IF NOT EXISTS idx_factor_data_trade_date ON factor_data(trade_date);
CREATE INDEX IF NOT EXISTS idx_factor_data_factor_name ON factor_data(factor_name);
CREATE INDEX IF NOT EXISTS idx_factor_data_composite ON factor_data(ts_code, trade_date, factor_name);

CREATE INDEX IF NOT EXISTS idx_factor_scores_ts_code ON factor_scores(ts_code);
CREATE INDEX IF NOT EXISTS idx_factor_scores_trade_date ON factor_scores(trade_date);
CREATE INDEX IF NOT EXISTS idx_factor_scores_composite ON factor_scores(ts_code, trade_date);

-- 资金流向表索引
CREATE INDEX IF NOT EXISTS idx_money_flow_ts_code ON money_flow(ts_code);
CREATE INDEX IF NOT EXISTS idx_money_flow_trade_date ON money_flow(trade_date);
CREATE INDEX IF NOT EXISTS idx_money_flow_composite ON money_flow(ts_code, trade_date);

-- 指数数据表索引
CREATE INDEX IF NOT EXISTS idx_index_daily_ts_code ON index_daily(ts_code);
CREATE INDEX IF NOT EXISTS idx_index_daily_trade_date ON index_daily(trade_date);
CREATE INDEX IF NOT EXISTS idx_index_daily_composite ON index_daily(ts_code, trade_date);

-- 行业数据表索引
CREATE INDEX IF NOT EXISTS idx_sw_industry_ts_code ON sw_industry_history(ts_code);
CREATE INDEX IF NOT EXISTS idx_sw_industry_trade_date ON sw_industry_history(trade_date);
CREATE INDEX IF NOT EXISTS idx_sw_industry_sw_l1 ON sw_industry_history(sw_l1);
CREATE INDEX IF NOT EXISTS idx_sw_industry_composite ON sw_industry_history(ts_code, trade_date);

-- ================================
-- 12. 分区表设置（TimescaleDB自动分区）
-- ================================

-- 设置数据保留策略（可选）
-- SELECT add_retention_policy('stock_daily', INTERVAL '5 years');
-- SELECT add_retention_policy('daily_basic', INTERVAL '5 years');
-- SELECT add_retention_policy('factor_data', INTERVAL '3 years');
-- SELECT add_retention_policy('factor_scores', INTERVAL '3 years');
-- SELECT add_retention_policy('money_flow', INTERVAL '3 years');

-- ================================
-- 13. 视图创建
-- ================================

-- 最新股票基本信息视图
CREATE OR REPLACE VIEW v_stock_latest_info AS
SELECT 
    sb.ts_code,
    sb.symbol,
    sb.name,
    sb.industry,
    sb.market,
    sd.close as latest_price,
    sd.pct_chg as latest_pct_chg,
    sd.trade_date as latest_trade_date,
    db.pe,
    db.pb,
    db.total_mv,
    db.circ_mv
FROM stock_basic sb
LEFT JOIN LATERAL (
    SELECT * FROM stock_daily 
    WHERE ts_code = sb.ts_code 
    ORDER BY trade_date DESC 
    LIMIT 1
) sd ON true
LEFT JOIN LATERAL (
    SELECT * FROM daily_basic 
    WHERE ts_code = sb.ts_code 
    ORDER BY trade_date DESC 
    LIMIT 1
) db ON true
WHERE sb.list_status = 'L';

-- 最新因子评分视图
CREATE OR REPLACE VIEW v_latest_factor_scores AS
SELECT 
    fs.*,
    sb.name,
    sb.industry
FROM factor_scores fs
JOIN stock_basic sb ON fs.ts_code = sb.ts_code
WHERE fs.trade_date = (
    SELECT MAX(trade_date) 
    FROM factor_scores 
    WHERE ts_code = fs.ts_code
)
AND sb.list_status = 'L'
ORDER BY fs.composite_score DESC;

-- 数据质量监控汇总视图
CREATE OR REPLACE VIEW v_data_quality_summary AS
SELECT 
    table_name,
    check_date,
    quality_score,
    status,
    CASE 
        WHEN total_records > 0 THEN ROUND((null_records::DECIMAL / total_records) * 100, 2)
        ELSE 0
    END as null_percentage,
    CASE 
        WHEN total_records > 0 THEN ROUND((duplicate_records::DECIMAL / total_records) * 100, 2)
        ELSE 0
    END as duplicate_percentage
FROM data_quality_monitor
WHERE check_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY check_date DESC, table_name;

-- 财务报表汇总表（用于统一存储三大财务报表数据）
CREATE TABLE IF NOT EXISTS financial_reports (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    ann_date DATE,                    -- 公告日期
    f_ann_date DATE,                  -- 实际公告日期
    end_date DATE NOT NULL,           -- 报告期
    report_type VARCHAR(10),          -- 报告类型
    comp_type VARCHAR(10),            -- 公司类型
    -- 利润表关键字段
    total_revenue DECIMAL(20,4),      -- 营业总收入
    revenue DECIMAL(20,4),            -- 营业收入
    operate_profit DECIMAL(20,4),     -- 营业利润
    total_profit DECIMAL(20,4),       -- 利润总额
    n_income DECIMAL(20,4),           -- 净利润
    n_income_attr_p DECIMAL(20,4),    -- 归属于母公司股东的净利润
    basic_eps DECIMAL(10,4),          -- 基本每股收益
    diluted_eps DECIMAL(10,4),        -- 稀释每股收益
    -- 资产负债表关键字段
    total_assets DECIMAL(20,4),       -- 资产总计
    total_liab DECIMAL(20,4),         -- 负债合计
    total_hldr_eqy_inc_min_int DECIMAL(20,4), -- 股东权益合计
    total_share DECIMAL(20,4),        -- 期末总股本
    -- 现金流量表关键字段
    c_fr_sale_sg DECIMAL(20,4),       -- 销售商品、提供劳务收到的现金
    c_paid_goods_s DECIMAL(20,4),     -- 购买商品、接受劳务支付的现金
    n_cashflow_act DECIMAL(20,4),     -- 经营活动产生的现金流量净额
    n_cashflow_inv_act DECIMAL(20,4), -- 投资活动产生的现金流量净额
    n_cashflow_fin_act DECIMAL(20,4), -- 筹资活动产生的现金流量净额
    c_cash_equ_end_period DECIMAL(20,4), -- 期末现金及现金等价物余额
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, end_date, report_type)
);

-- 因子库表
CREATE TABLE IF NOT EXISTS factor_library (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    trade_date DATE NOT NULL,         -- 交易日期
    factor_name VARCHAR(50) NOT NULL, -- 因子名称
    factor_value DECIMAL(15,8),       -- 因子值
    factor_category VARCHAR(20),      -- 因子分类（技术面/基本面/情绪面等）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date, factor_name)
);

SELECT create_hypertable('factor_library', 'trade_date', if_not_exists => TRUE);

-- 为factor_library表创建索引
CREATE INDEX IF NOT EXISTS idx_factor_library_ts_code ON factor_library(ts_code);
CREATE INDEX IF NOT EXISTS idx_factor_library_factor_name ON factor_library(factor_name);
CREATE INDEX IF NOT EXISTS idx_factor_library_category ON factor_library(factor_category);

-- 情绪数据表（用于存储市场情绪、资金流向等数据）
CREATE TABLE IF NOT EXISTS sentiment_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    trade_date DATE NOT NULL,         -- 交易日期
    ts_code VARCHAR(20),              -- 股票代码（可为空，表示市场整体数据）
    data_category VARCHAR(50) NOT NULL, -- 数据分类（market_sentiment/fund_flow/north_money等）
    data_type VARCHAR(50) NOT NULL,   -- 数据类型（具体的数据子类型）
    data_value TEXT,                  -- 数据值（JSON格式存储）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('sentiment_data', 'trade_date', if_not_exists => TRUE);

-- 为sentiment_data表创建索引
CREATE INDEX IF NOT EXISTS idx_sentiment_data_ts_code ON sentiment_data(ts_code);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_category ON sentiment_data(data_category);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_type ON sentiment_data(data_type);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_trade_date ON sentiment_data(trade_date);}]}}}

-- ================================
-- 14. 函数创建
-- ================================

-- 计算股票收益率函数
CREATE OR REPLACE FUNCTION calculate_return(
    p_ts_code VARCHAR(20),
    p_start_date DATE,
    p_end_date DATE
) RETURNS DECIMAL(10,6) AS $$
DECLARE
    start_price DECIMAL(10,3);
    end_price DECIMAL(10,3);
    return_rate DECIMAL(10,6);
BEGIN
    -- 获取起始价格
    SELECT close INTO start_price
    FROM stock_daily
    WHERE ts_code = p_ts_code AND trade_date = p_start_date;
    
    -- 获取结束价格
    SELECT close INTO end_price
    FROM stock_daily
    WHERE ts_code = p_ts_code AND trade_date = p_end_date;
    
    -- 计算收益率
    IF start_price IS NOT NULL AND end_price IS NOT NULL AND start_price > 0 THEN
        return_rate := (end_price - start_price) / start_price;
    ELSE
        return_rate := NULL;
    END IF;
    
    RETURN return_rate;
END;
$$ LANGUAGE plpgsql;

-- 数据质量检查函数
CREATE OR REPLACE FUNCTION check_data_quality(
    p_table_name VARCHAR(50),
    p_check_date DATE DEFAULT CURRENT_DATE
) RETURNS VOID AS $$
DECLARE
    total_count BIGINT;
    null_count BIGINT;
    duplicate_count BIGINT;
    quality_score DECIMAL(8,4);
    status VARCHAR(20);
BEGIN
    -- 这里可以根据不同表实现具体的数据质量检查逻辑
    -- 示例：检查stock_daily表
    IF p_table_name = 'stock_daily' THEN
        -- 总记录数
        EXECUTE format('SELECT COUNT(*) FROM %I WHERE trade_date = %L', p_table_name, p_check_date) INTO total_count;
        
        -- 空值记录数（检查关键字段）
        EXECUTE format('SELECT COUNT(*) FROM %I WHERE trade_date = %L AND (close IS NULL OR vol IS NULL)', p_table_name, p_check_date) INTO null_count;
        
        -- 重复记录数
        EXECUTE format('SELECT COUNT(*) - COUNT(DISTINCT (ts_code, trade_date)) FROM %I WHERE trade_date = %L', p_table_name, p_check_date) INTO duplicate_count;
        
        -- 计算质量评分
        IF total_count > 0 THEN
            quality_score := 100 - (null_count::DECIMAL / total_count * 50) - (duplicate_count::DECIMAL / total_count * 50);
        ELSE
            quality_score := 0;
        END IF;
        
        -- 确定状态
        IF quality_score >= 95 THEN
            status := 'PASS';
        ELSIF quality_score >= 80 THEN
            status := 'WARN';
        ELSE
            status := 'FAIL';
        END IF;
        
        -- 插入监控记录
        INSERT INTO data_quality_monitor (
            table_name, check_date, total_records, null_records, 
            duplicate_records, outlier_records, quality_score, status
        ) VALUES (
            p_table_name, p_check_date, total_count, null_count,
            duplicate_count, 0, quality_score, status
        ) ON CONFLICT (table_name, check_date) DO UPDATE SET
            total_records = EXCLUDED.total_records,
            null_records = EXCLUDED.null_records,
            duplicate_records = EXCLUDED.duplicate_records,
            quality_score = EXCLUDED.quality_score,
            status = EXCLUDED.status,
            created_at = CURRENT_TIMESTAMP;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ================================
-- 15. 触发器创建
-- ================================

-- 更新时间戳触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为相关表创建更新时间戳触发器
CREATE TRIGGER trigger_stock_daily_updated_at
    BEFORE UPDATE ON stock_daily
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_daily_basic_updated_at
    BEFORE UPDATE ON daily_basic
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_factor_data_updated_at
    BEFORE UPDATE ON factor_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_factor_scores_updated_at
    BEFORE UPDATE ON factor_scores
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ================================
-- 16. 权限设置
-- ================================

-- 创建只读用户（可选）
-- CREATE USER stockschool_readonly WITH PASSWORD 'readonly_password';
-- GRANT CONNECT ON DATABASE stockschool TO stockschool_readonly;
-- GRANT USAGE ON SCHEMA public TO stockschool_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO stockschool_readonly;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO stockschool_readonly;

-- 创建应用用户（可选）
-- CREATE USER stockschool_app WITH PASSWORD 'app_password';
-- GRANT CONNECT ON DATABASE stockschool TO stockschool_app;
-- GRANT USAGE ON SCHEMA public TO stockschool_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO stockschool_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO stockschool_app;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO stockschool_app;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO stockschool_app;

-- ================================
-- 完成
-- ================================

-- 数据库Schema创建完成
-- 包含：
-- 1. 基础信息表（股票基础信息、交易日历）
-- 2. 行情数据表（日线行情、每日指标、复权因子）
-- 3. 财务数据表（利润表、资产负债表、现金流量表、财务指标）
-- 4. 行业数据表（申万行业历史）
-- 5. 资金流向数据表
-- 6. 指数数据表
-- 7. 宏观经济数据表
-- 8. 因子数据表
-- 9. 数据质量监控表
-- 10. 任务调度表
-- 11. 索引优化
-- 12. 视图创建
-- 13. 函数创建
-- 14. 触发器创建
-- 15. 权限设置
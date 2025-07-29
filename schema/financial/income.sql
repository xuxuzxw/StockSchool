-- 利润表
CREATE TABLE IF NOT EXISTS income (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    ann_date DATE,                    -- 公告日期
    f_ann_date DATE,                  -- 实际公告日期
    end_date DATE NOT NULL,           -- 报告期
    report_type VARCHAR(10),          -- 报告类型
    comp_type VARCHAR(10),            -- 公司类型
    total_revenue DECIMAL(20,4),      -- 营业总收入
    revenue DECIMAL(20,4),            -- 营业收入
    int_income DECIMAL(20,4),         -- 利息收入
    premium_income DECIMAL(20,4),     -- 保费收入
    comm_income DECIMAL(20,4),        -- 手续费及佣金收入
    n_interest_income DECIMAL(20,4),  -- 利息净收入
    n_comm_income DECIMAL(20,4),      -- 手续费及佣金净收入
    invest_income DECIMAL(20,4),      -- 投资收益
    fairvalue_income DECIMAL(20,4),   -- 公允价值变动收益
    exchange_income DECIMAL(20,4),    -- 汇兑收益
    other_income DECIMAL(20,4),       -- 其他业务收入
    operate_cost DECIMAL(20,4),       -- 营业成本
    interest_expense DECIMAL(20,4),   -- 利息支出
    comm_expense DECIMAL(20,4),       -- 手续费及佣金支出
    biz_tax_surchg DECIMAL(20,4),     -- 营业税金及附加
    sell_expense DECIMAL(20,4),       -- 销售费用
    admin_expense DECIMAL(20,4),      -- 管理费用
    fin_expense DECIMAL(20,4),        -- 财务费用
    assets_impair_loss DECIMAL(20,4), -- 资产减值损失
    operate_profit DECIMAL(20,4),     -- 营业利润
    non_oper_income DECIMAL(20,4),    -- 营业外收入
    non_oper_expense DECIMAL(20,4),   -- 营业外支出
    non_oper_profit DECIMAL(20,4),    -- 营业外收支净额
    total_profit DECIMAL(20,4),       -- 利润总额
    income_tax DECIMAL(20,4),         -- 所得税
    n_income DECIMAL(20,4),           -- 净利润
    n_income_attr_p DECIMAL(20,4),    -- 归属于母公司股东的净利润
    n_income_attr_m DECIMAL(20,4),    -- 少数股东损益
    basic_eps DECIMAL(10,4),          -- 基本每股收益
    diluted_eps DECIMAL(10,4),        -- 稀释每股收益
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, end_date, report_type)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_income_ts_code ON income(ts_code);
CREATE INDEX IF NOT EXISTS idx_income_end_date ON income(end_date);
CREATE INDEX IF NOT EXISTS idx_income_ann_date ON income(ann_date);

-- 资产负债表
CREATE TABLE IF NOT EXISTS balance_sheet (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    ann_date DATE,                    -- 公告日期
    f_ann_date DATE,                  -- 实际公告日期
    end_date DATE NOT NULL,           -- 报告期
    report_type VARCHAR(10),          -- 报告类型
    comp_type VARCHAR(10),            -- 公司类型
    total_assets DECIMAL(20,4),       -- 资产总计
    total_liab DECIMAL(20,4),         -- 负债合计
    total_hldr_eqy_inc_min_int DECIMAL(20,4), -- 股东权益合计
    total_cash DECIMAL(20,4),         -- 货币资金
    trad_asset DECIMAL(20,4),         -- 交易性金融资产
    notes_receivable DECIMAL(20,4),   -- 应收票据
    accounts_receivable DECIMAL(20,4), -- 应收账款
    advance_payment DECIMAL(20,4),    -- 预付款项
    inventories DECIMAL(20,4),        -- 存货
    total_current_assets DECIMAL(20,4), -- 流动资产合计
    fix_assets DECIMAL(20,4),         -- 固定资产
    intang_assets DECIMAL(20,4),      -- 无形资产
    good_will DECIMAL(20,4),          -- 商誉
    lt_invest DECIMAL(20,4),          -- 长期投资
    deferred_tax_assets DECIMAL(20,4), -- 递延所得税资产
    total_non_current_assets DECIMAL(20,4), -- 非流动资产合计
    flow_liab DECIMAL(20,4),          -- 流动负债合计
    notes_payable DECIMAL(20,4),      -- 应付票据
    accounts_payable DECIMAL(20,4),   -- 应付账款
    advance_receipts DECIMAL(20,4),   -- 预收款项
    tax_payable DECIMAL(20,4),        -- 应交税费
    total_non_current_liab DECIMAL(20,4), -- 非流动负债合计
    lt_loan DECIMAL(20,4),            -- 长期借款
    bond_payable DECIMAL(20,4),       -- 应付债券
    defer_tax_liab DECIMAL(20,4),     -- 递延所得税负债
    total_share DECIMAL(20,4),        -- 总股本
    cap_reserve DECIMAL(20,4),        -- 资本公积
    surplus_reserve DECIMAL(20,4),    -- 盈余公积
    undist_profit DECIMAL(20,4),      -- 未分配利润
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, end_date, report_type)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_balance_sheet_ts_code ON balance_sheet(ts_code);
CREATE INDEX IF NOT EXISTS idx_balance_sheet_end_date ON balance_sheet(end_date);
CREATE INDEX IF NOT EXISTS idx_balance_sheet_ann_date ON balance_sheet(ann_date);
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
    finance_exp DECIMAL(20,4),        -- 财务费用
    c_fr_sale_sg DECIMAL(20,4),       -- 销售商品、提供劳务收到的现金
    recp_tax_refund DECIMAL(20,4),    -- 收到的税费返还
    c_other_recp DECIMAL(20,4),       -- 收到其他与经营活动有关的现金
    c_pay_goods_s DECIMAL(20,4),      -- 购买商品、接受劳务支付的现金
    c_pay_employee DECIMAL(20,4),     -- 支付给职工以及为职工支付的现金
    c_pay_tax DECIMAL(20,4),          -- 支付的各项税费
    c_other_pay DECIMAL(20,4),        -- 支付其他与经营活动有关的现金
    n_cashflow_act DECIMAL(20,4),     -- 经营活动产生的现金流量净额
    c_inv_act DECIMAL(20,4),          -- 投资活动现金流入小计
    c_recp_invest DECIMAL(20,4),      -- 收回投资收到的现金
    c_recp_disp_fa DECIMAL(20,4),     -- 处置固定资产、无形资产和其他长期资产收回的现金净额
    c_recp_other_inv DECIMAL(20,4),   -- 收到其他与投资活动有关的现金
    c_pay_inv DECIMAL(20,4),          -- 投资支付的现金
    c_pay_acq_fa DECIMAL(20,4),       -- 购建固定资产、无形资产和其他长期资产支付的现金
    c_pay_other_inv DECIMAL(20,4),    -- 支付其他与投资活动有关的现金
    n_cashflow_inv_act DECIMAL(20,4), -- 投资活动产生的现金流量净额
    c_fnc_act DECIMAL(20,4),          -- 筹资活动现金流入小计
    c_recp_fnc DECIMAL(20,4),         -- 吸收投资收到的现金
    c_recp_loan DECIMAL(20,4),        -- 取得借款收到的现金
    c_recp_other_fnc DECIMAL(20,4),   -- 收到其他与筹资活动有关的现金
    c_pay_fnc DECIMAL(20,4),          -- 筹资活动现金流出小计
    c_pay_div DECIMAL(20,4),          -- 分配股利、利润或偿付利息支付的现金
    c_pay_debt DECIMAL(20,4),         -- 偿还债务支付的现金
    c_pay_other_fnc DECIMAL(20,4),    -- 支付其他与筹资活动有关的现金
    n_cashflow_fin_act DECIMAL(20,4), -- 筹资活动产生的现金流量净额
    c_cash_incr DECIMAL(20,4),        -- 现金及现金等价物净增加额
    c_cash_beg_period DECIMAL(20,4),  -- 期初现金及现金等价物余额
    c_cash_end_period DECIMAL(20,4),  -- 期末现金及现金等价物余额
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, end_date, report_type)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_cash_flow_ts_code ON cash_flow(ts_code);
CREATE INDEX IF NOT EXISTS idx_cash_flow_end_date ON cash_flow(end_date);
CREATE INDEX IF NOT EXISTS idx_cash_flow_ann_date ON cash_flow(ann_date);

-- 财务指标表
CREATE TABLE IF NOT EXISTS financial_indicator (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    ann_date DATE,                    -- 公告日期
    f_ann_date DATE,                  -- 实际公告日期
    end_date DATE NOT NULL,           -- 报告期
    report_type VARCHAR(10),          -- 报告类型
    comp_type VARCHAR(10),            -- 公司类型
    eps DECIMAL(10,4),                -- 基本每股收益
    eps_ttm DECIMAL(10,4),            -- 每股收益TTM
    dt_eps DECIMAL(10,4),             -- 扣非每股收益
    revenue_per_share DECIMAL(10,4),  -- 每股营业收入
    op_income_per_share DECIMAL(10,4), -- 每股经营活动现金流量
    bps DECIMAL(10,4),                -- 每股净资产
    net_assets_per_share DECIMAL(10,4), -- 每股净资产
    pe DECIMAL(10,4),                 -- 市盈率
    pe_ttm DECIMAL(10,4),             -- 市盈率TTM
    pb DECIMAL(10,4),                 -- 市净率
    ps DECIMAL(10,4),                 -- 市销率
    ps_ttm DECIMAL(10,4),             -- 市销率TTM
    pcf DECIMAL(10,4),                -- 市现率
    roe DECIMAL(10,4),                -- 净资产收益率
    roe_waa DECIMAL(10,4),            -- 加权平均净资产收益率
    roe_dt DECIMAL(10,4),             -- 扣非净资产收益率
    roa DECIMAL(10,4),                -- 总资产收益率
    roa_ttm DECIMAL(10,4),            -- 总资产收益率TTM
    roic DECIMAL(10,4),               -- 投入资本回报率
    debt_to_assets DECIMAL(10,4),     -- 资产负债率
    assets_to_eqt DECIMAL(10,4),      -- 权益乘数
    current_ratio DECIMAL(10,4),      -- 流动比率
    quick_ratio DECIMAL(10,4),        -- 速动比率
    operating_cycle DECIMAL(10,4),    -- 营业周期
    inventory_turnover DECIMAL(10,4), -- 存货周转率
    ar_turnover DECIMAL(10,4),        -- 应收账款周转率
    fa_turnover DECIMAL(10,4),        -- 固定资产周转率
    assets_turnover DECIMAL(10,4),    -- 总资产周转率
    cash_flow_ratio DECIMAL(10,4),    -- 现金流量比率
    net_profit_margin DECIMAL(10,4),  -- 销售净利率
    grossprofit_margin DECIMAL(10,4), -- 销售毛利率
    cost_to_income_ratio DECIMAL(10,4), -- 成本费用利润率
    operating_profit_margin DECIMAL(10,4), -- 营业利润率
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, end_date, report_type)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_financial_indicator_ts_code ON financial_indicator(ts_code);
CREATE INDEX IF NOT EXISTS idx_financial_indicator_end_date ON financial_indicator(end_date);
CREATE INDEX IF NOT EXISTS idx_financial_indicator_ann_date ON financial_indicator(ann_date);
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
    UNIQUE(ts_code, trade_date)
);

-- 创建TimescaleDB超表
SELECT create_hypertable('money_flow', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_money_flow_ts_code ON money_flow(ts_code);
CREATE INDEX IF NOT EXISTS idx_money_flow_trade_date ON money_flow(trade_date);
CREATE INDEX IF NOT EXISTS idx_money_flow_composite ON money_flow(ts_code, trade_date);
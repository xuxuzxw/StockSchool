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

-- 为申万行业历史数据表创建唯一索引以支持高效UPSERT操作
CREATE UNIQUE INDEX IF NOT EXISTS idx_sw_industry_unique 
ON sw_industry_history (ts_code, trade_date, sw_l1, sw_l2, sw_l3);

-- 创建TimescaleDB超表
SELECT create_hypertable('sw_industry_history', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_sw_industry_ts_code ON sw_industry_history(ts_code);
CREATE INDEX IF NOT EXISTS idx_sw_industry_trade_date ON sw_industry_history(trade_date);
CREATE INDEX IF NOT EXISTS idx_sw_industry_sw_l1 ON sw_industry_history(sw_l1);
CREATE INDEX IF NOT EXISTS idx_sw_industry_composite ON sw_industry_history(ts_code, trade_date);
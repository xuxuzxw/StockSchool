-- 复权因子表
CREATE TABLE IF NOT EXISTS adj_factor (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    trade_date DATE NOT NULL,         -- 交易日期
    adj_factor DECIMAL(15,8),         -- 复权因子
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

-- 创建TimescaleDB超表
SELECT create_hypertable('adj_factor', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_adj_factor_ts_code ON adj_factor(ts_code);
CREATE INDEX IF NOT EXISTS idx_adj_factor_trade_date ON adj_factor(trade_date);
CREATE INDEX IF NOT EXISTS idx_adj_factor_composite ON adj_factor(ts_code, trade_date);
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

-- 创建TimescaleDB超表
SELECT create_hypertable('index_daily', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_index_daily_ts_code ON index_daily(ts_code);
CREATE INDEX IF NOT EXISTS idx_index_daily_trade_date ON index_daily(trade_date);
CREATE INDEX IF NOT EXISTS idx_index_daily_composite ON index_daily(ts_code, trade_date);

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

-- 创建TimescaleDB超表
SELECT create_hypertable('index_daily_basic', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_index_daily_basic_ts_code ON index_daily_basic(ts_code);
CREATE INDEX IF NOT EXISTS idx_index_daily_basic_trade_date ON index_daily_basic(trade_date);
CREATE INDEX IF NOT EXISTS idx_index_daily_basic_composite ON index_daily_basic(ts_code, trade_date);
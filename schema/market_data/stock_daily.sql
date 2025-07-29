-- 股票日线行情表
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
    pct_chg DECIMAL(8,4),             -- 涨跌幅(%)
    vol DECIMAL(15,2),                -- 成交量(手)
    amount DECIMAL(20,4),             -- 成交额(万元)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

-- 创建TimescaleDB超表
SELECT create_hypertable('stock_daily', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_stock_daily_ts_code ON stock_daily(ts_code);
CREATE INDEX IF NOT EXISTS idx_stock_daily_trade_date ON stock_daily(trade_date);
CREATE INDEX IF NOT EXISTS idx_stock_daily_composite ON stock_daily(ts_code, trade_date);

-- 股票每日指标表
CREATE TABLE IF NOT EXISTS daily_basic (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    trade_date DATE NOT NULL,         -- 交易日期
    close DECIMAL(10,3),              -- 当日收盘价
    turnover_rate DECIMAL(8,4),       -- 换手率(%)
    turnover_rate_f DECIMAL(8,4),     -- 换手率(自由流通股)(%)
    volume_ratio DECIMAL(8,4),        -- 量比
    pe DECIMAL(10,4),                 -- 市盈率(总市值/净利润)
    pe_ttm DECIMAL(10,4),             -- 市盈率TTM
    pb DECIMAL(10,4),                 -- 市净率(总市值/净资产)
    ps DECIMAL(10,4),                 -- 市销率
    ps_ttm DECIMAL(10,4),             -- 市销率TTM
    total_mv DECIMAL(20,4),           -- 总市值(万元)
    circ_mv DECIMAL(20,4),            -- 流通市值(万元)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

-- 创建TimescaleDB超表
SELECT create_hypertable('daily_basic', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_daily_basic_ts_code ON daily_basic(ts_code);
CREATE INDEX IF NOT EXISTS idx_daily_basic_trade_date ON daily_basic(trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_basic_composite ON daily_basic(ts_code, trade_date);
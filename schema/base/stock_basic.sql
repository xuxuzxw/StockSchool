-- 股票基本信息表
CREATE TABLE IF NOT EXISTS stock_basic (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,         -- 股票代码
    symbol VARCHAR(10) NOT NULL,          -- 股票简称
    name VARCHAR(20) NOT NULL,            -- 股票全称
    area VARCHAR(20),                     -- 地域
    industry VARCHAR(50),                 -- 行业
    market VARCHAR(10),                   -- 市场类型（主板/创业板/科创板等）
    list_date DATE NOT NULL,              -- 上市日期
    delist_date DATE,                     -- 退市日期
    list_status VARCHAR(10) NOT NULL,     -- 上市状态（L上市/D退市/P暂停上市）
    is_hs VARCHAR(10),                    -- 是否沪深港通标的
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_stock_basic_industry ON stock_basic(industry);
CREATE INDEX IF NOT EXISTS idx_stock_basic_market ON stock_basic(market);
CREATE INDEX IF NOT EXISTS idx_stock_basic_list_status ON stock_basic(list_status);

-- 交易日历表
CREATE TABLE IF NOT EXISTS trade_calendar (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    exchange VARCHAR(10) NOT NULL,        -- 交易所
    cal_date DATE NOT NULL,               -- 日历日期
    is_open SMALLINT NOT NULL,            -- 是否交易日（0休市，1开市）
    pretrade_date DATE,                   -- 上一个交易日
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(exchange, cal_date)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_trade_calendar_exchange ON trade_calendar(exchange);
CREATE INDEX IF NOT EXISTS idx_trade_calendar_cal_date ON trade_calendar(cal_date);
CREATE INDEX IF NOT EXISTS idx_trade_calendar_is_open ON trade_calendar(is_open);
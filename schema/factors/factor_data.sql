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

-- 创建TimescaleDB超表
SELECT create_hypertable('factor_data', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_factor_data_ts_code ON factor_data(ts_code);
CREATE INDEX IF NOT EXISTS idx_factor_data_trade_date ON factor_data(trade_date);
CREATE INDEX IF NOT EXISTS idx_factor_data_factor_name ON factor_data(factor_name);
CREATE INDEX IF NOT EXISTS idx_factor_data_composite ON factor_data(ts_code, trade_date, factor_name);

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

-- 创建TimescaleDB超表
SELECT create_hypertable('factor_scores', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_factor_scores_ts_code ON factor_scores(ts_code);
CREATE INDEX IF NOT EXISTS idx_factor_scores_trade_date ON factor_scores(trade_date);
CREATE INDEX IF NOT EXISTS idx_factor_scores_composite ON factor_scores(ts_code, trade_date);
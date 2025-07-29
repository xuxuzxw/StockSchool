-- 情感数据表
CREATE TABLE IF NOT EXISTS sentiment_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    trade_date DATE NOT NULL,         -- 交易日期
    news_count INT,                   -- 新闻数量
    positive_news INT,                -- 正面新闻数量
    negative_news INT,                -- 负面新闻数量
    neutral_news INT,                 -- 中性新闻数量
    sentiment_score DECIMAL(8,4),     -- 情感得分
    sentiment_label VARCHAR(10),      -- 情感标签
    source VARCHAR(50),               -- 数据来源
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, trade_date)
);

-- 创建TimescaleDB超表
SELECT create_hypertable('sentiment_data', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_sentiment_data_ts_code ON sentiment_data(ts_code);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_trade_date ON sentiment_data(trade_date);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_composite ON sentiment_data(ts_code, trade_date);
-- AkShare数据源相关表结构
-- 根据数据同步增强功能需求创建的新表

-- 新闻情绪数据表
CREATE TABLE IF NOT EXISTS news_sentiment (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    news_date DATE NOT NULL,
    sentiment_score DECIMAL(5,4),  -- 情绪分数 -1到1
    positive_count INTEGER,
    negative_count INTEGER,
    neutral_count INTEGER,
    news_volume INTEGER,
    source VARCHAR(50),  -- 数据来源
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, news_date, source)
);

-- 创建时序超表
SELECT create_hypertable('news_sentiment', 'news_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_news_sentiment_ts_code ON news_sentiment(ts_code);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_date ON news_sentiment(news_date);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_score ON news_sentiment(sentiment_score);

-- 用户关注度数据表
CREATE TABLE IF NOT EXISTS user_attention (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    attention_date DATE NOT NULL,
    attention_score DECIMAL(8,4),  -- 关注度分数
    search_volume INTEGER,
    discussion_volume INTEGER,
    view_count INTEGER,
    comment_count INTEGER,
    source VARCHAR(50),  -- 数据来源
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, attention_date, source)
);

-- 创建时序超表
SELECT create_hypertable('user_attention', 'attention_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_user_attention_ts_code ON user_attention(ts_code);
CREATE INDEX IF NOT EXISTS idx_user_attention_date ON user_attention(attention_date);
CREATE INDEX IF NOT EXISTS idx_user_attention_score ON user_attention(attention_score);

-- 人气榜数据表
CREATE TABLE IF NOT EXISTS popularity_ranking (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    ranking_date DATE NOT NULL,
    ranking_type VARCHAR(20),  -- 'hot', 'active', 'attention', 'volume'
    rank_position INTEGER,
    popularity_score DECIMAL(8,4),
    change_from_previous INTEGER,  -- 相比前一天的排名变化
    source VARCHAR(50),  -- 数据来源
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, ranking_date, ranking_type, source)
);

-- 创建时序超表
SELECT create_hypertable('popularity_ranking', 'ranking_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_popularity_ranking_ts_code ON popularity_ranking(ts_code);
CREATE INDEX IF NOT EXISTS idx_popularity_ranking_date ON popularity_ranking(ranking_date);
CREATE INDEX IF NOT EXISTS idx_popularity_ranking_type ON popularity_ranking(ranking_type);
CREATE INDEX IF NOT EXISTS idx_popularity_ranking_position ON popularity_ranking(rank_position);

-- 数据同步状态表（如果不存在）
CREATE TABLE IF NOT EXISTS sync_status (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    data_source VARCHAR(50) NOT NULL,  -- 'tushare', 'akshare'
    data_type VARCHAR(50) NOT NULL,    -- 'daily', 'financial', 'sentiment', 'news', 'attention', 'ranking'
    last_sync_date DATE,
    last_sync_time TIMESTAMP,
    sync_status VARCHAR(20),           -- 'success', 'failed', 'running', 'pending'
    records_processed INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    error_message TEXT,
    sync_duration_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(data_source, data_type)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_sync_status_source ON sync_status(data_source);
CREATE INDEX IF NOT EXISTS idx_sync_status_type ON sync_status(data_type);
CREATE INDEX IF NOT EXISTS idx_sync_status_status ON sync_status(sync_status);
CREATE INDEX IF NOT EXISTS idx_sync_status_date ON sync_status(last_sync_date);

-- 数据质量监控表
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    metric_date DATE NOT NULL,
    total_records INTEGER,
    missing_records INTEGER,
    duplicate_records INTEGER,
    anomaly_records INTEGER,
    completeness_rate DECIMAL(5,4),  -- 完整性比率 0-1
    accuracy_rate DECIMAL(5,4),      -- 准确性比率 0-1
    timeliness_score DECIMAL(5,4),   -- 时效性评分 0-1
    overall_quality_score DECIMAL(5,4), -- 综合质量评分 0-1
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(table_name, metric_date)
);

-- 创建时序超表
SELECT create_hypertable('data_quality_metrics', 'metric_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_data_quality_table ON data_quality_metrics(table_name);
CREATE INDEX IF NOT EXISTS idx_data_quality_date ON data_quality_metrics(metric_date);
CREATE INDEX IF NOT EXISTS idx_data_quality_score ON data_quality_metrics(overall_quality_score);
-- 申
万行业分类表
CREATE TABLE IF NOT EXISTS industry_classification (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    industry_code VARCHAR(20) NOT NULL,
    industry_name VARCHAR(100) NOT NULL,
    industry_level VARCHAR(10) NOT NULL,  -- 'L1', 'L2', 'L3'
    parent_code VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    source VARCHAR(50) DEFAULT 'tushare_sw2021',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(industry_code, source)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_industry_classification_code ON industry_classification(industry_code);
CREATE INDEX IF NOT EXISTS idx_industry_classification_level ON industry_classification(industry_level);
CREATE INDEX IF NOT EXISTS idx_industry_classification_parent ON industry_classification(parent_code);
CREATE INDEX IF NOT EXISTS idx_industry_classification_active ON industry_classification(is_active);

-- 股票行业归属映射表
CREATE TABLE IF NOT EXISTS stock_industry_mapping (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    industry_code VARCHAR(20) NOT NULL,
    in_date DATE NOT NULL,
    out_date DATE,
    is_current BOOLEAN DEFAULT TRUE,
    source VARCHAR(50) DEFAULT 'tushare_sw2021',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, industry_code, in_date, source)
);

-- 创建时序超表
SELECT create_hypertable('stock_industry_mapping', 'in_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_stock_industry_mapping_ts_code ON stock_industry_mapping(ts_code);
CREATE INDEX IF NOT EXISTS idx_stock_industry_mapping_industry ON stock_industry_mapping(industry_code);
CREATE INDEX IF NOT EXISTS idx_stock_industry_mapping_current ON stock_industry_mapping(is_current);
CREATE INDEX IF NOT EXISTS idx_stock_industry_mapping_date_range ON stock_industry_mapping(in_date, out_date);

-- 申万行业历史变更表（用于跟踪行业分类的历史变化）
CREATE TABLE IF NOT EXISTS sw_industry_history (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    old_industry_code VARCHAR(20),
    new_industry_code VARCHAR(20),
    change_date DATE NOT NULL,
    change_reason VARCHAR(200),
    source VARCHAR(50) DEFAULT 'tushare_sw2021',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, change_date, old_industry_code, new_industry_code)
);

-- 创建时序超表
SELECT create_hypertable('sw_industry_history', 'change_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_sw_industry_history_ts_code ON sw_industry_history(ts_code);
CREATE INDEX IF NOT EXISTS idx_sw_industry_history_date ON sw_industry_history(change_date);
CREATE INDEX IF NOT EXISTS idx_sw_industry_history_old_industry ON sw_industry_history(old_industry_code);
CREATE INDEX IF NOT EXISTS idx_sw_industry_history_new_industry ON sw_industry_history(new_industry_code);
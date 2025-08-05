-- StockSchool 验收测试数据库初始化脚本
-- 此脚本用于创建验收测试所需的数据库结构和基础数据

-- ==================== 扩展和配置 ====================
-- 启用 TimescaleDB 扩展
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 启用其他必要扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- 设置时区
SET timezone = 'Asia/Shanghai';

-- ==================== 基础表结构 ====================

-- 股票基础信息表
CREATE TABLE IF NOT EXISTS stock_basic (
    ts_code VARCHAR(20) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    name VARCHAR(100) NOT NULL,
    area VARCHAR(50),
    industry VARCHAR(100),
    market VARCHAR(20),
    list_date DATE,
    is_hs VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_stock_basic_symbol ON stock_basic(symbol);
CREATE INDEX IF NOT EXISTS idx_stock_basic_industry ON stock_basic(industry);
CREATE INDEX IF NOT EXISTS idx_stock_basic_market ON stock_basic(market);

-- 股票日线数据表
CREATE TABLE IF NOT EXISTS stock_daily (
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    open DECIMAL(10,3),
    high DECIMAL(10,3),
    low DECIMAL(10,3),
    close DECIMAL(10,3),
    pre_close DECIMAL(10,3),
    change DECIMAL(10,3),
    pct_chg DECIMAL(8,4),
    vol DECIMAL(20,2),
    amount DECIMAL(20,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);

-- 转换为 TimescaleDB 超表
SELECT create_hypertable('stock_daily', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_stock_daily_ts_code ON stock_daily(ts_code, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_daily_trade_date ON stock_daily(trade_date DESC);

-- 财务数据表
CREATE TABLE IF NOT EXISTS financial_data (
    ts_code VARCHAR(20) NOT NULL,
    end_date DATE NOT NULL,
    report_type VARCHAR(20) NOT NULL, -- annual, quarter
    revenue DECIMAL(20,2),
    net_profit DECIMAL(20,2),
    total_assets DECIMAL(20,2),
    total_equity DECIMAL(20,2),
    roe DECIMAL(8,4),
    roa DECIMAL(8,4),
    eps DECIMAL(8,4),
    bps DECIMAL(8,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, end_date, report_type)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_financial_data_ts_code ON financial_data(ts_code, end_date DESC);
CREATE INDEX IF NOT EXISTS idx_financial_data_end_date ON financial_data(end_date DESC);

-- 技术因子表
CREATE TABLE IF NOT EXISTS technical_factors (
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    rsi_14 DECIMAL(8,4),
    rsi_6 DECIMAL(8,4),
    macd DECIMAL(8,4),
    macd_signal DECIMAL(8,4),
    macd_hist DECIMAL(8,4),
    bb_upper DECIMAL(10,3),
    bb_middle DECIMAL(10,3),
    bb_lower DECIMAL(10,3),
    bb_width DECIMAL(8,4),
    sma_5 DECIMAL(10,3),
    sma_10 DECIMAL(10,3),
    sma_20 DECIMAL(10,3),
    sma_60 DECIMAL(10,3),
    ema_12 DECIMAL(10,3),
    ema_26 DECIMAL(10,3),
    volume_ratio DECIMAL(8,4),
    turnover_rate DECIMAL(8,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);

-- 转换为 TimescaleDB 超表
SELECT create_hypertable('technical_factors', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_technical_factors_ts_code ON technical_factors(ts_code, trade_date DESC);

-- 基本面因子表
CREATE TABLE IF NOT EXISTS fundamental_factors (
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    pe_ratio DECIMAL(8,4),
    pb_ratio DECIMAL(8,4),
    ps_ratio DECIMAL(8,4),
    pcf_ratio DECIMAL(8,4),
    market_cap DECIMAL(20,2),
    circulating_cap DECIMAL(20,2),
    total_share DECIMAL(20,2),
    float_share DECIMAL(20,2),
    free_share DECIMAL(20,2),
    revenue_growth DECIMAL(8,4),
    profit_growth DECIMAL(8,4),
    roe DECIMAL(8,4),
    roa DECIMAL(8,4),
    gross_margin DECIMAL(8,4),
    net_margin DECIMAL(8,4),
    debt_ratio DECIMAL(8,4),
    current_ratio DECIMAL(8,4),
    quick_ratio DECIMAL(8,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);

-- 转换为 TimescaleDB 超表
SELECT create_hypertable('fundamental_factors', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_fundamental_factors_ts_code ON fundamental_factors(ts_code, trade_date DESC);

-- 情绪因子表
CREATE TABLE IF NOT EXISTS sentiment_factors (
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    money_flow_5 DECIMAL(20,2),
    money_flow_10 DECIMAL(20,2),
    money_flow_20 DECIMAL(20,2),
    net_inflow DECIMAL(20,2),
    main_inflow DECIMAL(20,2),
    retail_inflow DECIMAL(20,2),
    attention_score DECIMAL(8,4),
    news_sentiment DECIMAL(8,4),
    social_sentiment DECIMAL(8,4),
    analyst_rating DECIMAL(8,4),
    institution_holding DECIMAL(8,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);

-- 转换为 TimescaleDB 超表
SELECT create_hypertable('sentiment_factors', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_sentiment_factors_ts_code ON sentiment_factors(ts_code, trade_date DESC);

-- AI 模型表
CREATE TABLE IF NOT EXISTS ai_models (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- lightgbm, xgboost, random_forest
    model_version VARCHAR(20) NOT NULL,
    training_start_date DATE NOT NULL,
    training_end_date DATE NOT NULL,
    feature_list TEXT[], -- 使用的特征列表
    hyperparameters JSONB,
    performance_metrics JSONB,
    model_path VARCHAR(500),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_ai_models_name_version ON ai_models(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_ai_models_type ON ai_models(model_type);
CREATE INDEX IF NOT EXISTS idx_ai_models_active ON ai_models(is_active);

-- AI 预测结果表
CREATE TABLE IF NOT EXISTS ai_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ai_models(model_id),
    ts_code VARCHAR(20) NOT NULL,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    predicted_return DECIMAL(8,4),
    predicted_price DECIMAL(10,3),
    confidence_score DECIMAL(8,4),
    feature_importance JSONB,
    shap_values JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 转换为 TimescaleDB 超表
SELECT create_hypertable('ai_predictions', 'prediction_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_ai_predictions_model_ts_code ON ai_predictions(model_id, ts_code, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_ts_code ON ai_predictions(ts_code, prediction_date DESC);

-- 外接AI分析结果表
CREATE TABLE IF NOT EXISTS external_ai_analysis (
    analysis_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    analysis_date DATE NOT NULL,
    analysis_type VARCHAR(50) NOT NULL, -- deep_analysis, backtest_optimization
    request_data JSONB,
    response_data JSONB,
    technical_analysis JSONB,
    fundamental_analysis JSONB,
    sentiment_analysis JSONB,
    investment_advice TEXT,
    risk_assessment JSONB,
    price_prediction JSONB,
    confidence_score DECIMAL(8,4),
    api_response_time DECIMAL(8,3),
    api_success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 转换为 TimescaleDB 超表
SELECT create_hypertable('external_ai_analysis', 'analysis_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_external_ai_analysis_ts_code ON external_ai_analysis(ts_code, analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_external_ai_analysis_type ON external_ai_analysis(analysis_type, analysis_date DESC);

-- 回测结果表
CREATE TABLE IF NOT EXISTS backtest_results (
    backtest_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    stock_pool TEXT[], -- 股票池
    factor_weights JSONB,
    strategy_params JSONB,
    total_return DECIMAL(8,4),
    annual_return DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    sortino_ratio DECIMAL(8,4),
    calmar_ratio DECIMAL(8,4),
    win_rate DECIMAL(8,4),
    profit_loss_ratio DECIMAL(8,4),
    daily_returns JSONB,
    positions JSONB,
    trades JSONB,
    performance_metrics JSONB,
    is_optimized BOOLEAN DEFAULT FALSE,
    optimization_source VARCHAR(50), -- manual, ai_optimized
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy ON backtest_results(strategy_name, start_date DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_results_performance ON backtest_results(total_return DESC, max_drawdown ASC);

-- 验收测试结果表
CREATE TABLE IF NOT EXISTS acceptance_test_results (
    test_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_session_id UUID NOT NULL,
    test_phase VARCHAR(50) NOT NULL,
    test_name VARCHAR(200) NOT NULL,
    test_status VARCHAR(20) NOT NULL, -- passed, failed, skipped
    execution_time DECIMAL(8,3),
    error_message TEXT,
    test_details JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_acceptance_test_session ON acceptance_test_results(test_session_id, test_phase);
CREATE INDEX IF NOT EXISTS idx_acceptance_test_status ON acceptance_test_results(test_status, created_at DESC);

-- ==================== 数据压缩和保留策略 ====================

-- 为时序表添加压缩策略（3个月后压缩）
SELECT add_compression_policy('stock_daily', INTERVAL '3 months');
SELECT add_compression_policy('technical_factors', INTERVAL '3 months');
SELECT add_compression_policy('fundamental_factors', INTERVAL '3 months');
SELECT add_compression_policy('sentiment_factors', INTERVAL '3 months');
SELECT add_compression_policy('ai_predictions', INTERVAL '3 months');
SELECT add_compression_policy('external_ai_analysis', INTERVAL '1 month');

-- 为测试数据添加保留策略（测试数据保留1年）
SELECT add_retention_policy('acceptance_test_results', INTERVAL '1 year');

-- ==================== 触发器和函数 ====================

-- 更新时间戳触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为需要的表添加更新时间戳触发器
CREATE TRIGGER update_stock_basic_updated_at BEFORE UPDATE ON stock_basic
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_models_updated_at BEFORE UPDATE ON ai_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==================== 视图创建 ====================

-- 股票综合信息视图
CREATE OR REPLACE VIEW stock_comprehensive_view AS
SELECT 
    sb.ts_code,
    sb.symbol,
    sb.name,
    sb.industry,
    sb.market,
    sd.trade_date,
    sd.close,
    sd.pct_chg,
    sd.vol,
    tf.rsi_14,
    tf.macd,
    ff.pe_ratio,
    ff.pb_ratio,
    ff.roe,
    sf.net_inflow,
    sf.attention_score
FROM stock_basic sb
LEFT JOIN stock_daily sd ON sb.ts_code = sd.ts_code
LEFT JOIN technical_factors tf ON sb.ts_code = tf.ts_code AND sd.trade_date = tf.trade_date
LEFT JOIN fundamental_factors ff ON sb.ts_code = ff.ts_code AND sd.trade_date = ff.trade_date
LEFT JOIN sentiment_factors sf ON sb.ts_code = sf.ts_code AND sd.trade_date = sf.trade_date;

-- 验收测试统计视图
CREATE OR REPLACE VIEW acceptance_test_summary AS
SELECT 
    test_session_id,
    test_phase,
    COUNT(*) as total_tests,
    COUNT(CASE WHEN test_status = 'passed' THEN 1 END) as passed_tests,
    COUNT(CASE WHEN test_status = 'failed' THEN 1 END) as failed_tests,
    COUNT(CASE WHEN test_status = 'skipped' THEN 1 END) as skipped_tests,
    ROUND(AVG(execution_time), 3) as avg_execution_time,
    MIN(created_at) as start_time,
    MAX(created_at) as end_time
FROM acceptance_test_results
GROUP BY test_session_id, test_phase;

-- ==================== 权限设置 ====================

-- 为验收测试用户授权
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO acceptance_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO acceptance_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO acceptance_user;

-- ==================== 初始化完成标记 ====================

-- 插入初始化完成标记
INSERT INTO acceptance_test_results (
    test_session_id,
    test_phase,
    test_name,
    test_status,
    execution_time,
    test_details
) VALUES (
    uuid_generate_v4(),
    'initialization',
    'database_schema_initialization',
    'passed',
    0.0,
    '{"message": "Database schema initialized successfully", "timestamp": "' || CURRENT_TIMESTAMP || '"}'
);

-- 输出初始化完成信息
DO $$
BEGIN
    RAISE NOTICE 'StockSchool 验收测试数据库初始化完成！';
    RAISE NOTICE '数据库: %', current_database();
    RAISE NOTICE '用户: %', current_user;
    RAISE NOTICE '时间: %', CURRENT_TIMESTAMP;
END $$;
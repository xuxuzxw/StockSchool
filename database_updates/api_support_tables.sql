-- API支持表结构
-- 用于支持因子API的认证、权限管理、因子定义等功能

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    roles TEXT[] DEFAULT ARRAY['viewer'],
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- 创建用户表索引
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);

-- 因子定义表
CREATE TABLE IF NOT EXISTS factor_definitions (
    factor_name VARCHAR(100) PRIMARY KEY,
    factor_type VARCHAR(50) NOT NULL,
    category VARCHAR(100) NOT NULL,
    description TEXT,
    formula TEXT,
    parameters JSONB DEFAULT '{}',
    data_requirements TEXT[] DEFAULT ARRAY[]::TEXT[],
    update_frequency VARCHAR(20) DEFAULT 'daily',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建因子定义表索引
CREATE INDEX IF NOT EXISTS idx_factor_definitions_type ON factor_definitions(factor_type);
CREATE INDEX IF NOT EXISTS idx_factor_definitions_category ON factor_definitions(category);
CREATE INDEX IF NOT EXISTS idx_factor_definitions_is_active ON factor_definitions(is_active);

-- 任务调度配置表
CREATE TABLE IF NOT EXISTS task_schedules (
    task_id VARCHAR(100) PRIMARY KEY,
    task_name VARCHAR(200) NOT NULL,
    task_type VARCHAR(50) NOT NULL,
    schedule_time VARCHAR(50) NOT NULL,
    priority VARCHAR(20) DEFAULT 'normal',
    dependencies TEXT[] DEFAULT ARRAY[]::TEXT[],
    parameters JSONB DEFAULT '{}',
    is_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run TIMESTAMP,
    next_run TIMESTAMP
);

-- 创建任务调度表索引
CREATE INDEX IF NOT EXISTS idx_task_schedules_type ON task_schedules(task_type);
CREATE INDEX IF NOT EXISTS idx_task_schedules_enabled ON task_schedules(is_enabled);
CREATE INDEX IF NOT EXISTS idx_task_schedules_next_run ON task_schedules(next_run);

-- 任务执行历史表
CREATE TABLE IF NOT EXISTS task_executions (
    execution_id VARCHAR(100) PRIMARY KEY,
    task_id VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    progress FLOAT DEFAULT 0.0,
    message TEXT,
    error_details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES task_schedules(task_id)
);

-- 创建任务执行历史表索引
CREATE INDEX IF NOT EXISTS idx_task_executions_task_id ON task_executions(task_id);
CREATE INDEX IF NOT EXISTS idx_task_executions_status ON task_executions(status);
CREATE INDEX IF NOT EXISTS idx_task_executions_started_at ON task_executions(started_at);

-- 监控告警规则表
CREATE TABLE IF NOT EXISTS monitoring_rules (
    rule_id VARCHAR(100) PRIMARY KEY,
    rule_name VARCHAR(200) NOT NULL,
    rule_type VARCHAR(50) NOT NULL,
    condition_expression TEXT NOT NULL,
    threshold_value FLOAT NOT NULL,
    alert_level VARCHAR(20) DEFAULT 'warning',
    notification_channels TEXT[] DEFAULT ARRAY[]::TEXT[],
    is_enabled BOOLEAN DEFAULT true,
    cooldown_minutes INTEGER DEFAULT 60,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建监控规则表索引
CREATE INDEX IF NOT EXISTS idx_monitoring_rules_type ON monitoring_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_monitoring_rules_enabled ON monitoring_rules(is_enabled);
CREATE INDEX IF NOT EXISTS idx_monitoring_rules_level ON monitoring_rules(alert_level);

-- 告警历史表
CREATE TABLE IF NOT EXISTS alert_history (
    alert_id VARCHAR(100) PRIMARY KEY,
    rule_id VARCHAR(100),
    alert_type VARCHAR(50) NOT NULL,
    alert_level VARCHAR(20) NOT NULL,
    title VARCHAR(500) NOT NULL,
    message TEXT,
    source VARCHAR(100),
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    is_resolved BOOLEAN DEFAULT false,
    resolution_notes TEXT,
    FOREIGN KEY (rule_id) REFERENCES monitoring_rules(rule_id)
);

-- 创建告警历史表索引
CREATE INDEX IF NOT EXISTS idx_alert_history_rule_id ON alert_history(rule_id);
CREATE INDEX IF NOT EXISTS idx_alert_history_type ON alert_history(alert_type);
CREATE INDEX IF NOT EXISTS idx_alert_history_level ON alert_history(alert_level);
CREATE INDEX IF NOT EXISTS idx_alert_history_triggered_at ON alert_history(triggered_at);
CREATE INDEX IF NOT EXISTS idx_alert_history_resolved ON alert_history(is_resolved);

-- 性能指标历史表
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(20),
    source VARCHAR(100),
    tags JSONB DEFAULT '{}',
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建性能指标表索引
CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_recorded_at ON performance_metrics(recorded_at);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_source ON performance_metrics(source);

-- 将性能指标表转换为时序表（如果使用TimescaleDB）
SELECT create_hypertable('performance_metrics', 'recorded_at', if_not_exists => TRUE);

-- API访问日志表
CREATE TABLE IF NOT EXISTS api_access_logs (
    log_id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(50),
    endpoint VARCHAR(500) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    ip_address INET,
    user_agent TEXT,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 创建API访问日志表索引
CREATE INDEX IF NOT EXISTS idx_api_access_logs_user_id ON api_access_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_api_access_logs_endpoint ON api_access_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_access_logs_accessed_at ON api_access_logs(accessed_at);
CREATE INDEX IF NOT EXISTS idx_api_access_logs_status_code ON api_access_logs(status_code);

-- 将API访问日志表转换为时序表（如果使用TimescaleDB）
SELECT create_hypertable('api_access_logs', 'accessed_at', if_not_exists => TRUE);

-- 因子计算请求表
CREATE TABLE IF NOT EXISTS factor_calculation_requests (
    request_id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(50),
    request_type VARCHAR(50) NOT NULL,
    ts_codes TEXT[] DEFAULT ARRAY[]::TEXT[],
    factor_names TEXT[] DEFAULT ARRAY[]::TEXT[],
    factor_types TEXT[] DEFAULT ARRAY[]::TEXT[],
    calculation_date DATE,
    start_date DATE,
    end_date DATE,
    force_recalculate BOOLEAN DEFAULT false,
    priority VARCHAR(20) DEFAULT 'normal',
    status VARCHAR(20) DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,
    result_summary JSONB DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 创建因子计算请求表索引
CREATE INDEX IF NOT EXISTS idx_factor_calc_requests_user_id ON factor_calculation_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_factor_calc_requests_status ON factor_calculation_requests(status);
CREATE INDEX IF NOT EXISTS idx_factor_calc_requests_created_at ON factor_calculation_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_factor_calc_requests_type ON factor_calculation_requests(request_type);

-- 因子有效性分析结果表
CREATE TABLE IF NOT EXISTS factor_effectiveness_analysis (
    analysis_id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(50),
    factor_names TEXT[] NOT NULL,
    analysis_types TEXT[] NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    return_periods INTEGER[] DEFAULT ARRAY[1,5,20],
    ic_results JSONB DEFAULT '{}',
    ir_results JSONB DEFAULT '{}',
    layered_backtest_results JSONB DEFAULT '{}',
    decay_analysis_results JSONB DEFAULT '{}',
    correlation_analysis_results JSONB DEFAULT '{}',
    overall_rating VARCHAR(10),
    recommendations TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 创建因子有效性分析表索引
CREATE INDEX IF NOT EXISTS idx_factor_effectiveness_user_id ON factor_effectiveness_analysis(user_id);
CREATE INDEX IF NOT EXISTS idx_factor_effectiveness_status ON factor_effectiveness_analysis(status);
CREATE INDEX IF NOT EXISTS idx_factor_effectiveness_created_at ON factor_effectiveness_analysis(created_at);

-- 系统配置表
CREATE TABLE IF NOT EXISTS system_configurations (
    config_key VARCHAR(200) PRIMARY KEY,
    config_value JSONB NOT NULL,
    config_type VARCHAR(50) NOT NULL,
    description TEXT,
    is_encrypted BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(50),
    FOREIGN KEY (updated_by) REFERENCES users(user_id)
);

-- 创建系统配置表索引
CREATE INDEX IF NOT EXISTS idx_system_configurations_type ON system_configurations(config_type);
CREATE INDEX IF NOT EXISTS idx_system_configurations_updated_at ON system_configurations(updated_at);

-- 插入默认因子定义
INSERT INTO factor_definitions (factor_name, factor_type, category, description, formula, parameters, data_requirements, update_frequency) VALUES
-- 技术面因子
('sma_5', 'technical', 'trend', '5日简单移动平均', 'SMA(close, 5)', '{"window": 5}', ARRAY['stock_daily'], 'daily'),
('sma_10', 'technical', 'trend', '10日简单移动平均', 'SMA(close, 10)', '{"window": 10}', ARRAY['stock_daily'], 'daily'),
('sma_20', 'technical', 'trend', '20日简单移动平均', 'SMA(close, 20)', '{"window": 20}', ARRAY['stock_daily'], 'daily'),
('sma_60', 'technical', 'trend', '60日简单移动平均', 'SMA(close, 60)', '{"window": 60}', ARRAY['stock_daily'], 'daily'),
('ema_12', 'technical', 'trend', '12日指数移动平均', 'EMA(close, 12)', '{"window": 12}', ARRAY['stock_daily'], 'daily'),
('ema_26', 'technical', 'trend', '26日指数移动平均', 'EMA(close, 26)', '{"window": 26}', ARRAY['stock_daily'], 'daily'),
('rsi_6', 'technical', 'momentum', '6日相对强弱指标', 'RSI(close, 6)', '{"window": 6}', ARRAY['stock_daily'], 'daily'),
('rsi_14', 'technical', 'momentum', '14日相对强弱指标', 'RSI(close, 14)', '{"window": 14}', ARRAY['stock_daily'], 'daily'),
('macd', 'technical', 'momentum', 'MACD指标', 'MACD(close, 12, 26, 9)', '{"fast": 12, "slow": 26, "signal": 9}', ARRAY['stock_daily'], 'daily'),
('macd_signal', 'technical', 'momentum', 'MACD信号线', 'MACD_SIGNAL(close, 12, 26, 9)', '{"fast": 12, "slow": 26, "signal": 9}', ARRAY['stock_daily'], 'daily'),
('macd_hist', 'technical', 'momentum', 'MACD柱状图', 'MACD_HIST(close, 12, 26, 9)', '{"fast": 12, "slow": 26, "signal": 9}', ARRAY['stock_daily'], 'daily'),
('williams_r', 'technical', 'momentum', '威廉指标', 'WILLIAMS_R(high, low, close, 14)', '{"window": 14}', ARRAY['stock_daily'], 'daily'),
('momentum_5', 'technical', 'momentum', '5日动量指标', 'MOMENTUM(close, 5)', '{"window": 5}', ARRAY['stock_daily'], 'daily'),
('momentum_10', 'technical', 'momentum', '10日动量指标', 'MOMENTUM(close, 10)', '{"window": 10}', ARRAY['stock_daily'], 'daily'),
('momentum_20', 'technical', 'momentum', '20日动量指标', 'MOMENTUM(close, 20)', '{"window": 20}', ARRAY['stock_daily'], 'daily'),
('roc_5', 'technical', 'momentum', '5日变化率', 'ROC(close, 5)', '{"window": 5}', ARRAY['stock_daily'], 'daily'),
('roc_10', 'technical', 'momentum', '10日变化率', 'ROC(close, 10)', '{"window": 10}', ARRAY['stock_daily'], 'daily'),
('roc_20', 'technical', 'momentum', '20日变化率', 'ROC(close, 20)', '{"window": 20}', ARRAY['stock_daily'], 'daily'),
('volatility_5', 'technical', 'volatility', '5日历史波动率', 'VOLATILITY(close, 5)', '{"window": 5}', ARRAY['stock_daily'], 'daily'),
('volatility_20', 'technical', 'volatility', '20日历史波动率', 'VOLATILITY(close, 20)', '{"window": 20}', ARRAY['stock_daily'], 'daily'),
('volatility_60', 'technical', 'volatility', '60日历史波动率', 'VOLATILITY(close, 60)', '{"window": 60}', ARRAY['stock_daily'], 'daily'),
('atr_14', 'technical', 'volatility', '14日平均真实波幅', 'ATR(high, low, close, 14)', '{"window": 14}', ARRAY['stock_daily'], 'daily'),
('bollinger_upper', 'technical', 'volatility', '布林带上轨', 'BOLLINGER_UPPER(close, 20, 2)', '{"window": 20, "std_dev": 2}', ARRAY['stock_daily'], 'daily'),
('bollinger_lower', 'technical', 'volatility', '布林带下轨', 'BOLLINGER_LOWER(close, 20, 2)', '{"window": 20, "std_dev": 2}', ARRAY['stock_daily'], 'daily'),
('bollinger_width', 'technical', 'volatility', '布林带宽度', 'BOLLINGER_WIDTH(close, 20, 2)', '{"window": 20, "std_dev": 2}', ARRAY['stock_daily'], 'daily'),
('bollinger_position', 'technical', 'volatility', '布林带位置', 'BOLLINGER_POSITION(close, 20, 2)', '{"window": 20, "std_dev": 2}', ARRAY['stock_daily'], 'daily'),
('volume_sma_5', 'technical', 'volume', '5日成交量均线', 'SMA(volume, 5)', '{"window": 5}', ARRAY['stock_daily'], 'daily'),
('volume_sma_20', 'technical', 'volume', '20日成交量均线', 'SMA(volume, 20)', '{"window": 20}', ARRAY['stock_daily'], 'daily'),
('volume_ratio', 'technical', 'volume', '量比指标', 'VOLUME_RATIO(volume, 5)', '{"window": 5}', ARRAY['stock_daily'], 'daily'),
('vpt', 'technical', 'volume', '量价趋势指标', 'VPT(close, volume)', '{}', ARRAY['stock_daily'], 'daily'),
('mfi_14', 'technical', 'volume', '14日资金流量指标', 'MFI(high, low, close, volume, 14)', '{"window": 14}', ARRAY['stock_daily'], 'daily'),

-- 基本面因子
('pe_ttm', 'fundamental', 'valuation', '市盈率TTM', 'PE_TTM(market_cap, net_profit_ttm)', '{}', ARRAY['stock_basic', 'financial_indicator'], 'quarterly'),
('pb', 'fundamental', 'valuation', '市净率', 'PB(market_cap, total_equity)', '{}', ARRAY['stock_basic', 'balance_sheet'], 'quarterly'),
('ps_ttm', 'fundamental', 'valuation', '市销率TTM', 'PS_TTM(market_cap, revenue_ttm)', '{}', ARRAY['stock_basic', 'income_statement'], 'quarterly'),
('pcf_ttm', 'fundamental', 'valuation', '市现率TTM', 'PCF_TTM(market_cap, cash_flow_ttm)', '{}', ARRAY['stock_basic', 'cash_flow'], 'quarterly'),
('ev_ebitda', 'fundamental', 'valuation', '企业价值倍数', 'EV_EBITDA(enterprise_value, ebitda)', '{}', ARRAY['stock_basic', 'financial_indicator'], 'quarterly'),
('peg', 'fundamental', 'valuation', 'PEG比率', 'PEG(pe_ttm, growth_rate)', '{}', ARRAY['stock_basic', 'financial_indicator'], 'quarterly'),
('roe', 'fundamental', 'profitability', '净资产收益率', 'ROE(net_profit, total_equity)', '{}', ARRAY['income_statement', 'balance_sheet'], 'quarterly'),
('roa', 'fundamental', 'profitability', '总资产收益率', 'ROA(net_profit, total_assets)', '{}', ARRAY['income_statement', 'balance_sheet'], 'quarterly'),
('roic', 'fundamental', 'profitability', '投入资本回报率', 'ROIC(nopat, invested_capital)', '{}', ARRAY['income_statement', 'balance_sheet'], 'quarterly'),
('gross_margin', 'fundamental', 'profitability', '毛利率', 'GROSS_MARGIN(gross_profit, revenue)', '{}', ARRAY['income_statement'], 'quarterly'),
('net_margin', 'fundamental', 'profitability', '净利率', 'NET_MARGIN(net_profit, revenue)', '{}', ARRAY['income_statement'], 'quarterly'),
('operating_margin', 'fundamental', 'profitability', '营业利润率', 'OPERATING_MARGIN(operating_profit, revenue)', '{}', ARRAY['income_statement'], 'quarterly'),

-- 情绪面因子
('money_flow_5', 'sentiment', 'money_flow', '5日资金流向', 'MONEY_FLOW(close, volume, 5)', '{"window": 5}', ARRAY['stock_daily'], 'daily'),
('money_flow_20', 'sentiment', 'money_flow', '20日资金流向', 'MONEY_FLOW(close, volume, 20)', '{"window": 20}', ARRAY['stock_daily'], 'daily'),
('net_inflow_rate', 'sentiment', 'money_flow', '净流入率', 'NET_INFLOW_RATE(close, volume)', '{}', ARRAY['stock_daily'], 'daily'),
('large_order_ratio', 'sentiment', 'money_flow', '大单占比', 'LARGE_ORDER_RATIO(volume, amount)', '{}', ARRAY['stock_daily'], 'daily'),
('medium_order_ratio', 'sentiment', 'money_flow', '中单占比', 'MEDIUM_ORDER_RATIO(volume, amount)', '{}', ARRAY['stock_daily'], 'daily'),
('small_order_ratio', 'sentiment', 'money_flow', '小单占比', 'SMALL_ORDER_RATIO(volume, amount)', '{}', ARRAY['stock_daily'], 'daily'),
('turnover_rate', 'sentiment', 'attention', '换手率', 'TURNOVER_RATE(volume, float_shares)', '{}', ARRAY['stock_daily', 'stock_basic'], 'daily'),
('volume_attention', 'sentiment', 'attention', '成交量关注度', 'VOLUME_ATTENTION(volume)', '{}', ARRAY['stock_daily'], 'daily'),
('price_attention', 'sentiment', 'attention', '价格关注度', 'PRICE_ATTENTION(close, high, low)', '{}', ARRAY['stock_daily'], 'daily'),
('comprehensive_attention', 'sentiment', 'attention', '综合关注度', 'COMPREHENSIVE_ATTENTION(turnover, volume, price)', '{}', ARRAY['stock_daily'], 'daily'),
('attention_change_rate', 'sentiment', 'attention', '关注度变化率', 'ATTENTION_CHANGE_RATE(attention, 5)', '{"window": 5}', ARRAY['stock_daily'], 'daily'),
('price_momentum_sentiment', 'sentiment', 'strength', '价格动量情绪', 'PRICE_MOMENTUM_SENTIMENT(close, 20)', '{"window": 20}', ARRAY['stock_daily'], 'daily'),
('volatility_sentiment', 'sentiment', 'strength', '波动率情绪', 'VOLATILITY_SENTIMENT(close, 20)', '{"window": 20}', ARRAY['stock_daily'], 'daily'),
('bull_bear_ratio', 'sentiment', 'strength', '看涨看跌比例', 'BULL_BEAR_RATIO(close, volume)', '{}', ARRAY['stock_daily'], 'daily'),
('sentiment_volatility', 'sentiment', 'strength', '情绪波动率', 'SENTIMENT_VOLATILITY(sentiment, 20)', '{"window": 20}', ARRAY['stock_daily'], 'daily'),
('abnormal_volume', 'sentiment', 'event', '异常成交量', 'ABNORMAL_VOLUME(volume, 20)', '{"window": 20}', ARRAY['stock_daily'], 'daily'),
('abnormal_return', 'sentiment', 'event', '异常收益率', 'ABNORMAL_RETURN(return, 20)', '{"window": 20}', ARRAY['stock_daily'], 'daily'),
('limit_up_signal', 'sentiment', 'event', '涨停信号', 'LIMIT_UP_SIGNAL(close, pre_close)', '{}', ARRAY['stock_daily'], 'daily'),
('limit_down_signal', 'sentiment', 'event', '跌停信号', 'LIMIT_DOWN_SIGNAL(close, pre_close)', '{}', ARRAY['stock_daily'], 'daily'),
('gap_up_signal', 'sentiment', 'event', '向上跳空信号', 'GAP_UP_SIGNAL(open, pre_close)', '{}', ARRAY['stock_daily'], 'daily'),
('gap_down_signal', 'sentiment', 'event', '向下跳空信号', 'GAP_DOWN_SIGNAL(open, pre_close)', '{}', ARRAY['stock_daily'], 'daily'),
('volume_spike', 'sentiment', 'event', '成交量异动', 'VOLUME_SPIKE(volume, 20)', '{"window": 20}', ARRAY['stock_daily'], 'daily')

ON CONFLICT (factor_name) DO UPDATE SET
    factor_type = EXCLUDED.factor_type,
    category = EXCLUDED.category,
    description = EXCLUDED.description,
    formula = EXCLUDED.formula,
    parameters = EXCLUDED.parameters,
    data_requirements = EXCLUDED.data_requirements,
    update_frequency = EXCLUDED.update_frequency,
    updated_at = CURRENT_TIMESTAMP;

-- 插入默认系统配置
INSERT INTO system_configurations (config_key, config_value, config_type, description) VALUES
('database.pool_size', '20', 'database', '数据库连接池大小'),
('database.max_overflow', '30', 'database', '数据库连接池最大溢出'),
('computation.max_workers', '4', 'computation', '最大计算工作进程数'),
('computation.batch_size', '1000', 'computation', '批量计算大小'),
('cache.redis_ttl', '3600', 'cache', 'Redis缓存过期时间(秒)'),
('cache.memory_limit', '1000', 'cache', '内存缓存条目限制'),
('monitoring.alert_enabled', 'true', 'monitoring', '是否启用告警'),
('monitoring.metrics_retention_days', '30', 'monitoring', '性能指标保留天数'),
('api.rate_limit_per_minute', '100', 'api', 'API每分钟请求限制'),
('api.jwt_expiration_hours', '24', 'api', 'JWT令牌过期时间(小时)')

ON CONFLICT (config_key) DO UPDATE SET
    config_value = EXCLUDED.config_value,
    updated_at = CURRENT_TIMESTAMP;

-- 插入默认监控规则
INSERT INTO monitoring_rules (rule_id, rule_name, rule_type, condition_expression, threshold_value, alert_level, notification_channels) VALUES
('cpu_high', 'CPU使用率过高', 'resource', 'cpu_percent > threshold', 80.0, 'warning', ARRAY['email']),
('memory_high', '内存使用率过高', 'resource', 'memory_percent > threshold', 85.0, 'warning', ARRAY['email']),
('disk_high', '磁盘使用率过高', 'resource', 'disk_percent > threshold', 90.0, 'error', ARRAY['email', 'webhook']),
('task_failure_rate_high', '任务失败率过高', 'performance', 'failure_rate > threshold', 10.0, 'error', ARRAY['email', 'webhook']),
('api_response_time_high', 'API响应时间过长', 'performance', 'avg_response_time > threshold', 5000.0, 'warning', ARRAY['email']),
('database_connection_high', '数据库连接数过高', 'database', 'connection_count > threshold', 50.0, 'warning', ARRAY['email'])

ON CONFLICT (rule_id) DO UPDATE SET
    rule_name = EXCLUDED.rule_name,
    condition_expression = EXCLUDED.condition_expression,
    threshold_value = EXCLUDED.threshold_value,
    updated_at = CURRENT_TIMESTAMP;

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为相关表创建更新时间触发器
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_factor_definitions_updated_at BEFORE UPDATE ON factor_definitions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_task_schedules_updated_at BEFORE UPDATE ON task_schedules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_monitoring_rules_updated_at BEFORE UPDATE ON monitoring_rules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_configurations_updated_at BEFORE UPDATE ON system_configurations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 创建数据保留策略（使用TimescaleDB）
-- 性能指标保留90天
SELECT add_retention_policy('performance_metrics', INTERVAL '90 days', if_not_exists => TRUE);

-- API访问日志保留180天
SELECT add_retention_policy('api_access_logs', INTERVAL '180 days', if_not_exists => TRUE);

-- 创建压缩策略
-- 性能指标7天后压缩
SELECT add_compression_policy('performance_metrics', INTERVAL '7 days', if_not_exists => TRUE);

-- API访问日志30天后压缩
SELECT add_compression_policy('api_access_logs', INTERVAL '30 days', if_not_exists => TRUE);

COMMIT;
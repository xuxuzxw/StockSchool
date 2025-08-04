-- 因子计算引擎相关表结构
-- 扩展现有的因子存储结构，支持新的因子计算引擎

-- ================================
-- 1. 因子元数据表
-- ================================

-- 因子元数据表
CREATE TABLE IF NOT EXISTS factor_metadata (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    factor_name VARCHAR(100) NOT NULL UNIQUE,  -- 因子名称
    factor_type VARCHAR(20) NOT NULL,          -- 因子类型（technical/fundamental/sentiment）
    category VARCHAR(30) NOT NULL,             -- 因子分类
    description TEXT,                          -- 因子描述
    formula TEXT,                              -- 计算公式
    parameters JSONB,                          -- 计算参数
    data_requirements TEXT[],                  -- 数据需求
    min_periods INTEGER DEFAULT 1,             -- 最小计算周期
    enabled BOOLEAN DEFAULT TRUE,              -- 是否启用
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_factor_metadata_type ON factor_metadata(factor_type);
CREATE INDEX IF NOT EXISTS idx_factor_metadata_category ON factor_metadata(category);

-- ================================
-- 2. 因子宽表设计
-- ================================

-- 技术面因子宽表
CREATE TABLE IF NOT EXISTS stock_factors_technical (
    ts_code VARCHAR(20) NOT NULL,              -- 股票代码
    trade_date DATE NOT NULL,                  -- 交易日期
    
    -- 动量类因子
    momentum_5 DECIMAL(15,8),                  -- 5日动量
    momentum_10 DECIMAL(15,8),                 -- 10日动量
    momentum_20 DECIMAL(15,8),                 -- 20日动量
    rsi_14 DECIMAL(15,8),                      -- 14日RSI
    rsi_6 DECIMAL(15,8),                       -- 6日RSI
    williams_r_14 DECIMAL(15,8),               -- 14日威廉指标
    roc_5 DECIMAL(15,8),                       -- 5日变化率
    roc_10 DECIMAL(15,8),                      -- 10日变化率
    roc_20 DECIMAL(15,8),                      -- 20日变化率
    
    -- 趋势类因子
    sma_5 DECIMAL(15,8),                       -- 5日简单移动平均
    sma_10 DECIMAL(15,8),                      -- 10日简单移动平均
    sma_20 DECIMAL(15,8),                      -- 20日简单移动平均
    sma_60 DECIMAL(15,8),                      -- 60日简单移动平均
    ema_12 DECIMAL(15,8),                      -- 12日指数移动平均
    ema_26 DECIMAL(15,8),                      -- 26日指数移动平均
    macd DECIMAL(15,8),                        -- MACD线
    macd_signal DECIMAL(15,8),                 -- MACD信号线
    macd_histogram DECIMAL(15,8),              -- MACD柱状图
    price_to_sma20 DECIMAL(15,8),              -- 价格相对20日均线
    price_to_sma60 DECIMAL(15,8),              -- 价格相对60日均线
    
    -- 波动率类因子
    volatility_5 DECIMAL(15,8),                -- 5日波动率
    volatility_20 DECIMAL(15,8),               -- 20日波动率
    volatility_60 DECIMAL(15,8),               -- 60日波动率
    atr_14 DECIMAL(15,8),                      -- 14日ATR
    bb_upper DECIMAL(15,8),                    -- 布林带上轨
    bb_middle DECIMAL(15,8),                   -- 布林带中轨
    bb_lower DECIMAL(15,8),                    -- 布林带下轨
    bb_width DECIMAL(15,8),                    -- 布林带宽度
    bb_position DECIMAL(15,8),                 -- 布林带位置
    
    -- 成交量类因子
    volume_sma_5 DECIMAL(15,8),                -- 5日成交量均值
    volume_sma_20 DECIMAL(15,8),               -- 20日成交量均值
    volume_ratio_5 DECIMAL(15,8),              -- 5日量比
    volume_ratio_20 DECIMAL(15,8),             -- 20日量比
    vpt DECIMAL(15,8),                         -- 量价趋势
    mfi DECIMAL(15,8),                         -- 资金流量指标
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (ts_code, trade_date)
);

-- 创建时间序列表
SELECT create_hypertable('stock_factors_technical', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_factors_technical_date ON stock_factors_technical(trade_date);
CREATE INDEX IF NOT EXISTS idx_factors_technical_code ON stock_factors_technical(ts_code);
-- 基本
面因子宽表
CREATE TABLE IF NOT EXISTS stock_factors_fundamental (
    ts_code VARCHAR(20) NOT NULL,              -- 股票代码
    report_date DATE NOT NULL,                 -- 报告期
    
    -- 估值类因子
    pe_ttm DECIMAL(15,8),                      -- 市盈率TTM
    pb DECIMAL(15,8),                          -- 市净率
    ps_ttm DECIMAL(15,8),                      -- 市销率TTM
    pcf_ttm DECIMAL(15,8),                     -- 市现率TTM
    ev_ebitda DECIMAL(15,8),                   -- EV/EBITDA
    peg DECIMAL(15,8),                         -- PEG比率
    
    -- 盈利能力类因子
    roe DECIMAL(15,8),                         -- 净资产收益率
    roa DECIMAL(15,8),                         -- 总资产收益率
    roic DECIMAL(15,8),                        -- 投入资本回报率
    gross_margin DECIMAL(15,8),                -- 毛利率
    net_margin DECIMAL(15,8),                  -- 净利率
    operating_margin DECIMAL(15,8),            -- 营业利润率
    
    -- 成长性类因子
    revenue_yoy DECIMAL(15,8),                 -- 营收同比增长率
    net_profit_yoy DECIMAL(15,8),              -- 净利润同比增长率
    revenue_qoq DECIMAL(15,8),                 -- 营收环比增长率
    net_profit_qoq DECIMAL(15,8),              -- 净利润环比增长率
    revenue_cagr_3y DECIMAL(15,8),             -- 营收3年复合增长率
    net_profit_cagr_3y DECIMAL(15,8),          -- 净利润3年复合增长率
    
    -- 财务质量类因子
    debt_to_equity DECIMAL(15,8),              -- 资产负债率
    current_ratio DECIMAL(15,8),               -- 流动比率
    quick_ratio DECIMAL(15,8),                 -- 速动比率
    cash_ratio DECIMAL(15,8),                  -- 现金比率
    interest_coverage DECIMAL(15,8),           -- 利息保障倍数
    asset_turnover DECIMAL(15,8),              -- 资产周转率
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (ts_code, report_date)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_factors_fundamental_date ON stock_factors_fundamental(report_date);
CREATE INDEX IF NOT EXISTS idx_factors_fundamental_code ON stock_factors_fundamental(ts_code);

-- 情绪面因子宽表
CREATE TABLE IF NOT EXISTS stock_factors_sentiment (
    ts_code VARCHAR(20) NOT NULL,              -- 股票代码
    trade_date DATE NOT NULL,                  -- 交易日期
    
    -- 资金流向类因子
    money_flow_5 DECIMAL(15,8),                -- 5日资金流向
    money_flow_20 DECIMAL(15,8),               -- 20日资金流向
    big_order_ratio DECIMAL(15,8),             -- 大单占比
    medium_order_ratio DECIMAL(15,8),          -- 中单占比
    small_order_ratio DECIMAL(15,8),           -- 小单占比
    net_inflow_rate DECIMAL(15,8),             -- 净流入率
    
    -- 关注度类因子
    search_volume DECIMAL(15,8),               -- 搜索量
    discussion_volume DECIMAL(15,8),           -- 讨论量
    attention_score DECIMAL(15,8),             -- 关注度评分
    attention_change_rate DECIMAL(15,8),       -- 关注度变化率
    
    -- 情绪强度类因子
    news_sentiment DECIMAL(15,8),              -- 新闻情绪
    sentiment_strength DECIMAL(15,8),          -- 情绪强度
    sentiment_volatility DECIMAL(15,8),        -- 情绪波动率
    bullish_ratio DECIMAL(15,8),               -- 看涨比例
    bearish_ratio DECIMAL(15,8),               -- 看跌比例
    
    -- 特殊事件类因子
    dragon_tiger_score DECIMAL(15,8),          -- 龙虎榜评分
    institutional_activity DECIMAL(15,8),      -- 机构活跃度
    northbound_flow DECIMAL(15,8),             -- 北向资金流向
    margin_trading_ratio DECIMAL(15,8),        -- 融资融券比例
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (ts_code, trade_date)
);

-- 创建时间序列表
SELECT create_hypertable('stock_factors_sentiment', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_factors_sentiment_date ON stock_factors_sentiment(trade_date);
CREATE INDEX IF NOT EXISTS idx_factors_sentiment_code ON stock_factors_sentiment(ts_code);

-- ================================
-- 3. 因子标准化统计表
-- ================================

-- 因子统计信息表
CREATE TABLE IF NOT EXISTS factor_statistics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    factor_name VARCHAR(100) NOT NULL,         -- 因子名称
    calculation_date DATE NOT NULL,            -- 计算日期
    total_count INTEGER,                       -- 总数量
    valid_count INTEGER,                       -- 有效数量
    mean_value DECIMAL(15,8),                  -- 均值
    std_value DECIMAL(15,8),                   -- 标准差
    min_value DECIMAL(15,8),                   -- 最小值
    max_value DECIMAL(15,8),                   -- 最大值
    percentile_5 DECIMAL(15,8),                -- 5%分位数
    percentile_25 DECIMAL(15,8),               -- 25%分位数
    percentile_50 DECIMAL(15,8),               -- 50%分位数
    percentile_75 DECIMAL(15,8),               -- 75%分位数
    percentile_95 DECIMAL(15,8),               -- 95%分位数
    coverage_rate DECIMAL(8,4),                -- 覆盖率
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(factor_name, calculation_date)
);

-- 创建时间序列表
SELECT create_hypertable('factor_statistics', 'calculation_date', if_not_exists => TRUE);

-- 因子标准化值表
CREATE TABLE IF NOT EXISTS factor_standardized_values (
    ts_code VARCHAR(20) NOT NULL,              -- 股票代码
    trade_date DATE NOT NULL,                  -- 交易日期
    factor_name VARCHAR(100) NOT NULL,         -- 因子名称
    raw_value DECIMAL(15,8),                   -- 原始值
    zscore_value DECIMAL(15,8),                -- Z-score标准化值
    percentile_rank DECIMAL(8,4),              -- 分位数排名
    industry_zscore DECIMAL(15,8),             -- 行业内Z-score
    industry_rank DECIMAL(8,4),                -- 行业内排名
    is_valid BOOLEAN DEFAULT TRUE,             -- 是否有效
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (ts_code, trade_date, factor_name)
);

-- 创建时间序列表
SELECT create_hypertable('factor_standardized_values', 'trade_date', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_standardized_values_factor ON factor_standardized_values(factor_name, trade_date);
CREATE INDEX IF NOT EXISTS idx_standardized_values_code ON factor_standardized_values(ts_code, trade_date);

-- ================================
-- 4. 因子计算任务表
-- ================================

-- 因子计算任务表
CREATE TABLE IF NOT EXISTS factor_calculation_tasks (
    task_id UUID PRIMARY KEY,                  -- 任务ID
    task_name VARCHAR(200),                    -- 任务名称
    factor_types TEXT[],                       -- 因子类型列表
    ts_codes TEXT[],                           -- 股票代码列表
    start_date DATE,                           -- 开始日期
    end_date DATE,                             -- 结束日期
    priority INTEGER DEFAULT 0,                -- 优先级
    status VARCHAR(20) DEFAULT 'PENDING',      -- 状态
    progress DECIMAL(5,2) DEFAULT 0.0,         -- 进度百分比
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,                      -- 开始时间
    completed_at TIMESTAMP,                    -- 完成时间
    error_message TEXT                         -- 错误信息
);

-- 因子计算结果表
CREATE TABLE IF NOT EXISTS factor_calculation_results (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    task_id UUID REFERENCES factor_calculation_tasks(task_id),
    ts_code VARCHAR(20) NOT NULL,              -- 股票代码
    factor_type VARCHAR(20) NOT NULL,          -- 因子类型
    status VARCHAR(20) NOT NULL,               -- 计算状态
    execution_time DECIMAL(10,4),              -- 执行时间（秒）
    data_points INTEGER,                       -- 数据点数量
    factors_calculated INTEGER,                -- 计算的因子数量
    error_message TEXT,                        -- 错误信息
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_calculation_tasks_status ON factor_calculation_tasks(status);
CREATE INDEX IF NOT EXISTS idx_calculation_tasks_created ON factor_calculation_tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_calculation_results_task ON factor_calculation_results(task_id);

-- ================================
-- 5. 因子有效性分析表
-- ================================

-- 因子IC分析表
CREATE TABLE IF NOT EXISTS factor_ic_analysis (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    factor_name VARCHAR(100) NOT NULL,         -- 因子名称
    analysis_date DATE NOT NULL,               -- 分析日期
    period_days INTEGER NOT NULL,              -- 预测周期（天）
    ic_value DECIMAL(8,6),                     -- IC值
    ic_ir DECIMAL(8,6),                        -- IC信息比率
    ic_mean DECIMAL(8,6),                      -- IC均值
    ic_std DECIMAL(8,6),                       -- IC标准差
    positive_ic_ratio DECIMAL(8,4),            -- 正IC比例
    abs_ic_mean DECIMAL(8,6),                  -- 绝对IC均值
    t_statistic DECIMAL(8,6),                  -- t统计量
    p_value DECIMAL(8,6),                      -- p值
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(factor_name, analysis_date, period_days)
);

-- 因子分层回测表
CREATE TABLE IF NOT EXISTS factor_layered_backtest (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    factor_name VARCHAR(100) NOT NULL,         -- 因子名称
    backtest_date DATE NOT NULL,               -- 回测日期
    layer_number INTEGER NOT NULL,             -- 分层编号
    stock_count INTEGER,                       -- 股票数量
    avg_return DECIMAL(8,6),                   -- 平均收益率
    cumulative_return DECIMAL(8,6),            -- 累计收益率
    volatility DECIMAL(8,6),                   -- 波动率
    sharpe_ratio DECIMAL(8,6),                 -- 夏普比率
    max_drawdown DECIMAL(8,6),                 -- 最大回撤
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(factor_name, backtest_date, layer_number)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_ic_analysis_factor ON factor_ic_analysis(factor_name, analysis_date);
CREATE INDEX IF NOT EXISTS idx_layered_backtest_factor ON factor_layered_backtest(factor_name, backtest_date);

-- ================================
-- 6. 视图定义
-- ================================

-- 因子概览视图
CREATE OR REPLACE VIEW factor_overview AS
SELECT 
    fm.factor_name,
    fm.factor_type,
    fm.category,
    fm.description,
    fm.enabled,
    fs.calculation_date,
    fs.total_count,
    fs.valid_count,
    fs.coverage_rate,
    fs.mean_value,
    fs.std_value
FROM factor_metadata fm
LEFT JOIN factor_statistics fs ON fm.factor_name = fs.factor_name
WHERE fs.calculation_date = (
    SELECT MAX(calculation_date) 
    FROM factor_statistics fs2 
    WHERE fs2.factor_name = fm.factor_name
);

-- 最新因子值视图（技术面）
CREATE OR REPLACE VIEW latest_technical_factors AS
SELECT *
FROM stock_factors_technical
WHERE trade_date = (SELECT MAX(trade_date) FROM stock_factors_technical);

-- 最新因子值视图（基本面）
CREATE OR REPLACE VIEW latest_fundamental_factors AS
SELECT *
FROM stock_factors_fundamental
WHERE report_date = (SELECT MAX(report_date) FROM stock_factors_fundamental);

-- 最新因子值视图（情绪面）
CREATE OR REPLACE VIEW latest_sentiment_factors AS
SELECT *
FROM stock_factors_sentiment
WHERE trade_date = (SELECT MAX(trade_date) FROM stock_factors_sentiment);

-- 添加注释
COMMENT ON TABLE factor_metadata IS '因子元数据表，存储因子的基本信息和配置';
COMMENT ON TABLE stock_factors_technical IS '技术面因子宽表，存储所有技术指标因子';
COMMENT ON TABLE stock_factors_fundamental IS '基本面因子宽表，存储所有基本面因子';
COMMENT ON TABLE stock_factors_sentiment IS '情绪面因子宽表，存储所有情绪面因子';
COMMENT ON TABLE factor_statistics IS '因子统计信息表，存储因子的统计特征';
COMMENT ON TABLE factor_standardized_values IS '因子标准化值表，存储标准化后的因子值';
COMMENT ON TABLE factor_calculation_tasks IS '因子计算任务表，记录计算任务的执行情况';
COMMENT ON TABLE factor_ic_analysis IS '因子IC分析表，存储因子有效性分析结果';
COMMENT ON TABLE factor_layered_backtest IS '因子分层回测表，存储分层回测结果';
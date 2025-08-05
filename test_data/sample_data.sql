-- StockSchool 验收测试示例数据
-- 此脚本插入验收测试所需的示例数据

-- ==================== 股票基础信息示例数据 ====================
INSERT INTO stock_basic (ts_code, symbol, name, area, industry, market, list_date, is_hs) VALUES
('000001.SZ', '000001', '平安银行', '深圳', '银行', '主板', '1991-04-03', 'S'),
('000002.SZ', '000002', '万科A', '深圳', '房地产开发', '主板', '1991-01-29', 'S'),
('600000.SH', '600000', '浦发银行', '上海', '银行', '主板', '1999-11-10', 'S'),
('600036.SH', '600036', '招商银行', '上海', '银行', '主板', '2002-04-09', 'S'),
('000858.SZ', '000858', '五粮液', '四川', '白酒', '主板', '1998-04-27', 'S')
ON CONFLICT (ts_code) DO NOTHING;

-- ==================== 股票日线数据示例 ====================
-- 生成最近30天的示例数据
DO $$
DECLARE
    stock_code TEXT;
    trade_date DATE;
    base_price DECIMAL(10,3);
    current_price DECIMAL(10,3);
    daily_return DECIMAL(8,4);
    vol_base DECIMAL(20,2);
BEGIN
    -- 为每只股票生成数据
    FOR stock_code IN SELECT ts_code FROM stock_basic LOOP
        -- 设置基础价格
        CASE stock_code
            WHEN '000001.SZ' THEN base_price := 12.50;
            WHEN '000002.SZ' THEN base_price := 18.30;
            WHEN '600000.SH' THEN base_price := 8.90;
            WHEN '600036.SH' THEN base_price := 35.20;
            WHEN '000858.SZ' THEN base_price := 158.80;
            ELSE base_price := 20.00;
        END CASE;
        
        current_price := base_price;
        vol_base := 1000000 + (RANDOM() * 5000000);
        
        -- 生成最近30天的数据
        FOR i IN 0..29 LOOP
            trade_date := CURRENT_DATE - INTERVAL '1 day' * i;
            
            -- 跳过周末
            IF EXTRACT(DOW FROM trade_date) NOT IN (0, 6) THEN
                -- 生成随机价格变动 (-5% 到 +5%)
                daily_return := (RANDOM() - 0.5) * 0.1;
                current_price := current_price * (1 + daily_return);
                
                INSERT INTO stock_daily (
                    ts_code, trade_date, open, high, low, close, pre_close,
                    change, pct_chg, vol, amount
                ) VALUES (
                    stock_code,
                    trade_date,
                    current_price * (0.995 + RANDOM() * 0.01), -- open
                    current_price * (1.000 + RANDOM() * 0.02), -- high
                    current_price * (0.980 + RANDOM() * 0.02), -- low
                    current_price, -- close
                    current_price / (1 + daily_return), -- pre_close
                    current_price - (current_price / (1 + daily_return)), -- change
                    daily_return * 100, -- pct_chg
                    vol_base * (0.5 + RANDOM()), -- vol
                    current_price * vol_base * (0.5 + RANDOM()) -- amount
                ) ON CONFLICT (ts_code, trade_date) DO NOTHING;
            END IF;
        END LOOP;
    END LOOP;
END $$;

-- ==================== 财务数据示例 ====================
INSERT INTO financial_data (
    ts_code, end_date, report_type, revenue, net_profit, total_assets, 
    total_equity, roe, roa, eps, bps
) VALUES
('000001.SZ', '2024-09-30', 'quarter', 45678900000, 8901230000, 567890000000, 234567000000, 0.1234, 0.0567, 1.23, 12.34),
('000002.SZ', '2024-09-30', 'quarter', 23456780000, 4567890000, 345678000000, 123456000000, 0.0987, 0.0432, 0.98, 15.67),
('600000.SH', '2024-09-30', 'quarter', 34567890000, 6789012000, 456789000000, 178901000000, 0.1098, 0.0543, 1.45, 13.21),
('600036.SH', '2024-09-30', 'quarter', 56789012000, 12345678000, 678901000000, 267890000000, 0.1456, 0.0678, 2.34, 18.90),
('000858.SZ', '2024-09-30', 'quarter', 78901234000, 23456789000, 234567000000, 156789000000, 0.1789, 0.0890, 4.56, 45.67)
ON CONFLICT (ts_code, end_date, report_type) DO NOTHING;

-- ==================== 技术因子示例数据 ====================
-- 为最近10天生成技术因子数据
DO $$
DECLARE
    stock_code TEXT;
    trade_date DATE;
    close_price DECIMAL(10,3);
BEGIN
    FOR stock_code IN SELECT ts_code FROM stock_basic LOOP
        FOR i IN 0..9 LOOP
            trade_date := CURRENT_DATE - INTERVAL '1 day' * i;
            
            -- 跳过周末
            IF EXTRACT(DOW FROM trade_date) NOT IN (0, 6) THEN
                -- 获取收盘价
                SELECT close INTO close_price 
                FROM stock_daily 
                WHERE ts_code = stock_code AND stock_daily.trade_date = trade_date;
                
                IF close_price IS NOT NULL THEN
                    INSERT INTO technical_factors (
                        ts_code, trade_date, rsi_14, rsi_6, macd, macd_signal, macd_hist,
                        bb_upper, bb_middle, bb_lower, bb_width,
                        sma_5, sma_10, sma_20, sma_60, ema_12, ema_26,
                        volume_ratio, turnover_rate
                    ) VALUES (
                        stock_code,
                        trade_date,
                        30 + RANDOM() * 40, -- rsi_14 (30-70)
                        25 + RANDOM() * 50, -- rsi_6 (25-75)
                        (RANDOM() - 0.5) * 2, -- macd
                        (RANDOM() - 0.5) * 1.5, -- macd_signal
                        (RANDOM() - 0.5) * 0.5, -- macd_hist
                        close_price * (1.02 + RANDOM() * 0.03), -- bb_upper
                        close_price, -- bb_middle
                        close_price * (0.95 + RANDOM() * 0.03), -- bb_lower
                        0.05 + RANDOM() * 0.05, -- bb_width
                        close_price * (0.98 + RANDOM() * 0.04), -- sma_5
                        close_price * (0.97 + RANDOM() * 0.06), -- sma_10
                        close_price * (0.95 + RANDOM() * 0.10), -- sma_20
                        close_price * (0.90 + RANDOM() * 0.20), -- sma_60
                        close_price * (0.99 + RANDOM() * 0.02), -- ema_12
                        close_price * (0.98 + RANDOM() * 0.04), -- ema_26
                        0.8 + RANDOM() * 0.4, -- volume_ratio
                        1.0 + RANDOM() * 3.0 -- turnover_rate
                    ) ON CONFLICT (ts_code, trade_date) DO NOTHING;
                END IF;
            END IF;
        END LOOP;
    END LOOP;
END $$;

-- ==================== 基本面因子示例数据 ====================
DO $$
DECLARE
    stock_code TEXT;
    trade_date DATE;
    close_price DECIMAL(10,3);
BEGIN
    FOR stock_code IN SELECT ts_code FROM stock_basic LOOP
        FOR i IN 0..9 LOOP
            trade_date := CURRENT_DATE - INTERVAL '1 day' * i;
            
            IF EXTRACT(DOW FROM trade_date) NOT IN (0, 6) THEN
                SELECT close INTO close_price 
                FROM stock_daily 
                WHERE ts_code = stock_code AND stock_daily.trade_date = trade_date;
                
                IF close_price IS NOT NULL THEN
                    INSERT INTO fundamental_factors (
                        ts_code, trade_date, pe_ratio, pb_ratio, ps_ratio, pcf_ratio,
                        market_cap, circulating_cap, total_share, float_share, free_share,
                        revenue_growth, profit_growth, roe, roa, gross_margin, net_margin,
                        debt_ratio, current_ratio, quick_ratio
                    ) VALUES (
                        stock_code,
                        trade_date,
                        8 + RANDOM() * 20, -- pe_ratio
                        0.8 + RANDOM() * 2.0, -- pb_ratio
                        1.0 + RANDOM() * 3.0, -- ps_ratio
                        5.0 + RANDOM() * 10.0, -- pcf_ratio
                        50000000000 + RANDOM() * 100000000000, -- market_cap
                        40000000000 + RANDOM() * 80000000000, -- circulating_cap
                        1000000000 + RANDOM() * 2000000000, -- total_share
                        800000000 + RANDOM() * 1500000000, -- float_share
                        600000000 + RANDOM() * 1200000000, -- free_share
                        -0.1 + RANDOM() * 0.3, -- revenue_growth
                        -0.2 + RANDOM() * 0.5, -- profit_growth
                        0.05 + RANDOM() * 0.15, -- roe
                        0.02 + RANDOM() * 0.08, -- roa
                        0.2 + RANDOM() * 0.3, -- gross_margin
                        0.1 + RANDOM() * 0.2, -- net_margin
                        0.3 + RANDOM() * 0.4, -- debt_ratio
                        1.0 + RANDOM() * 2.0, -- current_ratio
                        0.8 + RANDOM() * 1.5 -- quick_ratio
                    ) ON CONFLICT (ts_code, trade_date) DO NOTHING;
                END IF;
            END IF;
        END LOOP;
    END LOOP;
END $$;

-- ==================== 情绪因子示例数据 ====================
DO $$
DECLARE
    stock_code TEXT;
    trade_date DATE;
BEGIN
    FOR stock_code IN SELECT ts_code FROM stock_basic LOOP
        FOR i IN 0..9 LOOP
            trade_date := CURRENT_DATE - INTERVAL '1 day' * i;
            
            IF EXTRACT(DOW FROM trade_date) NOT IN (0, 6) THEN
                INSERT INTO sentiment_factors (
                    ts_code, trade_date, money_flow_5, money_flow_10, money_flow_20,
                    net_inflow, main_inflow, retail_inflow,
                    attention_score, news_sentiment, social_sentiment,
                    analyst_rating, institution_holding
                ) VALUES (
                    stock_code,
                    trade_date,
                    -50000000 + RANDOM() * 100000000, -- money_flow_5
                    -100000000 + RANDOM() * 200000000, -- money_flow_10
                    -200000000 + RANDOM() * 400000000, -- money_flow_20
                    -30000000 + RANDOM() * 60000000, -- net_inflow
                    -20000000 + RANDOM() * 40000000, -- main_inflow
                    -10000000 + RANDOM() * 20000000, -- retail_inflow
                    0.3 + RANDOM() * 0.4, -- attention_score
                    -0.2 + RANDOM() * 0.4, -- news_sentiment
                    -0.3 + RANDOM() * 0.6, -- social_sentiment
                    2.0 + RANDOM() * 3.0, -- analyst_rating
                    0.1 + RANDOM() * 0.3 -- institution_holding
                ) ON CONFLICT (ts_code, trade_date) DO NOTHING;
            END IF;
        END LOOP;
    END LOOP;
END $$;

-- ==================== AI模型示例数据 ====================
INSERT INTO ai_models (
    model_name, model_type, model_version, training_start_date, training_end_date,
    feature_list, hyperparameters, performance_metrics, model_path, is_active
) VALUES
(
    'StockPredictor_LightGBM_v1',
    'lightgbm',
    '1.0.0',
    '2023-01-01',
    '2024-09-30',
    ARRAY['rsi_14', 'macd', 'pe_ratio', 'pb_ratio', 'roe', 'net_inflow'],
    '{"num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": 0}',
    '{"r2_score": 0.156, "mse": 0.0234, "mae": 0.0987, "sharpe_ratio": 1.23, "max_drawdown": 0.08}',
    '/models/lightgbm_v1_20241201.pkl',
    true
),
(
    'StockPredictor_XGBoost_v1',
    'xgboost',
    '1.0.0',
    '2023-01-01',
    '2024-09-30',
    ARRAY['rsi_14', 'rsi_6', 'macd', 'bb_width', 'pe_ratio', 'pb_ratio', 'roe', 'roa'],
    '{"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8}',
    '{"r2_score": 0.142, "mse": 0.0267, "mae": 0.1023, "sharpe_ratio": 1.18, "max_drawdown": 0.09}',
    '/models/xgboost_v1_20241201.pkl',
    false
)
ON CONFLICT (model_id) DO NOTHING;

-- ==================== 回测结果示例数据 ====================
INSERT INTO backtest_results (
    strategy_name, start_date, end_date, stock_pool,
    factor_weights, strategy_params,
    total_return, annual_return, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio,
    win_rate, profit_loss_ratio, daily_returns, positions, trades, performance_metrics,
    is_optimized, optimization_source
) VALUES
(
    'Multi_Factor_Strategy_v1',
    '2024-01-01',
    '2024-11-30',
    ARRAY['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ'],
    '{"rsi_14": 0.2, "macd": 0.3, "pe_ratio": 0.25, "pb_ratio": 0.15, "roe": 0.1}',
    '{"rebalance_freq": "monthly", "position_limit": 0.2, "stop_loss": 0.05, "take_profit": 0.15}',
    0.1234,
    0.1456,
    0.0789,
    1.23,
    1.45,
    1.85,
    0.58,
    1.34,
    '{"2024-01-01": 0.0023, "2024-01-02": -0.0012, "2024-01-03": 0.0045}',
    '{"000001.SZ": 0.18, "000002.SZ": 0.22, "600000.SH": 0.19, "600036.SH": 0.21, "000858.SZ": 0.20}',
    '[{"date": "2024-01-15", "stock": "000001.SZ", "action": "buy", "quantity": 1000, "price": 12.34}]',
    '{"volatility": 0.156, "beta": 0.89, "alpha": 0.034, "information_ratio": 0.67}',
    false,
    'manual'
),
(
    'AI_Optimized_Strategy_v1',
    '2024-01-01',
    '2024-11-30',
    ARRAY['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ'],
    '{"rsi_14": 0.15, "macd": 0.35, "pe_ratio": 0.20, "pb_ratio": 0.20, "roe": 0.10}',
    '{"rebalance_freq": "monthly", "position_limit": 0.18, "stop_loss": 0.04, "take_profit": 0.18}',
    0.1567,
    0.1823,
    0.0634,
    1.56,
    1.78,
    2.87,
    0.62,
    1.67,
    '{"2024-01-01": 0.0034, "2024-01-02": -0.0008, "2024-01-03": 0.0056}',
    '{"000001.SZ": 0.16, "000002.SZ": 0.24, "600000.SH": 0.18, "600036.SH": 0.22, "000858.SZ": 0.20}',
    '[{"date": "2024-01-15", "stock": "000002.SZ", "action": "buy", "quantity": 1200, "price": 18.45}]',
    '{"volatility": 0.142, "beta": 0.92, "alpha": 0.045, "information_ratio": 0.78}',
    true,
    'ai_optimized'
)
ON CONFLICT (backtest_id) DO NOTHING;

-- ==================== 外接AI分析示例数据 ====================
INSERT INTO external_ai_analysis (
    ts_code, analysis_date, analysis_type, request_data, response_data,
    technical_analysis, fundamental_analysis, sentiment_analysis,
    investment_advice, risk_assessment, price_prediction,
    confidence_score, api_response_time, api_success
) VALUES
(
    '000001.SZ',
    CURRENT_DATE,
    'deep_analysis',
    '{"stock_code": "000001.SZ", "analysis_date": "2024-12-01", "analysis_types": ["technical", "fundamental", "sentiment"]}',
    '{"status": "success", "analysis_id": "ai_20241201_000001", "processing_time": 23.45}',
    '{"rsi_signal": "neutral", "macd_signal": "bullish", "trend": "upward", "support_level": 12.10, "resistance_level": 13.20}',
    '{"valuation": "reasonable", "pe_analysis": "below_average", "growth_potential": "moderate", "financial_health": "good"}',
    '{"market_sentiment": "positive", "news_sentiment": "neutral", "institutional_flow": "inflow", "retail_sentiment": "optimistic"}',
    '建议适度买入。技术面显示上升趋势，基本面估值合理，市场情绪偏正面。建议分批建仓，设置止损位于12.00元。',
    '{"risk_level": "medium", "volatility_risk": 0.15, "liquidity_risk": 0.05, "market_risk": 0.12, "specific_risk": 0.08}',
    '{"target_price": 13.50, "price_range": {"low": 12.80, "high": 14.20}, "time_horizon": "3_months", "probability": 0.72}',
    0.78,
    23.45,
    true
),
(
    '000858.SZ',
    CURRENT_DATE,
    'deep_analysis',
    '{"stock_code": "000858.SZ", "analysis_date": "2024-12-01", "analysis_types": ["technical", "fundamental", "sentiment"]}',
    '{"status": "success", "analysis_id": "ai_20241201_000858", "processing_time": 28.12}',
    '{"rsi_signal": "overbought", "macd_signal": "bearish", "trend": "sideways", "support_level": 155.00, "resistance_level": 165.00}',
    '{"valuation": "expensive", "pe_analysis": "above_average", "growth_potential": "stable", "financial_health": "excellent"}',
    '{"market_sentiment": "mixed", "news_sentiment": "positive", "institutional_flow": "neutral", "retail_sentiment": "cautious"}',
    '建议观望。技术面显示超买状态，估值偏高，但基本面优秀。建议等待回调至155元附近再考虑买入。',
    '{"risk_level": "medium_high", "volatility_risk": 0.18, "liquidity_risk": 0.03, "market_risk": 0.14, "specific_risk": 0.06}',
    '{"target_price": 162.00, "price_range": {"low": 152.00, "high": 168.00}, "time_horizon": "3_months", "probability": 0.65}',
    0.82,
    28.12,
    true
)
ON CONFLICT (analysis_id) DO NOTHING;

-- ==================== 输出示例数据统计 ====================
DO $$
DECLARE
    stock_count INTEGER;
    daily_count INTEGER;
    factor_count INTEGER;
    ai_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO stock_count FROM stock_basic;
    SELECT COUNT(*) INTO daily_count FROM stock_daily;
    SELECT COUNT(*) INTO factor_count FROM technical_factors;
    SELECT COUNT(*) INTO ai_count FROM external_ai_analysis;
    
    RAISE NOTICE '=== 示例数据插入完成 ===';
    RAISE NOTICE '股票基础信息: % 条', stock_count;
    RAISE NOTICE '日线数据: % 条', daily_count;
    RAISE NOTICE '技术因子: % 条', factor_count;
    RAISE NOTICE 'AI分析结果: % 条', ai_count;
    RAISE NOTICE '数据时间范围: % 到 %', CURRENT_DATE - INTERVAL '29 days', CURRENT_DATE;
END $$;
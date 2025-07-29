-- 最新股票信息视图
CREATE VIEW IF NOT EXISTS latest_stock_info AS
SELECT 
    sb.ts_code, 
    sb.name, 
    sb.industry, 
    sb.area, 
    sb.market, 
    sb.list_date, 
    db.close, 
    db.pct_chg, 
    db.volume, 
    db.amount, 
    fi.pe, 
    fi.pb, 
    fi.ps, 
    fi.dy
FROM 
    stock_basic sb
LEFT JOIN 
    daily_basic db ON sb.ts_code = db.ts_code
LEFT JOIN 
    financial_indicator fi ON sb.ts_code = fi.ts_code
WHERE 
    db.trade_date = (SELECT MAX(trade_date) FROM trade_calendar WHERE is_open = 1)
    AND fi.end_date = (SELECT MAX(end_date) FROM financial_indicator);

-- 因子评分汇总视图
CREATE VIEW IF NOT EXISTS factor_score_summary AS
SELECT 
    fs.ts_code, 
    sb.name, 
    fs.trade_date, 
    fs.technical_score, 
    fs.fundamental_score, 
    fs.sentiment_score, 
    fs.composite_score, 
    fs.rank_percentile,
    CASE 
        WHEN fs.rank_percentile <= 0.2 THEN 'A'
        WHEN fs.rank_percentile <= 0.4 THEN 'B'
        WHEN fs.rank_percentile <= 0.6 THEN 'C'
        WHEN fs.rank_percentile <= 0.8 THEN 'D'
        ELSE 'E'
    END AS score_grade
FROM 
    factor_scores fs
LEFT JOIN 
    stock_basic sb ON fs.ts_code = sb.ts_code
WHERE 
    fs.trade_date = (SELECT MAX(trade_date) FROM trade_calendar WHERE is_open = 1);

-- 数据质量监控视图
CREATE VIEW IF NOT EXISTS data_quality_overview AS
SELECT 
    table_name, 
    COUNT(*) AS total_records, 
    SUM(CASE WHEN is_valid = TRUE THEN 1 ELSE 0 END) AS valid_records, 
    SUM(CASE WHEN is_valid = FALSE THEN 1 ELSE 0 END) AS invalid_records, 
    ROUND(SUM(CASE WHEN is_valid = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS valid_percentage,
    MAX(last_updated) AS last_updated
FROM 
    data_quality_monitor
GROUP BY 
    table_name;

-- 创建视图索引（PostgreSQL不直接支持视图索引，此处为注释说明）
-- 注意：对于频繁查询的大型视图，建议创建物化视图并添加索引
-- CREATE MATERIALIZED VIEW mv_latest_stock_info AS SELECT * FROM latest_stock_info;
-- CREATE UNIQUE INDEX idx_mv_latest_stock_info_ts_code ON mv_latest_stock_info(ts_code);
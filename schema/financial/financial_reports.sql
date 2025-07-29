-- 财务报告表
CREATE TABLE IF NOT EXISTS financial_reports (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,     -- 股票代码
    ann_date DATE NOT NULL,           -- 公告日期
    f_ann_date DATE,                  -- 实际公告日期
    end_date DATE NOT NULL,           -- 报告期
    report_type VARCHAR(20),          -- 报告类型
    comp_type VARCHAR(20),            -- 公司类型
    is_audit INTEGER,                 -- 是否审计
    audit_agency VARCHAR(100),        -- 审计机构
    audit_opinion VARCHAR(100),       -- 审计意见
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, end_date, report_type)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_financial_reports_ts_code ON financial_reports(ts_code);
CREATE INDEX IF NOT EXISTS idx_financial_reports_end_date ON financial_reports(end_date);
CREATE INDEX IF NOT EXISTS idx_financial_reports_ann_date ON financial_reports(ann_date);
CREATE INDEX IF NOT EXISTS idx_financial_reports_composite ON financial_reports(ts_code, end_date);
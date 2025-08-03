-- 数据质量监控相关表结构

-- 数据质量指标表
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    report_date DATE NOT NULL,
    data_source VARCHAR(50) NOT NULL,  -- 'tushare', 'akshare'
    data_type VARCHAR(50) NOT NULL,    -- 'daily', 'financial', 'sentiment', 'attention', 'popularity'
    
    -- 质量指标分数 (0-100)
    completeness_score DECIMAL(5,2) NOT NULL,
    accuracy_score DECIMAL(5,2) NOT NULL,
    timeliness_score DECIMAL(5,2) NOT NULL,
    consistency_score DECIMAL(5,2) NOT NULL,
    overall_score DECIMAL(5,2) NOT NULL,
    
    -- 质量等级
    quality_level VARCHAR(20) NOT NULL, -- 'excellent', 'good', 'fair', 'poor', 'critical'
    
    -- 问题和建议
    issues TEXT,
    recommendations TEXT,
    
    -- 统计信息
    record_count INTEGER DEFAULT 0,
    null_count INTEGER DEFAULT 0,
    duplicate_count INTEGER DEFAULT 0,
    
    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 唯一约束
    UNIQUE(data_source, data_type, report_date)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_date ON data_quality_metrics(report_date);
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_source ON data_quality_metrics(data_source, data_type);
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_score ON data_quality_metrics(overall_score);

-- 数据异常记录表
CREATE TABLE IF NOT EXISTS data_anomalies (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    anomaly_date DATE NOT NULL,
    data_source VARCHAR(50) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    
    -- 异常类型
    anomaly_type VARCHAR(50) NOT NULL, -- 'logical_error', 'statistical_outlier', 'missing_value', 'format_error'
    
    -- 异常详情
    column_name VARCHAR(100),
    original_value TEXT,
    expected_value TEXT,
    anomaly_score DECIMAL(8,4), -- 异常程度分数
    
    -- 处理状态
    status VARCHAR(20) DEFAULT 'detected', -- 'detected', 'fixed', 'ignored'
    fix_method VARCHAR(100),
    fixed_value TEXT,
    
    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_data_anomalies_stock_date ON data_anomalies(ts_code, anomaly_date);
CREATE INDEX IF NOT EXISTS idx_data_anomalies_type ON data_anomalies(anomaly_type);
CREATE INDEX IF NOT EXISTS idx_data_anomalies_status ON data_anomalies(status);

-- 数据质量规则表
CREATE TABLE IF NOT EXISTS data_quality_rules (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL UNIQUE,
    data_source VARCHAR(50) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    
    -- 规则配置
    rule_type VARCHAR(50) NOT NULL, -- 'range_check', 'logic_check', 'format_check', 'completeness_check'
    rule_config JSONB NOT NULL,     -- 规则的具体配置参数
    
    -- 规则状态
    is_active BOOLEAN DEFAULT true,
    severity VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    
    -- 描述
    description TEXT,
    
    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_data_quality_rules_source ON data_quality_rules(data_source, data_type);
CREATE INDEX IF NOT EXISTS idx_data_quality_rules_active ON data_quality_rules(is_active);

-- 数据修复记录表
CREATE TABLE IF NOT EXISTS data_repair_log (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    repair_date DATE NOT NULL,
    data_source VARCHAR(50) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    
    -- 修复详情
    column_name VARCHAR(100) NOT NULL,
    original_value TEXT,
    repaired_value TEXT,
    repair_method VARCHAR(100) NOT NULL, -- 'forward_fill', 'industry_mean', 'interpolation', 'manual'
    
    -- 修复质量
    confidence_score DECIMAL(5,2), -- 修复的置信度 (0-100)
    
    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_data_repair_log_stock_date ON data_repair_log(ts_code, repair_date);
CREATE INDEX IF NOT EXISTS idx_data_repair_log_method ON data_repair_log(repair_method);

-- 插入默认的数据质量规则
INSERT INTO data_quality_rules (rule_name, data_source, data_type, rule_type, rule_config, description) VALUES
-- 股票日线数据规则
('daily_price_logic_check', 'tushare', 'daily', 'logic_check', 
 '{"checks": ["high >= low", "high >= open", "high >= close", "low <= open", "low <= close", "volume >= 0"]}',
 '检查日线数据的价格逻辑关系'),

('daily_price_range_check', 'tushare', 'daily', 'range_check',
 '{"columns": {"open": {"min": 0.01, "max": 10000}, "high": {"min": 0.01, "max": 10000}, "low": {"min": 0.01, "max": 10000}, "close": {"min": 0.01, "max": 10000}}}',
 '检查日线数据的价格范围'),

('daily_change_limit_check', 'tushare', 'daily', 'range_check',
 '{"columns": {"pct_chg": {"min": -20, "max": 20}}}',
 '检查日线数据的涨跌幅限制'),

-- 财务数据规则
('financial_logic_check', 'tushare', 'financial', 'logic_check',
 '{"checks": ["total_assets >= total_liab", "revenue >= 0", "total_share > 0"]}',
 '检查财务数据的逻辑关系'),

('financial_range_check', 'tushare', 'financial', 'range_check',
 '{"columns": {"revenue": {"min": 0}, "total_assets": {"min": 0}, "total_share": {"min": 0}}}',
 '检查财务数据的数值范围'),

-- 情绪数据规则
('sentiment_range_check', 'akshare', 'sentiment', 'range_check',
 '{"columns": {"sentiment_score": {"min": -1, "max": 1}, "positive_count": {"min": 0}, "negative_count": {"min": 0}, "neutral_count": {"min": 0}}}',
 '检查情绪数据的数值范围'),

-- 关注度数据规则
('attention_range_check', 'akshare', 'attention', 'range_check',
 '{"columns": {"attention_score": {"min": 0}, "search_volume": {"min": 0}, "discussion_volume": {"min": 0}}}',
 '检查关注度数据的数值范围'),

-- 人气榜数据规则
('popularity_range_check', 'akshare', 'popularity', 'range_check',
 '{"columns": {"rank_position": {"min": 1, "max": 1000}, "popularity_score": {"min": 0}}}',
 '检查人气榜数据的数值范围')

ON CONFLICT (rule_name) DO NOTHING;

-- 创建数据质量监控的视图
CREATE OR REPLACE VIEW v_data_quality_summary AS
SELECT 
    data_source,
    data_type,
    AVG(overall_score) as avg_score,
    MIN(overall_score) as min_score,
    MAX(overall_score) as max_score,
    COUNT(*) as report_count,
    COUNT(CASE WHEN quality_level = 'critical' THEN 1 END) as critical_count,
    COUNT(CASE WHEN quality_level = 'poor' THEN 1 END) as poor_count,
    COUNT(CASE WHEN quality_level = 'fair' THEN 1 END) as fair_count,
    COUNT(CASE WHEN quality_level = 'good' THEN 1 END) as good_count,
    COUNT(CASE WHEN quality_level = 'excellent' THEN 1 END) as excellent_count,
    MAX(report_date) as latest_report_date
FROM data_quality_metrics
WHERE report_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY data_source, data_type
ORDER BY data_source, data_type;

-- 创建异常统计视图
CREATE OR REPLACE VIEW v_anomaly_summary AS
SELECT 
    data_source,
    data_type,
    anomaly_type,
    COUNT(*) as anomaly_count,
    COUNT(CASE WHEN status = 'fixed' THEN 1 END) as fixed_count,
    COUNT(CASE WHEN status = 'detected' THEN 1 END) as pending_count,
    AVG(anomaly_score) as avg_anomaly_score,
    MAX(anomaly_date) as latest_anomaly_date
FROM data_anomalies
WHERE anomaly_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY data_source, data_type, anomaly_type
ORDER BY data_source, data_type, anomaly_type;

-- 创建质量趋势视图
CREATE OR REPLACE VIEW v_quality_trend AS
SELECT 
    report_date,
    data_source,
    data_type,
    overall_score,
    quality_level,
    LAG(overall_score) OVER (PARTITION BY data_source, data_type ORDER BY report_date) as prev_score,
    overall_score - LAG(overall_score) OVER (PARTITION BY data_source, data_type ORDER BY report_date) as score_change
FROM data_quality_metrics
WHERE report_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY data_source, data_type, report_date;

-- 添加注释
COMMENT ON TABLE data_quality_metrics IS '数据质量指标表，记录各数据源的质量评估结果';
COMMENT ON TABLE data_anomalies IS '数据异常记录表，记录检测到的数据异常';
COMMENT ON TABLE data_quality_rules IS '数据质量规则表，定义数据质量检查规则';
COMMENT ON TABLE data_repair_log IS '数据修复记录表，记录数据修复的历史';

COMMENT ON COLUMN data_quality_metrics.completeness_score IS '数据完整性分数 (0-100)';
COMMENT ON COLUMN data_quality_metrics.accuracy_score IS '数据准确性分数 (0-100)';
COMMENT ON COLUMN data_quality_metrics.timeliness_score IS '数据时效性分数 (0-100)';
COMMENT ON COLUMN data_quality_metrics.consistency_score IS '数据一致性分数 (0-100)';
COMMENT ON COLUMN data_quality_metrics.overall_score IS '综合质量分数 (0-100)';

COMMENT ON VIEW v_data_quality_summary IS '数据质量汇总视图，显示各数据源的质量统计';
COMMENT ON VIEW v_anomaly_summary IS '异常统计视图，显示各类异常的统计信息';
COMMENT ON VIEW v_quality_trend IS '质量趋势视图，显示质量分数的变化趋势';
-
- 同步任务状态表
CREATE TABLE IF NOT EXISTS sync_task_status (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    task_id VARCHAR(200) NOT NULL UNIQUE,
    data_source VARCHAR(50) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    target_date DATE NOT NULL,
    
    -- 任务状态
    status VARCHAR(20) NOT NULL, -- 'pending', 'running', 'completed', 'failed', 'cancelled', 'skipped'
    priority INTEGER NOT NULL,   -- 1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL
    
    -- 重试信息
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- 执行时间
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds INTEGER GENERATED ALWAYS AS (
        CASE 
            WHEN start_time IS NOT NULL AND end_time IS NOT NULL 
            THEN EXTRACT(EPOCH FROM (end_time - start_time))::INTEGER
            ELSE NULL
        END
    ) STORED,
    
    -- 错误信息
    error_message TEXT,
    
    -- 资源需求
    cpu_requirement DECIMAL(4,2) DEFAULT 1.0,
    memory_requirement INTEGER DEFAULT 512,
    api_calls_required INTEGER DEFAULT 10,
    
    -- 依赖关系（JSON格式存储）
    dependencies JSONB DEFAULT '[]',
    
    -- 执行结果（JSON格式存储）
    result JSONB,
    
    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_sync_task_status_date ON sync_task_status(target_date);
CREATE INDEX IF NOT EXISTS idx_sync_task_status_source ON sync_task_status(data_source, data_type);
CREATE INDEX IF NOT EXISTS idx_sync_task_status_status ON sync_task_status(status);
CREATE INDEX IF NOT EXISTS idx_sync_task_status_priority ON sync_task_status(priority);
CREATE INDEX IF NOT EXISTS idx_sync_task_status_created ON sync_task_status(created_at);

-- 资源使用记录表
CREATE TABLE IF NOT EXISTS resource_usage_log (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    task_id VARCHAR(200) NOT NULL,
    
    -- 资源使用情况
    cpu_cores_used DECIMAL(4,2) NOT NULL,
    memory_mb_used INTEGER NOT NULL,
    api_calls_made INTEGER NOT NULL,
    
    -- 使用时间
    allocation_time TIMESTAMP NOT NULL,
    release_time TIMESTAMP,
    
    -- 资源池状态快照
    total_cpu_available DECIMAL(4,2),
    total_memory_available INTEGER,
    concurrent_tasks INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (task_id) REFERENCES sync_task_status(task_id)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_resource_usage_log_task ON resource_usage_log(task_id);
CREATE INDEX IF NOT EXISTS idx_resource_usage_log_time ON resource_usage_log(allocation_time);

-- 任务依赖关系表
CREATE TABLE IF NOT EXISTS task_dependencies (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    task_id VARCHAR(200) NOT NULL,
    dependency_task_id VARCHAR(200) NOT NULL,
    dependency_type VARCHAR(50) DEFAULT 'hard', -- 'hard', 'soft'
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (task_id) REFERENCES sync_task_status(task_id),
    FOREIGN KEY (dependency_task_id) REFERENCES sync_task_status(task_id),
    UNIQUE(task_id, dependency_task_id)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_task_dependencies_task ON task_dependencies(task_id);
CREATE INDEX IF NOT EXISTS idx_task_dependencies_dep ON task_dependencies(dependency_task_id);

-- 调度器状态表
CREATE TABLE IF NOT EXISTS scheduler_status (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    scheduler_instance VARCHAR(100) NOT NULL,
    
    -- 调度器状态
    is_running BOOLEAN NOT NULL,
    max_workers INTEGER NOT NULL,
    current_workers INTEGER NOT NULL,
    
    -- 资源池状态
    max_cpu_cores DECIMAL(4,2) NOT NULL,
    used_cpu_cores DECIMAL(4,2) NOT NULL,
    max_memory_mb INTEGER NOT NULL,
    used_memory_mb INTEGER NOT NULL,
    
    -- API限制状态
    api_limits JSONB, -- 存储各数据源的API限制信息
    
    -- 统计信息
    total_tasks INTEGER DEFAULT 0,
    pending_tasks INTEGER DEFAULT 0,
    running_tasks INTEGER DEFAULT 0,
    completed_tasks INTEGER DEFAULT 0,
    failed_tasks INTEGER DEFAULT 0,
    
    -- 时间戳
    last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(scheduler_instance)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_scheduler_status_instance ON scheduler_status(scheduler_instance);
CREATE INDEX IF NOT EXISTS idx_scheduler_status_heartbeat ON scheduler_status(last_heartbeat);

-- 创建任务执行统计视图
CREATE OR REPLACE VIEW v_task_execution_stats AS
SELECT 
    data_source,
    data_type,
    status,
    COUNT(*) as task_count,
    AVG(duration_seconds) as avg_duration_seconds,
    MIN(duration_seconds) as min_duration_seconds,
    MAX(duration_seconds) as max_duration_seconds,
    AVG(retry_count) as avg_retry_count,
    COUNT(CASE WHEN retry_count > 0 THEN 1 END) as tasks_with_retries,
    DATE(created_at) as execution_date
FROM sync_task_status
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY data_source, data_type, status, DATE(created_at)
ORDER BY execution_date DESC, data_source, data_type;

-- 创建资源使用统计视图
CREATE OR REPLACE VIEW v_resource_usage_stats AS
SELECT 
    DATE(allocation_time) as usage_date,
    EXTRACT(HOUR FROM allocation_time) as usage_hour,
    AVG(cpu_cores_used) as avg_cpu_usage,
    MAX(cpu_cores_used) as peak_cpu_usage,
    AVG(memory_mb_used) as avg_memory_usage,
    MAX(memory_mb_used) as peak_memory_usage,
    SUM(api_calls_made) as total_api_calls,
    COUNT(DISTINCT task_id) as concurrent_tasks
FROM resource_usage_log
WHERE allocation_time >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY DATE(allocation_time), EXTRACT(HOUR FROM allocation_time)
ORDER BY usage_date DESC, usage_hour;

-- 创建任务依赖链视图
CREATE OR REPLACE VIEW v_task_dependency_chain AS
WITH RECURSIVE dependency_chain AS (
    -- 基础情况：没有依赖的任务
    SELECT 
        task_id,
        task_id as root_task,
        0 as depth,
        ARRAY[task_id] as chain_path
    FROM sync_task_status
    WHERE task_id NOT IN (SELECT task_id FROM task_dependencies)
    
    UNION ALL
    
    -- 递归情况：有依赖的任务
    SELECT 
        td.task_id,
        dc.root_task,
        dc.depth + 1,
        dc.chain_path || td.task_id
    FROM task_dependencies td
    JOIN dependency_chain dc ON td.dependency_task_id = dc.task_id
    WHERE NOT (td.task_id = ANY(dc.chain_path)) -- 防止循环依赖
)
SELECT 
    root_task,
    task_id,
    depth,
    chain_path,
    array_length(chain_path, 1) as chain_length
FROM dependency_chain
ORDER BY root_task, depth;

-- 添加表注释
COMMENT ON TABLE sync_task_status IS '同步任务状态表，记录所有同步任务的执行状态';
COMMENT ON TABLE resource_usage_log IS '资源使用记录表，记录任务的资源分配和使用情况';
COMMENT ON TABLE task_dependencies IS '任务依赖关系表，定义任务间的依赖关系';
COMMENT ON TABLE scheduler_status IS '调度器状态表，记录调度器实例的运行状态';

COMMENT ON VIEW v_task_execution_stats IS '任务执行统计视图，按数据源和类型统计任务执行情况';
COMMENT ON VIEW v_resource_usage_stats IS '资源使用统计视图，按时间统计资源使用情况';
COMMENT ON VIEW v_task_dependency_chain IS '任务依赖链视图，显示完整的任务依赖关系链';
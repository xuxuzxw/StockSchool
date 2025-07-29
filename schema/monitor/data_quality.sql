-- 数据质量监控表
CREATE TABLE IF NOT EXISTS data_quality_monitor (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,  -- 表名
    record_count INT,                 -- 记录总数
    valid_count INT,                  -- 有效记录数
    invalid_count INT,                -- 无效记录数
    check_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 检查时间
    error_details JSONB,              -- 错误详情
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(table_name, check_time)
);

-- 任务调度表
CREATE TABLE IF NOT EXISTS task_schedule (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    task_name VARCHAR(100) NOT NULL,  -- 任务名称
    task_type VARCHAR(50),            -- 任务类型
    cron_expression VARCHAR(50),      -- Cron表达式
    status VARCHAR(20) DEFAULT 'active', -- 任务状态
    priority INT DEFAULT 5,           -- 任务优先级
    last_execution_time TIMESTAMP,    -- 上次执行时间
    next_execution_time TIMESTAMP,    -- 下次执行时间
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(task_name)
);

-- 任务执行日志表
CREATE TABLE IF NOT EXISTS task_execution_log (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    task_id UUID NOT NULL,            -- 任务ID
    start_time TIMESTAMP NOT NULL,    -- 开始时间
    end_time TIMESTAMP,               -- 结束时间
    status VARCHAR(20) NOT NULL,      -- 执行状态
    execution_duration INT,           -- 执行时长(毫秒)
    error_message TEXT,               -- 错误信息
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES task_schedule(id)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_data_quality_table_name ON data_quality_monitor(table_name);
CREATE INDEX IF NOT EXISTS idx_task_schedule_status ON task_schedule(status);
CREATE INDEX IF NOT EXISTS idx_task_execution_task_id ON task_execution_log(task_id);
CREATE INDEX IF NOT EXISTS idx_task_execution_status ON task_execution_log(status);
---
inclusion: fileMatch
fileMatchPattern: 'database_schema.sql|**/database/**|**/data/**'
---

# 数据库设计和操作指南

## TimescaleDB配置要求

### 必须启用的扩展
```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### 时序表创建规范
```sql
-- 创建时序超表
SELECT create_hypertable('stock_daily', 'trade_date', if_not_exists => TRUE);

-- 启用压缩
ALTER TABLE stock_daily SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'ts_code, trade_date DESC'
);

-- 设置压缩策略
SELECT add_compression_policy('stock_daily', INTERVAL '7 days');
```

## 索引策略

### 复合索引规范
```sql
-- 必须按 stock_code + end_date 建立复合索引
CREATE INDEX idx_stock_daily_composite 
ON stock_daily (ts_code, trade_date DESC);

-- 行业分类复合索引
CREATE INDEX idx_industry_composite 
ON sw_industry_classify (ts_code, industry_level);

-- 高频访问表的部分索引
CREATE INDEX idx_daily_basic_latest 
ON daily_basic (end_date) 
WHERE end_date >= CURRENT_DATE - INTERVAL '30 days';
```

### 性能优化索引
```sql
-- 因子查询优化索引
CREATE INDEX idx_stock_factors_lookup 
ON stock_technical_factors (ts_code, trade_date DESC);

-- 财务数据查询索引
CREATE INDEX idx_financial_reports_period 
ON financial_reports (ts_code, end_date DESC, report_type);
```

## 数据完整性约束

### 主键和唯一约束
```sql
-- 股票基本信息主键
ALTER TABLE stock_basic ADD CONSTRAINT pk_stock_basic PRIMARY KEY (ts_code);

-- 日线数据唯一约束
ALTER TABLE stock_daily ADD CONSTRAINT uk_stock_daily UNIQUE (ts_code, trade_date);

-- 财务数据唯一约束
ALTER TABLE financial_reports ADD CONSTRAINT uk_financial_reports 
UNIQUE (ts_code, end_date, report_type);
```

### 外键约束
```sql
-- 因子数据外键约束
ALTER TABLE stock_technical_factors 
ADD CONSTRAINT fk_factors_stock 
FOREIGN KEY (ts_code) REFERENCES stock_basic(ts_code);

-- 行业分类外键约束
ALTER TABLE sw_industry_history 
ADD CONSTRAINT fk_industry_stock 
FOREIGN KEY (ts_code) REFERENCES stock_basic(ts_code);
```

## 数据类型规范

### 价格和金额字段
```sql
-- 价格字段使用DECIMAL(10,3) - 支持到分
open DECIMAL(10,3),
high DECIMAL(10,3),
low DECIMAL(10,3),
close DECIMAL(10,3),

-- 大金额字段使用DECIMAL(20,4) - 支持到万分位
total_revenue DECIMAL(20,4),
total_assets DECIMAL(20,4),

-- 比率字段使用DECIMAL(8,4) - 支持到万分位
pct_chg DECIMAL(8,4),
roe DECIMAL(8,4)
```

### 日期时间字段
```sql
-- 交易日期使用DATE类型
trade_date DATE NOT NULL,
end_date DATE NOT NULL,

-- 创建时间使用TIMESTAMP WITH TIME ZONE
created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
```

## 数据同步策略

### 增量更新模式
```sql
-- 使用UPSERT模式进行数据同步
INSERT INTO stock_daily (ts_code, trade_date, open, high, low, close, vol, amount)
VALUES (:ts_code, :trade_date, :open, :high, :low, :close, :vol, :amount)
ON CONFLICT (ts_code, trade_date) 
DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    vol = EXCLUDED.vol,
    amount = EXCLUDED.amount,
    updated_at = CURRENT_TIMESTAMP;
```

### 批量操作优化
```sql
-- 使用批量插入提高性能
COPY stock_daily (ts_code, trade_date, open, high, low, close, vol, amount)
FROM STDIN WITH (FORMAT CSV, HEADER);

-- 或使用VALUES子句批量插入
INSERT INTO stock_daily (ts_code, trade_date, open, high, low, close)
VALUES 
    ('000001.SZ', '2024-01-01', 10.0, 10.5, 9.8, 10.2),
    ('000001.SZ', '2024-01-02', 10.2, 10.8, 10.0, 10.6);
```

## 数据质量检查

### 数据完整性检查
```sql
-- 检查缺失的交易日数据
SELECT tc.cal_date, COUNT(sd.ts_code) as stock_count
FROM trade_calendar tc
LEFT JOIN stock_daily sd ON tc.cal_date = sd.trade_date
WHERE tc.is_open = 1 
  AND tc.cal_date >= '2024-01-01'
GROUP BY tc.cal_date
HAVING COUNT(sd.ts_code) < (SELECT COUNT(*) FROM stock_basic WHERE list_status = 'L');
```

### 数据异常检查
```sql
-- 检查价格异常数据
SELECT ts_code, trade_date, open, high, low, close
FROM stock_daily
WHERE high < low 
   OR open < 0 
   OR close < 0
   OR high/low > 1.2  -- 单日涨跌幅超过20%
ORDER BY trade_date DESC;
```

## 分区管理

### 时间分区策略
```sql
-- 按月分区存储历史数据
SELECT create_hypertable('stock_daily', 'trade_date', 
    chunk_time_interval => INTERVAL '1 month');

-- 设置数据保留策略
SELECT add_retention_policy('stock_daily', INTERVAL '5 years');
```

### 分区维护
```sql
-- 查看分区信息
SELECT * FROM timescaledb_information.chunks 
WHERE hypertable_name = 'stock_daily'
ORDER BY range_start DESC;

-- 手动压缩旧分区
SELECT compress_chunk(chunk_name) 
FROM timescaledb_information.chunks 
WHERE hypertable_name = 'stock_daily' 
  AND range_end < NOW() - INTERVAL '7 days';
```

## 备份和恢复

### 数据备份策略
```bash
# 全量备份
pg_dump -h localhost -U stockschool -d stockschool > backup_$(date +%Y%m%d).sql

# 增量备份（WAL归档）
archive_command = 'cp %p /backup/wal_archive/%f'
```

### 数据恢复
```bash
# 从备份恢复
psql -h localhost -U stockschool -d stockschool < backup_20240101.sql

# 时间点恢复
pg_basebackup -h localhost -U stockschool -D /backup/base_backup
```

## 监控和维护

### 性能监控查询
```sql
-- 查看慢查询
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
WHERE mean_time > 1000  -- 超过1秒的查询
ORDER BY mean_time DESC;

-- 查看表大小
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### 索引维护
```sql
-- 查看索引使用情况
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0  -- 未使用的索引
ORDER BY schemaname, tablename;

-- 重建索引
REINDEX INDEX CONCURRENTLY idx_stock_daily_composite;
```
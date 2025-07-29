-- 创建PostgreSQL扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 设置时区
SET TIME ZONE 'Asia/Shanghai';

-- 创建超表通用函数
CREATE OR REPLACE FUNCTION create_hypertable_if_not_exists(
    table_name TEXT,
    time_column_name TEXT
) RETURNS VOID AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM _timescaledb_catalog.hypertable
        WHERE table_name = create_hypertable_if_not_exists.table_name
    ) THEN
        PERFORM create_hypertable(table_name, time_column_name);
    END IF;
END;
$$ LANGUAGE plpgsql;
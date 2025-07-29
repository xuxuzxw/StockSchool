-- 因子库表
CREATE TABLE IF NOT EXISTS factor_library (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    factor_name VARCHAR(50) NOT NULL,  -- 因子名称
    factor_desc TEXT,                  -- 因子描述
    factor_type VARCHAR(20),           -- 因子类型（技术面、基本面、情绪面等）
    formula TEXT,                      -- 因子计算公式
    params JSONB,                      -- 因子参数
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(factor_name)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_factor_library_name ON factor_library(factor_name);
CREATE INDEX IF NOT EXISTS idx_factor_library_type ON factor_library(factor_type);
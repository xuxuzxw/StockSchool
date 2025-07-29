-- 宏观经济数据表

-- GDP数据表
CREATE TABLE IF NOT EXISTS macro_gdp (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    quarter VARCHAR(10) NOT NULL,     -- 季度
    gdp DECIMAL(15,2),                -- 国内生产总值(亿元)
    gdp_yoy DECIMAL(8,4),             -- GDP同比增长率
    pi DECIMAL(15,2),                 -- 第一产业(亿元)
    pi_yoy DECIMAL(8,4),              -- 第一产业同比增长率
    si DECIMAL(15,2),                 -- 第二产业(亿元)
    si_yoy DECIMAL(8,4),              -- 第二产业同比增长率
    ti DECIMAL(15,2),                 -- 第三产业(亿元)
    ti_yoy DECIMAL(8,4),              -- 第三产业同比增长率
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(quarter)
);

-- CPI数据表
CREATE TABLE IF NOT EXISTS macro_cpi (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    month VARCHAR(10) NOT NULL,       -- 月份
    nt_val DECIMAL(8,4),              -- 全国当月值
    nt_yoy DECIMAL(8,4),              -- 全国同比增长
    nt_mom DECIMAL(8,4),              -- 全国环比增长
    nt_accu DECIMAL(8,4),             -- 全国累计值
    town_val DECIMAL(8,4),            -- 城市当月值
    town_yoy DECIMAL(8,4),            -- 城市同比增长
    town_mom DECIMAL(8,4),            -- 城市环比增长
    town_accu DECIMAL(8,4),           -- 城市累计值
    cnt_val DECIMAL(8,4),             -- 农村当月值
    cnt_yoy DECIMAL(8,4),             -- 农村同比增长
    cnt_mom DECIMAL(8,4),             -- 农村环比增长
    cnt_accu DECIMAL(8,4),            -- 农村累计值
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(month)
);

-- PMI数据表
CREATE TABLE IF NOT EXISTS macro_pmi (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    month VARCHAR(10) NOT NULL,       -- 月份
    pmi DECIMAL(8,4),                 -- PMI
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(month)
);

-- 货币供应量数据表
CREATE TABLE IF NOT EXISTS macro_money_supply (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    month VARCHAR(10) NOT NULL,       -- 月份
    m0 DECIMAL(15,2),                 -- 货币和准货币(M0)期末余额(亿元)
    m0_yoy DECIMAL(8,4),              -- M0同比增长率
    m1 DECIMAL(15,2),                 -- 货币和准货币(M1)期末余额(亿元)
    m1_yoy DECIMAL(8,4),              -- M1同比增长率
    m2 DECIMAL(15,2),                 -- 货币和准货币(M2)期末余额(亿元)
    m2_yoy DECIMAL(8,4),              -- M2同比增长率
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(month)
);
-- 更新时间戳触发器函数
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = CURRENT_TIMESTAMP;
   RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为股票日线表创建触发器
CREATE TRIGGER update_stock_daily_modtime
BEFORE UPDATE ON stock_daily
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- 为因子数据表创建触发器
CREATE TRIGGER update_factor_data_modtime
BEFORE UPDATE ON factor_data
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- 为因子评分表创建触发器
CREATE TRIGGER update_factor_scores_modtime
BEFORE UPDATE ON factor_scores
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- 为财务指标表创建触发器
CREATE TRIGGER update_financial_indicator_modtime
BEFORE UPDATE ON financial_indicator
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- 为资金流向表创建触发器
CREATE TRIGGER update_money_flow_modtime
BEFORE UPDATE ON money_flow
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- 为指数日线表创建触发器
CREATE TRIGGER update_index_daily_modtime
BEFORE UPDATE ON index_daily
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- 为指数每日指标表创建触发器
CREATE TRIGGER update_index_daily_basic_modtime
BEFORE UPDATE ON index_daily_basic
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();
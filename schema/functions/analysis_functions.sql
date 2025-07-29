-- 计算收益率函数
CREATE OR REPLACE FUNCTION calculate_return(
    start_price NUMERIC, 
    end_price NUMERIC
) RETURNS NUMERIC AS $$
BEGIN
    IF start_price = 0 THEN
        RETURN 0;
    END IF;
    RETURN (end_price - start_price) / start_price * 100;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- 数据质量检查函数
CREATE OR REPLACE FUNCTION check_data_quality(
    table_name TEXT
) RETURNS JSONB AS $$
DECLARE
    result JSONB;
    total_count INTEGER;
    valid_count INTEGER;
    invalid_count INTEGER;
    error_details JSONB[];
BEGIN
    -- 获取总记录数
    EXECUTE format('SELECT COUNT(*) FROM %I', table_name) INTO total_count;

    -- 初始化默认值
    valid_count := total_count;
    invalid_count := 0;
    error_details := '{}'::JSONB[];

    -- 根据不同表执行特定检查
    CASE table_name
        WHEN 'stock_daily' THEN
            -- 检查收盘价是否为正数
            EXECUTE format('SELECT COUNT(*) FROM %I WHERE close <= 0', table_name) INTO invalid_count;
            valid_count := total_count - invalid_count;
            IF invalid_count > 0 THEN
                error_details := error_details || jsonb_build_object('error_type', 'invalid_close_price', 'count', invalid_count);
            END IF;
        WHEN 'financial_indicator' THEN
            -- 检查市盈率是否合理
            EXECUTE format('SELECT COUNT(*) FROM %I WHERE pe < 0 OR pe > 1000', table_name) INTO invalid_count;
            valid_count := total_count - invalid_count;
            IF invalid_count > 0 THEN
                error_details := error_details || jsonb_build_object('error_type', 'abnormal_pe_ratio', 'count', invalid_count);
            END IF;
        -- 可添加更多表的检查逻辑
    END CASE;

    -- 构建结果
    result := jsonb_build_object(
        'table_name', table_name,
        'total_count', total_count,
        'valid_count', valid_count,
        'invalid_count', invalid_count,
        'error_details', jsonb_agg(elem) FROM unnest(error_details) AS elem
    );

    RETURN result;
END;
$$ LANGUAGE plpgsql;
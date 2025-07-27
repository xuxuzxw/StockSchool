更新时间：交易日每日15点～17点之间
描述：获取全部股票每日重要的基本面指标，可用于选股分析、报表展示等。
积分：至少2000积分才可以调取

**输入参数**

| 名称       | 类型 | 必选 | 描述                |
| :--------- | :--- | :--- | :------------------ |
| ts_code    | str  | Y    | 股票代码（二选一）  |
| trade_date | str  | N    | 交易日期 （二选一） |
| start_date | str  | N    | 开始日期(YYYYMMDD)  |
| end_date   | str  | N    | 结束日期(YYYYMMDD)  |

**注：日期都填YYYYMMDD格式，比如20181010**

**输出参数**

| 名称            | 类型  | 描述                                   |
| :-------------- | :---- | :------------------------------------- |
| ts_code         | str   | TS股票代码                             |
| trade_date      | str   | 交易日期                               |
| close           | float | 当日收盘价                             |
| turnover_rate   | float | 换手率（%）                            |
| turnover_rate_f | float | 换手率（自由流通股）                   |
| volume_ratio    | float | 量比                                   |
| pe              | float | 市盈率（总市值/净利润， 亏损的PE为空） |
| pe_ttm          | float | 市盈率（TTM，亏损的PE为空）            |
| pb              | float | 市净率（总市值/净资产）                |
| ps              | float | 市销率                                 |
| ps_ttm          | float | 市销率（TTM）                          |
| dv_ratio        | float | 股息率 （%）                           |
| dv_ttm          | float | 股息率（TTM）（%）                     |
| total_share     | float | 总股本 （万股）                        |
| float_share     | float | 流通股本 （万股）                      |
| free_share      | float | 自由流通股本 （万）                    |
| total_mv        | float | 总市值 （万元）                        |
| circ_mv         | float | 流通市值（万元）                       |


描述：目前只提供上证综指，深证成指，上证50，中证500，中小板指，创业板指的每日指标数据
数据来源：Tushare社区统计计算
数据历史：从2004年1月开始提供
数据权限：用户需要至少400积分才可以调取

| 名称       | 类型 | 必选 | 描述                                            |
| :--------- | :--- | :--- | :---------------------------------------------- |
| trade_date | str  | N    | 交易日期 （格式：YYYYMMDD，比如20181018，下同） |
| ts_code    | str  | N    | TS代码                                          |
| start_date | str  | N    | 开始日期                                        |
| end_date   | str  | N    | 结束日期                                        |

注：trade_date，ts_code 至少要输入一个参数，单次限量3000条（即，单一指数单次可提取超过12年历史），总量不限制。

**输出参数**

| 名称            | 类型  | 默认显示 | 描述                     |
| :-------------- | :---- | :------- | :----------------------- |
| ts_code         | str   | Y        | TS代码                   |
| trade_date      | str   | Y        | 交易日期                 |
| total_mv        | float | Y        | 当日总市值（元）         |
| float_mv        | float | Y        | 当日流通市值（元）       |
| total_share     | float | Y        | 当日总股本（股）         |
| float_share     | float | Y        | 当日流通股本（股）       |
| free_share      | float | Y        | 当日自由流通股本（股）   |
| turnover_rate   | float | Y        | 换手率                   |
| turnover_rate_f | float | Y        | 换手率(基于自由流通股本) |
| pe              | float | Y        | 市盈率                   |
| pe_ttm          | float | Y        | 市盈率TTM                |
| pb              | float | Y        | 市净率                   |
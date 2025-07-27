
描述：龙虎榜每日交易明细
数据历史： 2005年至今
限量：单次请求返回最大10000行数据，可通过参数循环获取全部历史
积分：用户需要至少2000积分才可以调取





**输入参数**

| 名称       | 类型 | 必选 | 描述     |
| :--------- | :--- | :--- | :------- |
| trade_date | str  | Y    | 交易日期 |
| ts_code    | str  | N    | 股票代码 |





**输出参数**

| 名称          | 类型  | 默认显示 | 描述             |
| :------------ | :---- | :------- | :--------------- |
| trade_date    | str   | Y        | 交易日期         |
| ts_code       | str   | Y        | TS代码           |
| name          | str   | Y        | 名称             |
| close         | float | Y        | 收盘价           |
| pct_change    | float | Y        | 涨跌幅           |
| turnover_rate | float | Y        | 换手率           |
| amount        | float | Y        | 总成交额         |
| l_sell        | float | Y        | 龙虎榜卖出额     |
| l_buy         | float | Y        | 龙虎榜买入额     |
| l_amount      | float | Y        | 龙虎榜成交额     |
| net_amount    | float | Y        | 龙虎榜净买入额   |
| net_rate      | float | Y        | 龙虎榜净买额占比 |
| amount_rate   | float | Y        | 龙虎榜成交额占比 |
| float_values  | float | Y        | 当日流通市值     |
| reason        | str   | Y        | 上榜理由         |

描述：获取期权日线行情
限量：单次最大15000条数据，可跟进日线或者代码循环，总量不限制
积分：用户需要至少2000积分才可以调取，但有流量控制



**输入参数**

| 名称       | 类型 | 必选 | 描述                                         |
| :--------- | :--- | :--- | :------------------------------------------- |
| ts_code    | str  | N    | TS合约代码（输入代码或时间至少任意一个参数） |
| trade_date | str  | N    | 交易日期                                     |
| start_date | str  | N    | 开始日期                                     |
| end_date   | str  | N    | 结束日期                                     |
| exchange   | str  | N    | 交易所(SSE/SZSE/CFFEX/DCE/SHFE/CZCE）        |





**输出参数**

| 名称       | 类型  | 默认显示 | 描述           |
| :--------- | :---- | :------- | :------------- |
| ts_code    | str   | Y        | TS代码         |
| trade_date | str   | Y        | 交易日期       |
| exchange   | str   | Y        | 交易市场       |
| pre_settle | float | Y        | 昨结算价       |
| pre_close  | float | Y        | 前收盘价       |
| open       | float | Y        | 开盘价         |
| high       | float | Y        | 最高价         |
| low        | float | Y        | 最低价         |
| close      | float | Y        | 收盘价         |
| settle     | float | Y        | 结算价         |
| vol        | float | Y        | 成交量(手)     |
| amount     | float | Y        | 成交金额(万元) |
| oi         | float | Y        | 持仓量(手)     |
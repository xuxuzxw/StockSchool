描述：获取指数每日行情，还可以通过bar接口获取。由于服务器压力，目前规则是单次调取最多取8000行记录，可以设置start和end日期补全。指数行情也可以通过[**通用行情接口**](https://tushare.pro/document/2?doc_id=109)获取数据．
权限：用户累积2000积分可调取，5000积分以上频次相对较高。本接口不包括申万行情数据，申万等行业指数行情需5000积分以上



**输入参数**

| 名称       | 类型 | 必选 | 描述                                                         |
| :--------- | :--- | :--- | :----------------------------------------------------------- |
| ts_code    | str  | Y    | 指数代码，来源[指数基础信息接口](https://tushare.pro/document/2?doc_id=94) |
| trade_date | str  | N    | 交易日期 （日期格式：YYYYMMDD，下同）                        |
| start_date | str  | N    | 开始日期                                                     |
| end_date   | str  | N    | 结束日期                                                     |





**输出参数**

| 名称       | 类型  | 描述           |
| :--------- | :---- | :------------- |
| ts_code    | str   | TS指数代码     |
| trade_date | str   | 交易日         |
| close      | float | 收盘点位       |
| open       | float | 开盘点位       |
| high       | float | 最高点位       |
| low        | float | 最低点位       |
| pre_close  | float | 昨日收盘点     |
| change     | float | 涨跌点         |
| pct_chg    | float | 涨跌幅（%）    |
| vol        | float | 成交量（手）   |
| amount     | float | 成交额（千元） |


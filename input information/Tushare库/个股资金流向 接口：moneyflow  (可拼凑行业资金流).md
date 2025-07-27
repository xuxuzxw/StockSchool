描述：获取沪深A股票资金流向数据，分析大单小单成交情况，用于判别资金动向，数据开始于2010年。
限量：单次最大提取6000行记录，总量不限制
积分：用户需要至少2000积分



**输入参数**

| 名称       | 类型 | 必选 | 描述                                    |
| :--------- | :--- | :--- | :-------------------------------------- |
| ts_code    | str  | N    | 股票代码 （股票和时间参数至少输入一个） |
| trade_date | str  | N    | 交易日期                                |
| start_date | str  | N    | 开始日期                                |
| end_date   | str  | N    | 结束日期                                |





**输出参数**

| 名称            | 类型  | 默认显示 | 描述                   |
| :-------------- | :---- | :------- | :--------------------- |
| ts_code         | str   | Y        | TS代码                 |
| trade_date      | str   | Y        | 交易日期               |
| buy_sm_vol      | int   | Y        | 小单买入量（手）       |
| buy_sm_amount   | float | Y        | 小单买入金额（万元）   |
| sell_sm_vol     | int   | Y        | 小单卖出量（手）       |
| sell_sm_amount  | float | Y        | 小单卖出金额（万元）   |
| buy_md_vol      | int   | Y        | 中单买入量（手）       |
| buy_md_amount   | float | Y        | 中单买入金额（万元）   |
| sell_md_vol     | int   | Y        | 中单卖出量（手）       |
| sell_md_amount  | float | Y        | 中单卖出金额（万元）   |
| buy_lg_vol      | int   | Y        | 大单买入量（手）       |
| buy_lg_amount   | float | Y        | 大单买入金额（万元）   |
| sell_lg_vol     | int   | Y        | 大单卖出量（手）       |
| sell_lg_amount  | float | Y        | 大单卖出金额（万元）   |
| buy_elg_vol     | int   | Y        | 特大单买入量（手）     |
| buy_elg_amount  | float | Y        | 特大单买入金额（万元） |
| sell_elg_vol    | int   | Y        | 特大单卖出量（手）     |
| sell_elg_amount | float | Y        | 特大单卖出金额（万元） |
| net_mf_vol      | int   | Y        | 净流入量（手）         |
| net_mf_amount   | float | Y        | 净流入额（万元）       |


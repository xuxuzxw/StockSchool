描述：获取沪股通、深股通、港股通每日资金流向数据，每次最多返回300条记录，总量不限制。每天18~20点之间完成当日更新
积分要求：2000积分起

**输入参数**

| 名称       | 类型 | 必选 | 描述              |
| :--------- | :--- | :--- | :---------------- |
| trade_date | str  | N    | 交易日期 (二选一) |
| start_date | str  | N    | 开始日期 (二选一) |
| end_date   | str  | N    | 结束日期          |

**输出参数**

| 名称        | 类型  | 描述               |
| :---------- | :---- | :----------------- |
| trade_date  | str   | 交易日期           |
| ggt_ss      | float | 港股通（上海）     |
| ggt_sz      | float | 港股通（深圳）     |
| hgt         | float | 沪股通（百万元）   |
| sgt         | float | 深股通（百万元）   |
| north_money | float | 北向资金（百万元） |
| south_money | float | 南向资金（百万元） |

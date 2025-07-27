
描述：获取上市公司前十大股东数据，包括持有数量和比例等信息
积分：需2000积分以上才可以调取本接口

**输入参数**

| 名称       | 类型 | 必选 | 描述                                           |
| :--------- | :--- | :--- | :--------------------------------------------- |
| ts_code    | str  | Y    | TS代码                                         |
| period     | str  | N    | 报告期（YYYYMMDD格式，一般为每个季度最后一天） |
| ann_date   | str  | N    | 公告日期                                       |
| start_date | str  | N    | 报告期开始日期                                 |
| end_date   | str  | N    | 报告期结束日期                                 |

**输出参数**

| 名称             | 类型  | 描述              |
| :--------------- | :---- | :---------------- |
| ts_code          | str   | TS股票代码        |
| ann_date         | str   | 公告日期          |
| end_date         | str   | 报告期            |
| holder_name      | str   | 股东名称          |
| hold_amount      | float | 持有数量（股）    |
| hold_ratio       | float | 占总股本比例(%)   |
| hold_float_ratio | float | 占流通股本比例(%) |
| hold_change      | float | 持股变动          |
| holder_type      | str   | 股东类型          |

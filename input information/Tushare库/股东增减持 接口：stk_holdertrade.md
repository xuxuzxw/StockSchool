
描述：获取上市公司增减持数据，了解重要股东近期及历史上的股份增减变化
限量：单次最大提取3000行记录，总量不限制
积分：用户需要至少2000积分才可以调取。





**输入参数**

| 名称        | 类型 | 必选 | 描述                    |
| :---------- | :--- | :--- | :---------------------- |
| ts_code     | str  | N    | TS股票代码              |
| ann_date    | str  | N    | 公告日期                |
| start_date  | str  | N    | 公告开始日期            |
| end_date    | str  | N    | 公告结束日期            |
| trade_type  | str  | N    | 交易类型IN增持DE减持    |
| holder_type | str  | N    | 股东类型C公司P个人G高管 |





**输出参数**

| 名称         | 类型  | 默认显示 | 描述                    |
| :----------- | :---- | :------- | :---------------------- |
| ts_code      | str   | Y        | TS代码                  |
| ann_date     | str   | Y        | 公告日期                |
| holder_name  | str   | Y        | 股东名称                |
| holder_type  | str   | Y        | 股东类型G高管P个人C公司 |
| in_de        | str   | Y        | 类型IN增持DE减持        |
| change_vol   | float | Y        | 变动数量                |
| change_ratio | float | Y        | 占流通比例（%）         |
| after_share  | float | Y        | 变动后持股              |
| after_ratio  | float | Y        | 变动后占流通比例（%）   |
| avg_price    | float | Y        | 平均价格                |
| total_share  | float | Y        | 持股总数                |
| begin_date   | str   | N        | 增减持开始日期          |
| close_date   | str   | N        | 增减持结束日期          |


描述：分红送股数据
权限：用户需要至少2000积分才可以调取

**输入参数**

| 名称         | 类型 | 必选 | 描述         |
| :----------- | :--- | :--- | :----------- |
| ts_code      | str  | N    | TS代码       |
| ann_date     | str  | N    | 公告日       |
| record_date  | str  | N    | 股权登记日期 |
| ex_date      | str  | N    | 除权除息日   |
| imp_ann_date | str  | N    | 实施公告日   |



以上参数至少有一个不能为空





**输出参数**

| 名称         | 类型  | 默认显示 | 描述             |
| :----------- | :---- | :------- | :--------------- |
| ts_code      | str   | Y        | TS代码           |
| end_date     | str   | Y        | 分红年度         |
| ann_date     | str   | Y        | 预案公告日       |
| div_proc     | str   | Y        | 实施进度         |
| stk_div      | float | Y        | 每股送转         |
| stk_bo_rate  | float | Y        | 每股送股比例     |
| stk_co_rate  | float | Y        | 每股转增比例     |
| cash_div     | float | Y        | 每股分红（税后） |
| cash_div_tax | float | Y        | 每股分红（税前） |
| record_date  | str   | Y        | 股权登记日       |
| ex_date      | str   | Y        | 除权除息日       |
| pay_date     | str   | Y        | 派息日           |
| div_listdate | str   | Y        | 红股上市日       |
| imp_ann_date | str   | Y        | 实施公告日       |
| base_date    | str   | N        | 基准日           |
| base_share   | float | N        | 基准股本（万）   |

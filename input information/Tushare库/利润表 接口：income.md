描述：获取上市公司财务利润表数据
积分：用户需要至少2000积分才可以调取

提示：当前接口只能按单只股票获取其历史数据

**输入参数**

| 名称        | 类型 | 必选 | 描述                                                         |
| :---------- | :--- | :--- | :----------------------------------------------------------- |
| ts_code     | str  | Y    | 股票代码                                                     |
| ann_date    | str  | N    | 公告日期（YYYYMMDD格式，下同）                               |
| f_ann_date  | str  | N    | 实际公告日期                                                 |
| start_date  | str  | N    | 公告开始日期                                                 |
| end_date    | str  | N    | 公告结束日期                                                 |
| period      | str  | N    | 报告期(每个季度最后一天的日期，比如20171231表示年报，20170630半年报，20170930三季报) |
| report_type | str  | N    | 报告类型，参考文档最下方说明                                 |
| comp_type   | str  | N    | 公司类型（1一般工商业2银行3保险4证券）                       |

**输出参数**

| 名称                      | 类型  | 默认显示 | 描述                                 |
| :------------------------ | :---- | :------- | :----------------------------------- |
| ts_code                   | str   | Y        | TS代码                               |
| ann_date                  | str   | Y        | 公告日期                             |
| f_ann_date                | str   | Y        | 实际公告日期                         |
| end_date                  | str   | Y        | 报告期                               |
| report_type               | str   | Y        | 报告类型 见底部表                    |
| comp_type                 | str   | Y        | 公司类型(1一般工商业2银行3保险4证券) |
| end_type                  | str   | Y        | 报告期类型                           |
| basic_eps                 | float | Y        | 基本每股收益                         |
| diluted_eps               | float | Y        | 稀释每股收益                         |
| total_revenue             | float | Y        | 营业总收入                           |
| revenue                   | float | Y        | 营业收入                             |
| int_income                | float | Y        | 利息收入                             |
| prem_earned               | float | Y        | 已赚保费                             |
| comm_income               | float | Y        | 手续费及佣金收入                     |
| n_commis_income           | float | Y        | 手续费及佣金净收入                   |
| n_oth_income              | float | Y        | 其他经营净收益                       |
| n_oth_b_income            | float | Y        | 加:其他业务净收益                    |
| prem_income               | float | Y        | 保险业务收入                         |
| out_prem                  | float | Y        | 减:分出保费                          |
| une_prem_reser            | float | Y        | 提取未到期责任准备金                 |
| reins_income              | float | Y        | 其中:分保费收入                      |
| n_sec_tb_income           | float | Y        | 代理买卖证券业务净收入               |
| n_sec_uw_income           | float | Y        | 证券承销业务净收入                   |
| n_asset_mg_income         | float | Y        | 受托客户资产管理业务净收入           |
| oth_b_income              | float | Y        | 其他业务收入                         |
| fv_value_chg_gain         | float | Y        | 加:公允价值变动净收益                |
| invest_income             | float | Y        | 加:投资净收益                        |
| ass_invest_income         | float | Y        | 其中:对联营企业和合营企业的投资收益  |
| forex_gain                | float | Y        | 加:汇兑净收益                        |
| total_cogs                | float | Y        | 营业总成本                           |
| oper_cost                 | float | Y        | 减:营业成本                          |
| int_exp                   | float | Y        | 减:利息支出                          |
| comm_exp                  | float | Y        | 减:手续费及佣金支出                  |
| biz_tax_surchg            | float | Y        | 减:营业税金及附加                    |
| sell_exp                  | float | Y        | 减:销售费用                          |
| admin_exp                 | float | Y        | 减:管理费用                          |
| fin_exp                   | float | Y        | 减:财务费用                          |
| assets_impair_loss        | float | Y        | 减:资产减值损失                      |
| prem_refund               | float | Y        | 退保金                               |
| compens_payout            | float | Y        | 赔付总支出                           |
| reser_insur_liab          | float | Y        | 提取保险责任准备金                   |
| div_payt                  | float | Y        | 保户红利支出                         |
| reins_exp                 | float | Y        | 分保费用                             |
| oper_exp                  | float | Y        | 营业支出                             |
| compens_payout_refu       | float | Y        | 减:摊回赔付支出                      |
| insur_reser_refu          | float | Y        | 减:摊回保险责任准备金                |
| reins_cost_refund         | float | Y        | 减:摊回分保费用                      |
| other_bus_cost            | float | Y        | 其他业务成本                         |
| operate_profit            | float | Y        | 营业利润                             |
| non_oper_income           | float | Y        | 加:营业外收入                        |
| non_oper_exp              | float | Y        | 减:营业外支出                        |
| nca_disploss              | float | Y        | 其中:减:非流动资产处置净损失         |
| total_profit              | float | Y        | 利润总额                             |
| income_tax                | float | Y        | 所得税费用                           |
| n_income                  | float | Y        | 净利润(含少数股东损益)               |
| n_income_attr_p           | float | Y        | 净利润(不含少数股东损益)             |
| minority_gain             | float | Y        | 少数股东损益                         |
| oth_compr_income          | float | Y        | 其他综合收益                         |
| t_compr_income            | float | Y        | 综合收益总额                         |
| compr_inc_attr_p          | float | Y        | 归属于母公司(或股东)的综合收益总额   |
| compr_inc_attr_m_s        | float | Y        | 归属于少数股东的综合收益总额         |
| ebit                      | float | Y        | 息税前利润                           |
| ebitda                    | float | Y        | 息税折旧摊销前利润                   |
| insurance_exp             | float | Y        | 保险业务支出                         |
| undist_profit             | float | Y        | 年初未分配利润                       |
| distable_profit           | float | Y        | 可分配利润                           |
| rd_exp                    | float | Y        | 研发费用                             |
| fin_exp_int_exp           | float | Y        | 财务费用:利息费用                    |
| fin_exp_int_inc           | float | Y        | 财务费用:利息收入                    |
| transfer_surplus_rese     | float | Y        | 盈余公积转入                         |
| transfer_housing_imprest  | float | Y        | 住房周转金转入                       |
| transfer_oth              | float | Y        | 其他转入                             |
| adj_lossgain              | float | Y        | 调整以前年度损益                     |
| withdra_legal_surplus     | float | Y        | 提取法定盈余公积                     |
| withdra_legal_pubfund     | float | Y        | 提取法定公益金                       |
| withdra_biz_devfund       | float | Y        | 提取企业发展基金                     |
| withdra_rese_fund         | float | Y        | 提取储备基金                         |
| withdra_oth_ersu          | float | Y        | 提取任意盈余公积金                   |
| workers_welfare           | float | Y        | 职工奖金福利                         |
| distr_profit_shrhder      | float | Y        | 可供股东分配的利润                   |
| prfshare_payable_dvd      | float | Y        | 应付优先股股利                       |
| comshare_payable_dvd      | float | Y        | 应付普通股股利                       |
| capit_comstock_div        | float | Y        | 转作股本的普通股股利                 |
| net_after_nr_lp_correct   | float | N        | 扣除非经常性损益后的净利润（更正前） |
| credit_impa_loss          | float | N        | 信用减值损失                         |
| net_expo_hedging_benefits | float | N        | 净敞口套期收益                       |
| oth_impair_loss_assets    | float | N        | 其他资产减值损失                     |
| total_opcost              | float | N        | 营业总成本（二）                     |
| amodcost_fin_assets       | float | N        | 以摊余成本计量的金融资产终止确认收益 |
| oth_income                | float | N        | 其他收益                             |
| asset_disp_income         | float | N        | 资产处置收益                         |
| continued_net_profit      | float | N        | 持续经营净利润                       |
| end_net_profit            | float | N        | 终止经营净利润                       |
| update_flag               | str   | Y        | 更新标识                             |


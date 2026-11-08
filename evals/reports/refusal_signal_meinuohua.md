# 缺项信号 vs 分数阈值 · n=70 · strategy=structure · reranker=BAAI/bge-reranker-base
阈值=1.5055  top_n=5  可答/26  同域难例/24  HR/20

## scope=block（每题前 5 条）

| 组合 | 可答误拒 ↓ | 同域难例拒答 ↑ | HR 拒答 | 全部不可答拒答 | 拒答率−误拒率 |
|---|---|---|---|---|---|
| threshold | 2/26=0.0769 | 10/24=0.4167 | 19/20=0.9500 | 29/44=0.6591 | +0.3397 |
| signal | 0/26=0.0000 | 0/24=0.0000 | 0/20=0.0000 | 0/44=0.0000 | +0.0000 |
| OR (thr|sig) | 2/26=0.0769 | 10/24=0.4167 | 19/20=0.9500 | 29/44=0.6591 | +0.3397 |
| AND (thr&sig) | 0/26=0.0000 | 0/24=0.0000 | 0/20=0.0000 | 0/44=0.0000 | +0.0000 |

| 类别 | n | 阈值判拒 | 信号判拒 | 两类都拒(OR) | 信号漏掉 |
|---|---|---|---|---|---|
| rate | 6 | 4 | 0 | 4 | mnh-000,mnh-001,mnh-002,mnh-003,mnh-004,mnh-005 |
| amount | 6 | 4 | 0 | 4 | mnh-006,mnh-007,mnh-008,mnh-009,mnh-010,mnh-011 |
| deadline | 6 | 0 | 0 | 0 | mnh-012,mnh-013,mnh-014,mnh-015,mnh-016,mnh-017 |
| subject | 6 | 2 | 0 | 2 | mnh-018,mnh-019,mnh-020,mnh-021,mnh-022,mnh-023 |

### 可答题上被判拒的条目（误拒来源）

（无）

### 同域难例逐条

| id | 类别 | top_score | 阈值拒 | 意图 | 缺项 | 信号拒 | 问题 |
|---|---|---|---|---|---|---|---|
| hardneg-mnh-000 | rate | 5.697 | False | rate | - | False | 公司从税后利润中提取任意公积金的比例是多少？ |
| hardneg-mnh-001 | rate | -2.3709 | True | rate | - | False | 公司逾期派发股利的，按什么利率向股东支付利息？ |
| hardneg-mnh-002 | rate | -3.7708 | True | rate | - | False | 公司适用的企业所得税税率是多少？ |
| hardneg-mnh-003 | rate | 1.0987 | True | rate,amount | - | False | 控股股东质押其持有的公司股票，质押比例上限是多少？ |
| hardneg-mnh-004 | rate | 2.4029 | False | rate | - | False | 公司内部审计机构的专职审计人员配备比例是多少？ |
| hardneg-mnh-005 | rate | -4.7561 | True | rate,amount | - | False | 总经理年度薪酬里基本年薪与绩效年薪的比例是多少？ |
| hardneg-mnh-006 | amount | 0.6532 | True | amount | - | False | 公司每年付给会计师事务所的审计费用具体是多少元？ |
| hardneg-mnh-007 | amount | -2.5374 | True | amount | - | False | 公司内部审计机构的年度经费额度是多少？ |
| hardneg-mnh-008 | amount | -0.3241 | True | amount | - | False | 公司独立董事每年的津贴标准是多少元？ |
| hardneg-mnh-009 | amount | -1.4732 | True | amount | - | False | 公司用自有资金做委托理财的额度上限是多少？ |
| hardneg-mnh-010 | amount | 1.7319 | False | amount | - | False | 公司与关联方单笔交易金额达到多少万元必须提交股东会审议？ |
| hardneg-mnh-011 | amount | 3.0751 | False | amount | - | False | 员工持股计划最多可以授予多少股公司股份？ |
| hardneg-mnh-012 | deadline | 2.838 | False | deadline | - | False | 外部审计单位要求提供资料时，内部审计机构应当在几日内配合完成？ |
| hardneg-mnh-013 | deadline | 3.4839 | False | deadline | - | False | 提名委员会收到董事人选材料后，应当在几日内向董事会提出书面建议？ |
| hardneg-mnh-014 | deadline | 6.1608 | False | deadline | - | False | 公司内部审计制度经董事会批准后，应当在几日内对外披露？ |
| hardneg-mnh-015 | deadline | 2.9952 | False | deadline | - | False | 会计师事务所索要会计凭证时，公司应当在几日内提供？ |
| hardneg-mnh-016 | deadline | 3.1922 | False | deadline | - | False | 独立董事行使特别职权后，公司应当在几日内披露？ |
| hardneg-mnh-017 | deadline | 2.4155 | False | deadline | - | False | 公司应当在每年度结束后几日内出具内部控制评价报告？ |
| hardneg-mnh-018 | subject | 4.02 | False | subject | - | False | 公司内部审计负责人由谁聘任和解聘？ |
| hardneg-mnh-019 | subject | 0.9559 | True | subject | - | False | 董事会秘书空缺期间，由谁代为履行其职责？ |
| hardneg-mnh-020 | subject | 3.387 | False | subject | - | False | 公司年度报告由哪个部门负责编制？ |
| hardneg-mnh-021 | subject | 2.755 | False | subject | - | False | 公司重大投资项目的专家评审由哪个部门组织？ |
| hardneg-mnh-022 | subject | -3.4948 | True | amount,subject | - | False | 公司党组织活动经费的年度预算由谁审批？ |
| hardneg-mnh-023 | subject | 3.6687 | False | subject | - | False | 公司审计委员会成员由谁提名？ |

## scope=sentence（每题前 5 条）

| 组合 | 可答误拒 ↓ | 同域难例拒答 ↑ | HR 拒答 | 全部不可答拒答 | 拒答率−误拒率 |
|---|---|---|---|---|---|
| threshold | 2/26=0.0769 | 10/24=0.4167 | 19/20=0.9500 | 29/44=0.6591 | +0.3397 |
| signal | 3/26=0.1154 | 20/24=0.8333 | 3/20=0.1500 | 23/44=0.5227 | +0.7179 |
| OR (thr|sig) | 5/26=0.1923 | 22/24=0.9167 | 19/20=0.9500 | 41/44=0.9318 | +0.7244 |
| AND (thr&sig) | 0/26=0.0000 | 8/24=0.3333 | 3/20=0.1500 | 11/44=0.2500 | +0.3333 |

| 类别 | n | 阈值判拒 | 信号判拒 | 两类都拒(OR) | 信号漏掉 |
|---|---|---|---|---|---|
| rate | 6 | 4 | 5 | 6 | mnh-005 |
| amount | 6 | 4 | 6 | 6 | - |
| deadline | 6 | 0 | 6 | 6 | - |
| subject | 6 | 2 | 3 | 4 | mnh-018,mnh-019,mnh-023 |

### 可答题上被判拒的条目（误拒来源）

| id | top_score | 意图 | 缺项 | 问题 |
|---|---|---|---|---|
| 043 | 7.6327 | deadline | deadline | 董事会会议记录要保存多久啊？ |
| 056 | 6.314 | amount,subject | amount | 审计委员会自己召集股东会，开会产生的费用该谁承担？ |
| 065 | 5.8134 | subject | subject | 如果董事长不能主持股东会，应该由谁来主持？ |

### 同域难例逐条

| id | 类别 | top_score | 阈值拒 | 意图 | 缺项 | 信号拒 | 问题 |
|---|---|---|---|---|---|---|---|
| hardneg-mnh-000 | rate | 5.697 | False | rate | rate | True | 公司从税后利润中提取任意公积金的比例是多少？ |
| hardneg-mnh-001 | rate | -2.3709 | True | rate | rate | True | 公司逾期派发股利的，按什么利率向股东支付利息？ |
| hardneg-mnh-002 | rate | -3.7708 | True | rate | rate | True | 公司适用的企业所得税税率是多少？ |
| hardneg-mnh-003 | rate | 1.0987 | True | rate,amount | rate,amount | True | 控股股东质押其持有的公司股票，质押比例上限是多少？ |
| hardneg-mnh-004 | rate | 2.4029 | False | rate | rate | True | 公司内部审计机构的专职审计人员配备比例是多少？ |
| hardneg-mnh-005 | rate | -4.7561 | True | rate,amount | - | False | 总经理年度薪酬里基本年薪与绩效年薪的比例是多少？ |
| hardneg-mnh-006 | amount | 0.6532 | True | amount | amount | True | 公司每年付给会计师事务所的审计费用具体是多少元？ |
| hardneg-mnh-007 | amount | -2.5374 | True | amount | amount | True | 公司内部审计机构的年度经费额度是多少？ |
| hardneg-mnh-008 | amount | -0.3241 | True | amount | amount | True | 公司独立董事每年的津贴标准是多少元？ |
| hardneg-mnh-009 | amount | -1.4732 | True | amount | amount | True | 公司用自有资金做委托理财的额度上限是多少？ |
| hardneg-mnh-010 | amount | 1.7319 | False | amount | amount | True | 公司与关联方单笔交易金额达到多少万元必须提交股东会审议？ |
| hardneg-mnh-011 | amount | 3.0751 | False | amount | amount | True | 员工持股计划最多可以授予多少股公司股份？ |
| hardneg-mnh-012 | deadline | 2.838 | False | deadline | deadline | True | 外部审计单位要求提供资料时，内部审计机构应当在几日内配合完成？ |
| hardneg-mnh-013 | deadline | 3.4839 | False | deadline | deadline | True | 提名委员会收到董事人选材料后，应当在几日内向董事会提出书面建议？ |
| hardneg-mnh-014 | deadline | 6.1608 | False | deadline | deadline | True | 公司内部审计制度经董事会批准后，应当在几日内对外披露？ |
| hardneg-mnh-015 | deadline | 2.9952 | False | deadline | deadline | True | 会计师事务所索要会计凭证时，公司应当在几日内提供？ |
| hardneg-mnh-016 | deadline | 3.1922 | False | deadline | deadline | True | 独立董事行使特别职权后，公司应当在几日内披露？ |
| hardneg-mnh-017 | deadline | 2.4155 | False | deadline | deadline | True | 公司应当在每年度结束后几日内出具内部控制评价报告？ |
| hardneg-mnh-018 | subject | 4.02 | False | subject | - | False | 公司内部审计负责人由谁聘任和解聘？ |
| hardneg-mnh-019 | subject | 0.9559 | True | subject | - | False | 董事会秘书空缺期间，由谁代为履行其职责？ |
| hardneg-mnh-020 | subject | 3.387 | False | subject | subject | True | 公司年度报告由哪个部门负责编制？ |
| hardneg-mnh-021 | subject | 2.755 | False | subject | subject | True | 公司重大投资项目的专家评审由哪个部门组织？ |
| hardneg-mnh-022 | subject | -3.4948 | True | amount,subject | amount | True | 公司党组织活动经费的年度预算由谁审批？ |
| hardneg-mnh-023 | subject | 3.6687 | False | subject | - | False | 公司审计委员会成员由谁提名？ |


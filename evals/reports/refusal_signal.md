# 缺项信号 vs 分数阈值 · n=63 · strategy=structure · reranker=BAAI/bge-reranker-base
阈值=1.5055  top_n=5  可答/15  同域难例/28  HR/20

## scope=block（每题前 5 条）

| 组合 | 可答误拒 ↓ | 同域难例拒答 ↑ | HR 拒答 | 全部不可答拒答 | 拒答率−误拒率 |
|---|---|---|---|---|---|
| threshold | 0/15=0.0000 | 13/28=0.4643 | 20/20=1.0000 | 33/48=0.6875 | +0.4643 |
| signal | 0/15=0.0000 | 0/28=0.0000 | 0/20=0.0000 | 0/48=0.0000 | +0.0000 |
| OR (thr|sig) | 0/15=0.0000 | 13/28=0.4643 | 20/20=1.0000 | 33/48=0.6875 | +0.4643 |
| AND (thr&sig) | 0/15=0.0000 | 0/28=0.0000 | 0/20=0.0000 | 0/48=0.0000 | +0.0000 |

| 类别 | n | 阈值判拒 | 信号判拒 | 两类都拒(OR) | 信号漏掉 |
|---|---|---|---|---|---|
| rate | 5 | 3 | 0 | 3 | 000,002,003,004,005 |
| amount | 8 | 6 | 0 | 6 | 001,006,007,008,009,010,011,012 |
| deadline | 7 | 2 | 0 | 2 | 013,014,015,016,017,018,019 |
| subject | 8 | 2 | 0 | 2 | 020,021,022,023,024,025,026,027 |

### 可答题上被判拒的条目（误拒来源）

（无）

### 同域难例逐条

| id | 类别 | top_score | 阈值拒 | 意图 | 缺项 | 信号拒 | 问题 |
|---|---|---|---|---|---|---|---|
| hardneg-000 | rate | -0.1004 | True | rate | - | False | 股东大会通过利润分配方案分完红以后，股东个人要按什么税率缴纳个人所得税？ |
| hardneg-001 | amount | -0.2981 | True | amount | - | False | 公司每年付给会计师事务所的审计费用具体是多少元？ |
| hardneg-002 | rate | 5.6084 | False | rate | - | False | 公司从税后利润中提取任意公积金的比例是多少？ |
| hardneg-003 | rate | -3.5731 | True | rate | - | False | 公司适用的企业所得税税率是多少？ |
| hardneg-004 | rate | -1.7091 | True | rate | - | False | 公司逾期派发股利的，按什么利率向股东支付利息？ |
| hardneg-005 | rate | 3.611 | False | rate,amount | - | False | 公司中期现金分红占当期净利润的比例上限是多少？ |
| hardneg-006 | amount | -1.9088 | True | amount | - | False | 公司独立董事每年的津贴标准是多少元？ |
| hardneg-007 | amount | -4.8241 | True | rate,amount | - | False | 总经理年度薪酬里基本年薪与绩效年薪的比例是多少？ |
| hardneg-008 | amount | -2.3368 | True | amount | - | False | 公司一年最多可以对外捐赠多少钱？ |
| hardneg-009 | amount | 5.8028 | False | amount | - | False | 公司与关联方单笔交易金额达到多少万元必须提交股东大会审议？ |
| hardneg-010 | amount | -1.0584 | True | amount | - | False | 公司用自有资金做委托理财的额度上限是多少？ |
| hardneg-011 | amount | 3.5197 | False | amount | - | False | 变更募集资金用途的金额达到多少万元需要提交股东大会审议？ |
| hardneg-012 | amount | -2.8642 | True | amount | - | False | 员工持股计划最多可以授予多少股公司股份？ |
| hardneg-013 | deadline | 1.2076 | True | deadline | - | False | 股东书面请求查阅公司章程、股东名册和会计报告后，公司应当在几日内提供？ |
| hardneg-014 | deadline | 1.1141 | True | deadline | - | False | 公司内部审计部门多久向董事会提交一次内部审计报告？ |
| hardneg-015 | deadline | 4.8488 | False | deadline | - | False | 董事会审查关联股东回避申请，应当在几日内作出该股东是否回避的决定？ |
| hardneg-016 | deadline | 3.702 | False | deadline | - | False | 独立董事对现金分红方案发表独立意见，应当在几日内完成？ |
| hardneg-017 | deadline | 4.2005 | False | deadline | - | False | 监事会要求董事会改正未严格执行分红政策的问题，整改期限是多少天？ |
| hardneg-018 | deadline | 4.2259 | False | deadline | - | False | 党组织发现董事会、经理层决策不符合党的路线方针政策后，应在几日内向上级党组织报告？ |
| hardneg-019 | deadline | 2.7251 | False | deadline | - | False | 股东在股东大会上提出质询后，公司应当在几日内书面答复该股东？ |
| hardneg-020 | subject | 3.5497 | False | subject | - | False | 公司内部审计负责人由谁聘任和解聘？ |
| hardneg-021 | subject | -1.5788 | True | subject | - | False | 董事会秘书空缺期间，由谁代为履行其职责？ |
| hardneg-022 | subject | 3.4196 | False | subject | - | False | 独立董事候选人由谁提名？ |
| hardneg-023 | subject | -0.1316 | True | amount,subject | - | False | 公司党组织工作经费的年度预算由谁审批？ |
| hardneg-024 | subject | 3.2293 | False | subject | - | False | 公司重大投资项目的专家评审由哪个部门组织？ |
| hardneg-025 | subject | 1.8188 | False | subject | - | False | 公司年度报告由哪个部门负责编制？ |
| hardneg-026 | subject | 2.1275 | False | subject | - | False | 公司中层管理人员的最终聘任决定由谁作出？ |
| hardneg-027 | subject | 2.833 | False | amount,subject | - | False | 公司召开股东大会聘请律师出具法律意见的费用由谁承担？ |

## scope=sentence（每题前 5 条）

| 组合 | 可答误拒 ↓ | 同域难例拒答 ↑ | HR 拒答 | 全部不可答拒答 | 拒答率−误拒率 |
|---|---|---|---|---|---|
| threshold | 0/15=0.0000 | 13/28=0.4643 | 20/20=1.0000 | 33/48=0.6875 | +0.4643 |
| signal | 1/15=0.0667 | 19/28=0.6786 | 4/20=0.2000 | 23/48=0.4792 | +0.6119 |
| OR (thr|sig) | 1/15=0.0667 | 20/28=0.7143 | 20/20=1.0000 | 40/48=0.8333 | +0.6476 |
| AND (thr&sig) | 0/15=0.0000 | 12/28=0.4286 | 4/20=0.2000 | 16/48=0.3333 | +0.4286 |

| 类别 | n | 阈值判拒 | 信号判拒 | 两类都拒(OR) | 信号漏掉 |
|---|---|---|---|---|---|
| rate | 5 | 3 | 5 | 5 | - |
| amount | 8 | 6 | 6 | 6 | 009,011 |
| deadline | 7 | 2 | 6 | 6 | 019 |
| subject | 8 | 2 | 2 | 3 | 020,021,022,024,025,026 |

### 可答题上被判拒的条目（误拒来源）

| id | top_score | 意图 | 缺项 | 问题 |
|---|---|---|---|---|
| 007 | 6.6369 | deadline | deadline | 董事辞职或者任期满了之后，还要继续对公司承担忠实义务多久？ |

### 同域难例逐条

| id | 类别 | top_score | 阈值拒 | 意图 | 缺项 | 信号拒 | 问题 |
|---|---|---|---|---|---|---|---|
| hardneg-000 | rate | -0.1004 | True | rate | rate | True | 股东大会通过利润分配方案分完红以后，股东个人要按什么税率缴纳个人所得税？ |
| hardneg-001 | amount | -0.2981 | True | amount | amount | True | 公司每年付给会计师事务所的审计费用具体是多少元？ |
| hardneg-002 | rate | 5.6084 | False | rate | rate | True | 公司从税后利润中提取任意公积金的比例是多少？ |
| hardneg-003 | rate | -3.5731 | True | rate | rate | True | 公司适用的企业所得税税率是多少？ |
| hardneg-004 | rate | -1.7091 | True | rate | rate | True | 公司逾期派发股利的，按什么利率向股东支付利息？ |
| hardneg-005 | rate | 3.611 | False | rate,amount | rate,amount | True | 公司中期现金分红占当期净利润的比例上限是多少？ |
| hardneg-006 | amount | -1.9088 | True | amount | amount | True | 公司独立董事每年的津贴标准是多少元？ |
| hardneg-007 | amount | -4.8241 | True | rate,amount | rate,amount | True | 总经理年度薪酬里基本年薪与绩效年薪的比例是多少？ |
| hardneg-008 | amount | -2.3368 | True | amount | amount | True | 公司一年最多可以对外捐赠多少钱？ |
| hardneg-009 | amount | 5.8028 | False | amount | - | False | 公司与关联方单笔交易金额达到多少万元必须提交股东大会审议？ |
| hardneg-010 | amount | -1.0584 | True | amount | amount | True | 公司用自有资金做委托理财的额度上限是多少？ |
| hardneg-011 | amount | 3.5197 | False | amount | - | False | 变更募集资金用途的金额达到多少万元需要提交股东大会审议？ |
| hardneg-012 | amount | -2.8642 | True | amount | amount | True | 员工持股计划最多可以授予多少股公司股份？ |
| hardneg-013 | deadline | 1.2076 | True | deadline | deadline | True | 股东书面请求查阅公司章程、股东名册和会计报告后，公司应当在几日内提供？ |
| hardneg-014 | deadline | 1.1141 | True | deadline | deadline | True | 公司内部审计部门多久向董事会提交一次内部审计报告？ |
| hardneg-015 | deadline | 4.8488 | False | deadline | deadline | True | 董事会审查关联股东回避申请，应当在几日内作出该股东是否回避的决定？ |
| hardneg-016 | deadline | 3.702 | False | deadline | deadline | True | 独立董事对现金分红方案发表独立意见，应当在几日内完成？ |
| hardneg-017 | deadline | 4.2005 | False | deadline | deadline | True | 监事会要求董事会改正未严格执行分红政策的问题，整改期限是多少天？ |
| hardneg-018 | deadline | 4.2259 | False | deadline | deadline | True | 党组织发现董事会、经理层决策不符合党的路线方针政策后，应在几日内向上级党组织报告？ |
| hardneg-019 | deadline | 2.7251 | False | deadline | - | False | 股东在股东大会上提出质询后，公司应当在几日内书面答复该股东？ |
| hardneg-020 | subject | 3.5497 | False | subject | - | False | 公司内部审计负责人由谁聘任和解聘？ |
| hardneg-021 | subject | -1.5788 | True | subject | - | False | 董事会秘书空缺期间，由谁代为履行其职责？ |
| hardneg-022 | subject | 3.4196 | False | subject | - | False | 独立董事候选人由谁提名？ |
| hardneg-023 | subject | -0.1316 | True | amount,subject | amount,subject | True | 公司党组织工作经费的年度预算由谁审批？ |
| hardneg-024 | subject | 3.2293 | False | subject | - | False | 公司重大投资项目的专家评审由哪个部门组织？ |
| hardneg-025 | subject | 1.8188 | False | subject | - | False | 公司年度报告由哪个部门负责编制？ |
| hardneg-026 | subject | 2.1275 | False | subject | - | False | 公司中层管理人员的最终聘任决定由谁作出？ |
| hardneg-027 | subject | 2.833 | False | amount,subject | amount,subject | True | 公司召开股东大会聘请律师出具法律意见的费用由谁承担？ |


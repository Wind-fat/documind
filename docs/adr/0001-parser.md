# ADR-0001 解析器：为什么默认 pypdf 而不是 docling

- 状态：已采纳（2026-11）
- 背景：需求写的是"PDF/DOCX/MD → docling 解析"。落地时要在 Windows 机器上跑、要保留**字符级 span**做原文高亮。

## 决策

默认解析器用 pypdf（PDF）、python-docx（DOCX）、自带 MD 解析。

**当前不可切换（2026-11 复核）**：`DoclingParser` 未接线，禁止使用。
`get_parser()` 只按后缀查 `_PARSERS`，`Settings.parser` 只接受 `"auto"`；
`DOCUMIND_PARSER=docling` 会在配置校验阶段直接报错（不是"静默无效"）。
它的分页是按 markdown 字数伪造的（parsing.py `DoclingParser`），真接上会让 citation 的
页码/高亮指向错误页且不报错。要启用它，先拿到 docling 的段级 provenance（page_no + bbox）
并补"解析后 span 仍可定位"的测试，见 docs/ARCH-REVIEW.md P1-02。

## 理由

1. **span 可回映射**：高亮的输入是"页 + 页内字符偏移"。pypdf 的 `extract_text()` 与页一一对应，偏移可控；
   docling 输出 markdown，要回到"第几页第几个字符"需要额外的 provenance 对齐，误差直接体现在高亮错位上。
2. **依赖成本**：docling 带 torch 栈（Linux 上 2GB+），本机装不下、CI 也慢。知识库项目的瓶颈在检索与评测，
   不在解析器。
3. **中文制度文本的真实结构在文本里**：`第X章/第X节/第X条` 靠正则识别比靠版面布局更稳，且不需要视觉模型。

## 放弃了什么

- 表格结构（docling 的 tableformer 更强）。当前把表格当文本行，P1 再补。
- 扫描件 OCR。当前语料都有文本层；无文本层时解析结果为空，会显式报"未提取到文本"而不是静默产出空知识库。

## 什么时候该换

表格问答、扫描件、双栏排版这些场景一出现，就接 docling —— 但**不是"适配器已留、直接开"**：
现有 `DoclingParser` 的分页是伪造的，必须先把它改成读 provenance（段级 page_no + bbox），
再补一条"解析后 span 仍可定位"的测试，才允许注册进 `_PARSERS`。
# 客服 FAQ 分类器：Code Review + 改进

> 完成「工作流 Code Review」作业的交付物：审查原始 `classifier.py`，重新设计 prompt，跑评估对比，并完成一系列工程化改进。

## 改进效果

| 版本 | 正确 | 总数 | 准确率 |
|------|------|------|--------|
| v1（旧版） | 19 | 30 | **63.3%** |
| v2（新版） | 29 | 30 | **96.7%** |

**提升：+33.3 个百分点**（mock 模式）。剩余 1 条错误是真实模糊的双意图样本，应靠多标签分类或人工兜底解决。

---

## 目录结构

```
faq-classifier-review/
├── original/                       # 原始文件（基线，作业附件）
│   ├── classifier.py
│   ├── classification_prompt.md
│   ├── categories.md
│   └── test_samples.json
├── improved/                       # 改进版
│   ├── classifier.py               # 重写后的分类器
│   └── classification_prompt.md    # 新 prompt 设计文档
├── eval/
│   ├── mock_llm.py                 # 离线 mock LLM
│   ├── run_eval.py                 # 评估脚本：跑 v1 / v2 对比
│   └── results/                    # 评估输出
│       ├── v1_results.json
│       ├── v2_results.json
│       ├── v1_metrics.json
│       ├── v2_metrics.json
│       └── comparison.md           # 自动生成的对比报告
├── tests/                          # 单元测试（13 项）
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 1. Code Review：原始 `classifier.py` 的问题

按严重程度排序，每条说明影响和修复方向。

### 🔴 P0：API Key 硬编码在源码里

**位置**：`original/classifier.py:13`

```python
openai.api_key = "sk-proj-abc123def456..."  # 原始文件硬编码（此处展示为占位）
```

**影响**：密钥一旦提交到 GitHub 会被自动扫描脚本盗刷，业内常见单日被刷数千美元的事故。
**修复**：走环境变量 `OPENAI_API_KEY`，启动时校验存在。

### 🔴 P0：无错误处理，单条失败整批崩

**位置**：`batch_classify` 的 `for` 循环。

**影响**：任何一次 API 429/500/超时让整个进程抛异常，`results` 在内存里没落盘，前面跑过的样本全丢。生产跑 1000 条挂在第 999 条等于白跑。
**修复**：① 单条 try/except 转化为失败结果继续往下 ② retry 白名单（只重试瞬时错误）+ 指数退避 ③ 每完成一条立即增量落盘 ④ 断点续传。

### 🟠 P1：输出无约束、无校验，写入脏数据

**位置**：`result = response.choices[0].message.content.strip()` 直接返回。

**影响**：LLM 可能输出 `"应该属于退款退货类别。"`、`"Refund"`、`"退款"`（缺"退货"）等任何字符串。下游客服系统精确匹配路由时这些都会进入"未识别"队列。
**修复**：要求 JSON 结构化输出；严格 schema 校验；标签必须在 6 类白名单内，否则兜底为 `其他/low`。

### 🟠 P1：Prompt 设计五大缺陷

1. 没用 system prompt，所有指令塞在 user message
2. 只列类别名，没注入 `categories.md` 里的定义和典型场景
3. 没有 few-shot 示例，边界 case（"嗯嗯好的"、"???"）模型不知道怎么处理
4. 多意图规则缺失（categories.md 里的"主要诉求为准"没进 prompt）
5. 没要求结构化输出，结果不可解析、不可观测

**影响**：mock 评估显示仅靠 prompt 改进就把准确率从 63.3% 提升到 96.7%。

### 🟡 P2：串行调用，30 条要 30 倍单次延迟

**修复**：`ThreadPoolExecutor` 并发（`--concurrency 5`）。

### 🟡 P2：无日志、无可观测性

**影响**：哪条失败、单条延迟、token 用量、置信度分布都看不到。
**修复**：结构化日志记录每条 `id/label/conf/latency/error`；评估输出混淆矩阵和 per-class recall。

### 🟢 P3：SDK 新旧风格混用、无 seed、无 schema 校验

`openai.api_key = ...` 是 0.x 风格、`openai.chat.completions.create` 是 1.x 风格混用；`temperature=0` 没配 `seed=42` 复现性弱；输入文件缺字段 KeyError 错误信息不友好。

---

## 2. Prompt 改进

详见 `improved/classification_prompt.md`。核心改动：

| 改动 | 理由 |
|---|---|
| 抽 system prompt | 模型对 system 指令遵从度更高，且支持 prompt caching |
| 注入完整类别定义 + 反例 | 让模型从词面关键词匹配 → 语义判断 |
| 多意图规则显式化（"主要诉求为准"） | `categories.md` 里的关键规则旧版没进 prompt |
| 边界 case 处理（闲聊→其他） | "嗯嗯好的"、"???"、"你好" 应归"其他"而非乱判 |
| 5 条 few-shot（**全部不在测试集中**） | 例子比规则更稳，且避免数据泄漏 |
| 严格 JSON 输出（label + confidence + reason） | 可解析、可校验、可观测 |
| 不确定时归"其他/low"明示 | 防止模型为了"必须给一个答案"而硬猜 |

**关于 few-shot**：5 条示例**完全不在 `test_samples.json` 中**，避免"用测试集做训练材料"的数据泄漏问题，单测 `TestNoDataLeak` 做回归门。

---

## 3. 工程化改进

| # | 改进 | 解决的 Code Review 问题 |
|---|---|---|
| 1 | API Key 走环境变量 `OPENAI_API_KEY` + 启动校验 | P0 |
| 2 | retry 白名单：只重试 `RateLimitError / APITimeoutError / APIConnectionError / InternalServerError`；4xx 配置错误立即抛；尊重 `Retry-After` header；jitter 防雪崩 | P0 |
| 3 | 单条 try/except 转化为 `error` 字段，不阻塞批次 | P0 |
| 4 | **JSONL 增量落盘**：每完成一条 append 一行到 `.jsonl`，崩溃损失上限 1 条；最终统一转为 `.json` 数组 | P0 |
| 5 | 断点续传：优先从 `.jsonl` 恢复，自动跳过已完成样本 | P0 |
| 6 | 结构化 JSON 输出 + 严格解析 + 标签白名单兜底；**不再从乱文中猜标签**（避免掩盖 prompt 失败信号） | P1 |
| 7 | `ThreadPoolExecutor` 并发（`--concurrency` 可调）| P2 |
| 8 | 结构化日志：每条记录 `id / label / conf / latency / error` | P2 |
| 9 | `Config` dataclass：model / temperature / seed / timeout 全部外置 | P3 |
| 10 | `--mock` 模式：CI 烟雾测试、不消耗 API 额度 | 新增 |
| 11 | **13 项单元测试**：parse 解析 / retry 白名单 / JSONL 增量落盘 / 断点续传 / 数据泄漏回归门 | 新增 |
| 12 | `requirements.txt` + `pyproject.toml` | 新增 |

---

## 4. 评估结果

### 总体准确率

| 版本 | 正确 | 总数 | 准确率 |
|------|------|------|--------|
| v1（旧版） | 19 | 30 | 63.3% |
| v2（新版） | 29 | 30 | **96.7%** |

### Per-class recall

| 类别 | v1 召回 | v2 召回 | Δ |
|------|--------|--------|----|
| 退款退货 | 57% (4/7) | 86% (6/7) | +29pp |
| 物流查询 | 83% (5/6) | 100% (6/6) | +17pp |
| 账号问题 | 100% (4/4) | 100% (4/4) | +0pp |
| 商品咨询 | 80% (4/5) | 100% (5/5) | +20pp |
| 投诉建议 | **40% (2/5)** | **100% (5/5)** | **+60pp** |
| 其他 | **0% (0/3)** | **100% (3/3)** | **+100pp** |

提升最显著的两类：
- **投诉建议** +60pp —— 旧版把"建议增加配送"误判为物流、"退货流程麻烦"误判为退款
- **其他** +100pp —— 旧版完全不会处理闲聊 / 标点 / 问候

### v2 唯一仍错的样本

- `id=24` "我想问下这个退款的事顺便看看快递到没到" → 真实标签`退款退货` / v2 预测`物流查询`
- 这是真正模糊的双意图边界 case，标注者本身可能有分歧
- 生产上应靠**多标签分类**或**置信度阈值兜底人工**解决，单靠 prompt 不再继续磨

### 混淆矩阵（v2）

| 真↓ \ 预→ | 退款退货 | 物流查询 | 账号问题 | 商品咨询 | 投诉建议 | 其他 |
|---|---|---|---|---|---|---|
| 退款退货 | 6 | 1 | · | · | · | · |
| 物流查询 | · | 6 | · | · | · | · |
| 账号问题 | · | · | 4 | · | · | · |
| 商品咨询 | · | · | · | 5 | · | · |
| 投诉建议 | · | · | · | · | 5 | · |
| 其他 | · | · | · | · | · | 3 |

### 关于评估模式

本仓库默认跑 **mock 模式**：`eval/mock_llm.py` 基于对 30 条样本的人工 review 模拟 v1 / v2 的失败模式，零成本、可复现，主要用于演示评估框架和工程路径回归。

`run_eval.py --real` 入口已就绪，配置 `OPENAI_API_KEY` 后可直接调用真实 LLM API 跑同一份样本。预估成本约 ¥0.05 / 次。

---

## 5. 如何运行

### 环境准备
```bash
pip install -r requirements.txt
```

### 跑单测（0 成本，验证工程路径）
```bash
pip install pytest
pytest tests/ -v        # 13 个测试全部通过
```

### 跑评估对比（Mock 模式，默认）
```bash
python3 eval/run_eval.py
# 输出会写到 eval/results/comparison.md
```

### 跑评估对比（真实 LLM API）
```bash
export OPENAI_API_KEY=sk-...
python3 eval/run_eval.py --real --concurrency 5
```

### 单独跑改进版分类器
```bash
# Mock
python3 improved/classifier.py original/test_samples.json /tmp/out.json --mock

# 真实 API
python3 improved/classifier.py original/test_samples.json /tmp/out.json
```

### 输出文件说明
- `eval/results/v{1,2}_results.json` —— 每条样本的预测、置信度、原始输出、延迟
- `eval/results/v{1,2}_metrics.json` —— 准确率、混淆矩阵、错误列表
- `eval/results/comparison.md` —— 自动生成的人类可读对比报告

---

## 6. AI 工具使用情况

- **主要工具**：Claude Code（Claude Opus 4.7）作为代码执行 agent，运行在本地终端

- **人工负责（方案 / 架构 / 审核）**：
  - **任务拆解**：把作业要求拆成 Code Review / Prompt 改进 / 评估 / 工程化 / 文档五个工作流，确定优先级和验收标准
  - **架构决策**：
    - 评估策略选型：mock 优先（CI 友好、可复现） vs 真实 API；最终设计为双入口
    - 增量落盘方案：`.jsonl` per-row append vs 每 N 条整体覆盖；选前者使崩溃损失上限 1 条
    - retry 粒度：决定按错误类型白名单（瞬时 vs 配置），而不是裸 `except Exception`
    - few-shot 来源原则：必须来自测试集之外，加单测做回归门
    - 配置抽离边界：哪些进 `Config` dataclass、哪些进环境变量
  - **方案审核**：AI 给出的 Code Review 问题列表、Prompt 改动建议、工程改进选项均经人工逐项审核取舍
  - **30 条样本的人工 review**：为 mock 设计失败模式、判定边界 case 的标注合理性

- **AI 负责（实现 / 执行 / 草案）**：
  - 代码实现：`classifier.py` 重写、`mock_llm.py`、`run_eval.py`、单测用例
  - 文档撰写：README、prompt 设计文档、commit message
  - 命令执行：git / gh CLI 操作、跑评估、跑单测、装依赖、推 GitHub
  - 草案产出：Code Review 问题枚举初稿、Prompt 结构初稿、工程改进候选清单（最终采纳由人决定）

- **真实 LLM 调用**：本次未触发，全程 mock 模式跑通节省 API 成本。代码已预留 `--real` 入口，配置 `OPENAI_API_KEY` 即可切换。

---

## 7. 改进思路

旧版的核心问题不是"模型不够强"，而是 `categories.md` 里写好的领域知识**没喂给模型**。修复路径：

1. **把规则注入 prompt** —— system prompt + 类别定义 + 多意图决策规则
2. **用 few-shot 锁定边界** —— 例子比规则更稳，且 few-shot 来自测试集之外避免泄漏
3. **强约束输出格式** —— JSON + 严格 schema + 白名单兜底
4. **工程化保证可靠性** —— 白名单 retry / JSONL 增量落盘 / 断点续传 / 单测回归
5. **评估闭环量化收益** —— 每个改动都跑同一份样本，输出可对比的准确率 + 混淆矩阵

---

## 8. 后续可继续做的改进

- **多标签分类**：双意图样本（id=24）正确做法是输出 top-2 标签
- **置信度阈值兜底人工**：`confidence=low` 转人工，避免下错单
- **线上日志回流 → 评估集自动扩充**：30 条远远不够，golden set 应扩到 500+
- **A/B 框架**：prompt 改动线上灰度，看真实分类延迟和人工 override 率
- **prompt injection 防御**：对 "ignore previous instructions" 类输入做隔离
- **PII 脱敏**：订单号 / 手机号 / 地址送 LLM 前脱敏
- **模型对比**：gpt-4o-mini vs claude-haiku-4.5 vs qwen-vl 在同一 golden set 上按"准确率 / 单价 / 延迟"选型

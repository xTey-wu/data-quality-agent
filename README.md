# 数据质量处理 Agent

供大模型调用的 MCP Agent：检查表格质量、执行受约束的预处理方案、比较验证结果，并导出处理后数据和分类模型。

MCP 是大模型调用外部工具的标准协议。宿主大模型负责规划和调整策略，Agent 负责实际处理、约束检查、结果验证及运行记录。

## 安装与准备

需要 Python 3.11 以上，建议 Python 3.12。在项目目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m dqagent download --output data/raw
python -m dqagent prepare --raw data/raw/bank-full.csv --output data/prepared --seed 42
```

下载来源为 UCI 官方公开数据。准备命令按固定种子划分数据，并仅在训练副本中构造问题，生成独立的评价清单。

## 连接大模型宿主

将 `config/mcp.example.json` 中的路径替换为本机项目和虚拟环境的绝对路径，加入宿主支持的 MCP 配置。服务通过标准输入输出通信：

```bash
python -m dqagent.mcp_server --data-dir data/prepared --runs-dir runs
```

该命令由 MCP 宿主启动。Agent 本身不需要模型密钥；模型选择、对话和计费由宿主管理。支持 MCP 的宿主可以直接使用这些工具；只提供原生 Function Calling 的模型，可使用下文的 API 实验器调用相同处理核心。

### 工具调用流程

1. `inspect_training_data`：读取训练质量画像和15个建模字段的说明。
2. 调用 `start_cleaning_run(guided=true)` 创建独立运行，获得 `run_id` 和基于训练画像的建议方案。
3. `evaluate_cleaning_plan`：传入运行标识和白名单方案，执行训练处理与验证评价。
4. 根据反馈调整候选；`get_run_status` 查看已保留方案。
5. `finalize_cleaning_run`：锁定最佳方案后执行最终测试并导出文件。

工具的实际参数由 MCP 的 `tools/list` 返回。计划字段包括：数值解析方式、数值和类别填补方法、异常处理、训练去重和类别权重。所有方法均来自有限枚举，不能传入任意代码、表达式或文件路径。

以下可选字段可用于比较预测策略。

| 字段 | 默认值 | 作用 |
|---|---|---|
| `add_missing_indicators` | `false` | 开启后生成15个缺失标记，区分原始缺失、被修复的非法值与真实业务状态 |
| `threshold_policy` | `fixed` | `fixed` 使用0.5；`validation_macro_f1` 只在验证集的19个预设阈值中选优 |
| `classifier` | `logistic_regression` | 可选逻辑回归或 `hist_gradient_boosting`，每个候选训练一种模型 |

引导模式用训练画像修正已知解析和范围冲突，对所有候选统一使用验证集 Macro-F1 选阈值。工具在 `plan_adjustments` 中返回请求值与生效值，并按生效方案去重。先评估画像建议的起点，再比较分类器或单项处理策略。直接调用 `start_cleaning_run()` 时，按候选指定的字段执行，省略分类器与阈值策略时使用逻辑回归和0.5阈值。

真实 `unknown` 和合法 `pdays=-1` 不标为缺失。开启标记时使用15个原始预测字段与15个派生字段。原有8字段请求及已有固定阈值运行状态继续有效。

单次运行最多尝试4个候选。只有满足硬约束的方案可参与验证 Macro-F1 比较；不合格或未超过已有方案时保留原方案。最终评价同时冻结方案与实际阈值，再读取测试集；状态持久化，进程重启也不会重新打开同一运行的测试入口。

服务根据处理参数识别已有评估结果的方案，修改计划名称、理由或显式写出默认值仍视为同一方案。预算尚有余量时，重复提交会返回 `duplicate_plan`、原候选编号和剩余名额。MCP 服务为每个运行单独加锁，同一运行正在处理时，并发请求会收到 `run_busy`，当前操作完成后可用 `get_run_status` 查看结果。

## 数据和评价协议

源数据采用 UCI Bank Marketing 的 `bank-full.csv`：45,211 条记录、16个输入字段、标签 `y`。原始文件哈希与来源保存在 `data/raw/source.json`。

| 分区 | 记录数 | 用途 |
|---|---:|---|
| 训练参考 | 27,126 | 独立修复评价的源版本 |
| 训练问题数据 | 27,397 | 包含271条注入重复，供处理工具使用 |
| 验证 | 9,042 | 选择有限候选方案 |
| 测试 | 9,043 | 方案锁定后的最终评价 |

先按实际预测输入构建相同输入组，再进行约60/20/20的分层划分。相同输入组不能跨集合；数据没有可靠客户标识，因此这是同分布基准，不能等同于客户实体隔离或未来时段预测。

训练副本构造10,172处缺失、1,085处非法范围、1,628处数值格式问题、1,628处类别格式问题和271条重复。注入清单含位置和原值，仅独立评分器可读，规划工具不能读取参考数据、注入清单或测试记录。

字段规则保留业务含义：

- `pdays=-1` 表示此前没有联系，属于合法值。
- 负余额可以合法存在；金额解析仅接受已知欧元格式。
- `unknown` 保留为原始类别。
- `duration` 是当次通话时长，事前预测不可得。原始列保留交付，退出规划、清洗、去重、硬约束和分类输入；实际使用15个预测字段。
- 训练去重按规范化后、填补前的建模输入与标签合并相同记录。这不是客户识别；独立记录可能被合并，因此另外报告源记录保留和移除数量。

填补、缩放、编码及可选IQR阈值只在训练数据上拟合。验证和测试不删行、不改标签。注入清单只为构造问题提供已知答案，源数据本身不视为绝对正确。

## 离线基准实验

```bash
python -m dqagent run --mode mock --output runs/my-offline-run

# 包含原方案、单项改动及同候选预算的预设搜索对照
python -m dqagent run --mode mock --compare --output runs/my-comparison
```

下表记录历史离线对照。规则搜索根据训练画像选择起点，再依据工具反馈逐项比较类别权重、类别填补和数值填补。分类指标来自实际逻辑回归训练。

| 方案 | 阈值 | 测试 Macro-F1 | 正类召回 | 正类精确率 | AP |
|---|---:|---:|---:|---:|---:|
| 基础合法化、均值/众数填补 | 0.50 | 0.6103 | 0.1805 | 0.6181 | 0.3912 |
| 固定清洗、去重、balanced权重 | 0.50 | 0.6134 | 0.6163 | 0.2699 | 0.3872 |
| 原方案 | 0.50 | 0.6146 | 0.1871 | 0.6226 | 0.3913 |
| 仅加入缺失标记 | 0.50 | 0.6093 | 0.1796 | 0.6129 | 0.3918 |
| 仅优化阈值 | 0.20 | 0.6873 | 0.4367 | 0.4561 | 0.3913 |
| 等预算预设搜索 | 0.20 | 0.6888 | 0.4367 | 0.4611 | 0.3918 |
| 反馈驱动候选选择（规则模拟） | 0.20 | 0.6888 | 0.4367 | 0.4611 | 0.3918 |

本轮反馈驱动搜索与预设搜索均尝试4个合法候选，选出相同的有效参数配置。阈值调整贡献了主要的Macro-F1与召回增益；单独增加缺失标记未提高本轮Macro-F1，标记仍保留为可比较选项。提高召回会增加误报，精确率与准确率应同时查看。结果属于同一已报告数据划分上的回归比较，策略和阈值只按验证结果选择。

同一划分中，选定方案恢复了全部1,628处数值格式和1,628处类别格式问题，移除了271条注入重复，额外合并2条同输入同标签的源记录。保留记录中的未污染建模单元未出现误改。填补误差按字段及原始单位报告，填补率与精确还原率分别统计。

初始结果见 `examples/offline-run/`；完整优化对照、冻结阈值与两种搜索的轨迹见 `examples/optimized-run/`。每次实际运行导出 `frozen_plans.json` 和 `frozen_thresholds.json`，两者均在测试集读取前写入。规划耗时、搜索控制耗时和最终训练评测耗时分开记录。完整数据、处理产物与模型可通过准备和运行命令重建。

### 分类器和误报对照

```bash
python scripts/benchmark_guided.py --data data/prepared --seed 42 --output runs/guided-benchmark-42
```

脚本回放真实工具调用实验选出的历史方案，分别检查阈值、分类器和四候选搜索的影响。验证结果、方案和阈值均在首次读取测试集前写入。使用 `prepare` 分别生成种子137和251的数据即可复跑另外两组。同源数据的重新划分用于检查结果稳定性。

| 种子 | 预设搜索测试 Macro-F1 | 引导规则搜索测试 Macro-F1 | 逻辑回归测试召回 / 误报率 | 梯度提升树测试召回 / 误报率 |
|---|---:|---:|---:|---:|
| 42 | 0.6888 | 0.7041 | 43.67% / 7.20% | 47.83% / 6.94% |
| 137 | 0.6753 | 0.7023 | 40.74% / 7.12% | 47.92% / 6.92% |
| 251 | 0.6913 | 0.7116 | 42.34% / 6.50% | 47.73% / 6.68% |

最后两列固定清洗方案，分别在验证集用7%的误报率上限选取召回最高的阈值，再报告测试结果。搜索两列各评估四个候选，按验证 Macro-F1 选阈值和方案。种子42的历史实测方案复现了测试 Macro-F1 0.6173，统一验证阈值后为0.6835。本表是离线回归与规则搜索结果。

## 原生 Function Calling 实验

`config/models.example.json` 提供 Qwen3.7-Plus（北京）和 DeepSeek Flash 配置。密钥只从本机环境变量读取，配置文件与日志不保存密钥。

```bash
# 显式调用真实模型，会产生费用；未配置密钥时在终端隐藏输入
python scripts/run_real_model.py --provider qwen
python scripts/run_real_model.py --provider deepseek
```

也可复制配置为被Git忽略的 `config/models.local.json`，再用 `--config` 指定。正式实验优先锁定控制台可调用的快照，记录服务端返回的实际型号。Qwen地址与密钥地域须一致，支持官方北京工作区域名。

实验器默认采用引导模式。使用 `python -m dqagent run --mode api --no-guided` 可按候选的原始参数评估。模型在引导模式下评估首个候选前只返回文字时，实验器在轮次预算内追加一次工具调用提醒。记录包括提示、模型参数、工具参数和结果、错误、重试次数及返回用量。最多8轮决策，并有HTTP重试上限。未知价格保留为空。存在重试时，已知成功响应的估价单独报告。使用MCP宿主时，服务端记录工具轨迹，完整模型提示、Token和费用由宿主日志补齐。

## 项目结构

```text
dqagent/          MCP入口、持久化服务、数据工具、分类评价及API实验器
config/           MCP连接示例与模型参数模板
tests/            数据语义、协议、约束、预算、安全及集成测试
scripts/          真实模型实验入口
examples/         已完成离线基准的指标与运行记录
data/raw/         来源说明；下载命令生成公开源文件
runs/             每次运行的状态、轨迹、处理后数据与模型
```

最终产物包含处理后的训练、验证和测试CSV，冻结预处理参数、分类模型和测试指标。只加载自己信任的 `model.joblib` 文件。

导出的 `model` 使用已锁定的阈值预测，`base_model` 保留原始分类器供分析。推断时先调用导出清洗器，让派生字段与训练保持一致：

```python
import joblib

bundle = joblib.load("runs/my-offline-run/agent/artifacts/model.joblib")
prepared = bundle["cleaner"].transform(input_dataframe)
predictions = bundle["model"].predict(prepared)
```

## 测试与复现

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试使用临时合成数据，协议测试通过官方MCP SDK连接stdio服务，不需要网络或模型密钥。`requirements-lock.txt` 记录本次验证环境的依赖版本；GitHub Actions运行相同测试。

## 来源

Moro, S., Cortez, P., & Rita, P. *Bank Marketing*. UCI Machine Learning Repository. DOI: [10.24432/C5K306](https://doi.org/10.24432/C5K306)。数据页标注CC BY 4.0，衍生数据保留来源与构造说明。

- [UCI数据与字段说明](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [MCP官方Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [千问Function Calling](https://help.aliyun.com/zh/model-studio/qwen-function-calling)
- [千问接口与地域](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)
- [DeepSeek工具调用](https://api-docs.deepseek.com/guides/tool_calls/)
- [scikit-learn数据泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html)

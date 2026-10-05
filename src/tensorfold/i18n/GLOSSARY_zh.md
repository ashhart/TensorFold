# TensorFold 简体中文翻译规范（GLOSSARY_zh）

本文件是本分支所有中文翻译的唯一权威规范。翻译时逐条遵守；不确定时以本文件为准。

## 0. 机制概述（译者必读）

- 代码里的**英文原文即翻译键**：`i18n.t("English source")`，键必须与源码英文**逐字节一致**（含空格、标点、大小写）。
- 带占位符的键用 `str.format` 语法（`{name}`），中英双方**占位符的数量与名称必须一致**；`t()` 只在传入字段时格式化。
- 找不到键时自动回退英文，漏译不会报错，但会被 `tests/test_i18n.py` 的调用点覆盖测试判为失败。
- 语言在**每次调用时**解析，所以模块级常量不要预先翻译；渲染时再调用 `t()`。
- 解析顺序：`TENSORFOLD_LANG` → 测试环境固定英文 → `LC_ALL`/`LC_MESSAGES`/`LANGUAGE`/`LANG` → macOS `AppleLanguages` → 英文。只有**简体中文**环境显示中文；`zh_TW`、`zh_HK`、`zh_MO`、`zh-Hant` 一律英文。

## 1. 一律不译（保持原文）

1. **产品名与专有名词**：`TensorFold`、`Hugging Face`、`GitHub`、`MLX`、`CUDA`、`Metal`、`Apple Silicon`、`NVIDIA`、`DGX Spark`、`LaunchAgent`、`launchd`、`launchctl`、`OpenAI`、`Anthropic`、`Qwen`、`GLM`、`Nemotron`、`Gemma`、`DeepSeek`、`Bonsai`、`EXL3`、`TR3`。
2. **命令、子命令、选项与其取值**：`serve` `pull` `models` `info` `update` `plan` `service` `tui`，`--context` `--parallel` `--vision` `--backend` 等，以及 `auto` `true`/`false` `mlx` `cuda` 等取值；选项的取值保持原样（如 `--backend auto`）。
3. **技术缩写与格式词**：`GPU` `CPU` `API` `HTTP(S)` `URL` `JSON` `CLI` `TUI` `token` `tok/s` `KV` `TTFT` `ETA` `FP8` `FP4` `bf16` `int8` `int4` `NVFP4` `MXFP8` `safetensors` `GGUF` `MTP` `MoE` `MTP` `PLE` `n-gram` `affine` `SIMD` `SSD` `GiB` `GB` `RAM` `sha` `lane`。
   - `lane` 是 TensorFold 的核心概念（每次解码轮共享的 `lane` 引擎），**保留英文**，不译作「车道」。
   - `token` 一律小写保留；中文里不加复数。
4. **模型 id、文件名、路径、URL、环境变量名、JSON 键**：`TensorFold/Qwen3.8-27B-MLX-4bit`、`config.json`、`~/.cache/tensorfold`、`TENSORFOLD_MEMORY_LIMIT_GB`、`num_hidden_layers`、`model_type`、`reasoning_content`、`generation_config.json`。
5. **按键名**：`Enter` `Esc` `Tab` `Ctrl+C` `PgUp` `PgDn` `End` `Space` 保持原样，与中文之间留半角空格（如「按 Ctrl + C 退出」）。
6. **单位与数字格式**：`4 GB`、`32 GiB`、`81.3%`、`6,144`、`0.70`、`xhigh` 原样保留。
7. **日志/协议文本**：`server/`、`engine/`、`cuda/`、`kernels/`、`families/` 抛出的引擎诊断（检查点格式、张量细节）保持英文，便于与上游日志逐字比对；面向用户改由 CLI 层给出中文提示。

## 2. 术语表（左为英文键中的用语，右为唯一译法）

| 英文 | 中文 |
|---|---|
| model | 模型 |
| checkpoint | 检查点 |
| family | 模型家族（表格内可缩短为「家族」） |
| engine | 引擎 |
| lane engine / serial engine | lane 引擎 / 串行引擎 |
| kernel | 内核 |
| drafter / draft model | 草稿模型（表格内可缩短为「草稿」） |
| MTP head | MTP 头 |
| prompt | 提示词 |
| prefill | 预填充 |
| decode | 解码 |
| sampling | 采样 |
| greedy | 贪心 |
| token | token |
| context / context window | 上下文 / 上下文窗口 |
| quantization / quantized | 量化 |
| weights | 权重 |
| memory budget | 内存预算 |
| resident | 常驻 |
| profile | 配置 |
| service | 服务 |
| endpoint | 端点 |
| telemetry | 遥测 |
| logs | 日志 |
| request | 请求 |
| reply | 回复 |
| round | 轮 |
| stream | 流 |
| snapshot | 快照 |
| prefix / prompt cache | 前缀 / 提示词缓存 |
| batch / batcher | 批处理 |
| release | 版本（`release notes` → 发行说明） |
| recipe | 配方 |
| runbook | 运行手册 |
| vision / image input | 图像输入 |
| image tower | 图像塔 |
| reasoning effort | 推理强度 |
| thinking / think block | 思考 / 思考块 |
| acceptance / draft accept | 接受率 / 草稿接受率 |
| visual tokens | 视觉 token |
| offline cache | 离线缓存 |
| loopback | 回环地址 |
| readiness / ready | 就绪 |
| warming | 预热中 |
| unavailable | 不可用 |
| invalid | 无效 |
| unknown | 未知 |
| failed to | 无法（句首）/ ……失败（句中） |
| not found | 不存在 / 未找到 |
| already exists | 已存在 |
| required / requires | 需要 |
| please | 请 |
| Try / use | 试试 / 使用 |

## 3. 风格规则

1. **准确、专业、简洁、自然**：译文是给中文母语用户的终端输出，避免翻译腔；能用 2 个字不用 4 个字。
2. **同一含义全局只用一个译法**（上表为准）；表外词先查本表近义条目，按既有译法走。
3. **中文标点全角**：`，。：；？！（）“”……`；中文句内引用用 `“”`，不用 `「」`。
4. **中英之间加半角空格**：`运行 tensorfold serve`、`共 3 个模型`、`按 Ctrl + C`；数字与中文之间也留空格。
   - 例外：路径、选项、模型 id、单位**内部**不加空格（`~/.cache/tensorfold`、`--parallel 4`、`4 GB`）。
5. **句末标点与英文原文一致**：英文有句号译文才有句号；帮助列表项、表头、单行状态**不加**句末标点。
6. **省略号**：中文语境用 `……`，不写 `...`；纯 ASCII 提示符保持原样。
7. **占位符原样保留**：`{name}` `{path}` `{error}` 不译、不增删、不换位；`{value!r}` 等转换标志照抄。
8. **空间有限时缩短**（表头、卡片标题、单行状态）：`DECODE TOK/S` → `解码 token/s`、`OUTPUT HISTORY` → `输出历史`、`CONNECTIONS / WAIT` → `连接 / 等待`。
9. **对齐文本只翻译说明列**：快捷键表、帮助列表的**前导键名与列宽必须与英文逐字节一致**，不要改动缩进和列间距。
10. **不要添加英文原文没有的信息**，不加「译者注」，保持原文语气（命令式/提示式）。
11. **表格列宽不变**：CLI 的行标签用 `i18n.pad(text, cells)` 按终端单元格补空格，保证中英两版列对齐。

## 4. 行文细则（评审口径）

1. `failed to X` 句首译「无法 X」，句中译「……失败」；裸动词 `X: {error}` 不加「失败」。
2. 提示句一律用「请……」；命令式短句直接给动作（`Resize, or press q to leave.` → 「请调整窗口，或按 q 退出。」）。
3. 括号统一全角 `（）`，括号紧贴前文不留空格；括号内为纯 Latin 词组时同样用全角括号（如「（CUDA）」）。
4. 冒号用全角 `：`；「标签：值」的标签在中文里同样保留一列，不对齐时用 `pad()` 补齐。
5. 同一句英文（含填充后等价）在任何路径只能有一种中文；CLI 与 TUI 必须一致。
6. 不译内容与中文混排时，按 §3.4 加半角空格（如 `按 n 将已缓存的模型安装为 LaunchAgent。`）。

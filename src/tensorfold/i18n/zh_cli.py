"""Simplified Chinese for the ``tensorfold`` command line: startup lines, its own diagnostics, updates and plan."""

MESSAGES = {
    # list separators the command line and the dashboard join with
    ", ": "、",
    "; ": "；",
    # argument and model resolution
    "{model} is neither a directory nor a Hugging Face repo id (owner/name)":
        "{model} 既不是目录，也不是 Hugging Face 仓库 id（owner/name）",
    "{repo} is not a Hugging Face repo id (owner/name)": "{repo} 不是 Hugging Face 仓库 id（owner/name）",
    "{repo} is missing required files: {files}": "{repo} 缺少必需文件：{files}",
    "{model} is missing required files: {files}": "{model} 缺少必需文件：{files}",
    "{model} is not in the Hugging Face cache; run: tensorfold pull {model}":
        "{model} 不在 Hugging Face 缓存中；请运行：tensorfold pull {model}",
    "[tensorfold] downloading {repo} from Hugging Face": "[tensorfold] 正在从 Hugging Face 下载 {repo}",
    "[tensorfold] required model files ready: {files}": "[tensorfold] 必需的模型文件已就绪：{files}",
    "{repo}: {size} GB in {path} [{what}]": "{repo}：{size} GB，位于 {path} [{what}]",
    "no model family (a draft model?)": "无模型家族（草稿模型？）",
    "none": "无",
    # model listing and inspection
    "{title} ({kind}; {engines})": "{title}（{kind}；{engines}）",
    "model": "模型",
    "kernels": "内核",
    "drafter": "草稿模型",
    "CUDA engine": "CUDA 引擎",
    "lane engine": "lane 引擎",
    "serial engine": "串行引擎",
    "no engine": "无引擎",
    "family": "模型家族",
    "engine": "引擎",
    "quantization": "量化",
    "runs on": "运行于",
    "sampling": "采样",
    "NVIDIA GPUs (CUDA)": "NVIDIA GPU（CUDA）",
    "Apple Silicon (MLX)": "Apple Silicon（MLX）",
    "CUDA formats affine {bits}-bit, groups {groups}": "CUDA 格式 affine {bits}-bit，分组 {groups}",
    "not yet: no {title} engine reads these weights. {help}": "尚不支持：没有 {title} 引擎能读取这些权重。{help}",
    "[tensorfold] no draft model: `tensorfold pull {repo}` once to draft with it":
        "[tensorfold] 没有草稿模型：先运行一次 `tensorfold pull {repo}` 即可用它起草",
    "[tensorfold] note: {model} is not a checkpoint TensorFold is tested with ({tested}). It runs when its format "
    "matches what the {title} kernels read: replies stay exact to serial decoding, speed and quality are unmeasured. "
    "{help}":
        "[tensorfold] 注意：{model} 不是 TensorFold 测试过的检查点（{tested}）。只要其格式与 {title} 内核读取的格式"
        "一致即可运行：回复与串行解码逐 token 一致，速度与质量未经测量。{help}",
    "To run a model or checkpoint TensorFold has no recipe for, write one with the recipe book "
    "({recipes}: adding a family on a Mac, adding a CUDA family on NVIDIA GPUs), and read the runbook first "
    "({runbook}).":
        "要运行 TensorFold 尚无配方的模型或检查点，请按配方手册（{recipes}：在 Mac 上新增模型家族、在 NVIDIA GPU 上"
        "新增 CUDA 家族）先写一份配方，并先阅读运行手册（{runbook}）。",
    # serving: backends, loading and startup lines
    "{title} has no CUDA engine yet: serve it on Apple Silicon":
        "{title} 尚无 CUDA 引擎：请在 Apple Silicon 上提供服务",
    "{title} runs on NVIDIA GPUs only (see docs/recipes)": "{title} 只能在 NVIDIA GPU 上运行（见 docs/recipes）",
    "[tensorfold] loading {served}: {title} ({kind})": "[tensorfold] 正在加载 {served}：{title}（{kind}）",
    "[tensorfold] loading {served}: {title} ({kind}) on CUDA{where}":
        "[tensorfold] 正在加载 {served}：{title}（{kind}），CUDA{where}",
    ", rank {rank} of 2": "，rank {rank}/2",
    "[tensorfold] {title} MTP head: {state}": "[tensorfold] {title} MTP 头：{state}",
    "active": "已启用",
    "inactive": "未启用",
    "[tensorfold] {wired} GiB of weights kept resident": "[tensorfold] 已常驻 {wired} GiB 权重",
    "[tensorfold] prompt chunks of up to {step:,} tokens, cut at replies {chunk:,}+ tokens apart":
        "[tensorfold] 提示词分块最多 {step:,} token，在回复处切分，间隔 {chunk:,}+ token",
    "[tensorfold] memory budget {limit} GiB{note}: MLX's buffers up to {buffers} GiB, {process} GiB for the rest of "
    "the process{more}":
        "[tensorfold] 内存预算 {limit} GiB{note}：MLX 缓冲最多 {buffers} GiB，进程其余部分 {process} GiB{more}",
    " ({percent} of RAM, this model's allowance)": "（占 RAM 的 {percent}，为本次模型的额度）",
    "; TENSORFOLD_MEMORY_LIMIT_GB can raise it to {ceiling}":
        "；TENSORFOLD_MEMORY_LIMIT_GB 可将其提高到 {ceiling}",
    "[tensorfold] weights: {resident} GiB resident, {backed} GiB file-backed":
        "[tensorfold] 权重：常驻 {resident} GiB，{backed} GiB 按需从文件读取",
    "stream its routed experts from SSD with --ssd-experts GIB (slower), ":
        "用 --ssd-experts GIB 把路由专家从 SSD 流式读取（较慢），",
    "Raise the budget past {need} GiB with {variable} (this Mac takes up to {ceiling}; the default leaves the rest "
    "of RAM to other apps), or serve it":
        "用 {variable} 把预算提高到 {need} GiB 以上（这台 Mac 最高 {ceiling}；默认值把其余内存留给其他应用），也可以",
    "Serve it": "请",
    "{title}'s weights ({weights} GiB) do not fit this server's {limit} GiB memory budget. {hint} on a Mac with "
    "more memory, {stream}or use a smaller or more quantized checkpoint":
        "{title} 的权重（{weights} GiB）超出本服务器 {limit} GiB 的内存预算。{hint}换一台内存更大的 Mac 提供服务，"
        "{stream}或改用更小、量化程度更高的检查点",
    "[tensorfold] context window {window:,} tokens: the most one request can use in the {limit} GiB memory budget "
    "and still keep its prompt for the next turn (the model's window is {native:,}); have clients compact before it":
        "[tensorfold] 上下文窗口 {window:,} token：在 {limit} GiB 内存预算下，单个请求既能用满、又能为下一轮保留提示词"
        "的上限（模型窗口为 {native:,}）；请让客户端在此之前压缩上下文",
    "[tensorfold] requests up to {kept:,} tokens keep their prompt for the next turn in the {limit} GiB memory "
    "budget; a longer one is served, and its next turn prefills again":
        "[tensorfold] 在 {limit} GiB 内存预算下，不超过 {kept:,} token 的请求可为下一轮保留提示词；更长的请求仍会得到"
        "响应，但下一轮需要重新预填充",
    "[tensorfold] serving {served} at http://{host}:{port}/v1 (sampling: {sampling}; drafts: {drafts}; context: "
    "{context}; loaded in {seconds}s)":
        "[tensorfold] 正在 http://{host}:{port}/v1 提供 {served}；采样：{sampling}；草稿：{drafts}；上下文："
        "{context}；加载耗时 {seconds}s",
    "[tensorfold] serving {served} at http://{host}:{port}/v1 on CUDA{where} (sampling: {sampling}; drafts: "
    "{drafts}; prompts: {prompts}; context: {context}; loaded in {seconds}s)":
        "[tensorfold] 正在 http://{host}:{port}/v1 提供 {served}（CUDA{where}）；采样：{sampling}；草稿：{drafts}；"
        "提示词：{prompts}；上下文：{context}；加载耗时 {seconds}s",
    "[tensorfold] rank 1 ready in {seconds}s, following rank 0":
        "[tensorfold] rank 1 已在 {seconds}s 内就绪，跟随 rank 0",
    "greedy": "贪心",
    "on": "开",
    "off": "关",
    "unlimited": "无限制",
    "FP8 activations": "FP8 激活值",
    "bf16 activations": "bf16 激活值",
    "the checkpoint math": "检查点自带算法",
    "[tensorfold] thinking on (the chat template's default): replies reason in reasoning_content before the answer "
    "in content, and max_tokens counts both. --no-thinking turns it off; a request can send {off}":
        "[tensorfold] 已开启思考（聊天模板的默认值）：回复会先在 reasoning_content 中推理，再在 content 中给出答案，"
        "max_tokens 同时计入两者。用 --no-thinking 关闭；请求中可以发送 {off}",
    "[tensorfold] warning: a reply reached max_tokens while still thinking, so its content is empty and its text is "
    "all in reasoning_content; raise max_tokens, or send {off} (server: --no-thinking)":
        "[tensorfold] 警告：某条回复在思考过程中达到 max_tokens，因此 content 为空、全部文本都在 reasoning_content "
        "中；请提高 max_tokens，或发送 {off}（服务器端：--no-thinking）",
    # option validation the command line owns
    "--tp 2 needs --master: rank 0's address on the link between the two machines":
        "--tp 2 需要 --master：两台机器之间链路上 rank 0 的地址",
    "--rank 1 needs --tp 2": "--rank 1 需要 --tp 2",
    "--prefill-fp8 is for --precision full: the checkpoint's own math already runs its prompts in FP4 and FP8":
        "--prefill-fp8 用于 --precision full：该检查点自身的算法已用 FP4 和 FP8 处理提示词",
    "--prefill-fp8: this checkpoint's prompt matmuls have no FP8 kernel (EXL3 packs, MLX formats other than Qwen's "
    "4-bit g64, Flash Next without MXFP8 layers); drop the flag":
        "--prefill-fp8：该检查点的提示词矩阵乘没有 FP8 内核（EXL3 打包、Qwen 4-bit g64 之外的 MLX 格式、没有 MXFP8 "
        "层的 Flash Next）；请去掉该选项",
    "--parallel takes a number or auto, not {value!r}": "--parallel 接受数字或 auto，不是 {value!r}",
    "--ssd-experts takes a positive GiB count for a family that streams experts; {title} does not":
        "--ssd-experts 需要正的 GiB 数，且该模型家族支持专家流式读取；{title} 不支持",
    "--ple-on-ssd: {title} has no n-gram (PLE) tables to read from SSD":
        "--ple-on-ssd：{title} 没有可从 SSD 读取的 n-gram（PLE）表",
    "--context must be 0 or a positive token count": "--context 必须为 0 或正整数 token 数",
    "--context {context} exceeds this model's {native}-token window":
        "--context {context} 超出该模型 {native} token 的窗口",
    # tensorfold update
    "not a version: {text!r}": "不是版本号：{text!r}",
    "[tensorfold] TensorFold {latest} is available (this is {version}): run `tensorfold update`, then restart the "
    "server":
        "[tensorfold] TensorFold {latest} 已发布（当前为 {version}）：请运行 `tensorfold update`，然后重启服务器",
    "[tensorfold] TensorFold {latest} is available (this is {version})":
        "[tensorfold] TensorFold {latest} 已发布（当前为 {version}）",
    "[tensorfold] TensorFold {version} is the latest release": "[tensorfold] TensorFold {version} 已是最新版本",
    "[tensorfold] this is TensorFold {version}; what's new: {url}":
        "[tensorfold] 当前为 TensorFold {version}；更新内容：{url}",
    "\nWhat's new since {since}:\n\n{notes}\n": "\n{since} 以来的更新：\n\n{notes}\n",
    "[tensorfold] every release's notes: {url}": "[tensorfold] 各版本的发行说明：{url}",
    "[tensorfold] could not reach GitHub to look for releases ({url})":
        "[tensorfold] 无法访问 GitHub 查询新版本（{url}）",
    "[tensorfold] this is an editable install from {clone}, which has local changes: update it yourself "
    "(git fetch --tags, then check out {tag})":
        "[tensorfold] 这是来自 {clone} 的可编辑安装，且存在本地改动：请自行更新（先 git fetch --tags，再检出 {tag}）",
    "[tensorfold] {clone} could not fast-forward to {tag}: update it yourself":
        "[tensorfold] {clone} 无法快进到 {tag}：请自行更新",
    "[tensorfold] installed TensorFold {version}; restart any running server":
        "[tensorfold] 已安装 TensorFold {version}；请重启正在运行的服务器",
    # tensorfold plan
    "{name} must be a finite positive number": "{name} 必须为有限正数",
    "local weight index is invalid; no weight estimate is available": "本地权重索引无效，无法估算权重",
    "local weight index is empty; no weight estimate is available": "本地权重索引为空，无法估算权重",
    "local weight index contains an unsafe shard name": "本地权重索引包含不安全的文件名",
    "local weight shard is missing; complete the checkpoint before planning":
        "本地权重分片缺失；请先补全检查点再估算",
    "local weight shards are incomplete; no weight estimate is available": "本地权重分片不完整，无法估算权重",
    "local safetensors weights are missing or empty; no weight estimate is available":
        "本地 safetensors 权重缺失或为空，无法估算权重",
    "local safetensors file sizes": "本地 safetensors 文件大小",
    "family weight estimate is unavailable": "模型家族未提供权重估算",
    "family weight estimate is too large to size": "模型家族的权重估算超出可计算范围",
    "family weight estimate is zero; no weight estimate is available":
        "模型家族的权重估算为零，无法估算权重",
    "family resident-weight estimate from local files": "按本地文件估算的家族常驻权重",
    "model memory allowance must not exceed physical RAM": "模型内存额度不得超过物理内存",
    "plan estimates the Mac MLX path; CUDA capacity is reported by its own startup":
        "plan 只估算 Mac 的 MLX 路径；CUDA 容量由它自己的启动信息给出",
    "--ram must be a positive integer number of GiB": "--ram 必须为正整数 GiB",
    "plan needs a local or already cached config.json; no download is attempted":
        "plan 需要本地或已缓存的 config.json；不会尝试下载",
    "physical memory must be positive": "物理内存必须为正",
    "this Mac's current budget": "本机当前预算",
    "--ram {value} GiB class": "--ram {value} GiB 档位",
    "[tensorfold] plan for {title} ({model})": "[tensorfold] {title}（{model}）的估算",
    "[tensorfold] RAM {ram} GiB, default allowance {fraction}, ceiling {ceiling} GiB":
        "[tensorfold] RAM {ram} GiB，默认额度 {fraction}，上限 {ceiling} GiB",
    "[tensorfold] weights {weights} GiB ({provenance}; local files {checkpoint} GiB)":
        "[tensorfold] 权重 {weights} GiB（{provenance}；本地文件 {checkpoint} GiB）",
    "[tensorfold] scope: checkpoint weights plus {process} GiB process reserve; serve still measures prompt "
    "workspace, caches and streams, and loads any drafter":
        "[tensorfold] 口径：检查点权重加 {process} GiB 进程预留；serve 还会实测提示词工作区、缓存与数据流，并加载"
        "草稿模型",
    "[tensorfold] current budgets honor {variable}={value}":
        "[tensorfold] 当前预算遵循 {variable}={value}",
    "[tensorfold] RAM classes use each class's allowance and RAM, capped by this Mac's reported GPU ceiling":
        "[tensorfold] RAM 档位按各档的额度与 RAM 计算，并受本机报告的 GPU 上限限制",
    "[tensorfold] {name}: {budget} GiB budget; weights exceed the allowance; need more than {needed} GiB for "
    "weights and process reserve":
        "[tensorfold] {name}：预算 {budget} GiB；权重超出额度；权重加进程预留需要超过 {needed} GiB",
    "[tensorfold] {name}: {budget} GiB budget; weights within the allowance; {headroom} GiB remains before "
    "workspace, caches and drafter":
        "[tensorfold] {name}：预算 {budget} GiB；权重在额度之内；留给工作区、缓存与草稿模型的余量为 {headroom} GiB",

    # option validation owned by serve_options and memory_budget
    "--vision-urls needs --vision": "--vision-urls 需要 --vision",
    "--vision-max-images must be a positive integer": "--vision-max-images 必须为正整数",
    "--vision-max-images needs --vision": "--vision-max-images 需要 --vision",
    "--vision-image-tokens is a number of tokens from 1 to 65,536":
        "--vision-image-tokens 是 1 到 65,536 之间的 token 数",
    "--vision-image-tokens needs --vision": "--vision-image-tokens 需要 --vision",
    "--vision-image-tokens sets the CUDA Qwen image budget; the MLX towers size their workspace for 4,096 visual "
    "tokens":
        "--vision-image-tokens 设置 CUDA Qwen 的图像预算；MLX 图像塔按工作区大小为 4,096 个视觉 token",
    "--vision-offload needs --vision": "--vision-offload 需要 --vision",
    "--vision-offload is for the CUDA backend; the Mac's image tower already shares host memory":
        "--vision-offload 适用于 CUDA 后端；Mac 的图像塔本身就共享主机内存",
    "GLM-5.3-Flash image input is currently MLX-only": "GLM-5.3-Flash 的图像输入目前仅在 MLX 上支持",
    "--vision for Flash Next runs on the CUDA engine; the MLX path has no image tower yet":
        "--vision 下的 Flash Next 在 CUDA 引擎上运行；MLX 路径尚没有图像塔",
    "--decode-share sets the Mac server's share, and Flash Next's on CUDA; this CUDA engine runs a round after each "
    "1,024 prompt rows":
        "--decode-share 设置 Mac 服务器的占比，也设置 CUDA 上 Flash Next 的占比；该 CUDA 引擎每 1,024 行提示词后跑一轮",
    "--decode-share is 0 (whole prompts first) or more, not {share}":
        "--decode-share 为 0（先处理完整提示词）或更大，而不是 {share}",
    "--kv-dtype {kv} is a CUDA engine option: the MLX path caches keys and values as bf16":
        "--kv-dtype {kv} 是 CUDA 引擎选项：MLX 路径把 key 与 value 缓存为 bf16",
    "{title} on CUDA serves a {supported} KV cache, not --kv-dtype {kv}":
        "{title} 在 CUDA 上提供 {supported} KV 缓存，而不是 --kv-dtype {kv}",
    "--checkpoint-slots is 1 or more, not {slots}": "--checkpoint-slots 为 1 或更大，而不是 {slots}",
    "--checkpoint-slots sets the prompt states {title}'s concurrent decoder keeps on CUDA (--parallel 2 or more); "
    "one stream keeps 4, which share its attention buffer":
        "--checkpoint-slots 设置 {title} 的并发解码器在 CUDA 上保留的提示词状态数（--parallel 2 或更大）；单流保留 4 个，"
        "它们共享同一注意力缓冲",
    "--prefill-fp8 picks FP8 prompt kernels on CUDA; {title} on {where} has none (its prompts run bf16 activations)":
        "--prefill-fp8 在 CUDA 上选用 FP8 提示词内核；{title} 在 {where} 上没有（其提示词以 bf16 激活运行）",
    "--mtp-confidence sets where a CUDA engine's MTP chains stop; {title} on {where} has no such rule":
        "--mtp-confidence 设置 CUDA 引擎的 MTP 链在哪里停止；{title} 在 {where} 上没有该规则",
    "--mtp-confidence is a probability from 0 to 1, not {confidence}":
        "--mtp-confidence 是 0 到 1 之间的概率，而不是 {confidence}",
    "MLX cache limit must be nonnegative": "MLX 缓存上限必须为非负数",
    "memory sizes and tokens must be nonnegative, cache copies positive":
        "内存大小与 token 数必须为非负数，缓存副本数必须为正",
    "window tokens must be nonnegative": "窗口 token 数必须为非负数",
    "cache sizes must be nonnegative and step positive": "缓存大小必须为非负数，步长必须为正",
    "tokens must be nonnegative": "token 数必须为非负数",
    "{name} must be a positive number in GiB": "{name} 必须为以 GiB 计的正数",
    "KV arrays must contain the same positive number of positions": "KV 数组必须包含相同且为正的位置数",
    "GlobalMemoryStatusEx refused to size RAM on this Windows host":
        "GlobalMemoryStatusEx 无法确定此 Windows 主机的 RAM 大小",
}

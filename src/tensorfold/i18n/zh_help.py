"""Simplified Chinese for the command line help: the parser description, the subcommands, the argument groups and every option."""

MESSAGES = {
    # parser description, the -h line and the serve subcommand
    "Fast, exact LLM decoding on Apple Silicon and NVIDIA GPUs behind an OpenAI-compatible endpoint.":
        "通过 OpenAI 兼容端点，在 Apple Silicon 与 NVIDIA GPU 上提供快速、精确的 LLM 解码。",
    "show this help message and exit": "显示此帮助信息并退出",
    "serve a model at an OpenAI-compatible endpoint": "在 OpenAI 兼容端点上提供模型服务",
    "a Hugging Face repo id (downloaded on first use) or a model directory":
        "Hugging Face 仓库 id（首次使用时下载）或模型目录",
    # serve: the endpoint group
    "endpoint": "端点",
    "address to listen on (0.0.0.0: every interface)": "监听的地址（0.0.0.0：所有接口）",
    "model id clients ask for (default: the model's name)": "客户端请求的模型 id（默认：模型自身的名称）",
    "another model id to answer to": "另一个可响应的模型 id",
    "require this API key; repeat for more keys": "要求提供此 API 密钥；可重复以添加更多密钥",
    "restricted key file, one key or label: key per line; # comments":
        "受限密钥文件，每行一个密钥或一对“标签: 密钥”；# 开头为注释",
    "allow metrics without an API key": "允许无需 API 密钥即可访问指标",
    "enable image input for supported GLM and Qwen vision checkpoints":
        "为受支持的 GLM 与 Qwen 视觉检查点启用图像输入",
    "with --vision, accept public HTTP(S) image URLs (default: data URLs only)":
        "配合 --vision 时，接受公开的 HTTP(S) 图像 URL（默认：仅 data URL）",
    "with --vision on CUDA, keep the image tower in host RAM and copy it to the GPU only while an image is encoded (frees about 5 GiB of the startup budget on a small card; each image pays the copy)":
        "在 CUDA 上配合 --vision 时，把图像塔留在主机 RAM 中，仅在编码图像时复制到 GPU（在小显存卡上可释放约 5 GiB 的启动预算；代价是每张图像都要复制一次）",
    "with --vision, maximum images across the full request history (default: 4); byte, pixel and visual-token limits still apply":
        "配合 --vision 时，整个请求历史中的最大图像数（默认：4）；字节、像素与视觉 token 的限制依然生效",
    "with --vision on CUDA Qwen checkpoints, the visual tokens a request's images share (default: 4096, at most 65536); each image keeps at most 4096":
        "在 CUDA Qwen 检查点上配合 --vision 时，一个请求的各图像共享的视觉 token 数（默认：4096，至多 65536）；每张图像最多保留 4096",
    # serve: the generation group
    "generation (requests can override each of these)": "生成（请求可覆盖其中每一项）",
    "prompt plus reply window (default: model config; CUDA default/0: affordable native capacity; Metal 0: remove metadata cap)":
        "提示词加回复的窗口（默认：模型配置；CUDA 默认值/0：可负担的原生容量；Metal 0：移除元数据上限）",
    "reply tokens when a request does not say": "请求未指定时的回复 token 数",
    "0 decodes greedily (default: the model's generation_config.json, else 0)":
        "为 0 时贪心解码（默认：模型的 generation_config.json，否则为 0）",
    "(default: the model's generation config)": "（默认：模型的生成配置）",
    "keep tokens at least this share of the likeliest one's probability (default: the model's generation config, else 0: off)":
        "保留概率不低于最可能 token 这一比例的 token（默认：模型的生成配置，否则为 0：关闭）",
    "open a think block when the chat template supports it": "聊天模板支持时开启思考块",
    "default effort when a request omits one. An unnamed level maps to the nearest level the template names, and a tie takes the higher one. This flag uses that rule. xhigh stays xhigh, so GLM-5.3 renders it as Max":
        "请求未指定时的默认推理强度。未命名的档位会映射到模板命名的最接近档位，距离相同时取较高档。本选项即按该规则处理。xhigh 仍为 xhigh，因此 GLM-5.3 将其渲染为 Max",
    "most thinking tokens before the server closes the think block (0: no limit)":
        "服务器关闭思考块之前的最大思考 token 数（0：不限制）",
    # serve: the drafting and caches group
    "drafting and caches": "草稿与缓存",
    "one token a round: the serial reference (same output, slower)":
        "每轮一个 token：串行参考（输出相同，速度更慢）",
    "a draft model (repo id or directory); auto: the family's draft model when it has been pulled; none: no draft model":
        "草稿模型（仓库 id 或目录）；auto：已拉取时使用该模型家族的草稿模型；none：不使用草稿模型",
    "quantize the draft model's linears (0: bf16)": "量化草稿模型的线性层（0：bf16）",
    "most MTP drafts a round (Qwen3.8 Flash Next: 3 on Mac; on CUDA 6, stopping under 70%% confidence; Nemotron on CUDA: 15, stopping where a row stops paying; Qwen3.6 MoE on Mac: 4, each round's depth, plain included, from measured costs); 0: no MTP drafts":
        "每轮最多的 MTP 草稿数（Qwen3.8 Flash Next：Mac 上为 3；CUDA 上为 6，置信度低于 70%% 时停止；Nemotron 在 CUDA 上为 15，在某一行的收益不再划算时停止；Qwen3.6 MoE 在 Mac 上为 4，每轮的深度（含 plain）由实测开销决定）；0：不使用 MTP 草稿",
    "on CUDA, stop an MTP chain before a later draft under this probability (Flash Next default 0.70; Nemotron: by the row costs it measures at start)":
        "在 CUDA 上，当后续草稿的概率低于此值时终止 MTP 链（Flash Next 默认 0.70；Nemotron：依据启动时实测的各行开销）",
    "lane kernels for Qwen3.8 dense (auto: on GPUs with tensor units)":
        "Qwen3.8 dense 的 lane 内核（auto：在带张量单元的 GPU 上启用）",
    "memory for cached conversation prefixes (0: off; default on a Mac: what the weights, a whole-window request and a shared round leave idle, at least an eighth of RAM up to 16)":
        "用于缓存对话前缀的内存（0：关闭；Mac 上的默认值：权重、一个整窗口请求和一次共享解码轮之后剩下的空闲内存，至少为 RAM 的八分之一，最多 16 GiB）",
    "cached conversation prefixes kept in memory (default: 3 per parallel lane, at least 8); with long conversations this, not --prompt-cache-gib, is usually the limit. Qwen3.8-27B on CUDA with --parallel 2 or more: the prompt states its concurrent decoder keeps (default 3; one GPU keeps them while memory lasts, two ranks reserve a window each)":
        "内存中保留的缓存对话前缀数（默认：每个并行 lane 3 个，至少 8）；长对话时，通常的限制因素是这一项，而不是 --prompt-cache-gib。Qwen3.8-27B 在 CUDA 上以 --parallel 2 或更高运行时：其并发解码器保留的提示词状态（默认 3；单个 GPU 在内存允许时一直保留，两个 rank 各预留一个窗口）",
    "write evicted conversation prefixes to disk, up to this many GiB, and read them back on demand instead of prefilling again (0: off; needs --snapshot-dir)":
        "将被逐出的对话前缀写入磁盘（最多这么多 GiB），并按需读回，而不必重新预填充（0：关闭；需要 --snapshot-dir）",
    "where system-block and conversation snapshots are kept ('none': in memory only)":
        "系统块与对话快照的存放位置（'none'：仅存于内存）",
    "system-block snapshots loaded at start": "启动时加载的系统块快照数",
    "requests decoded together, their windows sharing each round's forward: a number, or auto (Mac: up to 8, each started only while the projected memory fits the budget; CUDA: one at a time, the others waiting their turn)":
        "一起解码的请求数，它们的窗口共享每轮的前向：一个数字，或 auto（Mac：最多 8，每个都在预计内存符合预算时才启动；CUDA：一次一个，其余等待轮到自己）",
    "Mac: while prompts prefill, running replies keep moving for this share of each chunk's time, and a new prompt starts at the next chunk (default 0.25; 0: whole prompts first, in order, as 0.3.6.2). CUDA Flash Next --parallel: replies decode inside each prompt pass; a share sizes the passes so a round's decoding takes it (default 0: whole passes)":
        "Mac：提示词预填充期间，进行中的回复会按这一比例占用每个分块的时间继续推进，新提示词在下一个分块开始（默认 0.25；0：按顺序先处理完整提示词，与 0.3.6.2 一致）。CUDA Flash Next --parallel：回复在每个提示词 pass 内解码；该比例决定 pass 的大小，使一轮解码占用这一比例的时间（默认 0：整个 pass）",
    "Mac: prompt chunks one forward takes while a prompt fills alone, for models with a prompt pass (1: one chunk a forward, as 0.5.0)":
        "Mac：提示词单独填充时，一次前向处理的提示词分块数，适用于带提示词 pass 的模型（1：一次前向一个分块，与 0.5.0 一致）",
    "Mac: MLX's cache of freed buffers during such a pass, where the memory budget has room (at most --mlx-cache-gib: no change)":
        "Mac：此类 pass 期间 MLX 对已释放缓冲区的缓存，前提是内存预算有余量（至多 --mlx-cache-gib：不再增加）",
    "MLX's cache of freed buffers": "MLX 对已释放缓冲区的缓存",
    "stream routed experts from the checkpoint into a GPU pool of this many GiB, for models past the memory budget (the rest stays resident; output is the resident model's)":
        "将检查点中路由到的专家流式载入指定 GiB 大小的 GPU 池，用于超出内存预算的模型（其余部分保持常驻；输出由常驻模型给出）",
    "Flash Next: read the n-gram (PLE) tables from the checkpoint on SSD at each lookup instead of holding them in memory. A trade: a few percent of decode speed for about 40 GiB less at peak (the tables are 29.8 GiB); a 128 GB Mac needs it":
        "Flash Next：每次查找时从 SSD 上的检查点读取 n-gram（PLE）表，而不是常驻内存。这是一种取舍：牺牲几个百分点的解码速度，换取峰值内存减少约 40 GiB（这些表为 29.8 GiB）；128 GB 的 Mac 需要这样做",
    "don't ask GitHub whether a newer release exists (also TENSORFOLD_NO_UPDATE_CHECK=1)":
        "不向 GitHub 查询是否有更新的版本（也可设置 TENSORFOLD_NO_UPDATE_CHECK=1）",
    # serve: the NVIDIA GPUs (DGX Spark) group
    "NVIDIA GPUs (DGX Spark)": "NVIDIA GPU（DGX Spark）",
    "auto: MLX on macOS, CUDA elsewhere": "auto：macOS 上使用 MLX，其他平台使用 CUDA",
    "GPUs (one per machine) the model is split over; run the same command on each":
        "模型拆分到的 GPU 数（每台机器一个）；在每台机器上运行相同命令",
    "with --tp 2: this machine's rank; rank 0 serves HTTP, rank 1 follows it":
        "配合 --tp 2 时：本机的 rank；rank 0 提供 HTTP 服务，rank 1 跟随它",
    "with --tp 2: rank 0's address on the link between the machines":
        "配合 --tp 2 时：机器之间链路上 rank 0 的地址",
    "with --tp 2: rank 0's rendezvous port": "配合 --tp 2 时：rank 0 的会合端口",
    "KV cache: bf16 (the default), int8, or int4. Quantized keys and values use one fp16 scale per 32 values (changes the output; Flash Next on CUDA only)":
        "KV 缓存：bf16（默认）、int8 或 int4。量化后的 key 与 value 每 32 个值共用一个 fp16 缩放因子（会改变输出；仅限 CUDA 上的 Flash Next）",
    "prompt matmuls take FP8 (e4m3) activations, one scale a row, where the checkpoint has an FP8 prompt kernel (Qwen3.8 27B and Qwen3.6 MLX 4-bit, NVFP4 checkpoints' FP8 and MXFP8 layers): faster prompts, lower precision (e4m3 keeps 3 mantissa bits, bf16 keeps 7; docs/recipes/cuda.md#prompt-precision has the measured cost). Default: bf16 activations. Replies equal this server's own serial decoding either way":
        "提示词矩阵乘使用 FP8（e4m3）激活，每行一个缩放因子，前提是检查点带有 FP8 提示词内核（Qwen3.8 27B 与 Qwen3.6 MLX 4-bit，以及 NVFP4 检查点的 FP8 和 MXFP8 层）：提示词更快，精度更低（e4m3 保留 3 个尾数位，bf16 保留 7 个；实测开销见 docs/recipes/cuda.md#prompt-precision）。默认：bf16 激活。无论哪种方式，回复都等同于本服务器自身的串行解码",
    "the math for checkpoints that name their activations' formats (NVFP4): checkpoint, the default, runs their own math as their runtimes do (FP4 x FP4 in NVFP4 layers on SM 12.x GPUs, FP8 x FP8 in FP8 layers from SM 8.9, under the checkpoint's static input scales; layers a GPU has no mma for run W4A16, and the startup line says which); full runs bf16 activations against the stored weights exactly. The weights never change, only the math; MLX checkpoints have one math. Replies equal this server's own serial decoding either way":
        "针对会声明自身激活格式的检查点（NVFP4）的运算方式：checkpoint（默认）像其运行时那样执行检查点自身的运算（在 SM 12.x GPU 的 NVFP4 层中为 FP4 x FP4，从 SM 8.9 起的 FP8 层中为 FP8 x FP8，均遵循检查点的静态输入缩放；GPU 没有对应 mma 的层改用 W4A16 运行，启动信息会说明是哪些层）；full 则用 bf16 激活严格针对存储的权重做运算。权重从不改变，改变的只是运算方式；MLX 检查点只有一种运算。无论哪种方式，回复都等同于本服务器自身的串行解码",
    # pull, models, update and info
    "download models (or draft models) from Hugging Face": "从 Hugging Face 下载模型（或草稿模型）",
    "repo ids, e.g. TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP":
        "仓库 id，例如 TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP",
    "list the model families and the checkpoints they are tested with":
        "列出模型家族及与其一同测试的检查点",
    "install the newest TensorFold release from GitHub": "从 GitHub 安装最新的 TensorFold 版本",
    "only say whether a newer release exists": "只说明是否存在更新的版本",
    "reinstall the newest release even when it is current": "即使已是最新版本，也重新安装",
    "show which family serves a model (reads its config.json only)":
        "显示某个模型由哪个家族提供（仅读取其 config.json）",
    "a Hugging Face repo id or a model directory": "Hugging Face 仓库 id 或模型目录",
    # plan
    "estimate local checkpoint weights against MLX budgets without loading a model":
        "不加载模型，对照 MLX 内存预算估算本地检查点权重",
    "a local model directory or already cached Hugging Face repo id":
        "本地模型目录或已缓存的 Hugging Face 仓库 id",
    "also check this explicit budget, as TENSORFOLD_MEMORY_LIMIT_GB would set it":
        "同时检查这个显式预算，效果同 TENSORFOLD_MEMORY_LIMIT_GB 的设置",
    "also estimate this RAM class under the current GPU ceiling (repeatable)":
        "同时在当前 GPU 上限下估算这个 RAM 档位（可重复）",
}

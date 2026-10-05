"""Simplified Chinese for the control plane: ``tensorfold service``, the ``tui`` command and profile validation."""

MESSAGES = {
    # service, tui and the control-parser help
    "manage per-user macOS launchd services": "管理每用户的 macOS launchd 服务",
    "install a private LaunchAgent (starts next login; --start for now)":
        "安装私有 LaunchAgent（下次登录时启动；加 --start 立即启动）",
    "cached Hugging Face model ID or absolute local model directory":
        "已缓存的 Hugging Face 模型 ID，或本地模型目录的绝对路径",
    "profile name; lowercase letters, digits, hyphens": "配置名；小写字母、数字、连字符",
    "absolute serving-venv Python path": "服务 venv 中 Python 的绝对路径",
    "serve --parallel value written on the service command (default auto)":
        "写入服务命令的 serve --parallel 取值（默认 auto）",
    "extra literal serve argument, e.g. --arg=--vision": "额外的 serve 字面参数，例如 --arg=--vision",
    "non-secret override only": "仅限非机密覆盖项",
    "absolute private JSON file for credentials/overrides; mode 0600":
        "存放凭据/覆盖项的私有 JSON 绝对路径，权限 0600",
    "restricted API key file passed to tensorfold serve": "传给 tensorfold serve 的受限 API 密钥文件",
    "acknowledge unauthenticated non-loopback binding": "确认允许未认证的非回环绑定",
    "allow model downloads when service starts": "允许服务启动时下载模型",
    "replace a stopped, owned profile": "替换已停止且属于本用户的配置",
    "start now as well as at login": "立即启动，并在登录时启动",
    "print plist without writing files or calling launchctl": "只打印 plist，不写文件、不调用 launchctl",
    "confirm removal; logs and models are retained": "确认移除；日志与模型保留",
    "list owned profiles without loading any models": "列出本用户的配置，不加载任何模型",
    "TensorFold terminal control room (install the tui extra)":
        "TensorFold 终端控制台（需安装 tui extra）",
    "initial local profile": "初始本地配置",
    "read-only HTTP(S) endpoint; repeat for more": "只读 HTTP(S) 端点；可重复指定",
    "environment variable with API token, never saved or put in URLs":
        "存放 API token 的环境变量名，绝不保存或写入 URL",
    "poll interval in seconds, 0.5–30": "轮询间隔（秒），0.5–30",
    "simulated preview; no network or service operations": "模拟预览；不进行网络或服务操作",
    "write one .svg/.html/.txt frame instead of opening a terminal":
        "写出一帧 .svg/.html/.txt，而不打开终端",
    "snapshot width": "快照宽度",
    "snapshot height": "快照高度",
    "TensorFold service and terminal control plane": "TensorFold 服务与终端控制平面",
    # service command output and doctor
    "No TensorFold services installed. Use tensorfold service install MODEL.":
        "尚未安装任何 TensorFold 服务。请使用 tensorfold service install MODEL。",
    "macOS user session": "macOS 用户会话",
    "launchd query": "launchd 查询",
    "private environment": "私有环境变量",
    "serving interpreter imports": "服务解释器可导入",
    "install TensorFold and control in this venv": "请在此 venv 中安装 TensorFold 与 control",
    "interpreter unavailable": "解释器不可用",
    "HTTP readiness": "HTTP 就绪状态",
    "query error: {error}": "查询出错：{error}",
    "--lines must be between 1 and 2000": "--lines 必须在 1 到 2000 之间",
    "--json and --follow cannot be combined": "--json 与 --follow 不能同时使用",
    "uninstall requires --yes; logs and model files will be retained":
        "卸载需要 --yes；日志与模型文件将保留",
    "tensorfold service: {error}": "tensorfold service：{error}",
    "tensorfold tui needs {missing}.": "tensorfold tui 需要 {missing}。",
    "--token-env must name an environment variable": "--token-env 必须指定一个环境变量",
    "the requested token environment variable is unset or empty":
        "所请求的 token 环境变量未设置或为空",
    "snapshot dimensions must be 72–240 by 23–100": "快照尺寸必须为 72–240 × 23–100",
    "snapshot must end in .svg, .html or .txt": "快照文件名必须以 .svg、.html 或 .txt 结尾",
    "Saved {path}": "已保存 {path}",
    "interactive TUI needs a terminal; use --demo --snapshot preview.svg for an offline preview":
        "交互式 TUI 需要终端；如需离线预览，请使用 --demo --snapshot preview.svg",
    "tensorfold tui: {error}": "tensorfold tui：{error}",
    "context must be nonnegative": "context 必须为非负数",
    "--env needs unique KEY=VALUE assignments": "--env 需要唯一的 KEY=VALUE 赋值",
    # profile and environment validation
    "profile name: 1–48 lowercase letters, digits or hyphens; start with a letter":
        "配置名：1–48 个小写字母、数字或连字符，且以字母开头",
    "profile name {name} is reserved": "配置名 {name} 为保留名称",
    "{label} must be nonempty text without control characters (max {maximum})":
        "{label} 必须为非空且不含控制字符的文本（最多 {maximum} 个字符）",
    "model may not start with a dash": "model 不能以短横线开头",
    "python must be an absolute path to the serving virtual environment":
        "python 必须是指向服务虚拟环境的绝对路径",
    "host must be a numeric IPv4 or IPv6 address": "host 必须是 IPv4 或 IPv6 数字地址",
    "non-loopback binding needs --allow-network; put authentication in front of it":
        "绑定非回环地址需要 --allow-network；并请在前面配置身份验证",
    "port must be an integer from 1024 through 65535": "port 必须是 1024 到 65535 的整数",
    "macOS LaunchAgents support backend mlx or auto; remote CUDA is monitor-only":
        "macOS LaunchAgent 仅支持 backend mlx 或 auto；远程 CUDA 只能监控",
    "unsupported profile schema (expected 1)": "不支持的配置 schema（应为 1）",
    "allow_network and allow_download must be booleans": "allow_network 与 allow_download 必须为布尔值",
    "args must be a list of at most 128 literal arguments": "args 必须是最多 128 个字面参数的列表",
    "{option} is managed by the profile; do not put it in --arg": "{option} 由配置管理；请勿放入 --arg",
    "secret-bearing flags belong in a private environment file": "含密钥的选项应放在私有环境变量文件中",
    "environment_file must be absolute": "environment_file 必须为绝对路径",
    "api_key_file must be absolute": "api_key_file 必须为绝对路径",
    "log_bytes must be 64 KiB through 64 MiB": "log_bytes 必须为 64 KiB 到 64 MiB",
    "log_backups must be 1 through 10": "log_backups 必须为 1 到 10",
    "invalid service profile: {error}": "无效的服务配置：{error}",
    "environment must be an object with at most 64 entries": "environment 必须是至多 64 项的 object",
    "environment names must be uppercase identifiers": "环境变量名必须是大写标识符",
    "environment override not allowed: {name}": "不允许覆盖该环境变量：{name}",
    "{name} must be in --env-file, not the profile": "{name} 必须放在 --env-file 中，不能写在配置里",
    "environment file must be a private JSON object": "环境变量文件必须是私有 JSON object",
    "no installed profile: {name}": "没有已安装的配置：{name}",
    "profile name disagrees with filename: {path}": "配置名与文件名不一致：{path}",
    # private files and locks
    "refusing symlink: {path}": "拒绝符号链接：{path}",
    "directory is not owned by this user: {path}": "目录不属于当前用户：{path}",
    "not a regular file: {path}": "不是普通文件：{path}",
    "file is not owned by this user: {path}": "文件不属于当前用户：{path}",
    "private file must be mode 0600: {path}": "私有文件权限必须为 0600：{path}",
    "file exceeds {maximum} bytes: {path}": "文件超过 {maximum} 字节：{path}",
    "invalid service lock": "服务锁无效",
    "another TensorFold service operation is in progress": "另一个 TensorFold 服务操作正在进行中",
    # logs
    "log is not a regular file": "日志不是普通文件",
    "— log rotated / truncated —": "— 日志已轮转 / 截断 —",
    "— log burst truncated to keep the UI responsive —": "— 日志量过大已截断，以保持界面响应 —",
    # launchd
    "launchctl could not complete: {error}": "launchctl 无法完成：{error}",
    "launchd control is macOS-only; use --url for read-only remote monitoring":
        "launchd 控制仅限 macOS；请用 --url 进行只读远程监控",
    "run as your logged-in user, not root or sudo": "请以登录用户身份运行，不要用 root 或 sudo",
    "launchctl {verb} failed ({code}): {detail}": "launchctl {verb} 失败（{code}）：{detail}",
    "launchd requires macOS": "launchd 仅适用于 macOS",
    "cannot inspect {label}: {detail}": "无法检查 {label}：{detail}",
    "managed plist is malformed; refusing to modify it": "受管理的 plist 格式错误；拒绝修改",
    "plist differs from the managed profile; refusing to overwrite or control it":
        "plist 与受管理的配置不一致；拒绝覆盖或控制",
    "loaded label has a different or unknown plist path; refusing to control it":
        "已加载标签的 plist 路径不同或未知；拒绝控制",
    "the profile's Python interpreter does not exist or is not executable":
        "配置中的 Python 解释器不存在或不可执行",
    "profile or plist already exists; stop it, then use install --replace":
        "配置或 plist 已存在；请先停止它，再使用 install --replace",
    "stop the service before replacing its profile": "替换配置前请先停止服务",
    "this label is already loaded without a managed profile; refusing installation":
        "该标签已加载但没有受管理的配置；拒绝安装",
    "fix malformed profiles before installing: {errors}": "安装前请先修复格式错误的配置：{errors}",
    "port {port} is already reserved by another TensorFold profile":
        "端口 {port} 已被另一个 TensorFold 配置占用",
    "starts at next login, or with service start": "将在下次登录时启动，也可用 service start 启动",
    "service is still unloading; restart was not attempted": "服务仍在卸载中；未尝试重启",
    "disabled until explicitly started": "已禁用，直到显式启动",
    "logs retained; no model files were removed": "日志已保留；未删除任何模型文件",
    # telemetry reads
    "endpoint must be an HTTP(S) URL without whitespace": "端点必须是不含空白字符的 HTTP(S) URL",
    "invalid endpoint URL": "端点 URL 无效",
    "endpoint must be HTTP(S), without embedded credentials": "端点必须是 HTTP(S)，且不能内嵌凭据",
    "endpoint must not contain a query, fragment, or invalid port":
        "端点不能包含查询、片段或无效端口",
    "HTTP timeout must be > 0 and <= 30 seconds": "HTTP 超时必须大于 0 且不超过 30 秒",
    "invalid API token": "API token 无效",
    "telemetry body deadline exceeded": "遥测响应体超时",
    "telemetry response exceeds 1 MiB": "遥测响应超过 1 MiB",
    "endpoint unreachable ({error})": "无法连接端点（{error}）",
    "HTTP {status}; check --token-env": "HTTP {status}；请检查 --token-env",
    "health must be an object": "health 必须是 object",
    "health HTTP {status}": "健康检查 HTTP {status}",
    "metrics unavailable (HTTP {status})": "metrics 不可用（HTTP {status}）",

    "[control] log unavailable: {error}": "[control] 日志不可用：{error}",
    "serve argument": "serve 参数",
    "expected object": "应为 object",
}

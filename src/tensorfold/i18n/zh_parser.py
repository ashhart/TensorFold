"""Simplified Chinese for the argparse framework itself: group titles, the usage prefix and usage errors."""

MESSAGES = {
    "usage: ": "用法：",
    "error:": "错误：",
    "{prog}: error: {detail}": "{prog}：错误：{detail}",
    "positional arguments": "位置参数",
    "options": "选项",
    "show this help message and exit": "显示此帮助信息并退出",
    "show program's version number and exit": "显示程序版本号并退出",
    " (default: {value})": "（默认：{value}）",
    "argument {name}: ": "参数 {name}：",
    "invalid {kind} value: {value}": "{kind} 值无效：{value}",
    "invalid choice: {value} (choose from {choices})": "选择无效：{value}（可选：{choices}）",
    "expected one argument": "需要一个值",
    "expected at most one argument": "最多需要一个值",
    "expected {count} argument(s)": "需要 {count} 个参数",
    "not allowed with argument {other}": "不能与参数 {other} 同时使用",
    "the following arguments are required: {names}": "缺少必需参数：{names}",
    "unrecognized arguments: {names}": "无法识别的参数：{names}",
    "ambiguous option: {option} could match {matches}": "选项含义不明确：{option} 可能匹配 {matches}",
    "ignored explicit argument {value}": "忽略了显式参数 {value}",
}

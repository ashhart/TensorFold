//! Cortheon's checks before a reply ends: changed files tested before "done", researched answers naming sources.
const std = @import("std");
const json = @import("json");
const tools = @import("tools.zig");
const Value = json.Value;

pub const Check = struct { name: []const u8, note: []const u8 };

pub const tests_note = "Cortheon check: files were changed after the last test or build in this task. Run the relevant test, build or program with your tools now and report what it showed. If nothing can run it, say so plainly.";
pub const sources_note = "Cortheon check: end with a short Sources list of the URLs your answer relies on. Do not repeat the answer.";

const Act = enum { edit, verify, other };

const verify_names = [_][]const u8{ "test", "check", "lint", "build", "compile", "verify" };
const edit_names = [_][]const u8{ "edit", "write", "patch", "apply", "replace", "create", "insert", "save", "str_replace", "multiedit", "delete", "rename" };
const run_names = [_][]const u8{ "bash", "shell", "exec", "terminal", "run", "command", "cmd", "powershell", "sh" };
const verify_commands = [_][]const u8{ "test", "pytest", "jest", "vitest", "mocha", "cargo ", "go build", "go run", "go vet", "zig build", "zig run", "npm ", "pnpm ", "yarn ", "bun ", "make", "ctest", "tsc", "mvn ", "gradle", "swift ", "xcodebuild", "python", "node ", "deno ", "ruff", "eslint", "mypy", "rspec", "dotnet ", "./" };
const edit_commands = [_][]const u8{ "sed -i", " > ", ">>", "tee ", "git apply", "patch ", "mv ", "cp ", "rm ", "cat >", "touch " };

fn holds(text: []const u8, words: []const []const u8) bool {
    for (words) |w| if (std.ascii.findIgnoreCase(text, w) != null) return true;
    return false;
}

/// What a harness tool call did, read from its name and, for shell tools, its command.
fn act(call: Value) Act {
    const name = tools.callName(call);
    if (holds(name, &verify_names)) return .verify;
    if (holds(name, &edit_names)) return .edit;
    if (!holds(name, &run_names)) return .other;
    const f = call.get("function") orelse return .other;
    const args = f.get("arguments") orelse return .other;
    const text = switch (args) {
        .string => |s| s,
        .object => |o| if (o.count() > 0) (if (o.values()[0] == .string) o.values()[0].string else "") else "",
        else => "",
    };
    if (holds(text, &edit_commands)) return .edit;
    if (holds(text, &verify_commands)) return .verify;
    return .other;
}

/// Whether the work since the latest user message changed a file after its last test, build or run.
pub fn untested(messages: []const Value) bool {
    var start = messages.len;
    while (start > 0) {
        start -= 1;
        if (std.mem.eql(u8, messages[start].strField("role") orelse "", "user")) break;
    }
    var last_edit: ?usize = null;
    var last_verify: ?usize = null;
    var n: usize = 0;
    for (messages[start..]) |m| {
        const calls = m.get("tool_calls") orelse continue;
        if (calls != .array) continue;
        for (calls.array) |c| {
            n += 1;
            switch (act(c)) {
                .edit => last_edit = n,
                .verify => last_verify = n,
                .other => {},
            }
        }
    }
    const e = last_edit orelse return false;
    return if (last_verify) |v| e > v else true;
}

/// Whether `text` names a web source.
pub fn cites(text: []const u8) bool {
    return std.mem.indexOf(u8, text, "http://") != null or std.mem.indexOf(u8, text, "https://") != null;
}

test "an edit after the last test is untested; a test after it settles it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const edited = (try json.parseText(a,
        \\[{"role": "user", "content": "fix it"},
        \\ {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "bash", "arguments": "{\"command\": \"pytest -q\"}"}}]},
        \\ {"role": "tool", "content": "1 failed"},
        \\ {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "edit", "arguments": "{\"path\": \"a.py\"}"}}]},
        \\ {"role": "tool", "content": "ok"}]
    )).ok;
    try std.testing.expect(untested(edited.array));
    const tested = (try json.parseText(a,
        \\[{"role": "user", "content": "fix it"},
        \\ {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "write_file", "arguments": {"path": "a.py"}}}]},
        \\ {"role": "tool", "content": "ok"},
        \\ {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "zig build test"}}}]},
        \\ {"role": "tool", "content": "pass"}]
    )).ok;
    try std.testing.expect(!untested(tested.array));
    const read_only = (try json.parseText(a, "[{\"role\": \"user\", \"content\": \"explain\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"function\": {\"name\": \"read\", \"arguments\": {}}}]}]")).ok;
    try std.testing.expect(!untested(read_only.array));
    const shell_edit = (try json.parseText(a, "[{\"role\": \"user\", \"content\": \"x\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"function\": {\"name\": \"bash\", \"arguments\": {\"command\": \"sed -i s/a/b/ f.c\"}}}]}]")).ok;
    try std.testing.expect(untested(shell_edit.array));
    try std.testing.expect(cites("see https://ziglang.org") and !cites("no links"));
}

// Port of DeepSeek's encoding_dsv4.py, as vendored by TensorFold, and its
// DeepSeekTokenizer wrapper. Keep prompt bytes aligned with that encoder.
const std = @import("std");
const V = std.json.Value;
const A = std.mem.Allocator;
const W = std.Io.Writer;

pub const assistant = "<｜Assistant｜>";
const think_start = "<think>";
const think_end = "</think>";
const maximum_effort =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n" ++
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n" ++
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
const tools_intro =
    \\## Tools
    \\
    \\You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:
    \\
    \\<｜DSML｜tool_calls>
    \\<｜DSML｜invoke name="$TOOL_NAME">
    \\<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
    \\...
    \\</｜DSML｜invoke>
    \\<｜DSML｜invoke name="$TOOL_NAME2">
    \\...
    \\</｜DSML｜invoke>
    \\</｜DSML｜tool_calls>
    \\
    \\String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.
    \\
    \\If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.
    \\
    \\Otherwise, output directly after </think> with tool calls or final response.
    \\
    \\### Available Tool Schemas
    \\
    \\
;

fn get(v: V, key: []const u8) V {
    return if (v == .object) v.object.get(key) orelse .null else .null;
}

fn is(v: V, s: []const u8) bool {
    return v == .string and std.mem.eql(u8, v.string, s);
}

fn truth(v: V) bool {
    return switch (v) {
        .null => false,
        .bool => v.bool,
        .string, .number_string => |s| s.len > 0,
        .integer => v.integer != 0,
        .float => v.float != 0,
        .array => v.array.items.len > 0,
        .object => v.object.count() > 0,
    };
}

fn content(w: *W, v: V) !void {
    if (!truth(v)) return;
    if (v != .string) return error.InvalidContent;
    try w.writeAll(v.string);
}

// Python json.dumps uses spaces, preserves Unicode and retains a decimal point
// on integral floats. Its scientific notation threshold differs from Zig's.
fn json(w: *W, v: V) anyerror!void {
    switch (v) {
        .object => |obj| {
            try w.writeByte('{');
            var it = obj.iterator();
            var n: usize = 0;
            while (it.next()) |entry| : (n += 1) {
                if (n > 0) try w.writeAll(", ");
                try std.json.Stringify.value(entry.key_ptr.*, .{}, w);
                try w.writeAll(": ");
                try json(w, entry.value_ptr.*);
            }
            try w.writeByte('}');
        },
        .array => |arr| {
            try w.writeByte('[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try w.writeAll(", ");
                try json(w, item);
            }
            try w.writeByte(']');
        },
        .float => |f| {
            if (!std.math.isFinite(f)) return w.writeAll(if (std.math.isNan(f)) "NaN" else if (f < 0) "-Infinity" else "Infinity");
            var buf: [512]u8 = undefined;
            const sci = try std.fmt.bufPrint(&buf, "{e}", .{f});
            const at = std.mem.indexOfScalar(u8, sci, 'e') orelse return error.InvalidFloat;
            const exponent = try std.fmt.parseInt(i32, sci[at + 1 ..], 10);
            if (exponent >= -4 and exponent < 16) {
                var decimal: [512]u8 = undefined;
                const s = try std.fmt.bufPrint(&decimal, "{d}", .{f});
                try w.writeAll(s);
                if (std.mem.indexOfScalar(u8, s, '.') == null) try w.writeAll(".0");
            } else {
                try w.writeAll(sci[0..at]);
                try w.print("e{c}{d:0>2}", .{ @as(u8, if (exponent < 0) '-' else '+'), @abs(exponent) });
            }
        },
        else => try std.json.Stringify.value(v, .{}, w),
    }
}

fn merge(a: A, input: []const V) ![]V {
    var out: std.ArrayList(V) = .empty;
    for (input) |msg| {
        if (msg != .object) return error.InvalidMessage;
        const role = get(msg, "role");
        const tool = is(role, "tool");
        if (!tool and !is(role, "user")) {
            try out.append(a, msg);
            continue;
        }
        var block = V{ .object = .empty };
        try block.object.put(a, "type", .{ .string = if (tool) "tool_result" else "text" });
        try block.object.put(a, if (tool) "content" else "text", msg.object.get("content") orelse .{ .string = "" });
        if (tool) try block.object.put(a, "tool_use_id", get(msg, "tool_call_id"));
        if (out.items.len > 0) {
            const prev = &out.items[out.items.len - 1];
            if (is(get(prev.*, "role"), "user") and (tool or get(prev.*, "task") == .null)) {
                if (prev.object.getPtr("content_blocks")) |blocks| {
                    try blocks.array.append(block);
                    continue;
                }
            }
        }
        var item = V{ .object = .empty };
        try item.object.put(a, "role", .{ .string = "user" });
        var blocks = V{ .array = std.json.Array.init(a) };
        try blocks.array.append(block);
        try item.object.put(a, "content_blocks", blocks);
        if (!tool) {
            try item.object.put(a, "content", get(msg, "content"));
            for ([_][]const u8{ "task", "wo_eos", "mask" }) |key| if (msg.object.get(key)) |value| try item.object.put(a, key, value);
        }
        try out.append(a, item);
    }
    var order: std.StringHashMapUnmanaged(usize) = .empty;
    for (out.items) |*msg| {
        const calls = get(msg.*, "tool_calls");
        if (is(get(msg.*, "role"), "assistant") and truth(calls)) {
            if (calls != .array) return error.InvalidToolCall;
            order.clearRetainingCapacity();
            for (calls.array.items, 0..) |call, i| {
                var id = get(call, "id");
                if (!truth(id)) id = get(get(call, "function"), "id");
                if (truth(id)) {
                    if (id != .string) return error.InvalidToolCall;
                    try order.put(a, id.string, i);
                }
            }
        } else if (is(get(msg.*, "role"), "user") and order.count() > 0) {
            const blocks = msg.object.getPtr("content_blocks") orelse continue;
            var results: std.ArrayList(V) = .empty;
            for (blocks.array.items) |b| if (is(get(b, "type"), "tool_result")) try results.append(a, b);
            std.mem.sort(V, results.items, order, struct {
                fn less(map: std.StringHashMapUnmanaged(usize), lhs: V, rhs: V) bool {
                    const l = get(lhs, "tool_use_id");
                    const r = get(rhs, "tool_use_id");
                    return (map.get(if (l == .string) l.string else "") orelse 0) < (map.get(if (r == .string) r.string else "") orelse 0);
                }
            }.less);
            var i: usize = 0;
            for (blocks.array.items) |*b| if (is(get(b.*, "type"), "tool_result")) {
                b.* = results.items[i];
                i += 1;
            };
        }
    }
    return out.items;
}

fn lastUser(messages: []const V) ?usize {
    var last: ?usize = null;
    for (messages, 0..) |m, i| if (is(get(m, "role"), "user") or is(get(m, "role"), "developer")) {
        last = i;
    };
    return last;
}

fn parameter(w: *W, key: []const u8, value: V) !void {
    try w.print("<｜DSML｜parameter name=\"{s}\" string=\"{s}\">", .{ key, if (value == .string) "true" else "false" });
    if (value == .string) try w.writeAll(value.string) else try json(w, value);
    try w.writeAll("</｜DSML｜parameter>");
}

fn callArguments(a: A, w: *W, args: V) !void {
    const parsed: ?V = if (args == .string) blk: {
        const p = std.json.parseFromSlice(V, a, args.string, .{}) catch |err| {
            if (err == error.OutOfMemory) return err;
            break :blk null;
        };
        break :blk p.value;
    } else null;
    if (parsed) |value| {
        if (value != .object) return error.InvalidToolArguments;
        var it = value.object.iterator();
        var n: usize = 0;
        while (it.next()) |entry| : (n += 1) {
            if (n > 0) try w.writeByte('\n');
            try parameter(w, entry.key_ptr.*, entry.value_ptr.*);
        }
    } else try parameter(w, "arguments", args);
}

fn renderMessage(a: A, w: *W, messages: []const V, index: usize, thinking: bool, drop: bool, last: ?usize) !void {
    const msg = messages[index];
    const role = get(msg, "role");
    if (is(role, "system") or is(role, "developer")) {
        if (is(role, "developer")) {
            if (!truth(get(msg, "content"))) return error.InvalidContent;
            try w.writeAll("<｜User｜>");
        }
        try content(w, get(msg, "content"));
        const tools = get(msg, "tools");
        if (truth(tools)) {
            if (tools != .array) return error.InvalidTools;
            try w.writeAll("\n\n" ++ tools_intro);
            for (tools.array.items, 0..) |tool, i| {
                if (i > 0) try w.writeByte('\n');
                const function = get(tool, "function");
                if (function == .null) return error.InvalidTools;
                try json(w, function);
            }
            try w.writeAll("\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n");
        }
        const format = get(msg, "response_format");
        if (truth(format)) {
            try w.writeAll("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
            try json(w, format);
        }
    } else if (is(role, "user")) {
        try w.writeAll("<｜User｜>");
        const blocks = get(msg, "content_blocks");
        for (blocks.array.items, 0..) |block, i| {
            if (i > 0) try w.writeAll("\n\n");
            if (is(get(block, "type"), "text")) {
                try content(w, get(block, "text"));
            } else {
                try w.writeAll("<tool_result>");
                const value = get(block, "content");
                if (value == .array) {
                    for (value.array.items, 0..) |part, j| {
                        if (j > 0) try w.writeAll("\n\n");
                        if (is(get(part, "type"), "text")) try content(w, get(part, "text")) else {
                            const kind = get(part, "type");
                            try w.print("[Unsupported {s}]", .{if (kind == .string) kind.string else "None"});
                        }
                    }
                } else try content(w, value);
                try w.writeAll("</tool_result>");
            }
        }
    } else if (is(role, "latest_reminder")) {
        try w.writeAll("<｜latest_reminder｜>");
        try content(w, get(msg, "content"));
    } else if (is(role, "assistant")) {
        const prev_task = index > 0 and get(messages[index - 1], "task") != .null;
        if (thinking and !prev_task and (!drop or last == null or index > last.?)) {
            try content(w, get(msg, "reasoning_content"));
            try w.writeAll(think_end);
        }
        try content(w, get(msg, "content"));
        const calls = get(msg, "tool_calls");
        if (truth(calls)) {
            if (calls != .array) return error.InvalidToolCall;
            try w.writeAll("\n\n<｜DSML｜tool_calls>\n");
            for (calls.array.items, 0..) |call, i| {
                if (i > 0) try w.writeByte('\n');
                const function = get(call, "function");
                const name = get(function, "name");
                if (name != .string) return error.InvalidToolCall;
                try w.print("<｜DSML｜invoke name=\"{s}\">\n", .{name.string});
                try callArguments(a, w, get(function, "arguments"));
                try w.writeAll("\n</｜DSML｜invoke>");
            }
            try w.writeAll("\n</｜DSML｜tool_calls>");
        }
        if (!truth(get(msg, "wo_eos"))) try w.writeAll("<｜end▁of▁sentence｜>");
    } else return error.InvalidRole;
    if (index + 1 < messages.len) {
        const next_role = get(messages[index + 1], "role");
        if (!is(next_role, "assistant") and !is(next_role, "latest_reminder")) return;
    }
    const task = get(msg, "task");
    if (task != .null) {
        var valid = false;
        for ([_][]const u8{ "action", "query", "authority", "domain", "title", "read_url" }) |name| valid = valid or is(task, name);
        if (!valid) return error.InvalidTask;
        if (is(task, "action")) try w.writeAll(if (thinking) assistant ++ think_start else assistant ++ think_end);
        try w.print("<｜{s}｜>", .{task.string});
    } else if (is(role, "user") or is(role, "developer")) {
        try w.writeAll(assistant);
        try w.writeAll(if (thinking and (!drop or last == null or index >= last.?)) think_start else think_end);
    }
}

pub fn render(a: A, input: V, tools: V, thinking: bool, effort: ?[]const u8, generation: bool) ![]u8 {
    if (input != .array) return error.InvalidMessages;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const temp = arena.allocator();
    var messages: std.ArrayList(V) = .empty;
    try messages.appendSlice(temp, input.array.items);
    if (truth(tools)) {
        if (tools != .array) return error.InvalidTools;
        if (messages.items.len == 0 or !is(get(messages.items[0], "role"), "system")) {
            var system = V{ .object = .empty };
            try system.object.put(temp, "role", .{ .string = "system" });
            try messages.insert(temp, 0, system);
        }
        var first = V{ .object = try messages.items[0].object.clone(temp) };
        var wrapped = V{ .array = std.json.Array.init(temp) };
        for (tools.array.items) |tool| {
            if (tool != .object) return error.InvalidTools;
            if (tool.object.contains("function")) {
                try wrapped.array.append(tool);
            } else {
                var entry = V{ .object = .empty };
                try entry.object.put(temp, "type", .{ .string = "function" });
                try entry.object.put(temp, "function", tool);
                try wrapped.array.append(entry);
            }
        }
        try first.object.put(temp, "tools", wrapped);
        messages.items[0] = first;
    }
    const merged = try merge(temp, messages.items);
    var drop = true;
    for (merged) |msg| if (truth(get(msg, "tools"))) {
        drop = false;
    };
    var filtered: std.ArrayList(V) = .empty;
    const last = lastUser(merged);
    for (merged, 0..) |msg, i| {
        const role = get(msg, "role");
        if (thinking and drop and last != null and i < last.? and !is(role, "user") and !is(role, "system") and !is(role, "assistant") and !is(role, "tool") and !is(role, "latest_reminder") and !is(role, "direct_search_results")) continue;
        try filtered.append(temp, msg);
    }
    var out: W.Allocating = .init(a);
    defer out.deinit();
    const w = &out.writer;
    w.writeAll("<｜begin▁of▁sentence｜>") catch return error.OutOfMemory;
    if (thinking and effort != null and (std.mem.eql(u8, effort.?, "max") or std.mem.eql(u8, effort.?, "xhigh")) and filtered.items.len > 0) w.writeAll(maximum_effort) catch return error.OutOfMemory;
    const filtered_last = lastUser(filtered.items);
    for (filtered.items, 0..) |_, i| renderMessage(temp, w, filtered.items, i, thinking, drop, filtered_last) catch |err| return if (err == error.WriteFailed) error.OutOfMemory else err;
    const prefix = if (thinking) assistant ++ think_start else assistant ++ think_end;
    const ended_assistant = input.array.items.len > 0 and is(get(input.array.items[input.array.items.len - 1], "role"), "assistant");
    if (generation and ended_assistant) w.writeAll(prefix) catch return error.OutOfMemory else if (!generation and !ended_assistant and std.mem.endsWith(u8, out.written(), prefix)) out.shrinkRetainingCapacity(out.written().len - prefix.len);
    return out.toOwnedSlice();
}

fn allocationCheck(a: A) !void {
    const source =
        \\{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":null,"tool_calls":[{"id":"a","function":{"name":"f","arguments":"{\"n\":1.0}"}},{"id":"b","function":{"name":"f","arguments":{}}}]},{"role":"tool","tool_call_id":"b","content":"second"},{"role":"tool","tool_call_id":"a","content":"first"}],"tools":[{"name":"f","parameters":{"type":"object"}}]}
    ;
    const parsed = try std.json.parseFromSlice(V, a, source, .{});
    defer parsed.deinit();
    const result = try render(a, get(parsed.value, "messages"), get(parsed.value, "tools"), true, "xhigh", true);
    defer a.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "<tool_result>first</tool_result>\n\n<tool_result>second</tool_result>") != null);
    try std.testing.expect(std.mem.endsWith(u8, result, assistant ++ think_start));
    const unchanged = try std.json.Stringify.valueAlloc(a, parsed.value, .{});
    defer a.free(unchanged);
    try std.testing.expectEqualStrings(source, unchanged);
}

test "DeepSeek rendering owns scratch allocations and preserves caller messages" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCheck, .{});
}

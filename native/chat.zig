const std = @import("std");
const V = std.json.Value;
const Tokenizer = @import("vendor/tokenizer.zig").Tokenizer;
extern "c" fn jinja_render_chat([*:0]const u8, [*:0]const u8, ?[*:0]const u8, [*:0]const u8, c_int, *usize) ?[*]u8;
extern "c" fn jinja_str_free([*]u8) void;
extern "c" fn jinja_last_error() ?[*:0]const u8;

pub const Rendered = struct { text: []u8, images: V, thinking: bool };
pub const CallForm = struct { opener: []const u8, lead: ?[]const u8 = null, tail: []const u8 = "" };

pub fn encode(a: std.mem.Allocator, tokenizer: *Tokenizer, rendered: Rendered) ![]i32 {
    const tokens = try tokenizer.encode(a, rendered.text);
    defer a.free(tokens);
    var ids: std.ArrayList(i32) = .empty;
    errdefer ids.deinit(a);
    for (tokens) |id| try ids.append(a, @intCast(id));
    if (!rendered.thinking and tokens.len > 0) {
        const last = try tokenizer.decode(a, tokens[tokens.len - 1 ..], false);
        defer a.free(last);
        if (std.mem.eql(u8, std.mem.trim(u8, last, " \r\n\t"), "<think>")) {
            const close = try tokenizer.encode(a, "</think>");
            defer a.free(close);
            if (close.len == 1) try ids.append(a, @intCast(close[0]));
        }
    }
    return ids.toOwnedSlice(a);
}

pub fn check(io: std.Io, directory: []const u8, fixture: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var template = try Template.load(a, io, directory);
    defer template.deinit();
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, directory, a);
    var tokenizer = try @import("vendor/tokenizer.zig").loadTokenizer(io, a, path);
    defer tokenizer.deinit();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, fixture, a, .limited(16 * 1024 * 1024));
    const cases = try std.json.parseFromSlice(V, a, bytes, .{});
    for (cases.value.array.items, 0..) |case, index| {
        const rendered = try template.render(a, case.object.get("body").?);
        const ids = try encode(a, &tokenizer, rendered);
        const expected = case.object.get("tokens").?.array.items;
        var same = ids.len == expected.len;
        for (ids[0..@min(ids.len, expected.len)], expected[0..@min(ids.len, expected.len)], 0..) |id, want, at| if (id != want.integer) {
            std.debug.print("Chat fixture {d}, token {d}: native {d}, upstream {d}\n", .{ index, at, id, want.integer });
            same = false;
            break;
        };
        if (!same) {
            std.debug.print("Chat fixture {d}: native {d} tokens, upstream {d}\nRendered: {s}\n", .{ index, ids.len, expected.len, rendered.text });
            return error.ChatTemplateMismatch;
        }
        if (case.object.get("gates")) |gates| for (gates.array.items) |fixture_gate| {
            const form = (try template.callForm(a, &tokenizer)) orelse return error.MissingCallForm;
            var names: std.ArrayList([]const u8) = .empty;
            for (fixture_gate.object.get("names").?.array.items) |name| try names.append(a, name.string);
            var gate = try @import("call_gate.zig").Gate.init(a, &tokenizer, ids, form, names.items, std.mem.indexOf(u8, directory, "gemma") != null);
            var budget = try @import("thinking_budget.zig").Budget.init(a, &tokenizer, if (fixture_gate.object.get("budget")) |v| v.integer else 0);
            const required = if (fixture_gate.object.get("required")) |v| v.bool else true;
            for (fixture_gate.object.get("proposed").?.array.items, fixture_gate.object.get("expected").?.array.items, 0..) |proposed, expected_token, token_index| {
                const got = try budget.next(if (required) &gate else null, token_index, @intCast(proposed.integer), proposed.integer == fixture_gate.object.get("eos").?.integer);
                if (got != expected_token.integer) {
                    std.debug.print("Call gate fixture {d}, token {d}: native {d}, upstream {d}\n", .{ index, token_index, got, expected_token.integer });
                    return error.CallGateMismatch;
                }
            }
        };
    }
    std.debug.print("PASS: {d} chat prompts, thinking budgets and required-call gates match upstream token IDs\n", .{cases.value.array.items.len});
}
pub const Template = struct {
    arena: std.heap.ArenaAllocator,
    source: [:0]const u8,
    context: V,
    late_system: []const u8 = "system",

    pub fn load(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Template {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const owned = arena.allocator();
        const path = try std.fmt.allocPrint(owned, "{s}/tokenizer_config.json", .{dir});
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, owned, .limited(16 * 1024 * 1024));
        const config = try std.json.parseFromSlice(V, owned, bytes, .{});
        if (config.value != .object) return error.InvalidTokenizerConfig;
        var context = V{ .object = .empty };
        var it = config.value.object.iterator();
        while (it.next()) |entry| {
            if (!std.mem.endsWith(u8, entry.key_ptr.*, "_token")) continue;
            const value = entry.value_ptr.*;
            if (value == .string) try context.object.put(owned, entry.key_ptr.*, value) else if (value == .object) {
                if (value.object.get("content")) |content| if (content == .string) try context.object.put(owned, entry.key_ptr.*, content);
            }
        }
        const file = try std.fmt.allocPrint(owned, "{s}/chat_template.jinja", .{dir});
        const source = std.Io.Dir.cwd().readFileAlloc(io, file, owned, .limited(1024 * 1024)) catch |err| blk: {
            if (err != error.FileNotFound) return err;
            break :blk try templateSource(config.value.object.get("chat_template") orelse return error.MissingChatTemplate);
        };
        var result = Template{ .arena = undefined, .source = try numericMembers(owned, source), .context = context };
        const probe = try std.json.parseFromSlice(V, owned,
            \\[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"},{"role":"system","content":"tensorfold-late-system-probe"},{"role":"user","content":"v"}]
        , .{});
        const rendered = result.raw(owned, probe.value, .null, context, false, false) catch null;
        if (rendered == null or std.mem.indexOf(u8, rendered.?, "tensorfold-late-system-probe") == null) result.late_system = "user";
        result.arena = arena;
        return result;
    }
    pub fn deinit(t: *Template) void {
        t.arena.deinit();
    }
    pub fn callForm(t: *const Template, a: std.mem.Allocator, tokenizer: *Tokenizer) !?CallForm {
        const probe = try std.json.parseFromSlice(V, a,
            \\[{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[{"id":"call_0","type":"function","function":{"name":"tfprobe_fn","arguments":{}}}]}]
        , .{});
        var context = V{ .object = try t.context.object.clone(a) };
        try context.object.put(a, "enable_thinking", .{ .bool = false });
        try context.object.put(a, "thinking_mode", .{ .string = "chat" });
        const text = t.raw(a, probe.value, .null, context, false, false) catch "";
        const openers: []const []const u8 = &.{ "<tool_call>", "<|tool_call>", "<｜DSML｜tool_calls>" };
        var best: ?usize = null;
        var form: ?CallForm = null;
        if (std.mem.lastIndexOf(u8, text, "tfprobe_fn")) |name| for (openers) |opener| {
            if (std.mem.lastIndexOf(u8, text[0..name], opener)) |start| if (best == null or start > best.?) {
                best = start;
                const end = name + "tfprobe_fn".len;
                form = .{ .opener = opener, .lead = text[start + opener.len .. name], .tail = text[end..@min(end + 1, text.len)] };
            };
        };
        if (form != null) return form;
        for (openers) |opener| {
            const ids = try tokenizer.encode(a, opener);
            if (ids.len == 1) return .{ .opener = opener };
        }
        return null;
    }
    fn raw(t: *const Template, a: std.mem.Allocator, messages: V, tools: V, extra: V, generation: bool, report_error: bool) ![]u8 {
        const msg = try asJson(a, messages);
        const tool: ?[*:0]const u8 = if (tools == .null) null else (try asJson(a, tools)).ptr;
        const context = try asJson(a, extra);
        var len: usize = 0;
        const output = jinja_render_chat(t.source, msg, tool, context, @intFromBool(generation), &len) orelse {
            if (report_error) if (jinja_last_error()) |message| std.debug.print("Chat template: {s}\n", .{message});
            return error.ChatTemplateFailed;
        };
        defer jinja_str_free(output);
        return a.dupe(u8, output[0..len]);
    }
    pub fn render(t: *const Template, a: std.mem.Allocator, body: V) !Rendered {
        return t.renderWithDefaults(a, body, false, null);
    }

    pub fn renderWithDefaults(t: *const Template, a: std.mem.Allocator, body: V, default_thinking: bool, default_effort: ?[]const u8) !Rendered {
        if (body != .object) return error.InvalidRequest;
        const input = body.object.get("messages") orelse return error.MissingMessages;
        var images = V{ .array = std.json.Array.init(a) };
        const messages = try normalize(a, input, t.late_system, &images);
        var context = V{ .object = try t.context.object.clone(a) };
        var thinking = default_thinking;
        var effort = default_effort;
        const kwargs = body.object.get("chat_template_kwargs") orelse .null;
        var value = body.object.get("reasoning_effort") orelse .null;
        if (value == .null and kwargs == .object) value = kwargs.object.get("reasoning_effort") orelse .null;
        if (value != .null) {
            if (value != .string) return error.InvalidReasoningEffort;
            const choices = [_][]const u8{ "none", "minimal", "low", "medium", "high", "xhigh" };
            var valid = false;
            for (choices) |choice| valid = valid or std.mem.eql(u8, choice, value.string);
            if (!valid) return error.InvalidReasoningEffort;
            thinking = !std.mem.eql(u8, value.string, "none");
            effort = if (std.mem.eql(u8, value.string, "high")) "xhigh" else if (std.mem.eql(u8, value.string, "minimal")) "low" else value.string;
        }
        if (kwargs == .object) {
            var entries = kwargs.object.iterator();
            while (entries.next()) |entry| {
                if (std.mem.eql(u8, entry.key_ptr.*, "messages") or std.mem.eql(u8, entry.key_ptr.*, "tools") or std.mem.eql(u8, entry.key_ptr.*, "add_generation_prompt")) continue;
                try context.object.put(a, entry.key_ptr.*, entry.value_ptr.*);
            }
            if (kwargs.object.get("enable_thinking")) |enabled| {
                if (enabled != .bool) return error.InvalidThinking;
                thinking = enabled.bool;
                if (thinking and effort != null and std.mem.eql(u8, effort.?, "none")) effort = default_effort;
            }
        }
        try context.object.put(a, "enable_thinking", .{ .bool = thinking });
        try context.object.put(a, "thinking_mode", .{ .string = if (thinking) "thinking" else "chat" });
        if (thinking) {
            if (effort) |e| try context.object.put(a, "reasoning_effort", .{ .string = e });
        } else _ = context.object.swapRemove("reasoning_effort");
        const tools = try activeTools(a, body);
        return .{ .text = try t.raw(a, messages, tools, context, true, true), .images = images, .thinking = thinking };
    }
};

fn numericMembers(a: std.mem.Allocator, source: []const u8) ![:0]u8 {
    var out: std.ArrayList(u8) = .empty;
    var at: usize = 0;
    var closing: ?[]const u8 = null;
    var quote: ?u8 = null;
    while (at < source.len) {
        if (closing) |end| {
            if (quote) |q| {
                try out.append(a, source[at]);
                if (source[at] == '\\' and at + 1 < source.len) {
                    at += 1;
                    try out.append(a, source[at]);
                } else if (source[at] == q) quote = null;
            } else if (std.mem.startsWith(u8, source[at..], end)) {
                try out.appendSlice(a, end);
                at += end.len;
                closing = null;
                continue;
            } else if (source[at] == '\'' or source[at] == '"') {
                quote = source[at];
                try out.append(a, source[at]);
            } else if (source[at] == '.' and at > 0 and at + 1 < source.len and std.ascii.isDigit(source[at + 1]) and (std.ascii.isAlphabetic(source[at - 1]) or source[at - 1] == '_' or source[at - 1] == ']' or source[at - 1] == ')')) {
                var last = at + 1;
                while (last < source.len and std.ascii.isDigit(source[last])) : (last += 1) {}
                try out.append(a, '[');
                try out.appendSlice(a, source[at + 1 .. last]);
                try out.append(a, ']');
                at = last;
                continue;
            } else try out.append(a, source[at]);
        } else if (std.mem.startsWith(u8, source[at..], "{{") or std.mem.startsWith(u8, source[at..], "{%")) {
            closing = if (source[at + 1] == '{') "}}" else "%}";
            try out.appendSlice(a, source[at..][0..2]);
            at += 2;
            continue;
        } else if (std.mem.startsWith(u8, source[at..], "{#")) {
            const last = if (std.mem.indexOfPos(u8, source, at + 2, "#}")) |end| end + 2 else source.len;
            try out.appendSlice(a, source[at..last]);
            at = last;
            continue;
        } else try out.append(a, source[at]);
        at += 1;
    }
    return out.toOwnedSliceSentinel(a, 0);
}

fn asJson(a: std.mem.Allocator, value: V) ![:0]u8 {
    const bytes = try std.json.Stringify.valueAlloc(a, value, .{});
    return a.dupeSentinel(u8, bytes, 0);
}
fn templateSource(value: V) ![]const u8 {
    if (value == .string) return value.string;
    if (value == .object) if (value.object.get("default")) |source| if (source == .string) return source.string;
    if (value == .array) for (value.array.items) |entry| {
        if (entry != .object) continue;
        const name = entry.object.get("name") orelse continue;
        const source = entry.object.get("template") orelse continue;
        if (name == .string and std.mem.eql(u8, name.string, "default") and source == .string) return source.string;
    };
    return error.MissingChatTemplate;
}
pub fn activeTools(a: std.mem.Allocator, body: V) !V {
    const tools = body.object.get("tools") orelse .null;
    if (tools != .null and tools != .array) return error.InvalidTools;
    const choice = body.object.get("tool_choice") orelse .null;
    const mode = if (choice == .object) choice.object.get("type") orelse choice.object.get("mode") orelse .null else choice;
    if (mode == .string and std.ascii.eqlIgnoreCase(std.mem.trim(u8, mode.string, " \r\n\t"), "none")) return .null;
    var named: ?[]const u8 = null;
    var required = mode == .string and std.ascii.eqlIgnoreCase(mode.string, "required");
    if (mode == .string and std.ascii.eqlIgnoreCase(mode.string, "function")) {
        if (choice != .object) return error.InvalidToolChoice;
        const function = choice.object.get("function") orelse return error.InvalidToolChoice;
        named = try toolName(function);
        required = true;
    }
    var selected = V{ .array = std.json.Array.init(a) };
    if (tools == .array) for (tools.array.items) |tool| {
        const name = try toolName(tool);
        if (named == null or std.mem.eql(u8, name, named.?)) try selected.array.append(tool);
    };
    if (required and selected.array.items.len == 0) return error.MissingRequiredTool;
    return if (selected.array.items.len > 0) selected else .null;
}
pub fn requiresCall(body: V) bool {
    const choice = body.object.get("tool_choice") orelse return false;
    const mode = if (choice == .object) choice.object.get("type") orelse choice.object.get("mode") orelse return false else choice;
    if (mode != .string) return false;
    const value = std.mem.trim(u8, mode.string, " \r\n\t");
    return std.ascii.eqlIgnoreCase(value, "required") or (choice == .object and std.ascii.eqlIgnoreCase(value, "function"));
}
pub fn toolName(tool: V) ![]const u8 {
    if (tool != .object) return error.InvalidTools;
    const function = tool.object.get("function") orelse tool;
    if (function != .object) return error.InvalidTools;
    const name = function.object.get("name") orelse return error.InvalidTools;
    if (name != .string or std.mem.trim(u8, name.string, " \r\n\t").len == 0) return error.InvalidTools;
    return std.mem.trim(u8, name.string, " \r\n\t");
}

fn normalize(a: std.mem.Allocator, messages: V, late_system: []const u8, images: *V) !V {
    if (messages != .array or messages.array.items.len == 0) return error.InvalidMessages;
    var out = V{ .array = std.json.Array.init(a) };
    var instructions: std.ArrayList([]const u8) = .empty;
    var leading: ?V = null;
    for (messages.array.items) |message| {
        if (message != .object) return error.InvalidMessage;
        var item = V{ .object = try message.object.clone(a) };
        const role = item.object.get("role") orelse return error.InvalidRole;
        if (role != .string) return error.InvalidRole;
        var valid = false;
        for ([_][]const u8{ "system", "developer", "user", "assistant", "tool" }) |name| valid = valid or std.mem.eql(u8, role.string, name);
        if (!valid) return error.InvalidRole;
        var content = item.object.get("content") orelse .null;
        if (content == .null) content = .{ .string = "" };
        if (content == .array) {
            var parts = std.json.Array.init(a);
            var strings: std.ArrayList([]const u8) = .empty;
            var has_image = false;
            for (content.array.items) |part| {
                if (part != .object) return error.InvalidContentPart;
                const kind = part.object.get("type") orelse return error.InvalidContentPart;
                if (kind != .string) return error.InvalidContentPart;
                if (std.mem.eql(u8, kind.string, "text")) {
                    const value = part.object.get("text") orelse return error.InvalidContentPart;
                    if (value != .string) return error.InvalidContentPart;
                    try strings.append(a, value.string);
                    try parts.append(part);
                } else if (std.mem.eql(u8, kind.string, "image_url")) {
                    if (!std.mem.eql(u8, role.string, "user") or images.array.items.len >= 4) return error.InvalidImageMessage;
                    const source = part.object.get("image_url") orelse return error.InvalidImageMessage;
                    try images.array.append(source);
                    var replacement = V{ .object = .empty };
                    try replacement.object.put(a, "type", .{ .string = "image" });
                    try parts.append(replacement);
                    has_image = true;
                } else return error.UnsupportedModality;
            }
            content = if (has_image) .{ .array = parts } else .{ .string = try std.mem.join(a, "", strings.items) };
        } else if (content != .string) return error.InvalidContent;
        try item.object.put(a, "content", content);
        if (item.object.getPtr("tool_calls")) |calls| if (calls.* == .array) {
            var copied = std.json.Array.init(a);
            try copied.appendSlice(calls.array.items);
            calls.array = copied;
            for (calls.array.items) |*call| {
                if (call.* != .object) return error.InvalidToolCall;
                call.* = .{ .object = try call.object.clone(a) };
                if (call.object.getPtr("function")) |function| if (function.* == .object) {
                    function.* = .{ .object = try function.object.clone(a) };
                    if (function.object.getPtr("arguments")) |arguments| if (arguments.* == .string) {
                        if (std.json.parseFromSlice(V, a, arguments.string, .{})) |parsed| {
                            if (parsed.value == .object) arguments.* = parsed.value;
                        } else |_| {}
                    };
                };
            }
        };
        if (std.mem.eql(u8, role.string, "system") or std.mem.eql(u8, role.string, "developer")) {
            if (out.array.items.len == 0) {
                if (content != .string) return error.InvalidContent;
                if (leading == null) leading = item;
                try instructions.append(a, content.string);
                continue;
            }
            try item.object.put(a, "role", .{ .string = late_system });
        }
        try out.array.append(item);
    }
    if (leading) |*first| {
        try first.object.put(a, "role", .{ .string = "system" });
        try first.object.put(a, "content", .{ .string = try std.mem.join(a, "\n\n", instructions.items) });
        try out.array.insert(0, first.*);
    }
    return out;
}

test "Jinja bridge preserves NUL content and structured arguments" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const config = Template{ .arena = undefined, .source = "{% for m in messages %}{{m.role}}:{{m.content}}{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}", .context = .{ .object = .empty } };
    const body = try std.json.parseFromSlice(V, a,
        \\{"messages":[{"role":"developer","content":"first"},{"role":"system","content":"second"},{"role":"user","content":"a\u0000b"}]}
    , .{});
    const rendered = try config.render(a, body.value);
    try std.testing.expectEqualStrings("system:first\n\nseconduser:a\x00bassistant:", rendered.text);
}

test "numeric Jinja members preserve literals, decimals and comments" {
    const a = std.testing.allocator;
    const source = "raw.0 {# {{ a.0 }} #} {{ a.0.output }} {% if b.12 and 1.0 == 1.0 %}{{ 'c.0' }}{% endif %}";
    const result = try numericMembers(a, source);
    defer a.free(result);
    try std.testing.expectEqualStrings("raw.0 {# {{ a.0 }} #} {{ a[0].output }} {% if b[12] and 1.0 == 1.0 %}{{ 'c.0' }}{% endif %}", result);
}

test "server thinking defaults, request effort and explicit kwargs precedence" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const template = Template{ .arena = undefined, .source = "{{ 'on' if enable_thinking else 'off' }}:{{ reasoning_effort|default('unset') }}", .context = .{ .object = .empty } };
    const inputs = [_][]const u8{
        \\{"messages":[{"role":"user","content":"Hi"}]}
        ,
        \\{"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":null}
        ,
        \\{"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"none"}
        ,
        \\{"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":true}}
        ,
        \\{"messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":false}}
        ,
        \\{"messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"reasoning_effort":"minimal"}}
        ,
    };
    for (inputs, [_][]const u8{ "on:medium", "on:medium", "off:unset", "on:medium", "off:unset", "on:low" }) |source, want| {
        const body = try std.json.parseFromSlice(V, a, source, .{});
        const rendered = try template.renderWithDefaults(a, body.value, true, "medium");
        try std.testing.expectEqualStrings(want, rendered.text);
    }
}

//! Inkling 2's full native reply, split by control-token identity rather than text.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const native_reply = @import("native_reply.zig");
const tool_specs = @import("tool_specs.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;
pub const Result = native_reply.Result;

const Kind = enum {
    message_model,
    thinking,
    content_text,
    invoke_tool_text,
    tool_error,
    invoke_tool_json,
    end_message,
    end_sampling,
};

const Control = struct { kind: Kind, spelling: []const u8 };
const controls = [_]Control{
    .{ .kind = .message_model, .spelling = "<|message_model|>" },
    .{ .kind = .thinking, .spelling = "<|content_thinking|>" },
    .{ .kind = .content_text, .spelling = "<|content_text|>" },
    .{ .kind = .invoke_tool_text, .spelling = "<|content_invoke_tool_text|>" },
    .{ .kind = .tool_error, .spelling = "<|content_tool_error|>" },
    .{ .kind = .invoke_tool_json, .spelling = "<|content_invoke_tool_json|>" },
    .{ .kind = .end_message, .spelling = "<|end_message|>" },
    .{ .kind = .end_sampling, .spelling = "<|content_model_end_sampling|>" },
};
const Resolved = [controls.len]?u32;
const State = enum { none, thinking, content, tool_json };

fn resolve(a: Allocator, text: model_text.Text) Allocator.Error!Resolved {
    var out: Resolved = undefined;
    for (&out) |*entry| entry.* = null;
    for (controls, 0..) |control, i| {
        const id = text.tokenId(control.spelling) orelse continue;
        if (!text.isAdded(id)) continue;
        const vocabulary_spelling = try text.tokenString(a, id);
        defer a.free(vocabulary_spelling);
        if (std.mem.eql(u8, vocabulary_spelling, control.spelling)) out[i] = id;
    }
    return out;
}

fn controlKind(text: model_text.Text, ids: Resolved, id: u32) ?Kind {
    if (!text.isAdded(id)) return null;
    for (ids, controls) |control_id, control| if (control_id == id) return control.kind;
    return null;
}

fn appendDecoded(a: Allocator, out: *std.ArrayList(u8), text: model_text.Text, ids: []const u32) Allocator.Error!void {
    if (ids.len == 0) return;
    const bytes = try text.decode(a, ids);
    defer a.free(bytes);
    try out.appendSlice(a, bytes);
}

fn finishTool(a: Allocator, payload: []const u8, tools: []const Value, calls: *std.ArrayList(Value), rejected: *usize) Allocator.Error!void {
    const document = switch (try json.parse(a, payload)) {
        .ok => |value| value,
        .err => {
            rejected.* += 1;
            return;
        },
    };
    if (document != .object) {
        rejected.* += 1;
        return;
    }
    const name_value = document.get("name") orelse {
        rejected.* += 1;
        return;
    };
    const arguments = document.get("args") orelse {
        rejected.* += 1;
        return;
    };
    if (name_value != .string or arguments != .object) {
        rejected.* += 1;
        return;
    }
    const name = try a.dupe(u8, name_value.string); // Calls outlive the reusable payload.
    var offered = false;
    for (tools) |tool| {
        if (std.mem.eql(u8, name, try tool_specs.toolName(a, tool))) {
            offered = true;
            break;
        }
    }
    if (!offered) {
        rejected.* += 1;
        return;
    }

    const function = try json.newObject(a);
    try function.put(a, "name", .{ .string = name });
    try function.put(a, "arguments", .{ .string = try json.stringify(a, arguments, .{ .ascii = false, .compact = true }) });
    const call = try json.newObject(a);
    try call.put(a, "id", .{ .string = try std.fmt.allocPrint(a, "call_{d}", .{calls.items.len}) });
    try call.put(a, "type", .{ .string = "function" });
    try call.put(a, "function", .{ .object = function });
    try calls.append(a, .{ .object = call });
}

/// Owns the reply parser between growing token prefixes; results borrow its arena-backed lists.
pub const Stream = struct {
    a: Allocator,
    text: model_text.Text,
    resolved: Resolved,
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    tool_payload: std.ArrayList(u8) = .empty,
    calls: std.ArrayList(Value) = .empty,
    state: State = .none,
    reasoning_tokens: usize = 0,
    rejected: usize = 0, // tool calls refused: malformed, unoffered, or left open by a new header
    next: usize = 0,

    pub fn init(a: Allocator, text: model_text.Text) Allocator.Error!Stream {
        return .{ .a = a, .text = text, .resolved = try resolve(a, text) };
    }

    fn result(s: *const Stream) Result {
        return .{
            .content = s.content.items,
            .reasoning = s.reasoning.items,
            .calls = if (s.calls.items.len == 0) null else s.calls.items,
            .reasoning_tokens = s.reasoning_tokens,
            .rejected = s.rejected > 0 or s.state == .tool_json,
            .rejected_calls = s.rejected + @intFromBool(s.state == .tool_json),
            .open_think = s.state == .thinking,
        };
    }

    /// Neither a listed control nor any other special token of the tokenizer: text a block holds.
    fn payload(s: *const Stream, id: u32) bool {
        return controlKind(s.text, s.resolved, id) == null and !s.text.isSpecial(id);
    }

    /// Consume only the new suffix; ids is a monotonically growing full prefix, and EOS stripping must never shrink it.
    pub fn extend(s: *Stream, ids: []const u32, tools: []const Value) Allocator.Error!Result {
        std.debug.assert(ids.len >= s.next);
        while (s.next < ids.len) {
            const kind = controlKind(s.text, s.resolved, ids[s.next]);
            if (kind == .end_sampling) {
                // closes nothing: an open channel stays reported, and with ignore_eos the reply goes on
                s.next += 1;
                continue;
            }
            if (kind == .message_model and s.state != .none) {
                // a new message inside an open block: the block was never closed, and a call in it never finished
                if (s.state == .tool_json) s.rejected += 1;
                s.state = .none;
                s.next += 1;
                continue;
            }
            if (kind == .end_message) {
                if (s.state == .tool_json) try finishTool(s.a, s.tool_payload.items, tools, &s.calls, &s.rejected);
                s.state = .none;
                s.next += 1;
                continue;
            }
            if (s.state == .none) {
                if (kind) |found| s.state = switch (found) {
                    .thinking => .thinking,
                    .content_text, .invoke_tool_text, .tool_error => .content,
                    .invoke_tool_json => blk: {
                        s.tool_payload.clearRetainingCapacity();
                        break :blk .tool_json;
                    },
                    else => .none,
                };
                s.next += 1;
                continue;
            }
            if (!s.payload(ids[s.next])) {
                // a control or any other special token inside a block is never payload
                s.next += 1;
                continue;
            }

            const start = s.next;
            s.next += 1;
            while (s.next < ids.len and s.payload(ids[s.next])) s.next += 1;
            const run = ids[start..s.next];
            switch (s.state) {
                .thinking => {
                    try appendDecoded(s.a, &s.reasoning, s.text, run);
                    s.reasoning_tokens += run.len;
                },
                .content => try appendDecoded(s.a, &s.content, s.text, run),
                .tool_json => try appendDecoded(s.a, &s.tool_payload, s.text, run),
                .none => unreachable,
            }
        }
        return s.result();
    }
};

/// Parse a whole reply through the same cursor used by streamed replies.
pub fn parse(a: Allocator, text: model_text.Text, ids: []const u32, tools: []const Value) Allocator.Error!Result {
    var stream = try Stream.init(a, text);
    return stream.extend(ids, tools);
}

//! DeepSeek per-request state: the reasoning memory across turns and one reply's text and DSML stream.
const std = @import("std");
const json = @import("json");
const serve = @import("dsv41_serve");
const family = @import("family.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const tokenizer = serve.tokenizer;
const deepseek = @import("deepseek.zig");
const DeepSeek = deepseek.DeepSeek;
const tools_mod = deepseek.tools_mod;

/// Reasoning memory: reasoning by tool-call id or content digest, oldest dropped first.
pub const Memory = struct {
    gpa: Allocator,
    entries: usize,
    lock: std.atomic.Value(bool) = .init(false),
    map: std.StringArrayHashMapUnmanaged([]const u8) = .empty,

    fn acquire(m: *Memory) void {
        while (m.lock.swap(true, .acquire)) std.atomic.spinLoopHint();
    }
    fn release(m: *Memory) void {
        m.lock.store(false, .release);
    }

    fn contentKey(buf: *[66]u8, content: []const u8) []const u8 {
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(content, &d, .{});
        buf[0] = 'c';
        buf[1] = ':';
        _ = std.fmt.bufPrint(buf[2..], "{x}", .{d}) catch unreachable;
        return buf[0..66];
    }

    pub fn put(m: *Memory, reasoning: []const u8, content: []const u8, calls: []const Value) void {
        if (reasoning.len == 0 or m.entries == 0) return;
        m.acquire();
        defer m.release();
        var buf: [66]u8 = undefined;
        var keys_buf: [64][]const u8 = undefined;
        var keys: []const []const u8 = keys_buf[0..0];
        var n: usize = 0;
        for (calls) |c| if (c.strField("id")) |id| if (n < keys_buf.len) {
            keys_buf[n] = id;
            n += 1;
        };
        keys = keys_buf[0..n];
        if (n == 0) {
            keys_buf[0] = contentKey(&buf, content);
            keys = keys_buf[0..1];
        }
        const value = m.gpa.dupe(u8, reasoning) catch return;
        var used = false;
        for (keys) |k| {
            if (m.map.fetchOrderedRemove(k)) |old| {
                m.gpa.free(old.key);
                m.gpa.free(old.value);
            }
            const key = m.gpa.dupe(u8, k) catch continue;
            const v = if (used) m.gpa.dupe(u8, reasoning) catch {
                m.gpa.free(key);
                continue;
            } else value;
            used = true;
            m.map.put(m.gpa, key, v) catch {
                m.gpa.free(key);
                m.gpa.free(v);
            };
        }
        if (!used) m.gpa.free(value);
        while (m.map.count() > m.entries) {
            const k = m.map.keys()[0];
            const v = m.map.values()[0];
            m.map.orderedRemoveAt(0);
            m.gpa.free(k);
            m.gpa.free(v);
        }
    }

    /// A copy of ``messages`` with remembered reasoning put back where an assistant message lacks it.
    pub fn restore(m: *Memory, a: Allocator, messages: Value) Allocator.Error!Value {
        if (messages != .array or m.map.count() == 0) return messages;
        var out: ?[]Value = null;
        m.acquire();
        defer m.release();
        for (messages.array, 0..) |msg, i| {
            if (msg != .object) continue;
            const role = msg.get("role") orelse continue;
            if (role != .string or !std.mem.eql(u8, role.string, "assistant")) continue;
            const rc: Value = msg.get("reasoning_content") orelse .null;
            const rs: Value = msg.get("reasoning") orelse .null;
            if (rc.truthy() or rs.truthy()) continue;
            var key: ?[]const u8 = null;
            var buf: [66]u8 = undefined;
            if (msg.get("tool_calls")) |tc| if (tc == .array and tc.array.len > 0 and tc.array[0] == .object) if (tc.array[0].get("id")) |id| if (id == .string and id.string.len > 0) {
                key = id.string;
            };
            if (key == null) if (msg.get("content")) |c| if (c == .string) {
                key = contentKey(&buf, c.string);
            };
            const got = m.map.get(key orelse continue) orelse continue;
            if (out == null) out = try a.dupe(Value, messages.array);
            const copy = try json.copyObject(a, msg.object);
            try copy.put(a, "reasoning_content", .{ .string = try a.dupe(u8, got) });
            out.?[i] = .{ .object = copy };
        }
        return if (out) |o| .{ .array = o } else messages;
    }

    pub fn deinit(m: *Memory) void {
        for (m.map.keys(), m.map.values()) |k, v| {
            m.gpa.free(k);
            m.gpa.free(v);
        }
        m.map.deinit(m.gpa);
    }
};

/// One reply's text and DSML stream (``family.Reader``).
pub const Reply = struct {
    a: Allocator,
    d: *DeepSeek,
    detok: tokenizer.Detokenizer,
    text: std.ArrayList(u8) = .empty,
    stream: tools_mod.Stream,
    deltas: std.ArrayList(tools_mod.Delta) = .empty,
    thinking: bool,
    offered: tools_mod.Tools,
    single: bool,
    flushed: bool = false,

    fn self(ctx: *anyopaque) *Reply {
        return @ptrCast(@alignCast(ctx));
    }

    pub fn push(ctx: *anyopaque, ids: []const u32) Allocator.Error!void {
        const r = self(ctx);
        for (ids) |id| try r.detok.push(r.a, id, &r.text);
    }

    pub fn pending(ctx: *anyopaque) bool {
        return self(ctx).detok.pending();
    }

    pub fn feed(ctx: *anyopaque, finished: bool, out: *std.ArrayList(Value)) Allocator.Error!void {
        const r = self(ctx);
        const a = r.a;
        if (finished and !r.flushed) {
            try r.detok.flush(a, &r.text);
            r.flushed = true;
        }
        r.deltas.clearRetainingCapacity();
        try r.stream.feed(r.text.items, finished, &r.deltas);
        const fam = r.d.family_();
        for (r.deltas.items) |delta| switch (delta) {
            .reasoning => |t| try out.append(a, try fam.reasoningDelta(a, t)),
            .content => |t| try out.append(a, .{ .string = t }),
            .call => |c| {
                if (r.single and c.index > 0) continue;
                const f = try json.newObject(a);
                try f.put(a, "name", .{ .string = c.name });
                try f.put(a, "arguments", .{ .string = "" });
                const call = try json.newObject(a);
                try call.put(a, "index", try json.intValue(a, c.index));
                try call.put(a, "id", .{ .string = c.id });
                try call.put(a, "type", .{ .string = "function" });
                try call.put(a, "function", .{ .object = f });
                try out.append(a, try toolCallsDelta(a, .{ .object = call }));
            },
            .arguments => |g| {
                if (r.single and g.index > 0) continue;
                const f = try json.newObject(a);
                try f.put(a, "arguments", .{ .string = try a.dupe(u8, g.text) });
                const call = try json.newObject(a);
                try call.put(a, "index", try json.intValue(a, g.index));
                try call.put(a, "function", .{ .object = f });
                try out.append(a, try toolCallsDelta(a, .{ .object = call }));
            },
        };
    }

    fn toolCallsDelta(a: Allocator, call: Value) Allocator.Error!Value {
        const list = try a.alloc(Value, 1);
        list[0] = call;
        const o = try json.newObject(a);
        try o.put(a, "tool_calls", .{ .array = list });
        return .{ .object = o };
    }

    fn openai(a: Allocator, c: *const tools_mod.Call) Allocator.Error!Value {
        const f = try json.newObject(a);
        try f.put(a, "name", .{ .string = c.name });
        try f.put(a, "arguments", .{ .string = try c.arguments(a) });
        const o = try json.newObject(a);
        try o.put(a, "id", .{ .string = c.id });
        try o.put(a, "type", .{ .string = "function" });
        try o.put(a, "function", .{ .object = f });
        return .{ .object = o };
    }

    pub fn markup(ctx: *anyopaque) Allocator.Error!?family.Markup {
        const r = self(ctx);
        const n = try tools_mod.note(r.a, r.text.items, r.thinking, r.offered) orelse return null;
        return .{ .at = n.at, .reason = n.reason, .window = n.window };
    }

    pub fn parse(ctx: *anyopaque) Allocator.Error!family.Parsed {
        const r = self(ctx);
        if (!r.flushed) {
            try r.detok.flush(r.a, &r.text);
            r.flushed = true;
        }
        const p = try tools_mod.parse(r.a, r.text.items, r.thinking, r.offered, .{ .make = DeepSeek.callId });
        const n = if (r.single) @min(1, p.calls.len) else p.calls.len;
        const list = try r.a.alloc(Value, n);
        for (p.calls[0..n], list) |*c, *slot| slot.* = try openai(r.a, c);
        return .{ .reasoning = p.reasoning, .content = p.content, .calls = list };
    }

    pub fn calls(ctx: *anyopaque) Allocator.Error![]Value {
        const r = self(ctx);
        const all = r.stream.calls.items;
        const n = if (r.single) @min(1, all.len) else all.len;
        const out = try r.a.alloc(Value, n);
        for (all[0..n], out) |*c, *slot| slot.* = try openai(r.a, c);
        return out;
    }
};

//! The parity tests' tokenizer (bytes, then whole pieces) and template, byte for byte as fake_text.py.
const std = @import("std");
const server = @import("server");
const json = server.json;
const model_text = server.model_text;
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const FakeText = struct {
    pieces: []const []const u8,
    eos: [2]u32,

    pub const source = "{# fake template: 'low' 'medium' 'xhigh' enable_thinking #}";

    pub fn init(pieces: []const []const u8) FakeText {
        var t: FakeText = .{ .pieces = pieces, .eos = .{ 0, 0 } };
        t.eos = .{ t.pieceId("<|im_end|>").?, t.pieceId("<|endoftext|>").? };
        return t;
    }

    fn pieceId(t: *const FakeText, piece: []const u8) ?u32 {
        for (t.pieces, 0..) |p, i| if (std.mem.eql(u8, p, piece)) return @intCast(256 + i);
        return null;
    }

    /// Greedy longest piece at each byte, else the byte itself.
    pub fn encode(t: *const FakeText, a: Allocator, s: []const u8) Allocator.Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        while (i < s.len) {
            var best: ?usize = null;
            for (t.pieces, 0..) |p, j| {
                if (p.len < 2 or !std.mem.startsWith(u8, s[i..], p)) continue;
                if (best == null or p.len > t.pieces[best.?].len) best = j;
            }
            if (best) |j| {
                try out.append(a, @intCast(256 + j));
                i += t.pieces[j].len;
            } else {
                try out.append(a, s[i]);
                i += 1;
            }
        }
        return out.items;
    }

    /// The pieces' bytes as Python's ``bytes.decode("utf-8", "replace")`` reads them.
    pub fn decode(t: *const FakeText, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var raw: std.ArrayList(u8) = .empty;
        for (ids) |id| {
            if (id < 256) try raw.append(a, @intCast(id)) else if (id - 256 < t.pieces.len) try raw.appendSlice(a, t.pieces[id - 256]);
        }
        return replaceInvalid(a, raw.items);
    }

    fn textOf(v: ?Value) []const u8 {
        const x = v orelse return "";
        return switch (x) {
            .string => |s| s,
            else => "",
        };
    }

    /// The fake chat template: tools, each message, then the assistant header and an open think block.
    pub fn render(t: *const FakeText, a: Allocator, messages: Value, o: model_text.RenderOptions) Allocator.Error![]u8 {
        _ = t;
        var out: std.ArrayList(u8) = .empty;
        if (o.tools) |tools| if (tools.truthy()) {
            try out.appendSlice(a, "<|im_start|>system\n# Tools\n");
            try out.appendSlice(a, try json.stringify(a, tools, .{ .ascii = false }));
            try out.appendSlice(a, "<|im_end|>\n");
        };
        for (messages.array) |m| {
            const role = textOf(m.get("role"));
            try out.print(a, "<|im_start|>{s}\n", .{role});
            if (m.get("reasoning_content")) |r| if (r.truthy()) try out.print(a, "<think>\n{s}\n</think>\n\n", .{textOf(r)});
            const content = m.get("content") orelse Value.null;
            switch (content) {
                .string => |s| try out.appendSlice(a, s),
                .array => |parts| for (parts) |p| try out.appendSlice(a, textOf(p.get("text"))),
                else => {},
            }
            if (m.get("tool_calls")) |calls| if (calls == .array) for (calls.array) |call| {
                const f = call.get("function") orelse Value.null;
                const o2 = try json.newObject(a);
                try o2.put(a, "name", f.get("name") orelse .null);
                try o2.put(a, "arguments", f.get("arguments") orelse .null);
                try out.print(a, "\n<tool_call>\n{s}\n</tool_call>", .{try json.stringify(a, .{ .object = o2 }, .{ .ascii = false })});
            };
            if (std.mem.eql(u8, role, "tool")) try out.print(a, "\n[call {s}]", .{try server.tool_specs.pyStr(a, m.get("tool_call_id") orelse .null)});
            try out.appendSlice(a, "<|im_end|>\n");
        }
        if (o.add_generation_prompt) {
            try out.appendSlice(a, "<|im_start|>assistant\n");
            if (o.enable_thinking) {
                try out.appendSlice(a, "<think>\n");
                if (o.reasoning_effort) |e| try out.print(a, "[effort {s}]\n", .{e});
            }
        }
        return out.items;
    }

    pub fn text(t: *FakeText) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encodeFn, .decode = decodeFn, .token_id = idFn, .token_string = strFn, .vocab_size = vocabFn, .eos_ids = eosFn, .render = renderFn, .template_source = sourceFn } };
    }

    fn self(ctx: *anyopaque) *FakeText {
        return @ptrCast(@alignCast(ctx));
    }

    fn encodeFn(ctx: *anyopaque, a: Allocator, s: []const u8, _: bool) model_text.Error![]u32 {
        return self(ctx).encode(a, s);
    }

    fn decodeFn(ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        return self(ctx).decode(a, ids);
    }

    fn idFn(ctx: *anyopaque, piece: []const u8) ?u32 {
        const t = self(ctx);
        if (t.pieceId(piece)) |id| return id;
        return if (piece.len == 1 and piece[0] < 0x80) piece[0] else null;
    }

    fn strFn(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        const t = self(ctx);
        if (id >= 256) return a.dupe(u8, if (id - 256 < t.pieces.len) t.pieces[id - 256] else "");
        if (id >= 0x20 and id < 0x7f) return a.dupe(u8, &.{@intCast(id)});
        return std.fmt.allocPrint(a, "<0x{X:0>2}>", .{id});
    }

    fn vocabFn(ctx: *anyopaque) u32 {
        return @intCast(256 + self(ctx).pieces.len);
    }

    fn eosFn(ctx: *anyopaque) []const u32 {
        return &self(ctx).eos;
    }

    fn renderFn(ctx: *anyopaque, a: Allocator, messages: Value, o: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return self(ctx).render(a, messages, o);
    }

    fn sourceFn(_: *anyopaque) []const u8 {
        return source;
    }
};

/// CPython's UTF-8 decode with errors="replace": one U+FFFD for each invalid sequence the decoder reports.
pub fn replaceInvalid(a: Allocator, s: []const u8) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    outer: while (i < s.len) {
        const c = s[i];
        if (c < 0x80) {
            try out.append(a, c);
            i += 1;
            continue;
        }
        const need: usize = if (c >= 0xc2 and c <= 0xdf) 1 else if (c >= 0xe0 and c <= 0xef) 2 else if (c >= 0xf0 and c <= 0xf4) 3 else 0;
        if (need == 0) {
            try out.appendSlice(a, "\u{fffd}");
            i += 1;
            continue;
        }
        var k: usize = 1;
        while (k <= need) : (k += 1) {
            if (i + k >= s.len) {
                try out.appendSlice(a, "\u{fffd}");
                break :outer;
            }
            const b = s[i + k];
            const lo: u8, const hi: u8 = if (k == 1) switch (c) {
                0xe0 => .{ 0xa0, 0xbf },
                0xed => .{ 0x80, 0x9f },
                0xf0 => .{ 0x90, 0xbf },
                0xf4 => .{ 0x80, 0x8f },
                else => .{ 0x80, 0xbf },
            } else .{ 0x80, 0xbf };
            if (b < lo or b > hi) {
                try out.appendSlice(a, "\u{fffd}");
                i += k;
                continue :outer;
            }
        }
        try out.appendSlice(a, s[i .. i + need + 1]);
        i += need + 1;
    }
    return out.items;
}

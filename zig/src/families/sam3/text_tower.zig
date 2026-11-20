const std = @import("std");
const st = @import("safetensors");
const tokenizer = @import("tokenizer");
const runtime = @import("runtime.zig");
const elementwise = @import("elementwise.zig");
const projection = @import("projection.zig");
const norms = @import("norms.zig");
const attention = @import("attention.zig");
const dense_small = @import("dense_small.zig");

const Runtime = runtime.Runtime;
const Scope = runtime.Scope;
const Tensor = runtime.Tensor;

pub const max_tokens = 32;
pub const Mlp = enum { plain, compensated };

const start_id: u32 = 49406;
const end_id: u32 = 49407;
const width = 1024;
const blocks = 24;
const heads = 16;
const eps: f32 = 1e-5;

/// Cleaned text as BPE ids framed by the start and end ids, at most `max_tokens` (over-long text is truncated).
pub fn frame(allocator: std.mem.Allocator, tok: *const tokenizer.Tokenizer, text: []const u8) ![]u32 {
    const cleaned = try clean(allocator, text);
    defer allocator.free(cleaned);
    const ids = if (cleaned.len == 0) try allocator.alloc(u32, 0) else try tok.encode(allocator, cleaned);
    defer allocator.free(ids);
    return framed(allocator, ids);
}

fn framed(allocator: std.mem.Allocator, ids: []const u32) ![]u32 {
    const kept: usize = @min(ids.len, max_tokens - 2);
    const out = try allocator.alloc(u32, kept + 2);
    out[0] = start_id;
    @memcpy(out[1..][0..kept], ids[0..kept]);
    out[kept + 1] = end_id;
    return out;
}

/// HTML references replaced in two passes, whitespace runs collapsed to one space, ends trimmed, ASCII lowercased.
fn clean(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const once = try unescape(allocator, text);
    defer allocator.free(once);
    const unescaped = try unescape(allocator, once);
    defer allocator.free(unescaped);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var at: usize = 0;
    var space = false;
    while (at < unescaped.len) {
        const len, const point = decode(unescaped[at..]);
        if (point) |code| if (whitespace(code)) {
            space = true;
            at += len;
            continue;
        };
        if (space and out.items.len > 0) try out.append(allocator, ' ');
        space = false;
        for (unescaped[at..][0..len]) |byte| try out.append(allocator, std.ascii.toLower(byte));
        at += len;
    }
    return out.toOwnedSlice(allocator);
}

// Invalid UTF-8 advances one byte without a code point, so it passes through unchanged.
fn decode(bytes: []const u8) struct { usize, ?u21 } {
    const len = std.unicode.utf8ByteSequenceLength(bytes[0]) catch return .{ 1, null };
    if (len > bytes.len) return .{ 1, null };
    const point = std.unicode.utf8Decode(bytes[0..len]) catch return .{ 1, null };
    return .{ len, point };
}

fn whitespace(point: u21) bool {
    return switch (point) {
        0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

const named = [_]struct { []const u8, u21 }{
    .{ "&amp;", '&' },
    .{ "&lt;", '<' },
    .{ "&gt;", '>' },
    .{ "&quot;", '"' },
    .{ "&apos;", '\'' },
    .{ "&nbsp;", 0xA0 },
};

fn unescape(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var at: usize = 0;
    while (at < text.len) {
        if (text[at] == '&') if (reference(text[at..])) |found| {
            var bytes: [4]u8 = undefined;
            const len = std.unicode.utf8Encode(found[1], &bytes) catch unreachable;
            try out.appendSlice(allocator, bytes[0..len]);
            at += found[0];
            continue;
        };
        try out.append(allocator, text[at]);
        at += 1;
    }
    return out.toOwnedSlice(allocator);
}

// The reference at the start of `text` (which begins with '&'): its length and code point, or null to keep it as typed.
fn reference(text: []const u8) ?struct { usize, u21 } {
    for (named) |entry| if (std.mem.startsWith(u8, text, entry[0])) return .{ entry[0].len, entry[1] };
    if (text.len < 3 or text[1] != '#') return null;
    const hex = text[2] == 'x' or text[2] == 'X';
    const base: u32 = if (hex) 16 else 10;
    const first: usize = if (hex) 3 else 2;
    var at = first;
    var number: u32 = 0;
    while (at < text.len) : (at += 1) {
        const digit = std.fmt.charToDigit(text[at], @intCast(base)) catch break;
        number = if (number > 0x10FFFF) number else number * base + digit;
    }
    if (at == first or at >= text.len or text[at] != ';') return null;
    if (number > 0x10FFFF or (number >= 0xD800 and number <= 0xDFFF)) return null;
    return .{ at + 1, @intCast(number) };
}

/// Text features `[n, 256]` for 1 … 32 token ids through the 24-block causal text tower.
pub fn encode(rt: *Runtime, ids: []const u32, mlp: Mlp) !Tensor {
    if (ids.len == 0 or ids.len > max_tokens) return error.SamTextLength;
    const text = Scope.root(rt, rt.config.roots.text);
    const encoder = try text.sub("encoder");
    const tokens = try encoder.info("token_embedding.weight");
    if (tokens.rank != 2 or tokens.dim(1) != width) return error.SamTextShape;
    for (ids) |id| if (id >= tokens.dim(0)) return error.SamTokenOutOfRange;
    const positions = try encoder.info("positional_embedding");
    const n = ids.len;
    if (positions.numel() < n * width) return error.SamTextShape;
    const input = try rt.hostTensor(n * width);
    try embed(input.slice(f32, n * width), tokens, positions, ids);
    try rt.observe("text.input", input, &.{ 1, n, width });

    try rt.gpu.beginBatch();
    errdefer if (rt.gpu.batch != null) rt.gpu.endBatch() catch {};
    const start = rt.mark();
    errdefer rt.rewind(start, &.{});
    const stack = try encoder.sub("transformer");
    var h = input;
    for (0..blocks) |b| {
        h = try block(rt, try stack.item("resblocks", b), h, n, mlp, b == 0);
        try rt.observe(try rt.print("text.layer.{d}", .{b}), h, &.{ 1, n, width });
        rt.rewind(start, &.{h});
    }
    const final = try norms.layerNorm(encoder, "ln_final", h, eps, .plain);
    const out = try projection.affine(text, "resizer", final);
    try rt.observe("text.output", out, &.{ 1, n, out.count / n });
    try rt.gpu.endBatch();
    rt.rewind(start, &.{out});
    return out;
}

fn block(rt: *Runtime, s: Scope, h: Tensor, n: usize, mlp: Mlp, first: bool) !Tensor {
    const norm1 = try norms.layerNorm(s, "ln_1", h, eps, .plain);
    if (first) try rt.observe("text.first.norm1", norm1, &.{ 1, n, width });
    const attended = try attention.packedHeads(try s.sub("attn"), norm1, norm1, norm1, heads, .{ .causal = true });
    if (first) try rt.observe("text.first.projected", attended, &.{ 1, n, width });
    const residual = try elementwise.add(rt, h, attended);
    if (first) try rt.observe("text.first.residual", residual, &.{ 1, n, width });
    const norm2 = try norms.layerNorm(s, "ln_2", residual, eps, .plain);
    if (first) try rt.observe("text.first.norm2", norm2, &.{ 1, n, width });
    const expanded = try blockProjection(rt, s, "mlp.c_fc", norm2, n, mlp);
    if (first) try rt.observe("text.first.expanded", expanded, &.{ 1, n, expanded.count / n });
    const activated = try elementwise.gelu(rt, expanded);
    if (first) try rt.observe("text.first.gelu", activated, &.{ 1, n, activated.count / n });
    const contracted = try blockProjection(rt, s, "mlp.c_proj", activated, n, mlp);
    if (first) try rt.observe("text.first.contracted", contracted, &.{ 1, n, width });
    return elementwise.add(rt, residual, contracted);
}

fn blockProjection(rt: *Runtime, s: Scope, leaf: []const u8, x: Tensor, rows: usize, mlp: Mlp) !Tensor {
    switch (mlp) {
        .plain => return projection.affine(s, leaf, x),
        .compensated => {
            const p = try s.sub(leaf);
            const shape = try p.info("weight");
            if (shape.rank != 2) return error.SamTextShape;
            const weight = try p.param("weight");
            const bias = try p.param("bias");
            return rt.hold(try dense_small.run(rt.gpu, x, weight, bias, rows, shape.dim(0), shape.dim(1)));
        },
    }
}

// Row r is the token row ids[r] plus the position row r, one f32 addition with the token term first.
fn embed(out: []f32, tokens: st.Tensor, positions: st.Tensor, ids: []const u32) !void {
    if (out.len != ids.len * width) return error.SamTextShape;
    for (ids, 0..) |id, r| for (0..width) |c| {
        out[r * width + c] = try tableValue(tokens, @as(usize, id) * width + c) + try tableValue(positions, r * width + c);
    };
}

fn tableValue(t: st.Tensor, index: usize) !f32 {
    const size = t.dtype.size();
    if ((index + 1) * size > t.bytes.len) return error.SamTextShape;
    const bytes = t.bytes[index * size ..];
    return switch (t.dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, bytes[0..4], .little)),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[0..2], .little)))),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[0..2], .little)) << 16),
        else => error.SamTextShape,
    };
}

const testing = std.testing;

fn expectClean(expected: []const u8, text: []const u8) !void {
    const once = try clean(testing.allocator, text);
    defer testing.allocator.free(once);
    try testing.expectEqualStrings(expected, once);
    if (std.mem.indexOfScalar(u8, once, '&') != null) return;
    const twice = try clean(testing.allocator, once);
    defer testing.allocator.free(twice);
    try testing.expectEqualStrings(once, twice);
}

test "SAM text clean-up unescapes twice, collapses whitespace, trims and lowercases ASCII" {
    try expectClean("a&b c", "  A&amp;B \n\t c ");
    try expectClean("hi", "&#72;&#x49;");
    try expectClean("<b>", "&amp;lt;b&amp;gt;");
    try expectClean("&lt;", "&amp;amp;lt;");
    try expectClean("&bogus;", "&bogus;");
    try expectClean("a b", "a\u{3000}b");
    try expectClean("a b", "a&nbsp;b");
    try expectClean("<\"'>", "&lt;&quot;&apos;&gt;");
    try expectClean("\u{1F600}", "&#X1f600;");
    try expectClean("&#xd800; &#1114112; &#; &#12 &amp", "&#xD800; &#1114112; &#; &#12 &amp");
    try expectClean("&#99999999999999999999;", "&#99999999999999999999;");
    try expectClean("", " \u{2028}\t ");
    try expectClean("caf\u{C9}", "CAF\u{C9}");
    try expectClean("a\xffb", "A\xffB");
}

test "SAM text framing adds start and end ids and keeps at most 30 BPE ids" {
    const json =
        \\{"pre_tokenizer":{"type":"WhitespaceSplit"},"model":{"type":"WordLevel","unk_token":"?","vocab":{"?":0,"w":1,"a&b":2,"c":3}}}
    ;
    var tok = try tokenizer.parse(testing.allocator, json);
    defer tok.deinit();
    const cleaned = try frame(testing.allocator, &tok, "  A&amp;B \n\t c ");
    defer testing.allocator.free(cleaned);
    try testing.expectEqualSlices(u32, &.{ start_id, 2, 3, end_id }, cleaned);
    const empty = try frame(testing.allocator, &tok, " \n ");
    defer testing.allocator.free(empty);
    try testing.expectEqualSlices(u32, &.{ start_id, end_id }, empty);
    var words: [80]u8 = undefined;
    for (0..40) |i| {
        words[2 * i] = 'w';
        words[2 * i + 1] = ' ';
    }
    for ([_]usize{ 40, 31, 30, 3 }) |count| {
        const ids = try frame(testing.allocator, &tok, words[0 .. 2 * count]);
        defer testing.allocator.free(ids);
        const kept: usize = @min(count, 30);
        try testing.expectEqual(kept + 2, ids.len);
        try testing.expectEqual(start_id, ids[0]);
        try testing.expectEqual(end_id, ids[ids.len - 1]);
        for (ids[1 .. ids.len - 1]) |id| try testing.expectEqual(@as(u32, 1), id);
    }
}

fn table(bytes: []const u8, dtype: st.DType, rows: usize) st.Tensor {
    return .{ .dtype = dtype, .rank = 2, .shape = .{ rows, width, 1, 1, 1 }, .bytes = bytes };
}

test "SAM text embedding adds token row ids[r] and position row r" {
    const token_values = try testing.allocator.alloc(f32, 5 * width);
    defer testing.allocator.free(token_values);
    for (token_values, 0..) |*v, i| v.* = @floatFromInt(1000 * (i / width) + i % width);
    const position_values = try testing.allocator.alloc(f16, 3 * width);
    defer testing.allocator.free(position_values);
    for (position_values, 0..) |*v, i| v.* = @floatCast(0.5 * @as(f32, @floatFromInt(i / width)));
    const tokens = table(std.mem.sliceAsBytes(token_values), .f32, 5);
    const positions = table(std.mem.sliceAsBytes(position_values), .f16, 3);
    const ids = [_]u32{ 4, 0, 2 };
    var out: [3 * width]f32 = undefined;
    try embed(&out, tokens, positions, &ids);
    for (ids, 0..) |id, r| for ([_]usize{ 0, 7, width - 1 }) |c| {
        const expected = @as(f32, @floatFromInt(1000 * id + c)) + 0.5 * @as(f32, @floatFromInt(r));
        try testing.expectEqual(expected, out[r * width + c]);
    };
    try testing.expectError(error.SamTextShape, embed(&out, tokens, positions, &.{ 4, 0, 5 }));
}

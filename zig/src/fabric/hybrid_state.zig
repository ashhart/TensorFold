//! #1594 step 2b: a hybrid model's prompt state in one order both backends read, for a CUDA prefill and a Metal decode.
const std = @import("std");

/// `len` bytes at `off` of a backend's image (rows_image.Range's shape, kept here so the fabric needs no core import).
pub const Range = struct { off: u64, len: u64 };

/// The sizes that fix the order: Qwen3.8-27B has 48 linear and 16 attention layers.
pub const Geometry = struct {
    linear: u32, // linear (GDN) layers
    attention: u32, // full-attention layers
    rec_bytes: u64, // one recurrence: fp32 [value heads][dv][dk]
    conv_bytes: u64, // one convolution tail: bf16 [taps - 1][q, k and v channels]
    kv_heads: u32,
    row_bytes: u64, // one head's row at one position: bf16 head_dim

    /// Bytes every state has, whatever its length: the recurrences and tails.
    pub fn fixed(g: Geometry) u64 {
        return @as(u64, g.linear) * (g.rec_bytes + g.conv_bytes);
    }

    /// Bytes of one position: every attention layer's key and value rows.
    pub fn perToken(g: Geometry) u64 {
        return @as(u64, g.attention) * 2 * g.kv_heads * g.row_bytes;
    }

    /// Bytes of a state after `at` prompt tokens.
    pub fn bytes(g: Geometry, at: u32) u64 {
        return g.fixed() + @as(u64, at) * g.perToken();
    }

    /// Key/value row pieces: (attention layer, keys then values, head).
    pub fn rowPieces(g: Geometry) u64 {
        return @as(u64, g.attention) * 2 * g.kv_heads;
    }
};

/// Qwen3.8-27B (70ae7fa): 48 linear and 16 attention layers; 48 value heads, conv q|k|v of 16, 16, 48; 4 heads of 256.
pub const qwen27: Geometry = .{
    .linear = 48,
    .attention = 16,
    .rec_bytes = 48 * 128 * 128 * 4,
    .conv_bytes = 3 * (16 + 16 + 48) * 128 * 2,
    .kv_heads = 4,
    .row_bytes = 256 * 2,
};

/// What a piece of the neutral order holds.
pub const Kind = enum(u8) { recurrence, tail, keys, values };

/// One piece of the neutral order: `len` bytes at `off`; `layer` counts within its kind (linear or attention).
pub const Piece = struct { kind: Kind, layer: u32, head: u32 = 0, off: u64, len: u64 };

/// Every piece at `at`: per linear layer its recurrence then tail, then per attention layer keys then values by head.
pub fn pieces(g: Geometry, at: u32, out: *std.ArrayList(Piece), a: std.mem.Allocator) !void {
    var off: u64 = 0;
    for (0..g.linear) |i| {
        try out.append(a, .{ .kind = .recurrence, .layer = @intCast(i), .off = off, .len = g.rec_bytes });
        off += g.rec_bytes;
        try out.append(a, .{ .kind = .tail, .layer = @intCast(i), .off = off, .len = g.conv_bytes });
        off += g.conv_bytes;
    }
    const rows = @as(u64, at) * g.row_bytes;
    for (0..g.attention) |l| for ([_]Kind{ .keys, .values }) |k| for (0..g.kv_heads) |h| {
        try out.append(a, .{ .kind = k, .layer = @intCast(l), .head = @intCast(h), .off = off, .len = rows });
        off += rows;
    };
}

/// A CUDA copy's image (cuda_disk.zig) in this order: its words, every recurrence, every tail, then the same rows.
pub fn cudaRanges(g: Geometry, at: u32, words: u64, out: *std.ArrayList(Range), a: std.mem.Allocator) !void {
    const tails = words + @as(u64, g.linear) * g.rec_bytes;
    for (0..g.linear) |i| {
        try out.append(a, .{ .off = words + i * g.rec_bytes, .len = g.rec_bytes });
        try out.append(a, .{ .off = tails + i * g.conv_bytes, .len = g.conv_bytes });
    }
    const kv = @as(u64, at) * g.perToken();
    if (kv > 0) try out.append(a, .{ .off = words + g.fixed(), .len = kv });
}

/// A Metal state's image (qwen27_disk.zig) in this order: one range after its words, since the order is Metal's own.
pub fn metalRanges(g: Geometry, at: u32, words: u64, out: *std.ArrayList(Range), a: std.mem.Allocator) !void {
    try out.append(a, .{ .off = words, .len = g.bytes(at) });
}

/// A frame of at most a frame's bytes inside one piece: a recurrence or tail whole, or whole key or value rows.
pub const Frame = struct { piece: u32, row: u64 = 0, rows: u64 = 0, off: u64, len: u64 };

pub fn frames(g: Geometry, at: u32, max: u64, out: *std.ArrayList(Frame), a: std.mem.Allocator) !void {
    if (max < g.rec_bytes or max < g.conv_bytes or max < g.row_bytes) return error.FrameTooSmall;
    var list: std.ArrayList(Piece) = .empty;
    defer list.deinit(a);
    try pieces(g, at, &list, a);
    const per = max / g.row_bytes; // whole rows a frame takes
    for (list.items, 0..) |p, i| switch (p.kind) {
        .recurrence, .tail => try out.append(a, .{ .piece = @intCast(i), .off = p.off, .len = p.len }),
        .keys, .values => {
            var r: u64 = 0;
            while (r < at) {
                const n = @min(per, at - r);
                try out.append(a, .{ .piece = @intCast(i), .row = r, .rows = n, .off = p.off + r * g.row_bytes, .len = n * g.row_bytes });
                r += n;
            }
        },
    };
}

/// What a state's receiver checks before it lands a byte: the order's version, the sizes, the length and the prompt.
pub const Header = extern struct {
    magic: [4]u8 = magic,
    version: u16 = version,
    pad: u16 = 0,
    at: u32,
    linear: u32,
    attention: u32,
    kv_heads: u32,
    rec_bytes: u64,
    conv_bytes: u64,
    row_bytes: u64,
    total: u64, // Geometry.bytes(at)
    token_sha256: [32]u8, // of the prompt's first `at` token IDs as little-endian u32

    pub fn of(g: Geometry, at: u32, prompt: []const u32) Header {
        return .{ .at = at, .linear = g.linear, .attention = g.attention, .kv_heads = g.kv_heads, .rec_bytes = g.rec_bytes, .conv_bytes = g.conv_bytes, .row_bytes = g.row_bytes, .total = g.bytes(at), .token_sha256 = tokenSha(prompt[0..at]) };
    }

    /// Refuses a state for another order, model shape, length or prompt, by name.
    pub fn check(h: Header, g: Geometry, prompt: []const u32) !void {
        if (!std.mem.eql(u8, &h.magic, &magic)) return error.NotAHybridState;
        if (h.version != version) return error.HybridStateVersion;
        if (h.linear != g.linear or h.attention != g.attention or h.kv_heads != g.kv_heads or h.rec_bytes != g.rec_bytes or
            h.conv_bytes != g.conv_bytes or h.row_bytes != g.row_bytes) return error.HybridStateGeometry;
        if (h.at == 0 or h.at >= prompt.len) return error.HybridStateLength; // the receiver prefills at least one token
        if (h.total != g.bytes(h.at)) return error.HybridStateGeometry;
        if (!std.mem.eql(u8, &h.token_sha256, &tokenSha(prompt[0..h.at]))) return error.HybridStatePrompt;
    }
};

pub const magic = [4]u8{ 'T', 'F', 'H', 'S' };
pub const version: u16 = 1;

pub fn tokenSha(tokens: []const u32) [32]u8 {
    var s = std.crypto.hash.sha2.Sha256.init(.{});
    for (tokens) |t| {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, t, .little);
        s.update(&b);
    }
    return s.finalResult();
}

/// Copies `ranges` of `src` back to back into `out`, as an image's gather does.
pub fn gather(src: []const u8, ranges: []const Range, out: []u8) !void {
    var o: usize = 0;
    for (ranges) |r| {
        if (r.off + r.len > src.len or o + r.len > out.len) return error.BadRange;
        @memcpy(out[o..][0..@intCast(r.len)], src[@intCast(r.off)..][0..@intCast(r.len)]);
        o += @intCast(r.len);
    }
    if (o != out.len) return error.BadRange;
}

// --- tests --------------------------------------------------------------------------------------------------------

const testing = std.testing;

test "Qwen3.8-27B's state: about 147 MiB whatever its length, plus 64 KiB a token" {
    try testing.expectEqual(@as(u64, 153_944_064), qwen27.fixed());
    try testing.expectEqual(@as(u64, 65_536), qwen27.perToken());
    try testing.expectEqual(@as(u64, 153_944_064 + 16_384 * 65_536), qwen27.bytes(16_384));
    try testing.expectEqual(@as(u64, 128), qwen27.rowPieces()); // 16 layers, keys and values, 4 heads
}

/// A small model's state, every byte naming what it holds, laid out as each backend's image lays it out.
const Fake = struct {
    const g: Geometry = .{ .linear = 3, .attention = 2, .rec_bytes = 12, .conv_bytes = 6, .kv_heads = 2, .row_bytes = 4 };
    const at: u32 = 5;

    fn byte(kind: Kind, layer: u64, head: u64, i: u64) u8 {
        return @truncate(@backingInt(kind) * 61 + layer * 17 + head * 7 + i * 3 + 1);
    }

    fn rec(out: []u8, layer: u64) void {
        for (out, 0..) |*b, i| b.* = byte(.recurrence, layer, 0, i);
    }

    fn tail(out: []u8, layer: u64) void {
        for (out, 0..) |*b, i| b.* = byte(.tail, layer, 0, i);
    }

    fn kv(out: []u8, kind: Kind, layer: u64, head: u64) void {
        for (out, 0..) |*b, i| b.* = byte(kind, layer, head, i);
    }

    /// cuda_disk.zig's image: 16 words, every recurrence, every tail, then (layer, keys then values, head) rows.
    fn cuda(a: std.mem.Allocator) ![]u8 {
        const out = try a.alloc(u8, 16 + g.bytes(at));
        @memset(out[0..16], 0xee);
        var o: usize = 16;
        for (0..g.linear) |l| {
            rec(out[o..][0..g.rec_bytes], l);
            o += g.rec_bytes;
        }
        for (0..g.linear) |l| {
            tail(out[o..][0..g.conv_bytes], l);
            o += g.conv_bytes;
        }
        o = kvRows(out, o);
        std.debug.assert(o == out.len);
        return out;
    }

    /// qwen27_disk.zig's image: the drafter's 8-byte end, per layer its recurrence then its tail, then the same rows.
    fn metal(a: std.mem.Allocator) ![]u8 {
        const out = try a.alloc(u8, 8 + g.bytes(at));
        @memset(out[0..8], 0);
        var o: usize = 8;
        for (0..g.linear) |l| {
            rec(out[o..][0..g.rec_bytes], l);
            o += g.rec_bytes;
            tail(out[o..][0..g.conv_bytes], l);
            o += g.conv_bytes;
        }
        o = kvRows(out, o);
        std.debug.assert(o == out.len);
        return out;
    }

    fn kvRows(out: []u8, from: usize) usize {
        var o = from;
        const n: usize = @intCast(@as(u64, at) * g.row_bytes);
        for (0..g.attention) |l| for ([_]Kind{ .keys, .values }) |k| for (0..g.kv_heads) |h| {
            kv(out[o..][0..n], k, l, h);
            o += n;
        };
        return o;
    }
};

test "a CUDA copy read through its ranges is byte for byte the Metal image after its word" {
    const a = testing.allocator;
    const c = try Fake.cuda(a);
    defer a.free(c);
    const m = try Fake.metal(a);
    defer a.free(m);
    var cr: std.ArrayList(Range) = .empty;
    defer cr.deinit(a);
    try cudaRanges(Fake.g, Fake.at, 16, &cr, a);
    const neutral = try a.alloc(u8, @intCast(Fake.g.bytes(Fake.at)));
    defer a.free(neutral);
    try gather(c, cr.items, neutral);
    var mr: std.ArrayList(Range) = .empty;
    defer mr.deinit(a);
    try metalRanges(Fake.g, Fake.at, 8, &mr, a);
    const want = try a.alloc(u8, neutral.len);
    defer a.free(want);
    try gather(m, mr.items, want);
    try testing.expectEqualSlices(u8, want, neutral);
}

test "each piece of the neutral order holds what it names" {
    const a = testing.allocator;
    const m = try Fake.metal(a);
    defer a.free(m);
    var list: std.ArrayList(Piece) = .empty;
    defer list.deinit(a);
    try pieces(Fake.g, Fake.at, &list, a);
    try testing.expectEqual(@as(usize, 2 * Fake.g.linear + Fake.g.rowPieces()), list.items.len);
    var end: u64 = 0;
    for (list.items) |p| {
        try testing.expectEqual(end, p.off);
        end = p.off + p.len;
        for (m[@intCast(8 + p.off)..][0..@intCast(p.len)], 0..) |b, i| try testing.expectEqual(Fake.byte(p.kind, p.layer, p.head, i), b);
    }
    try testing.expectEqual(Fake.g.bytes(Fake.at), end);
}

test "frames cover the state once, each inside one piece: states whole, rows whole and at most the frame's size" {
    const a = testing.allocator;
    var fs: std.ArrayList(Frame) = .empty;
    defer fs.deinit(a);
    try frames(Fake.g, Fake.at, 12, &fs, a); // 3 rows a frame: 5 rows go as 3 and 2
    var list: std.ArrayList(Piece) = .empty;
    defer list.deinit(a);
    try pieces(Fake.g, Fake.at, &list, a);
    var end: u64 = 0;
    for (fs.items) |f| {
        try testing.expectEqual(end, f.off);
        end += f.len;
        try testing.expect(f.len <= 12);
        const p = list.items[f.piece];
        try testing.expect(f.off >= p.off and f.off + f.len <= p.off + p.len);
        switch (p.kind) {
            .recurrence, .tail => try testing.expectEqual(p.len, f.len),
            .keys, .values => try testing.expectEqual(f.rows * Fake.g.row_bytes, f.len),
        }
    }
    try testing.expectEqual(Fake.g.bytes(Fake.at), end);
    try testing.expectError(error.FrameTooSmall, frames(Fake.g, Fake.at, 8, &fs, a)); // a recurrence is 12 bytes
}

test "the header refuses another order, model, length or prompt by name" {
    const prompt = [_]u32{ 11, 22, 33, 44, 55, 66 };
    const h = Header.of(Fake.g, 5, &prompt);
    try h.check(Fake.g, &prompt);
    var other = prompt;
    other[2] = 34;
    try testing.expectError(error.HybridStatePrompt, h.check(Fake.g, &other));
    try testing.expectError(error.HybridStateLength, h.check(Fake.g, prompt[0..5])); // nothing left to prefill
    var g2 = Fake.g;
    g2.kv_heads = 3;
    try testing.expectError(error.HybridStateGeometry, h.check(g2, &prompt));
    var v = h;
    v.version = 2;
    try testing.expectError(error.HybridStateVersion, v.check(Fake.g, &prompt));
    var bad = h;
    bad.magic[0] = 'X';
    try testing.expectError(error.NotAHybridState, bad.check(Fake.g, &prompt));
    try testing.expectEqual(@as(usize, 88), @sizeOf(Header)); // 8 + 4 words + 4 sizes + the sha
}

//! #1594 step 2b: a hybrid state's neutral image (hybrid_state.zig) as an MKVH protocol 2 export, and landed back.
const std = @import("std");
const hs = @import("hybrid_state.zig");
const manifest = @import("manifest.zig");
const frames = @import("frames.zig");
const Allocator = std.mem.Allocator;

/// What the manifest names beyond the geometry's sizes; `Plan.check` holds them to those sizes.
pub const Shapes = struct {
    rec: [3]u64, // fp32 [value heads][dv][dk]
    conv: [2]u64, // bf16 [taps - 1][channels]
    head_dim: u64, // bf16
    interval: u32, // every interval-th layer attends, the last of its group (Qwen3.8: three linear, then one)
};

pub const qwen27: Shapes = .{ .rec = .{ 48, 128, 128 }, .conv = .{ 3, 10240 }, .head_dim = 256, .interval = 4 };

/// Recurrent parts per linear layer (neutral, Metal's) or all recurrences, then all tails (a CUDA copy's); rows alike.
pub const Order = enum { neutral, cuda };

/// One state's export: the geometry, its shapes, `at` prompt tokens and `block` tokens to an attention page row.
pub const Plan = struct {
    g: hs.Geometry,
    s: Shapes,
    at: u32,
    block: u32,
    order: Order = .neutral, // the image `build` reads; a landing writes the neutral order

    /// Refuses a length off the page rows, an interleave the layer counts do not fit, or shapes off the sizes.
    pub fn check(p: Plan) !void {
        if (p.block == 0 or p.at == 0 or p.at % p.block != 0) return error.HybridBlock;
        if (p.s.interval < 2 or p.g.linear != p.g.attention * (p.s.interval - 1)) return error.HybridLayout;
        if (p.s.rec[0] * p.s.rec[1] * p.s.rec[2] * 4 != p.g.rec_bytes) return error.HybridLayout;
        if (p.s.conv[0] * p.s.conv[1] * 2 != p.g.conv_bytes or p.s.head_dim * 2 != p.g.row_bytes) return error.HybridLayout;
    }

    pub fn layers(p: Plan) u32 {
        return p.g.linear + p.g.attention;
    }

    /// Model layer `index` as linear layer n or attention layer n; null past the model's layers.
    pub fn which(p: Plan, index: u32) ?Which {
        if (index >= p.layers()) return null;
        const group = index / p.s.interval;
        const pos = index % p.s.interval;
        if (pos == p.s.interval - 1) return .{ .attention = true, .n = group };
        return .{ .attention = false, .n = group * (p.s.interval - 1) + pos };
    }

    fn blocks(p: Plan) u64 {
        return p.at / p.block;
    }

    /// An attention layer's page shape: [keys and values][heads][page rows][tokens][head_dim].
    fn pageShape(p: Plan) [5]u64 {
        return .{ 2, p.g.kv_heads, p.blocks(), p.block, p.s.head_dim };
    }

    fn linearOff(p: Plan, n: u32) u64 {
        return @as(u64, n) * (p.g.rec_bytes + p.g.conv_bytes);
    }

    /// Where linear layer n's recurrence (part 0) or tail (part 1) sits in an image of this plan's order.
    fn partOff(p: Plan, n: u32, part: u1) u64 {
        return switch (p.order) {
            .neutral => p.linearOff(n) + if (part == 0) 0 else p.g.rec_bytes,
            .cuda => if (part == 0) @as(u64, n) * p.g.rec_bytes else @as(u64, p.g.linear) * p.g.rec_bytes + @as(u64, n) * p.g.conv_bytes,
        };
    }

    fn attentionBytes(p: Plan) u64 {
        return 2 * @as(u64, p.g.kv_heads) * p.at * p.g.row_bytes;
    }

    fn attentionOff(p: Plan, n: u32) u64 {
        return p.g.fixed() + @as(u64, n) * p.attentionBytes();
    }
};

pub const Which = struct { attention: bool, n: u32 };

const rec_dims = [_]manifest.PartDim{ .head, .value, .key };
const conv_dims = [_]manifest.PartDim{ .tap, .channel };
const page_dims = [_]manifest.Dim{ .kv, .head, .block, .token, .head_dim };

/// The export's layers in model order, and each one's source in `image`; slices live in the caller's allocator.
pub const Built = struct { layers: []manifest.Layer, sources: []frames.Source };

/// Layers and sources over `image`, a state in the plan's order (Geometry.bytes(at) long); nothing is copied.
pub fn build(a: Allocator, p: Plan, image: []const u8) !Built {
    try p.check();
    if (image.len != p.g.bytes(p.at)) return error.HybridStateLength;
    const layers = try a.alloc(manifest.Layer, p.layers());
    const sources = try a.alloc(frames.Source, p.layers());
    const rec = try a.dupe(u64, &p.s.rec);
    const conv = try a.dupe(u64, &p.s.conv);
    const shape = try a.dupe(u64, &p.pageShape());
    const rows = try a.alloc(u64, @intCast(p.blocks()));
    for (rows, 0..) |*r, i| r.* = i;
    const parts = [_]manifest.Part{
        .{ .name = "recurrence", .shape = rec, .dims = &rec_dims, .dtype = .float32 },
        .{ .name = "conv", .shape = conv, .dims = &conv_dims, .dtype = .bfloat16 },
    };
    const kept_parts = try a.dupe(manifest.Part, &parts);
    for (layers, sources, 0..) |*l, *s, i| {
        const index: u32 = @intCast(i);
        const w = p.which(index).?;
        if (!w.attention) {
            const pieces = try a.alloc([]const u8, 2);
            pieces[0] = image[@intCast(p.partOff(w.n, 0))..][0..@intCast(p.g.rec_bytes)];
            pieces[1] = image[@intCast(p.partOff(w.n, 1))..][0..@intCast(p.g.conv_bytes)];
            l.* = .{ .index = index, .kind = .recurrent, .shape = &.{}, .dims = &.{}, .dtype = .float32, .parts = kept_parts };
            s.* = .{ .bytes = &.{}, .shape = &.{}, .block_dim = 0, .elem = 1, .rows = &.{}, .parts = pieces };
        } else {
            const off: usize = @intCast(p.attentionOff(w.n));
            const heads = p.g.kv_heads;
            l.* = .{ .index = index, .kind = .attention, .shape = shape, .dims = &page_dims, .dtype = .bfloat16, .heads = heads, .total_heads = heads, .head_size = p.s.head_dim };
            s.* = .{ .bytes = image[off..][0..@intCast(p.attentionBytes())], .shape = shape, .block_dim = 2, .elem = 2, .rows = rows };
        }
    }
    return .{ .layers = layers, .sources = sources };
}

/// A state landing in `image` frame by frame; whole once every byte of the plan has landed (frames never repeat).
pub const Landing = struct {
    plan: Plan,
    image: []u8,
    landed: u64 = 0,

    /// Lands rows [row_start, row_start + rows) of `layer` in the neutral order, after checking the layer is ours.
    pub fn put(d: *Landing, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) !void {
        var p = d.plan;
        p.order = .neutral;
        const w = p.which(layer.index) orelse return error.HybridLayer;
        if (!w.attention) {
            if (layer.kind != .recurrent or !ourParts(p, layer.parts)) return error.HybridLayer;
            if (rows != 1 or row_start > 1) return error.HybridFrame;
            const len = if (row_start == 0) p.g.rec_bytes else p.g.conv_bytes;
            if (bytes.len != len) return error.HybridFrame;
            const off = p.partOff(w.n, @intCast(row_start));
            @memcpy(d.image[@intCast(off)..][0..bytes.len], bytes);
        } else {
            const shape = p.pageShape();
            if (layer.kind != .attention or layer.dtype != .bfloat16) return error.HybridLayer;
            if (!std.mem.eql(u64, layer.shape, &shape) or !std.mem.eql(manifest.Dim, layer.dims, &page_dims)) return error.HybridLayer;
            if (rows == 0 or rows > p.blocks() or row_start > p.blocks() - rows) return error.HybridFrame;
            const inner = @as(u64, p.block) * p.g.row_bytes;
            const outer = 2 * @as(u64, p.g.kv_heads);
            if (bytes.len != outer * rows * inner) return error.HybridFrame;
            const base = p.attentionOff(w.n);
            const run: usize = @intCast(rows * inner);
            for (0..outer) |o| {
                const to: usize = @intCast(base + (o * p.blocks() + row_start) * inner);
                @memcpy(d.image[to..][0..run], bytes[o * run ..][0..run]);
            }
        }
        d.landed += bytes.len;
    }

    /// `put` as a decoder Sink's or a StateSink's put_fn, `ptr` being the landing.
    pub fn putAny(ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void {
        const d: *Landing = @ptrCast(@alignCast(ptr));
        return d.put(layer, row_start, rows, bytes);
    }

    pub fn whole(d: *const Landing) bool {
        return d.landed == d.plan.g.bytes(d.plan.at);
    }
};

fn ourParts(p: Plan, parts: []const manifest.Part) bool {
    if (parts.len != 2) return false;
    const r = parts[0];
    const c = parts[1];
    return std.mem.eql(u8, r.name, "recurrence") and r.dtype == .float32 and std.mem.eql(u64, r.shape, &p.s.rec) and
        std.mem.eql(manifest.PartDim, r.dims, &rec_dims) and std.mem.eql(u8, c.name, "conv") and c.dtype == .bfloat16 and
        std.mem.eql(u64, c.shape, &p.s.conv) and std.mem.eql(manifest.PartDim, c.dims, &conv_dims);
}

/// A small hybrid: three linear layers, then one attention layer, twice; 8 tokens in page rows of 2.
const tiny: Plan = .{
    .g = .{ .linear = 6, .attention = 2, .rec_bytes = 2 * 2 * 2 * 4, .conv_bytes = 3 * 4 * 2, .kv_heads = 2, .row_bytes = 4 * 2 },
    .s = .{ .rec = .{ 2, 2, 2 }, .conv = .{ 3, 4 }, .head_dim = 4, .interval = 4 },
    .at = 8,
    .block = 2,
};

test "Qwen3.8-27B's plan holds its sizes, and layers interleave three linear then one attention" {
    const p: Plan = .{ .g = hs.qwen27, .s = qwen27, .at = 4096, .block = 16 };
    try p.check();
    try std.testing.expectEqual(Which{ .attention = false, .n = 0 }, p.which(0).?);
    try std.testing.expectEqual(Which{ .attention = true, .n = 0 }, p.which(3).?);
    try std.testing.expectEqual(Which{ .attention = false, .n = 3 }, p.which(4).?);
    try std.testing.expectEqual(Which{ .attention = true, .n = 15 }, p.which(63).?);
    try std.testing.expectEqual(@as(?Which, null), p.which(64));
    var off = p;
    off.at = 4100;
    try std.testing.expectError(error.HybridBlock, off.check());
    var wrong = p;
    wrong.s.head_dim = 128;
    try std.testing.expectError(error.HybridLayout, wrong.check());
}

test "every frame of a built export, through its manifest text, lands the image byte for byte" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const image = try a.alloc(u8, @intCast(tiny.g.bytes(tiny.at)));
    for (image, 0..) |*b, i| b.* = @truncate(i *% 29 +% 11);
    const built = try build(a, tiny, image);
    const m: manifest.Manifest = .{ .protocol = 2, .handoff = @splat(1), .model = "tiny", .prompt_tokens = 8, .first_token = 0, .token_sha256 = @splat('0'), .block_size = 2, .tp_rank = 0, .tp_size = 1, .layers = built.layers, .frames = 0 };
    var text: std.Io.Writer.Allocating = .init(a);
    try manifest.write(m, &text.writer);
    const back = try manifest.parseUpTo(a, text.written(), 2);
    const rows = try a.alloc(frames.LayerRows, back.layers.len);
    var sizes: std.ArrayList(u64) = .empty;
    for (back.layers) |l| for (l.parts) |part| try sizes.append(a, part.bytes());
    var at: usize = 0;
    for (back.layers, rows) |l, *x| {
        x.* = .{ .rows = l.rows(), .row_bytes = l.rowBytes(), .sizes = sizes.items[at..][0..l.parts.len] };
        at += l.parts.len;
    }
    const plan = try frames.plan(a, rows, 64); // an attention layer's 4 page rows go 2 a frame
    const got = try a.alloc(u8, image.len);
    @memset(got, 0);
    var landing: Landing = .{ .plan = tiny, .image = got };
    var buf: [64]u8 = undefined;
    for (plan) |f| {
        const n = try built.sources[f.position].gather(f.row_start, f.rows, &buf);
        try landing.put(&back.layers[f.position], f.row_start, f.rows, buf[0..@intCast(n)]);
    }
    try std.testing.expect(landing.whole());
    try std.testing.expectEqualSlices(u8, image, got);
}

test "a frame for another layer kind, shape or size is refused before it lands" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const image = try a.alloc(u8, @intCast(tiny.g.bytes(tiny.at)));
    @memset(image, 7);
    const built = try build(a, tiny, image);
    const got = try a.alloc(u8, image.len);
    @memset(got, 0);
    var landing: Landing = .{ .plan = tiny, .image = got };
    const rec: [32]u8 = @splat(1);
    var swapped = built.layers[0];
    swapped.index = 3; // a recurrent layer named as layer 3, which attends
    try std.testing.expectError(error.HybridLayer, landing.put(&swapped, 0, 1, &rec));
    try std.testing.expectError(error.HybridFrame, landing.put(&built.layers[3], 0, 1, &rec));
    try std.testing.expectError(error.HybridFrame, landing.put(&built.layers[0], 0, 1, rec[0..31]));
    try std.testing.expectError(error.HybridFrame, landing.put(&built.layers[0], 2, 1, &rec));
    var moved = built.layers[3];
    moved.index = 8;
    try std.testing.expectError(error.HybridLayer, landing.put(&moved, 0, 1, &.{}));
    var long = built.layers[3];
    long.shape = &.{ 2, 2, 5, 2, 4 };
    try std.testing.expectError(error.HybridLayer, landing.put(&long, 0, 1, &.{}));
    const page: [4 * 2 * 16]u8 = @splat(2); // 4 key and value heads, 2 rows of 2 tokens of 8 bytes
    try std.testing.expectError(error.HybridFrame, landing.put(&built.layers[3], 3, 2, &page));
    try landing.put(&built.layers[3], 2, 2, &page);
    try std.testing.expect(!landing.whole());
    try std.testing.expect(std.mem.allEqual(u8, got[0..@intCast(tiny.g.fixed())], 0));
}

test "a CUDA copy's order exports the same frames as the neutral order, and lands as the neutral image" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const neutral = try a.alloc(u8, @intCast(tiny.g.bytes(tiny.at)));
    for (neutral, 0..) |*b, i| b.* = @truncate(i *% 37 +% 5);
    var ranges: std.ArrayList(hs.Range) = .empty;
    try hs.cudaRanges(tiny.g, tiny.at, 0, &ranges, a); // where each neutral piece sits in a CUDA copy
    const cuda = try a.alloc(u8, neutral.len);
    var from: usize = 0;
    for (ranges.items) |r| {
        @memcpy(cuda[@intCast(r.off)..][0..@intCast(r.len)], neutral[from..][0..@intCast(r.len)]);
        from += @intCast(r.len);
    }
    var cp = tiny;
    cp.order = .cuda;
    const built = try build(a, cp, cuda);
    const got = try a.alloc(u8, neutral.len);
    @memset(got, 0);
    var landing: Landing = .{ .plan = cp, .image = got };
    var buf: [256]u8 = undefined;
    for (built.layers, built.sources) |*l, src| {
        var row: u64 = 0;
        while (row < l.rows()) : (row += 1) {
            const n = try src.gather(row, 1, &buf);
            try landing.put(l, row, 1, buf[0..@intCast(n)]);
        }
    }
    try std.testing.expect(landing.whole());
    try std.testing.expectEqualSlices(u8, neutral, got);
}

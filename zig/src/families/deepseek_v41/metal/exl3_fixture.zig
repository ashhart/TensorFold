//! A synthetic EXL3 pack's trellis groups for tests: random words, +-1 input scales, +-1/sqrt(K) output scales.
const std = @import("std");
const host = @import("dsv41");
const x3 = @import("core").exl3_projection;

const Config = host.config.Config;
const Spec = host.pack.Spec;
pub const Book = x3.format.Codebook;

/// One group under its prefix: K x N, width in half-bits and codebook (a 3INST group has no marker).
pub const Group = struct { name: []const u8, k: usize, n: usize, halves: u16, book: Book };

/// The widths groups cycle over: mcg 2-bit, mul1 2.5-bit, 3INST 3-bit, mul1 4-bit.
pub const widths = [_]struct { u16, Book }{ .{ 4, .mcg }, .{ 5, .mul1 }, .{ 6, .inst3 }, .{ 8, .mul1 } };

/// How a tensor's bytes are drawn.
pub const Fill = enum { words, su, sv, mul1, mcg };

fn add(a: std.mem.Allocator, out: *std.ArrayList(Group), turn: *usize, name: []const u8, k: usize, n: usize) !void {
    const w = widths[turn.* % widths.len];
    turn.* += 1;
    try out.append(a, .{ .name = try a.dupe(u8, name), .k = k, .n = n, .halves = w[0], .book = w[1] });
}

/// Every trellis group layer `layer` of `c` holds under `block`, widths cycled from `turn`; names live in `a`.
pub fn groups(a: std.mem.Allocator, c: *const Config, block: []const u8, layer: u32, out: *std.ArrayList(Group), turn: *usize) !void {
    var nb: [160]u8 = undefined;
    const kg = c.heads / c.o_groups * c.head_dim;
    const W = c.expert_width;
    const dense = [_]struct { []const u8, usize, usize, bool }{
        .{ "attn.wq_a", c.hidden, c.q_lora, true },
        .{ "attn.wkv", c.hidden, c.head_dim, true },
        .{ "attn.wq_b", c.q_lora, c.heads * c.head_dim, true },
        .{ "attn.wo_b", c.o_groups * c.o_lora, c.hidden, true },
        .{ "attn.compressor.wkv", c.hidden, c.head_dim, c.isKvSource(layer) },
        .{ "attn.compressor.wgate", c.hidden, c.head_dim, c.isKvSource(layer) and c.compressRatio(layer) > 1 },
        .{ "attn.indexer.wk", c.head_dim, c.index_dim, c.isKvSource(layer) },
        .{ "attn.indexer.wq_b", c.q_lora, c.index_heads * c.index_dim, c.isIndexSource(layer) },
        .{ "ffn.shared_experts.w1", c.hidden, W, true },
        .{ "ffn.shared_experts.w2", W, c.hidden, true },
        .{ "ffn.shared_experts.w3", c.hidden, W, true },
        .{ "engram.wkv", c.hashCols() * c.engram_head_dim, (c.hc_mult + 1) * c.hidden, c.isEngram(layer) },
    };
    for (dense) |d| if (d[3]) try add(a, out, turn, try std.fmt.bufPrint(&nb, "{s}.{s}", .{ block, d[0] }), d[1], d[2]);
    for (0..c.o_groups) |g| try add(a, out, turn, try std.fmt.bufPrint(&nb, "{s}.attn.wo_a.slice.{d}", .{ block, g }), kg, c.o_lora);
    for (0..c.expertsOf(layer).count) |e| for ([_][]const u8{ "w1", "w2", "w3" }) |w| {
        const two = std.mem.eql(u8, w, "w2");
        try add(a, out, turn, try std.fmt.bufPrint(&nb, "{s}.ffn.experts.{d}.{s}", .{ block, e, w }), if (two) W else c.hidden, if (two) c.hidden else W);
    };
}

/// The head's group, [hidden] in, [vocab] out.
pub fn head(a: std.mem.Allocator, c: *const Config, out: *std.ArrayList(Group), turn: *usize) !void {
    try add(a, out, turn, "head", c.hidden, c.vocab);
}

/// One tensor of a group: its spec and how its bytes are drawn (`k` scales the output scales).
pub const Tensor = struct { spec: Spec, fill: Fill, k: usize };

/// `g`'s tensors: trellis [K/16][N/16][8 halves] i16, suh [K] and svh [N] f16, its marker as a scalar i32.
pub fn tensors(a: std.mem.Allocator, g: Group, out: *std.ArrayList(Tensor)) !void {
    var nb: [192]u8 = undefined;
    const sh = try a.dupe(usize, &.{ g.k / 16, g.n / 16, 8 * @as(usize, g.halves) });
    try out.append(a, .{ .spec = .{ .name = try a.dupe(u8, try std.fmt.bufPrint(&nb, "{s}.trellis", .{g.name})), .dtype = "I16", .shape = sh }, .fill = .words, .k = g.k });
    try out.append(a, .{ .spec = .{ .name = try a.dupe(u8, try std.fmt.bufPrint(&nb, "{s}.suh", .{g.name})), .dtype = "F16", .shape = try a.dupe(usize, &.{g.k}) }, .fill = .su, .k = g.k });
    try out.append(a, .{ .spec = .{ .name = try a.dupe(u8, try std.fmt.bufPrint(&nb, "{s}.svh", .{g.name})), .dtype = "F16", .shape = try a.dupe(usize, &.{g.n}) }, .fill = .sv, .k = g.k });
    const mk: ?Fill = switch (g.book) {
        .mul1 => .mul1,
        .mcg => .mcg,
        .inst3 => null,
    };
    if (mk) |f| try out.append(a, .{ .spec = .{ .name = try a.dupe(u8, try std.fmt.bufPrint(&nb, "{s}.{t}", .{ g.name, f })), .dtype = "I32", .shape = &.{} }, .fill = f, .k = g.k });
}

/// A tensor's bytes.
pub fn draw(fill: Fill, k: usize, out: []u8, rng: std.Random) void {
    const sv: f16 = @floatCast(1.0 / @sqrt(@as(f32, @floatFromInt(k))));
    switch (fill) {
        .words => rng.bytes(out),
        .su => for (0..out.len / 2) |i| std.mem.writeInt(u16, out[2 * i ..][0..2], @bitCast(if (rng.boolean()) @as(f16, 1) else @as(f16, -1)), .little),
        .sv => for (0..out.len / 2) |i| std.mem.writeInt(u16, out[2 * i ..][0..2], @bitCast(if (rng.boolean()) sv else -sv), .little),
        .mul1 => std.mem.writeInt(u32, out[0..4], x3.format.Mul1Mul, .little),
        .mcg => std.mem.writeInt(u32, out[0..4], x3.format.McgMul, .little),
    }
}

/// The tensors as one safetensors image, drawn from `seed` in order.
pub fn image(a: std.mem.Allocator, ts: []const Tensor, seed: u64) ![]u8 {
    const specs = try a.alloc(Spec, ts.len);
    defer a.free(specs);
    for (specs, ts) |*s, t| s.* = t.spec;
    const bytes = try host.pack.image(a, specs);
    var at = 8 + @as(usize, @intCast(std.mem.readInt(u64, bytes[0..8], .little)));
    var prng = std.Random.DefaultPrng.init(seed);
    for (ts) |t| {
        var n = host.pack.DType.parse(t.spec.dtype).?.size();
        for (t.spec.shape) |d| n *= d;
        draw(t.fill, t.k, bytes[at..][0..n], prng.random());
        at += n;
    }
    return bytes;
}

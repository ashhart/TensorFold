//! The checkpoint's weights onto the device: affine projections packed as qlinear tiles of their served width.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");

const Geometry = geometry.Geometry;
const Slot = geometry.Slot;
const Engine = engine.Engine;
const Proj = engine.Proj;
const Affine4 = cuda.qlinear.Affine4;
const AffineN = cuda.qlinear.AffineN;
const servedBits = cuda.qlinear.servedBits;
const rowBits = cuda.qlinear.rowBits;
const A4 = cuda.experts.Affine4Experts;
const A3 = cuda.experts3.Affine3Experts;
const DeviceBuffer = cuda.DeviceBuffer;
const aligned = engine.aligned;
const head_dim = geometry.head_dim;

/// The packed tiles' bytes of an n-by-k projection of `bits` codes (qlinear's words, scales and biases).
pub fn packedBytes(n: usize, k: usize, bits: u8) usize {
    if (servedBits(bits) == 4) {
        const l = Affine4.layout(n, k);
        return aligned(l.words) + 2 * aligned(l.scales);
    }
    const l = AffineN.layout(n, k, servedBits(bits));
    return aligned(l.words) + 2 * aligned(l.scales);
}

/// The checkpoint's own bytes of an n-by-k projection of `bits` codes: words, then its scales and biases.
pub fn rawBytes(n: usize, k: usize, bits: u8) usize {
    return n * k * bits / 8 + 2 * (n * k / 64 * 2);
}

/// A slot's output and input widths.
fn shape(g: Geometry, slot: Slot) [2]usize {
    return switch (slot) {
        .gate, .up => .{ g.intermediate, g.hidden },
        .down => .{ g.hidden, g.intermediate },
        .qkv => .{ g.convWidth(), g.hidden },
        .z => .{ g.valueWidth(), g.hidden },
        .a, .b => .{ g.v_heads, g.hidden },
        .out => .{ g.hidden, g.valueWidth() },
        .q => .{ 2 * g.attnWidth(), g.hidden },
        .k, .v => .{ g.kv_heads * head_dim, g.hidden },
        .o => .{ g.hidden, g.attnWidth() },
    };
}

/// The slots layer `i` loads: a dense MLP's (routed experts load apart) and its mixer's.
fn slots(g: Geometry, i: usize) []const Slot {
    if (g.moe()) return if (g.linear[i]) &.{ .qkv, .z, .a, .b, .out } else &.{ .q, .k, .v, .o };
    return if (g.linear[i]) &.{ .gate, .up, .down, .qkv, .z, .a, .b, .out } else &.{ .gate, .up, .down, .q, .k, .v, .o };
}

/// The token embedding as the embed kernels read it: MLX rows at the width it is served (4 or 8 bits).
fn embedBytes(g: Geometry) usize {
    return aligned(g.vocab * g.hidden * rowBits(g.bits.embed) / 8) + 2 * aligned(g.vocab * g.hidden / 64 * 2);
}

/// A layer's routed experts' device bytes: pair table, down table, router rows (3-bit: the shared tables apart).
pub fn routedBytes(experts: usize, intermediate: usize, hidden: usize, bits: u8) usize {
    const rows = aligned((experts + 1) * hidden * 2);
    if (bits == 3) return aligned(A3.pairBytes(experts, intermediate, hidden)) + aligned(A3.bytes(experts, hidden, intermediate)) + aligned(A4.pairBytes(1, intermediate, hidden)) + aligned(A4.bytes(1, hidden, intermediate)) + rows;
    return aligned(A4.pairBytes(experts + 1, intermediate, hidden)) + aligned(A4.bytes(experts + 1, hidden, intermediate)) + rows;
}

/// Experts a 3-bit pack takes at once: its widened words and their 4-bit blocks sit in the stage beside the raw parts.
const chunk3 = 32;

/// The checkpoint's bytes of `count` experts' n-by-k 3-bit projection, then the scratch a chunk of them packs through.
fn raw3Bytes(count: usize, n: usize, k: usize) usize {
    const s = A3.scratchBytes(chunk3, n, k);
    return aligned(count * (n * k * 3 / 8 + 2 * (n * k / 64 * 2))) + aligned(s[0]) + s[1];
}

/// What `load` takes from the arena: every packed projection and vector, each allocation aligned.
pub fn total(g: Geometry) usize {
    var t: usize = embedBytes(g);
    t += packedBytes(g.vocab, g.hidden, g.bits.head); // the head, tied or not
    t += aligned(g.hidden * 2);
    for (0..g.layers) |i| {
        t += 2 * aligned(g.hidden * 2);
        if (g.moe()) t += routedBytes(g.experts, g.intermediate, g.hidden, g.expert_bits);
        for (slots(g, i)) |slot| {
            const nk = shape(g, slot);
            t += packedBytes(nk[0], nk[1], g.bits.of(i, slot));
        }
        if (g.linear[i]) {
            t += aligned(g.convWidth() * geometry.taps * 2) + aligned(geometry.dk * 2) + aligned(g.v_heads * 4) + aligned(g.v_heads * 2);
        } else {
            t += 2 * aligned(head_dim * 2);
        }
    }
    return t;
}

/// The largest projection's bytes as stored: the staging buffer the load packs each one from (and widens through).
pub fn stageBytes(g: Geometry) usize {
    var most = rawBytes(g.vocab, g.hidden, g.bits.head);
    if (rowBits(g.bits.embed) != g.bits.embed) most = @max(most, rawBytes(g.vocab, g.hidden, g.bits.embed));
    for (0..g.layers) |i| for (slots(g, i)) |slot| {
        const nk = shape(g, slot);
        most = @max(most, rawBytes(nk[0], nk[1], g.bits.of(i, slot)));
    };
    if (g.moe()) most = @max(most, @max(rawBytes(g.experts * g.intermediate, g.hidden, 4), rawBytes(g.experts * g.hidden, g.intermediate, 4)));
    if (g.moe() and g.expert_bits == 3) most = @max(most, @max(raw3Bytes(g.experts, g.intermediate, g.hidden), raw3Bytes(g.experts, g.hidden, g.intermediate)));
    return most;
}

/// `into` is the engine or a drafter's weights: anything with `alloc(bytes)` out of its `arena`.
pub fn upload(into: anytype, bytes: []const u8) !u64 {
    const p = try into.alloc(bytes.len);
    try into.arena.upload(p - into.arena.ptr, bytes);
    return p;
}

fn put(into: anytype, t: core.checkpoint.Tensor) !u64 {
    return upload(into, t.bytes);
}

/// A vector the kernels read as f32, stored as f32 (the 2B) or bf16 (the 27B's A_log).
fn f32vec(e: *Engine, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, n: usize) !u64 {
    var buf: [160]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&buf, fmt, args));
    if (t.is(.f32, &.{n})) return put(e, t);
    if (!t.is(.bf16, &.{n})) return error.UnexpectedTensor;
    const words = try e.gpa.alloc(u32, n);
    defer e.gpa.free(words);
    for (words, std.mem.bytesAsSlice(u16, t.bytes)) |*w, h| w.* = @as(u32, h) << 16;
    return upload(e, std.mem.sliceAsBytes(words));
}

pub fn tensor(into: anytype, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: core.checkpoint.DType, shape_: []const usize) !u64 {
    var buf: [160]u8 = undefined;
    return put(into, try ck.expect(try std.fmt.bufPrint(&buf, fmt, args), dtype, shape_));
}

/// Raw words, scales and biases staged in order; each part's device pointer.
fn stageParts(stage: *DeviceBuffer, parts: [3][]const u8) ![3]u64 {
    var raw: [3]u64 = undefined;
    var at: usize = 0;
    for (parts, &raw) |p, *r| {
        if (at + p.len > stage.len) return error.StageTooSmall;
        try stage.upload(at, p);
        r.* = stage.ptr + at;
        at += p.len;
    }
    return raw;
}

/// Staged affine-4 parts packed into qlinear's tiles out of `into`'s arena (the caller waits before restaging).
fn pack4(e: *Engine, into: anytype, raw: [3]u64, n: usize, k: usize) !Affine4 {
    const l = Affine4.layout(n, k);
    const q: Affine4 = .{ .w = try into.alloc(l.words), .s = try into.alloc(l.scales), .b = try into.alloc(l.scales), .n = n, .k = k, .npad = l.npad };
    try e.affine.pack(e.stream, raw[0], raw[1], raw[2], q);
    return q;
}

/// An MLX projection of `bits` codes packed from the staging buffer into `into`'s arena; 4-bit modules take affine-4.
pub fn linear(e: *Engine, into: anytype, ck: *core.Checkpoint, stage: *DeviceBuffer, prefix: []const u8, name: []const u8, n: usize, k: usize, bits: u8) !Proj {
    var buf: [160]u8 = undefined;
    var parts: [3][]const u8 = undefined;
    const specs = [_]struct { []const u8, core.checkpoint.DType, usize }{ .{ "weight", .u32, k * bits / 32 }, .{ "scales", .bf16, k / 64 }, .{ "biases", .bf16, k / 64 } };
    for (specs, &parts) |p, *part| part.* = (try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ prefix, name, p[0] }), p[1], &.{ n, p[2] })).bytes;
    const raw = try stageParts(stage, parts);
    const served = servedBits(bits);
    const out: Proj = if (served == 4) .{ .a4 = try pack4(e, into, raw, n, k) } else blk: {
        const l = AffineN.layout(n, k, served);
        const q: AffineN = .{ .w = try into.alloc(l.words), .s = try into.alloc(l.scales), .b = try into.alloc(l.scales), .n = n, .k = k, .npad = l.npad, .bits = served };
        try e.wide.packWords(e.stream, raw[0], bits, n, k, q.w);
        try e.wide.packTable(e.stream, raw[1], n, k, q.npad, q.s);
        try e.wide.packTable(e.stream, raw[2], n, k, q.npad, q.b);
        break :blk .{ .an = q };
    };
    try e.stream.synchronize(); // the next projection's staging overwrites this one's
    return out;
}

/// Inputs [from, from + len) of an n-by-k affine-4 projection as their own packed projection (the MTP's fc halves).
pub fn linearSlice(e: *Engine, into: anytype, ck: *core.Checkpoint, stage: *DeviceBuffer, prefix: []const u8, name: []const u8, n: usize, k: usize, from: usize, len: usize) !Affine4 {
    var buf: [160]u8 = undefined;
    const specs = [_]struct { []const u8, core.checkpoint.DType, usize, usize }{ .{ "weight", .u32, k / 8, 4 }, .{ "scales", .bf16, k / 64, 2 }, .{ "biases", .bf16, k / 64, 2 } };
    var parts: [3][]u8 = undefined;
    var taken: usize = 0;
    errdefer for (parts[0..taken]) |p| e.gpa.free(p);
    for (specs, &parts) |p, *part| {
        const t = try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ prefix, name, p[0] }), p[1], &.{ n, p[2] });
        const per = k / p[2]; // inputs a column holds
        const cols = len / per;
        part.* = try e.gpa.alloc(u8, n * cols * p[3]);
        taken += 1;
        for (0..n) |r| @memcpy(part.*[r * cols * p[3] ..][0 .. cols * p[3]], t.bytes[(r * p[2] + from / per) * p[3] ..][0 .. cols * p[3]]);
    }
    defer for (parts) |p| e.gpa.free(p);
    const q = try pack4(e, into, try stageParts(stage, .{ parts[0], parts[1], parts[2] }), n, len);
    try e.stream.synchronize(); // the next projection's staging overwrites this one's
    return q;
}

fn bf16f(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// f32 to bf16, round to nearest even (finite values).
fn f32bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// 8-bit affine rows (four codes a word, a scale and bias a group of 64) dequantized to bf16: the router rows.
fn dequant8(dst: []u16, words: []align(1) const u32, scales: []align(1) const u16, biases: []align(1) const u16, k: usize) void {
    const groups = k / 64;
    for (0..dst.len / k) |r| {
        for (0..k) |i| {
            const code: f32 = @floatFromInt((words[r * (k / 4) + i / 4] >> @intCast(8 * (i % 4))) & 0xff);
            dst[r * k + i] = f32bf16(code * bf16f(scales[r * groups + i / 64]) + bf16f(biases[r * groups + i / 64]));
        }
    }
}

/// The router's rows as bf16 [experts + 1][hidden]: the routed experts' gate, then the shared expert's gate row.
fn routerRows(e: *Engine, into: anytype, ck: *core.Checkpoint, prefix: []const u8) !u64 {
    const g = e.g;
    const rows = try e.gpa.alloc(u16, (g.experts + 1) * g.hidden);
    defer e.gpa.free(rows);
    var buf: [160]u8 = undefined;
    const names = [_]struct { []const u8, usize, usize }{ .{ "mlp.gate", 0, g.experts }, .{ "mlp.shared_expert_gate", g.experts, 1 } };
    for (names) |n| {
        const w = try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ prefix, n[0] }), .u32, &.{ n[2], g.hidden / 4 });
        const s = try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.scales", .{ prefix, n[0] }), .bf16, &.{ n[2], g.hidden / 64 });
        const b = try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.biases", .{ prefix, n[0] }), .bf16, &.{ n[2], g.hidden / 64 });
        dequant8(rows[n[1] * g.hidden ..][0 .. n[2] * g.hidden], std.mem.bytesAsSlice(u32, w.bytes), std.mem.bytesAsSlice(u16, s.bytes), std.mem.bytesAsSlice(u16, b.bytes), g.hidden);
    }
    return upload(into, std.mem.sliceAsBytes(rows));
}

/// `count` stacked experts' `bits`-wide projection (one at count 0) staged as stored, packed as `out`'s mi-th of m.
fn expertTable(e: *Engine, ck: *core.Checkpoint, stage: *DeviceBuffer, prefix: []const u8, name: []const u8, count: usize, n: usize, k: usize, out: u64, m: usize, mi: usize, bits: u8) !void {
    var buf: [160]u8 = undefined;
    var parts: [3][]const u8 = undefined;
    const specs = [_]struct { []const u8, core.checkpoint.DType, usize }{ .{ "weight", .u32, k * bits / 32 }, .{ "scales", .bf16, k / 64 }, .{ "biases", .bf16, k / 64 } };
    for (specs, &parts) |p, *part| {
        const stacked: []const usize = &.{ count, n, p[2] };
        const one: []const usize = &.{ n, p[2] };
        part.* = (try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ prefix, name, p[0] }), p[1], if (count == 0) one else stacked)).bytes;
    }
    const raw = try stageParts(stage, parts);
    if (bits == 4) {
        try e.experts.packInto(e.stream, raw[0], raw[1], raw[2], out, @max(count, 1), n, k, m, mi);
    } else {
        const Pack = struct {
            fn run(owner: *Engine, w: u64, sc: u64, bias: u64, tmp: [2]u64, dst: u64, chunk_count: usize, nn: usize, kk: usize, mm: usize, index: usize) !void {
                try owner.experts3.packInto(&owner.experts, owner.stream, w, sc, bias, tmp, dst, chunk_count, nn, kk, mm, index);
            }
        };
        try @import("cuda_expert_pack.zig").chunks(e, Pack.run, stage.ptr, stage.len, raw, parts[0].len + parts[1].len + parts[2].len, out, count, n, k, m, mi);
    }
    try e.stream.synchronize(); // the next table's staging overwrites this one's
}

/// A layer's routed experts: gate and up as one pair table, down apart, the shared one last (3-bit: in 4-bit tables).
pub fn routed(e: *Engine, into: anytype, ck: *core.Checkpoint, stage: *DeviceBuffer, p: []const u8, bits: u8) !engine.Mlp {
    const g = e.g;
    const ni = g.intermediate;
    const three = bits == 3;
    const pair = try into.alloc(if (three) A3.pairBytes(g.experts, ni, g.hidden) else A4.pairBytes(g.experts + 1, ni, g.hidden));
    const down = try into.alloc(if (three) A3.bytes(g.experts, g.hidden, ni) else A4.bytes(g.experts + 1, g.hidden, ni));
    const shared_pair = if (three) try into.alloc(A4.pairBytes(1, ni, g.hidden)) else pair + A4.pairBytes(g.experts, ni, g.hidden);
    const shared_down = if (three) try into.alloc(A4.bytes(1, g.hidden, ni)) else down + A4.bytes(g.experts, g.hidden, ni);
    try expertTable(e, ck, stage, p, "mlp.switch_mlp.gate_proj", g.experts, ni, g.hidden, pair, 2, 0, bits);
    try expertTable(e, ck, stage, p, "mlp.switch_mlp.up_proj", g.experts, ni, g.hidden, pair, 2, 1, bits);
    try expertTable(e, ck, stage, p, "mlp.switch_mlp.down_proj", g.experts, g.hidden, ni, down, 1, 0, bits);
    try expertTable(e, ck, stage, p, "mlp.shared_expert.gate_proj", 0, ni, g.hidden, shared_pair, 2, 0, 4);
    try expertTable(e, ck, stage, p, "mlp.shared_expert.up_proj", 0, ni, g.hidden, shared_pair, 2, 1, 4);
    try expertTable(e, ck, stage, p, "mlp.shared_expert.down_proj", 0, g.hidden, ni, shared_down, 1, 0, 4);
    var layer: cuda.experts.Layer = .{ .format = .affine4, .up = pair, .down = down, .width = ni, .dims = g.hidden, .experts = g.experts + 1, .swiglu = true };
    if (three) {
        layer.format = .affine3;
        layer.shared = @intCast(g.experts);
        layer.shared_up = shared_pair;
        layer.shared_down = shared_down;
    }
    return .{ .moe = .{ .router = try routerRows(e, into, ck, p), .experts = layer } };
}

/// The embedding's words as the embed kernels read them: as stored at 4 or 8 bits, else widened through staging.
fn embedWords(e: *Engine, ck: *core.Checkpoint, stage: *DeviceBuffer, m: []const u8) !u64 {
    const g = e.g;
    const bits = g.bits.embed;
    const wide = rowBits(bits);
    var buf: [160]u8 = undefined;
    const t = try ck.expect(try std.fmt.bufPrint(&buf, "{s}embed_tokens.weight", .{m}), .u32, &.{ g.vocab, g.hidden * bits / 32 });
    if (bits == wide) return put(e, t);
    try stage.upload(0, t.bytes);
    const out = try e.alloc(g.vocab * g.hidden * wide / 8);
    try e.wide.widen(e.stream, stage.ptr, bits, g.vocab, g.hidden, out, wide);
    try e.stream.synchronize();
    return out;
}

pub fn load(e: *Engine, ck: *core.Checkpoint, stage: *DeviceBuffer) !void {
    const g = e.g;
    const m = "language_model.model.";
    e.embed_bits = rowBits(g.bits.embed);
    e.embed_w = try embedWords(e, ck, stage, m);
    e.embed_s = try tensor(e, ck, "{s}embed_tokens.scales", .{m}, .bf16, &.{ g.vocab, g.hidden / 64 });
    e.embed_b = try tensor(e, ck, "{s}embed_tokens.biases", .{m}, .bf16, &.{ g.vocab, g.hidden / 64 });
    e.head = if (g.tied) try linear(e, e, ck, stage, m, "embed_tokens", g.vocab, g.hidden, g.bits.head) else try linear(e, e, ck, stage, "language_model.", "lm_head", g.vocab, g.hidden, g.bits.head);
    e.final_norm = try tensor(e, ck, "{s}norm.weight", .{m}, .bf16, &.{g.hidden});
    const conv = g.convWidth();
    for (e.layers, 0..) |*l, i| {
        var pbuf: [96]u8 = undefined;
        const p = try std.fmt.bufPrint(&pbuf, "{s}layers.{d}.", .{ m, i });
        var ps: std.enums.EnumArray(Slot, Proj) = .initUndefined();
        for (slots(g, i)) |slot| {
            const nk = shape(g, slot);
            ps.set(slot, try linear(e, e, ck, stage, p, geometry.slot_modules.get(slot), nk[0], nk[1], g.bits.of(i, slot)));
        }
        l.* = .{
            .input_norm = try tensor(e, ck, "{s}input_layernorm.weight", .{p}, .bf16, &.{g.hidden}),
            .post_norm = try tensor(e, ck, "{s}post_attention_layernorm.weight", .{p}, .bf16, &.{g.hidden}),
            .mlp = if (g.moe()) try routed(e, e, ck, stage, p, g.expert_bits) else .{ .dense = .{ .gate = ps.get(.gate), .up = ps.get(.up), .down = ps.get(.down) } },
            .mixer = undefined,
        };
        if (g.linear[i]) {
            l.mixer = .{ .delta = .{
                .qkv = ps.get(.qkv),
                .z = ps.get(.z),
                .a = ps.get(.a),
                .b = ps.get(.b),
                .out = ps.get(.out),
                .conv = try tensor(e, ck, "{s}linear_attn.conv1d.weight", .{p}, .bf16, &.{ conv, geometry.taps, 1 }),
                .norm = try tensor(e, ck, "{s}linear_attn.norm.weight", .{p}, .bf16, &.{geometry.dk}),
                .a_log = try f32vec(e, ck, "{s}linear_attn.A_log", .{p}, g.v_heads),
                .dt_bias = try tensor(e, ck, "{s}linear_attn.dt_bias", .{p}, .bf16, &.{g.v_heads}),
            } };
        } else {
            l.mixer = .{ .attention = .{
                .q = ps.get(.q),
                .k = ps.get(.k),
                .v = ps.get(.v),
                .o = ps.get(.o),
                .q_norm = try tensor(e, ck, "{s}self_attn.q_norm.weight", .{p}, .bf16, &.{head_dim}),
                .k_norm = try tensor(e, ck, "{s}self_attn.k_norm.weight", .{p}, .bf16, &.{head_dim}),
            } };
        }
    }
}

test "a uniform 4-bit checkpoint keeps its sizes; other modules take their own width" {
    var g: Geometry = undefined;
    g = .{ .hidden = 2048, .intermediate = 6144, .layers = 2, .vocab = 248320, .k_heads = 16, .v_heads = 16, .q_heads = 8, .kv_heads = 2, .eps = 1e-6, .tied = true, .context = 4096, .linear = @splat(false) };
    g.linear[0] = true;
    const uniform = total(g);
    try std.testing.expectEqual(rawBytes(g.vocab, g.hidden, 4), stageBytes(g));
    g.bits.proj[1][@backingInt(Slot.down)] = 8;
    const nk = shape(g, .down);
    try std.testing.expectEqual(uniform - packedBytes(nk[0], nk[1], 4) + packedBytes(nk[0], nk[1], 8), total(g));
    g.bits.proj[1][@backingInt(Slot.down)] = 3; // on 3-bit tiles: 3 bits a weight
    try std.testing.expectEqual(uniform - packedBytes(nk[0], nk[1], 4) + packedBytes(nk[0], nk[1], 3), total(g));
    try std.testing.expect(packedBytes(nk[0], nk[1], 3) < packedBytes(nk[0], nk[1], 4));
    g.bits.proj[1][@backingInt(Slot.down)] = 4;
    g.bits.embed = 6;
    g.bits.head = 6;
    try std.testing.expect(total(g) > uniform);
    try std.testing.expectEqual(rawBytes(g.vocab, g.hidden, 6), stageBytes(g));
}

test {
    _ = @import("cuda_expert_pack.zig");
}

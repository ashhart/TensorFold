//! A ModelOpt checkpoint's bf16 MTP layer as the drafter's affine-4 tables, quantized at load (drafts only).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const engine = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");
const mtp = @import("cuda_mtp_weights.zig");

const host4 = core.affine4_host;
const A4 = cuda.experts.Affine4Experts;
const DeviceBuffer = cuda.DeviceBuffer;
const Engine = engine.Engine;
const head_dim = geometry.head_dim;

/// MLX affine-4 parts of `count` stacked n-by-k matrices: words [count][n][k/8], bf16 scales and biases [..][k/64].
const Parts = struct {
    words: []u32,
    scales: []u16,
    biases: []u16,

    fn init(gpa: std.mem.Allocator, count: usize, n: usize, k: usize) !Parts {
        const words = try gpa.alloc(u32, count * n * k / 8);
        errdefer gpa.free(words);
        const scales = try gpa.alloc(u16, count * n * k / host4.group);
        errdefer gpa.free(scales);
        return .{ .words = words, .scales = scales, .biases = try gpa.alloc(u16, scales.len) };
    }

    fn deinit(p: Parts, gpa: std.mem.Allocator) void {
        gpa.free(p.biases);
        gpa.free(p.scales);
        gpa.free(p.words);
    }

    /// Matrix `x` of the stack from bf16 rows [n][k] (`values`: n * k f32 of scratch).
    fn put(p: Parts, x: usize, n: usize, k: usize, rows: []const u8, values: []f32) void {
        const v = values[0 .. n * k];
        for (v, 0..) |*o, i| o.* = host4.f32of(std.mem.readInt(u16, rows[2 * i ..][0..2], .little));
        const g = n * k / host4.group;
        host4.quantize(v, p.words[x * n * k / 8 ..][0 .. n * k / 8], p.scales[x * g ..][0..g], p.biases[x * g ..][0..g]);
    }

    fn raw(p: Parts) [3][]const u8 {
        return .{ std.mem.sliceAsBytes(p.words), std.mem.sliceAsBytes(p.scales), std.mem.sliceAsBytes(p.biases) };
    }
};

/// Columns [from, from + k) of bf16 rows [n][k2] as rows [n][k] (fc's halves).
fn columns(out: []u8, rows: []const u8, n: usize, k2: usize, from: usize, k: usize) void {
    for (0..n) |r| @memcpy(out[r * k * 2 ..][0 .. k * 2], rows[(r * k2 + from) * 2 ..][0 .. k * 2]);
}

/// Gate (rows [0, ni)) or up (rows [ni, 2 ni)) of expert `x` in HF's fused gate_up_proj [E][2 ni][d].
fn half(fused: []const u8, x: usize, ni: usize, d: usize, up: bool) []const u8 {
    return fused[(x * 2 * ni + if (up) ni else 0) * d * 2 ..][0 .. ni * d * 2];
}

const Loader = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    w: *mtp.Weights,
    ck: *core.Checkpoint,
    stage: *DeviceBuffer,
    values: []f32,

    /// An n-by-k bf16 projection quantized and packed.
    fn linear(l: *Loader, name: []const u8, n: usize, k: usize) !engine.Proj {
        const t = try l.ck.expect(name, .bf16, &.{ n, k });
        return l.pack(t.bytes, n, k);
    }

    fn pack(l: *Loader, rows: []const u8, n: usize, k: usize) !engine.Proj {
        const p = try Parts.init(l.gpa, 1, n, k);
        defer p.deinit(l.gpa);
        p.put(0, n, k, rows, l.values);
        return weights.packLinear(l.e, l.w, l.stage, p.raw(), n, k);
    }

    /// An RMSNorm's weights as the engine's norm kernels take them (y = x * w): bf16(1 + w) of HF's zero-centred w.
    fn norm(l: *Loader, name: []const u8, n: usize) !u64 {
        const t = try l.ck.expect(name, .bf16, &.{n});
        const h = try l.gpa.alloc(u16, n);
        defer l.gpa.free(h);
        for (h, 0..) |*o, i| o.* = host4.bf16of(host4.f32of(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) + 1.0);
        return weights.upload(l.w, std.mem.sliceAsBytes(h));
    }

    /// `count` stacked experts' tables from `src(x)`'s bf16 rows, packed as `out`'s mi-th of m (one at count 0).
    fn table(l: *Loader, src: anytype, count: usize, n: usize, k: usize, out: u64, m: usize, mi: usize) !void {
        const p = try Parts.init(l.gpa, @max(count, 1), n, k);
        defer p.deinit(l.gpa);
        for (0..@max(count, 1)) |x| p.put(x, n, k, src.rows(x), l.values);
        try weights.packTable(l.e, l.stage, p.raw(), count, n, k, out, m, mi);
    }
};

/// The MTP layer's experts: fused gate_up_proj split into the pair table, down apart, the shared expert last.
fn experts(l: *Loader) !engine.Mlp {
    const g = l.e.g;
    const ni = g.intermediate;
    const d = g.hidden;
    const p = "mtp.layers.0.mlp.";
    const pair = try l.w.alloc(A4.pairBytes(g.experts + 1, ni, d));
    const down = try l.w.alloc(A4.bytes(g.experts + 1, d, ni));
    const gu = (try l.ck.expect(p ++ "experts.gate_up_proj", .bf16, &.{ g.experts, 2 * ni, d })).bytes;
    const dn = (try l.ck.expect(p ++ "experts.down_proj", .bf16, &.{ g.experts, d, ni })).bytes;
    const Fused = struct {
        bytes: []const u8,
        ni: usize,
        d: usize,
        up: bool,
        fn rows(f: @This(), x: usize) []const u8 {
            return half(f.bytes, x, f.ni, f.d, f.up);
        }
    };
    const Stacked = struct {
        bytes: []const u8,
        per: usize,
        fn rows(s: @This(), x: usize) []const u8 {
            return s.bytes[x * s.per ..][0..s.per];
        }
    };
    try l.table(Fused{ .bytes = gu, .ni = ni, .d = d, .up = false }, g.experts, ni, d, pair, 2, 0);
    try l.table(Fused{ .bytes = gu, .ni = ni, .d = d, .up = true }, g.experts, ni, d, pair, 2, 1);
    try l.table(Stacked{ .bytes = dn, .per = d * ni * 2 }, g.experts, d, ni, down, 1, 0);
    const shared = [_]struct { []const u8, usize, usize, u64, usize, usize }{
        .{ p ++ "shared_expert.gate_proj.weight", ni, d, pair + A4.pairBytes(g.experts, ni, d), 2, 0 },
        .{ p ++ "shared_expert.up_proj.weight", ni, d, pair + A4.pairBytes(g.experts, ni, d), 2, 1 },
        .{ p ++ "shared_expert.down_proj.weight", d, ni, down + A4.bytes(g.experts, d, ni), 1, 0 },
    };
    for (shared) |s| {
        const t = try l.ck.expect(s[0], .bf16, &.{ s[1], s[2] });
        try l.table(Stacked{ .bytes = t.bytes, .per = t.bytes.len }, 0, s[1], s[2], s[3], s[4], s[5]);
    }
    // the router's rows as stored: the routed experts' gate, then the shared expert's gate row
    const gate = try l.ck.expect(p ++ "gate.weight", .bf16, &.{ g.experts, d });
    const sg = try l.ck.expect(p ++ "shared_expert_gate.weight", .bf16, &.{ 1, d });
    const router = try l.w.alloc((g.experts + 1) * d * 2);
    try l.w.arena.upload(router - l.w.arena.ptr, gate.bytes);
    try l.w.arena.upload(router - l.w.arena.ptr + gate.bytes.len, sg.bytes);
    return .{ .moe = .{ .router = router, .experts = .{ .format = .affine4, .up = pair, .down = down, .width = ni, .dims = d, .experts = g.experts + 1, .swiglu = true } } };
}

/// The `mtp.*` tensors of the checkpoint in `dir` into the same tables `mtp.load` packs from mtp-4bit.safetensors.
pub fn load(gpa: std.mem.Allocator, io: std.Io, e: *Engine, dir: []const u8) !mtp.Weights {
    const g = e.g;
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, "mtp.");
    defer ck.close();
    if (!ck.has("mtp.fc.weight")) return error.NoMtpLayer;
    std.log.info("qwen3.6 mtp: the checkpoint's bf16 MTP layer drafts as affine-4 tables quantized at load (drafts only)", .{});
    var w: mtp.Weights = .{ .arena = try DeviceBuffer.alloc(e.d, mtp.total(g)), .fc_e = undefined, .fc_h = undefined, .norm_e = 0, .norm_h = 0, .input_norm = 0, .post_norm = 0, .q_norm = 0, .k_norm = 0, .norm = 0, .q = undefined, .k = undefined, .v = undefined, .o = undefined, .mlp = undefined };
    errdefer w.arena.free();
    try w.arena.fill8(0, null);
    var stage = try DeviceBuffer.alloc(e.d, mtp.stageBytes(g));
    defer stage.free();
    const aw = g.attnWidth();
    const kvw = g.kv_heads * head_dim;
    // the largest matrix put stages: fc's halves, q (with its gate), k and v, o, or an expert
    const values = try gpa.alloc(f32, g.hidden * @max(g.hidden, 2 * aw, kvw, g.intermediate));
    defer gpa.free(values);
    var l: Loader = .{ .gpa = gpa, .e = e, .w = &w, .ck = &ck, .stage = &stage, .values = values };
    const fc = try ck.expect("mtp.fc.weight", .bf16, &.{ g.hidden, 2 * g.hidden });
    const cut = try gpa.alloc(u8, g.hidden * g.hidden * 2);
    defer gpa.free(cut);
    columns(cut, fc.bytes, g.hidden, 2 * g.hidden, 0, g.hidden); // the embedding's half first, as fc reads them
    w.fc_e = (try l.pack(cut, g.hidden, g.hidden)).a4;
    columns(cut, fc.bytes, g.hidden, 2 * g.hidden, g.hidden, g.hidden);
    w.fc_h = (try l.pack(cut, g.hidden, g.hidden)).a4;
    w.norm_e = try l.norm("mtp.pre_fc_norm_embedding.weight", g.hidden);
    w.norm_h = try l.norm("mtp.pre_fc_norm_hidden.weight", g.hidden);
    w.norm = try l.norm("mtp.norm.weight", g.hidden);
    w.input_norm = try l.norm("mtp.layers.0.input_layernorm.weight", g.hidden);
    w.post_norm = try l.norm("mtp.layers.0.post_attention_layernorm.weight", g.hidden);
    w.q_norm = try l.norm("mtp.layers.0.self_attn.q_norm.weight", head_dim);
    w.k_norm = try l.norm("mtp.layers.0.self_attn.k_norm.weight", head_dim);
    w.q = try l.linear("mtp.layers.0.self_attn.q_proj.weight", 2 * aw, g.hidden);
    w.k = try l.linear("mtp.layers.0.self_attn.k_proj.weight", kvw, g.hidden);
    w.v = try l.linear("mtp.layers.0.self_attn.v_proj.weight", kvw, g.hidden);
    w.o = try l.linear("mtp.layers.0.self_attn.o_proj.weight", g.hidden, aw);
    w.mlp = try experts(&l);
    return w;
}

test "fc's halves are its column ranges, and gate is the first half of each fused expert" {
    var rows: [3 * 256]u16 = undefined;
    for (&rows, 0..) |*r, i| r.* = @intCast(i);
    var out: [3 * 128 * 2]u8 = undefined;
    columns(&out, std.mem.sliceAsBytes(&rows), 3, 256, 128, 128);
    const got = std.mem.bytesAsSlice(u16, &out);
    for (0..3) |r| for (0..128) |c| try std.testing.expectEqual(@as(u16, @intCast(r * 256 + 128 + c)), got[r * 128 + c]);
    var fused: [2 * 2 * 4 * 64]u16 = undefined; // E 2, 2 ni = 4 rows, d 64
    for (&fused, 0..) |*f, i| f.* = @intCast(i);
    const bytes = std.mem.sliceAsBytes(&fused);
    const up1 = std.mem.bytesAsSlice(u16, half(bytes, 1, 2, 64, true));
    try std.testing.expectEqual(@as(u16, (1 * 4 + 2) * 64), up1[0]); // expert 1's row ni
    try std.testing.expectEqual(@as(usize, 2 * 64), up1.len);
    try std.testing.expectEqual(@as(u16, 0), std.mem.bytesAsSlice(u16, half(bytes, 0, 2, 64, false))[0]);
}

test "a stacked part is the shared quantizer's codes, scales and biases of that matrix" {
    const a = std.testing.allocator;
    const n = 2;
    const k = 128;
    var rows: [2 * n * k]u16 = undefined;
    for (&rows, 0..) |*r, i| r.* = host4.bf16of(@as(f32, @floatFromInt(@as(i32, @intCast(i % 97)) - 48)) / 16.0);
    const p = try Parts.init(a, 2, n, k);
    defer p.deinit(a);
    const values = try a.alloc(f32, n * k);
    defer a.free(values);
    const bytes = std.mem.sliceAsBytes(&rows);
    for (0..2) |x| p.put(x, n, k, bytes[x * n * k * 2 ..][0 .. n * k * 2], values);
    var v: [n * k]f32 = undefined;
    for (&v, rows[n * k ..]) |*o, r| o.* = host4.f32of(r);
    var words: [n * k / 8]u32 = undefined;
    var scales: [n * k / 64]u16 = undefined;
    var biases: [n * k / 64]u16 = undefined;
    host4.quantize(&v, &words, &scales, &biases);
    try std.testing.expectEqualSlices(u32, &words, p.words[n * k / 8 ..]);
    try std.testing.expectEqualSlices(u16, &scales, p.scales[n * k / 64 ..]);
    try std.testing.expectEqualSlices(u16, &biases, p.biases[n * k / 64 ..]);
}

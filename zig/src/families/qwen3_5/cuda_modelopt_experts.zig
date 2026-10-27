//! A ModelOpt Qwen3.5 MoE checkpoint's NVFP4 experts on Nvfp4Experts: gate-up and down tables, the shared expert last.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");
const modelopt = @import("cuda_modelopt.zig");

const X = cuda.experts.Nvfp4Experts;
const Engine = engine.Engine;
const DeviceBuffer = cuda.DeviceBuffer;
const aligned = engine.aligned;

/// A layer's device bytes: the gate-up and down tables, their global scales and the router's bf16 rows.
pub fn routedBytes(g: geometry.Geometry) usize {
    const e1 = g.experts + 1;
    const tables = aligned(X.bytes(e1, g.intermediate, g.hidden, 2)) + aligned(X.bytes(e1, g.hidden, g.intermediate, 1));
    return tables + aligned(e1 * 2 * 4) + aligned(e1 * 4) + aligned(e1 * g.hidden * 2);
}

/// One projection of every expert as stored, codes then block scales: the stage its table packs from.
pub fn stageBytes(g: geometry.Geometry) usize {
    const e1 = g.experts + 1;
    return e1 * g.intermediate * g.hidden / 2 + e1 * g.intermediate * g.hidden / 16;
}

/// A projection of the experts: its module name, output and input widths, and its part of the table it packs into.
const Part = struct { name: []const u8, n: usize, k: usize, parts: usize, m: usize };

/// `bytes` into the engine's arena.
fn put(e: *Engine, bytes: []const u8) !u64 {
    const p = try e.alloc(bytes.len);
    try e.arena.upload(p - e.arena.ptr, bytes);
    return p;
}

/// A layer's experts under prefix `p`: each projection of every expert staged as stored, packed on the GPU.
pub fn routed(e: *Engine, ck: *core.Checkpoint, stage: *DeviceBuffer, p: []const u8) !engine.Mlp {
    const g = e.g;
    const e1 = g.experts + 1;
    const ni = g.intermediate;
    const d = g.hidden;
    const pack = &(e.mo orelse return error.FormatNotLoaded).pack;
    const up = try e.alloc(X.bytes(e1, ni, d, 2));
    const down = try e.alloc(X.bytes(e1, d, ni, 1));
    const up_g = try e.gpa.alloc(f32, e1 * 2);
    defer e.gpa.free(up_g);
    const down_g = try e.gpa.alloc(f32, e1);
    defer e.gpa.free(down_g);
    const parts = [_]Part{ .{ .name = "gate_proj", .n = ni, .k = d, .parts = 2, .m = 0 }, .{ .name = "up_proj", .n = ni, .k = d, .parts = 2, .m = 1 }, .{ .name = "down_proj", .n = d, .k = ni, .parts = 1, .m = 0 } };
    var mb: [160]u8 = undefined;
    var tb: [192]u8 = undefined;
    for (parts) |q| {
        const cb = q.n * q.k / 2;
        const sb = q.n * q.k / 16;
        if (e1 * (cb + sb) > stage.len) return error.StageTooSmall;
        for (0..e1) |x| {
            const module = if (x < g.experts) try std.fmt.bufPrint(&mb, "{s}mlp.experts.{d}.{s}", .{ p, x, q.name }) else try std.fmt.bufPrint(&mb, "{s}mlp.shared_expert.{s}", .{ p, q.name });
            const codes = try ck.expect(try std.fmt.bufPrint(&tb, "{s}.weight", .{module}), .u8, &.{ q.n, q.k / 2 });
            try stage.upload(x * cb, codes.bytes);
            const blocks = try ck.expect(try std.fmt.bufPrint(&tb, "{s}.weight_scale", .{module}), .f8_e4m3, &.{ q.n, q.k / 16 });
            try stage.upload(e1 * cb + x * sb, blocks.bytes);
            const global = try modelopt.scalar(ck, try std.fmt.bufPrint(&tb, "{s}.weight_scale_2", .{module}));
            if (!std.math.isFinite(global) or global <= 0) return error.UnsupportedQwenQuantization;
            if (q.parts == 2) up_g[x * 2 + q.m] = global else down_g[x] = global;
        }
        try pack.fp4Experts(e.stream, stage.ptr, stage.ptr + e1 * cb, if (q.parts == 2) up else down, e1, q.n, q.k, 0, q.k, q.parts, q.m);
        try e.stream.synchronize(); // the next projection's staging overwrites this one's
    }
    const up_scale = try put(e, std.mem.sliceAsBytes(up_g));
    const down_scale = try put(e, std.mem.sliceAsBytes(down_g));
    // the router's rows as stored: the routed experts' gate, then the shared expert's gate row
    const gate = try ck.expect(try std.fmt.bufPrint(&tb, "{s}mlp.gate.weight", .{p}), .bf16, &.{ g.experts, d });
    const shared = try ck.expect(try std.fmt.bufPrint(&tb, "{s}mlp.shared_expert_gate.weight", .{p}), .bf16, &.{ 1, d });
    const router = try e.alloc(e1 * d * 2);
    try e.arena.upload(router - e.arena.ptr, gate.bytes);
    try e.arena.upload(router - e.arena.ptr + gate.bytes.len, shared.bytes);
    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = up, .down = down, .up_scale = up_scale, .down_scale = down_scale, .width = ni, .dims = d, .experts = e1 };
    return .{ .moe = .{ .router = router, .experts = layer } };
}

test "Qwen3.6-35B-A3B's NVFP4 layer: 455,810,560 device bytes, a 151,584,768-byte stage" {
    var g: geometry.Geometry = .{ .hidden = 2048, .intermediate = 512, .layers = 40, .vocab = 248320, .k_heads = 16, .v_heads = 32, .q_heads = 16, .kv_heads = 2, .eps = 1e-6, .tied = false, .context = 4096, .linear = @splat(false) };
    g.experts = 256;
    g.top_k = 8;
    try std.testing.expectEqual(@as(usize, 455_810_560), routedBytes(g));
    try std.testing.expectEqual(@as(usize, 151_584_768), stageBytes(g));
}

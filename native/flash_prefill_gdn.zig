const std = @import("std");
const mx = @import("mlx.zig");
const c = mx.c;
const A = mx.Array;
const src = @import("kernel_sources.zig");
const affine = @import("flash_ops.zig");
const mm = @import("flash_prefill_ops.zig").matmul;
const Ops = @import("prefill_ops.zig").Ops;

pub const Config = struct {
    key_heads: i32,
    value_heads: i32,
    key_dims: i32,
    value_dims: i32,
    activation: enum { sigmoid, silu } = .sigmoid,
    epsilon: f32 = 1e-6,
};
pub const Weights = struct {
    qkv: affine.Weight,
    z: affine.Weight,
    b: affine.Weight,
    a: affine.Weight,
    out: affine.Weight,
    stacked: ?affine.Weight = null,
    conv: A,
    a_log: A,
    dt_bias: A,
    norm: A,
};
pub const Cache = struct { conv: A = mx.empty, state: A = mx.empty };
pub const Result = struct { output: A, cache: Cache, q: A, k: A, v: A, recurrent: A, gated: A };

pub fn stack(s: *mx.Scope, w: Weights) !?affine.Weight {
    // Upstream only installs a prefill stack when all four projections share a bit width.
    const parts = [_]affine.Weight{ w.qkv, w.z, w.b, w.a };
    for (parts) |part| if (part.format.bits != w.qkv.format.bits) return null;
    return try affine.stack(s, &parts);
}

fn l2(s: *mx.Scope, x: A) !A {
    var summed = c.mlx_array_new();
    const rc = c.mlx_sum_axis(&summed, try s.unary(c.mlx_square, x), -1, true, mx.stream);
    const eps = try s.cast(try s.scalar(1e-6), mx.dtype(x));
    const inv = try s.unary(c.mlx_rsqrt, try s.binary(c.mlx_add, try s.result(rc, summed), eps));
    return s.binary(c.mlx_multiply, x, inv);
}

pub fn forward(kernels: *mx.Kernels, ops: *Ops, s: *mx.Scope, x: A, w: Weights, cfg: Config, cache: Cache) !Result {
    if (mx.shape(x).len != 3 or mx.dtype(x) != mx.bf16) return error.InvalidTensorShape;
    const batch = mx.dim(x, 0);
    const rows = mx.dim(x, 1);
    if (batch < 1 or rows < 1 or rows > 2048 or cfg.key_heads < 1 or cfg.value_heads < 1 or @mod(cfg.value_heads, cfg.key_heads) != 0 or cfg.key_dims < 32 or @mod(cfg.key_dims, 32) != 0 or cfg.value_dims < 4 or @mod(cfg.value_dims, 4) != 0) return error.InvalidTensorShape;
    const key_width = cfg.key_heads * cfg.key_dims;
    const value_width = cfg.value_heads * cfg.value_dims;
    const conv_width = 2 * key_width + value_width;
    if (mx.shape(w.conv).len != 3 or mx.dim(w.conv, 0) != conv_width or mx.dim(w.conv, 2) != 1 or mx.dim(w.conv, 1) < 1) return error.InvalidTensorShape;
    const taps = mx.dim(w.conv, 1);
    if (!std.mem.eql(i32, mx.shape(w.a_log), &.{cfg.value_heads}) or !std.mem.eql(i32, mx.shape(w.dt_bias), &.{cfg.value_heads}) or !std.mem.eql(i32, mx.shape(w.norm), &.{cfg.value_dims})) return error.InvalidTensorShape;
    if (mx.dtype(w.conv) != mx.bf16 or mx.dtype(w.a_log) != mx.bf16 or mx.dtype(w.dt_bias) != mx.bf16 or mx.dtype(w.norm) != mx.bf16) return error.InvalidTensorDType;
    for ([_]affine.Weight{ w.qkv, w.z, w.b, w.a }, [_]i32{ conv_width, value_width, cfg.value_heads, cfg.value_heads }) |weight, n| {
        const g = try weight.geometry(2);
        if (g.n != n or g.k != mx.dim(x, 2)) return error.InvalidTensorShape;
    }
    const og = try w.out.geometry(2);
    if (og.k != value_width or og.n != mx.dim(x, 2)) return error.InvalidTensorShape;
    if (cache.conv.ctx != null and (!std.mem.eql(i32, mx.shape(cache.conv), &.{ batch, taps - 1, conv_width }) or mx.dtype(cache.conv) != mx.bf16)) return error.InvalidTensorShape;
    if (cache.state.ctx != null and (!std.mem.eql(i32, mx.shape(cache.state), &.{ batch, cfg.value_heads, cfg.value_dims, cfg.key_dims }) or mx.dtype(cache.state) != mx.f32t)) return error.InvalidTensorShape;
    var qkv: A = undefined;
    var z: A = undefined;
    var b: A = undefined;
    var a: A = undefined;
    if (w.stacked != null and batch * rows >= 64) {
        const sg = try w.stacked.?.geometry(2);
        if (sg.n != conv_width + value_width + 2 * cfg.value_heads or sg.k != mx.dim(x, 2)) return error.InvalidTensorShape;
        const all = try @import("flash_prefill_mm.zig").linear(kernels, s, x, w.stacked.?);
        qkv = try s.slice(all, 2, 0, conv_width);
        z = try s.slice(all, 2, conv_width, conv_width + value_width);
        b = try s.slice(all, 2, conv_width + value_width, conv_width + value_width + cfg.value_heads);
        a = try s.slice(all, 2, conv_width + value_width + cfg.value_heads, sg.n);
    } else {
        qkv = try mm(s, x, w.qkv);
        z = try mm(s, x, w.z);
        b = try mm(s, x, w.b);
        a = try mm(s, x, w.a);
    }
    const tail = if (cache.conv.ctx != null) cache.conv else try s.zeros(&.{ batch, taps - 1, conv_width }, mx.bf16);
    const seq = try s.cat(&.{ tail, qkv }, 1);
    const next_conv = try s.contiguous(try s.slice(seq, 1, rows, rows + taps - 1));
    var conv = c.mlx_array_new();
    const rc = c.mlx_conv1d(&conv, seq, w.conv, 1, 0, 1, conv_width, mx.stream);
    const activated = try ops.call(s, .silu, &.{try s.result(rc, conv)});
    var q = try l2(s, try s.reshape(try s.slice(activated, 2, 0, key_width), &.{ batch, rows, cfg.key_heads, cfg.key_dims }));
    q = try s.binary(c.mlx_multiply, q, try s.cast(try s.scalar(1 / @sqrt(@as(f32, @floatFromInt(cfg.key_dims)))), mx.bf16));
    const k = try l2(s, try s.reshape(try s.slice(activated, 2, key_width, 2 * key_width), &.{ batch, rows, cfg.key_heads, cfg.key_dims }));
    const v = try s.reshape(try s.slice(activated, 2, 2 * key_width, conv_width), &.{ batch, rows, cfg.value_heads, cfg.value_dims });
    const g = try ops.call(s, .decay, &.{ w.a_log, a, w.dt_bias });
    const beta = try s.unary(c.mlx_sigmoid, b);
    const initial = if (cache.state.ctx != null) cache.state else try s.zeros(&.{ batch, cfg.value_heads, cfg.value_dims, cfg.key_dims }, mx.f32t);
    const result = try kernels.run(s, src.flash_prefill_gdn, &.{ q, k, v, g, beta, initial, try s.reshape(try s.ints(&.{rows}), &.{}) }, &.{ mx.td("InT", mx.bf16), mx.td("StT", mx.f32t), mx.ti("Dk", cfg.key_dims), mx.ti("Dv", cfg.value_dims), mx.ti("Hk", cfg.key_heads), mx.ti("Hv", cfg.value_heads) }, .{ 32, cfg.value_dims, batch * cfg.value_heads }, .{ 32, 4, 1 }, &.{ .{ .shape = &.{ batch, rows, cfg.value_heads, cfg.value_dims } }, .{ .shape = &.{ batch, cfg.value_heads, cfg.value_dims, cfg.key_dims }, .dtype = mx.f32t } });
    var normed = c.mlx_array_new();
    const nr = c.mlx_fast_rms_norm(&normed, result[0], w.norm, cfg.epsilon, mx.stream);
    const y = try s.cast(try s.result(nr, normed), mx.f32t);
    const gate = try s.cast(try s.reshape(z, mx.shape(result[0])), mx.f32t);
    const activated_gate = if (cfg.activation == .sigmoid) try s.unary(c.mlx_sigmoid, gate) else try ops.call(s, .silu, &.{gate});
    const gated = try s.reshape(try s.cast(try s.binary(c.mlx_multiply, y, activated_gate), mx.bf16), &.{ batch, rows, value_width });
    return .{ .output = try @import("flash_prefill_mm.zig").linear(kernels, s, gated, w.out), .cache = .{ .conv = next_conv, .state = result[1] }, .q = q, .k = k, .v = v, .recurrent = result[0], .gated = gated };
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var ops = Ops{};
    defer ops.deinit();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/gdn.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { name: []const u8, config: Config, cached: bool, bits: [5]i32, groups: [5]i32, stacked: bool };
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    if (cases.value.len == 0) return error.EmptyFixtures;
    for (cases.value) |case| {
        errdefer std.debug.print("Flash prefill GDN fixture failed: {s}\n", .{case.name});
        var store = @import("checkpoint.zig").Store.init(32);
        defer store.deinit();
        try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
        var s = mx.Scope{};
        defer s.deinit();
        var w: Weights = undefined;
        inline for (.{ "qkv", "z", "b", "a", "out" }, 0..) |key, i| @field(w, key) = .{ .arrays = .{ try store.get(key ++ ".weight"), try store.get(key ++ ".scales"), try store.get(key ++ ".biases") }, .format = .{ .bits = case.bits[i], .group_size = case.groups[i] } };
        inline for (.{ "conv", "a_log", "dt_bias", "norm" }) |key| @field(w, key) = try store.get(key);
        w.stacked = null;
        w.stacked = try stack(&s, w);
        try std.testing.expectEqual(case.stacked, w.stacked != null);
        const cache = if (case.cached) Cache{ .conv = try store.get("previous.conv"), .state = try store.get("previous.state") } else Cache{};
        const result = try forward(&kernels, &ops, &s, try store.get("input"), w, case.config, cache);
        const equal = @import("variant_checks.zig").equalBits;
        inline for (.{ "output", "q", "k", "v", "recurrent", "gated" }) |key| {
            errdefer std.debug.print("Mismatch in {s}\n", .{key});
            try equal(&s, @field(result, key), try store.get(key));
        }
        try equal(&s, result.cache.conv, try store.get("next.conv"));
        try equal(&s, result.cache.state, try store.get("next.state"));
    }
    std.debug.print("PASS: {d} complete Flash prefill GDN layers, scalar recurrence, exact intermediate arrays and convolution/recurrent cache continuation\n", .{cases.value.len});
}

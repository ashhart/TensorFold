//! Vision towers' shared glue (vit.cu), torch Linears through cuBLASLt, and full attention over a picture's rows.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const cublaslt = @import("cublaslt.zig");
const memory = @import("memory.zig");

const Driver = @import("driver.zig").Driver;
const Allocator = std.mem.Allocator;
const Stream = stream_.Stream;
const Args = launch_.Args;
const Linear = cublaslt.Linear;

const threads = 256; // vit.cu's NT
/// The widest LayerNorm row: NT * MAXE.
pub const norm_most = 8 * threads;
const workspace_bytes = 32 << 20;

/// What tf_vit_pointwise applies to a Linear's output: nothing, torch's tanh GELU, its exact (erf) GELU, or relu^2.
pub const Act = enum(i32) { none = 0, gelu_tanh = 1, gelu = 2, relu2 = 3 };

pub const Kernels = struct {
    mod: module.Module,
    layer_norm_fn: module.Function,
    pointwise_fn: module.Function,
    bias_fn: module.Function,
    unshuffle_fn: module.Function,
    cast_fn: module.Function,
    position_fn: module.Function,

    pub fn load(d: *const Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try module.Module.load(d, kernels.vit);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .layer_norm_fn = try mod.function("tf_vit_layer_norm"),
            .pointwise_fn = try mod.function("tf_vit_pointwise"),
            .bias_fn = try mod.function("tf_vit_bias"),
            .unshuffle_fn = try mod.function("tf_vit_unshuffle"),
            .cast_fn = try mod.function("tf_vit_cast"),
            .position_fn = try mod.function("tf_vit_position_add"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    /// y = LayerNorm(x) * w + b over rows up to 2048 wide.
    pub fn layerNorm(k: *const Kernels, s: Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, width: usize, eps: f32) !void {
        if (width > norm_most) return error.RowShape;
        var a: Args = .{};
        for ([_]u64{ x, w, b, y }) |p| a.add(p);
        a.add(int(width));
        a.add(eps);
        try go(k.layer_norm_fn, s, rows, 1, &a);
    }

    /// o [count] = x (a Linear's bf16 output) through `act`, then plus r when given; x or r may be o.
    pub fn pointwise(k: *const Kernels, s: Stream, x: u64, r: ?u64, o: u64, count: usize, act: Act) !void {
        var a: Args = .{};
        for ([_]u64{ x, r orelse 0, o }) |p| a.add(p);
        a.add(@as(i64, @intCast(count)));
        a.add(@backingInt(act));
        try go(k.pointwise_fn, s, blocks(count, threads), 1, &a);
    }

    /// o [rows, n] = bf16(acc + b), acc f32.
    pub fn bias(k: *const Kernels, s: Stream, acc: u64, b: u64, o: u64, rows: usize, n: usize) !void {
        var a: Args = .{};
        for ([_]u64{ acc, b, o }) |p| a.add(p);
        a.add(int(n));
        try go(k.bias_fn, s, rows, blocks(n, threads), &a);
    }

    /// y [h/2 * w/2, 4c] = the projector's pixel unshuffle of x [h * w, c], an even grid.
    pub fn unshuffle(k: *const Kernels, s: Stream, x: u64, y: u64, h: usize, w: usize, c: usize) !void {
        if (h % 2 != 0 or w % 2 != 0) return error.RowShape;
        var a: Args = .{};
        a.add(x);
        a.add(y);
        a.add(int(w));
        a.add(int(c));
        try go(k.unshuffle_fn, s, h * w / 4, 1, &a);
    }

    /// y [count] = bf16(x), x f32, rounded to nearest even.
    pub fn cast(k: *const Kernels, s: Stream, x: u64, y: u64, count: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(count)));
        try go(k.cast_fn, s, blocks(count, threads), 1, &a);
    }

    /// o [h * w, c] = bf16(x + bf16(the table [side * side, c] resized to the longer side's square, rows y < h, x < w)).
    pub fn positionAdd(k: *const Kernels, s: Stream, x: u64, table: u64, o: u64, h: usize, w: usize, side: usize, c: usize) !void {
        var a: Args = .{};
        for ([_]u64{ x, table, o }) |p| a.add(p);
        for ([_]usize{ w, @max(h, w), side, c }) |n| a.add(int(n));
        try go(k.position_fn, s, h * w, 1, &a);
    }
};

fn int(x: usize) i32 {
    return @intCast(x);
}

fn blocks(n: usize, by: usize) usize {
    return (n + by - 1) / by;
}

fn go(f: module.Function, s: Stream, x: usize, y: usize, a: *Args) !void {
    if (x == 0) return;
    try launch_.launch(f, .{ .grid = .{ .x = @intCast(x), .y = @intCast(y), .z = 1 }, .block = .{ .x = threads } }, s, a);
}

/// torch Linears through cuBLASLt: a plan kept a shape, one workspace.
pub const Gemms = struct {
    gpa: Allocator,
    lt: cublaslt.Library,
    workspace: memory.DeviceBuffer,
    plans: std.AutoHashMapUnmanaged([3]u64, Linear) = .empty,

    pub fn init(gpa: Allocator, d: *const Driver) !Gemms {
        var lt = try cublaslt.Library.open();
        errdefer lt.close();
        return .{ .gpa = gpa, .lt = lt, .workspace = try memory.DeviceBuffer.alloc(d, workspace_bytes) };
    }

    pub fn deinit(g: *Gemms) void {
        var it = g.plans.valueIterator();
        while (it.next()) |l| l.deinit();
        g.plans.deinit(g.gpa);
        g.workspace.free();
        g.lt.close();
    }

    /// y [rows, n] = x [rows, k] . w [n, k]^T, f32 accumulation, `out` rows.
    pub fn run(g: *Gemms, s: Stream, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize, out: cublaslt.DataType) !void {
        try (try g.plan(rows, n, k, out, .none)).run(x, w, y, g.workspace.ptr, workspace_bytes, s.handle);
    }

    /// Whether the shape's own algorithm carries the bias in cuBLASLt's epilogue; else `linear` sums through f32.
    pub fn fused(g: *Gemms, rows: usize, n: usize, k: usize) !bool {
        return (try g.plan(rows, n, k, .bf16, .bias)).epilogue == .bias;
    }

    fn plan(g: *Gemms, rows: usize, n: usize, k: usize, out: cublaslt.DataType, epilogue: cublaslt.Epilogue) !*const Linear {
        const key = [3]u64{ rows, n, k | @as(u64, @intCast(@backingInt(out))) << 40 | @as(u64, @backingInt(epilogue)) << 48 };
        const got = try g.plans.getOrPut(g.gpa, key);
        if (!got.found_existing) got.value_ptr.* = Linear.initEpilogue(&g.lt, rows, n, k, out, workspace_bytes, epilogue) catch |e| {
            g.plans.removeByPtr(got.key_ptr);
            return e;
        };
        return got.value_ptr;
    }
};

/// y [rows, n] = bf16(x . w^T + b), the bias in cuBLASLt's epilogue, or else through f32 `acc` and tf_vit_bias.
pub fn linear(g: *Gemms, k: *const Kernels, s: Stream, acc: u64, x: u64, w: u64, b: u64, y: u64, rows: usize, n: usize, kk: usize) !void {
    const p = try g.plan(rows, n, kk, .bf16, .bias);
    if (p.epilogue == .bias) return p.runBias(x, w, b, y, g.workspace.ptr, workspace_bytes, s.handle);
    try g.run(s, x, w, acc, rows, n, kk, .f32);
    try k.bias(s, acc, b, y, rows, n);
}

/// full_kernel<D, 8, 8> of prefill_attention.cu: eight warps as eight 16-row tiles of one head.
const Full = struct {
    dim: usize,
    sym: [:0]const u8,
    const rows_per_block = 128;
    const smem: u32 = 65536; // eight 32-key slots of 80 dims (72 pads to 80), rows padded to 128
};

/// The head dims a tower's attention runs: SigLIP2's 72 and RADIO's 80.
const fulls = [_]Full{
    .{ .dim = 72, .sym = "_ZN20tf_prefill_attention11full_kernelILi72ELi8ELi8EEEvPK13__nv_bfloat16PS1_iif" },
    .{ .dim = 80, .sym = "_ZN20tf_prefill_attention11full_kernelILi80ELi8ELi8EEEvPK13__nv_bfloat16PS1_iif" },
};

/// Every row of a picture over every key (no mask), a block a head and 128 rows.
pub const Attention = struct {
    mod: module.Module,
    fns: [fulls.len]module.Function,

    pub fn load(d: *const Driver) !Attention {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try module.Module.load(d, kernels.prefill_attention);
        errdefer mod.unload();
        var a: Attention = .{ .mod = mod, .fns = undefined };
        for (fulls, &a.fns) |f, *fn_| {
            fn_.* = try mod.function(f.sym);
            try fn_.allowDynamicShared(Full.smem);
        }
        return a;
    }

    pub fn unload(a: *Attention) void {
        a.mod.unload();
    }

    /// q, k, v heads of qkv [rows, 3 H D] bf16 -> out [rows, H D] bf16, scaled by `scale`.
    pub fn full(a: *const Attention, s: Stream, qkv: u64, out: u64, rows: usize, heads: usize, dim: usize, scale: f32) !void {
        const i = for (fulls, 0..) |f, i| {
            if (f.dim == dim) break i;
        } else return error.UnsupportedShape;
        if (rows == 0) return;
        var args: Args = .{};
        args.add(qkv);
        args.add(out);
        for ([_]usize{ rows, heads }) |n| args.add(@as(c_int, @intCast(n)));
        args.add(scale);
        const grid: launch_.Dim3 = .{ .x = @intCast(blocks(rows, Full.rows_per_block)), .y = @intCast(heads) };
        try launch_.launch(a.fns[i], .{ .grid = grid, .block = .{ .x = threads }, .shared = Full.smem }, s, &args);
    }
};

/// cvt.rn.bf16.f32 of a finite value.
pub fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

test "bf16 rounds to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16(@bitCast(@as(u32, 0x3f818000))));
    try std.testing.expectEqual(@as(u16, 0x3f81), bf16(@bitCast(@as(u32, 0x3f808001))));
}

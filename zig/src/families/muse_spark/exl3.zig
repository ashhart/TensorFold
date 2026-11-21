//! Muse Spark's EXL3 layers on the shared operators (cuda.exl3 Linear, cuda.exl3_experts) plus its 32-member tiles.

const std = @import("std");
const cuda = @import("cuda");
const shared = cuda.exl3;
const sx = cuda.exl3_experts;
const launch_ = cuda.launch;

const Driver = cuda.Driver;
const Module = cuda.Module;
const Function = cuda.Function;
const Stream = cuda.Stream;
const Args = cuda.Args;

pub const DType = shared.DType;
pub const Scratch = sx.Scratch;

/// (K splits, warps a program) for a K x N layer: the shared linear's plan, the shape's alone.
pub fn plan(k: usize, n: usize) [2]usize {
    const p = shared.plan(k, n, 192, 8);
    return .{ p.sk, p.wk };
}

/// One EXL3 linear layer in the strips layout ([N/128, K/16, 8, 4 * K2] words), mul1 codebook.
pub const Linear = struct {
    words: u64,
    suh: u64, // fp16 [K]
    svh: u64, // fp16 [N]
    bias: u64 = 0, // fp16 [N] or 0
    counters: u64, // int32 [8 * N / 128], zeroed once (the split tickets)
    k: usize,
    n: usize,
    k2: u32,
    split: [2]usize, // plan(k, n)

    /// (words between k tiles, words between 128-column blocks) of the strips layout.
    pub fn strides(l: Linear) [2]i64 {
        const st = shared.strides(l.k, l.n, l.k2, true);
        return .{ @intCast(st.k), @intCast(st.nb) };
    }

    /// The shared linear's view of this layer.
    pub fn weight(l: Linear) shared.Weight {
        const st = shared.strides(l.k, l.n, l.k2, true);
        return .{ .trellis = l.words, .suh = l.suh, .svh = l.svh, .bias = l.bias, .k = @intCast(l.k), .n = @intCast(l.n), .halves = l.k2, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    }
};

/// One layer's routed experts: device tables of trellis pointers and widths, stacked scales.
pub const Experts = struct {
    gate_ptr: u64, // int64 [E]
    up_ptr: u64,
    down_ptr: u64,
    gate_k2: u64, // int32 [E]
    up_k2: u64,
    down_k2: u64,
    suh_g: u64, // fp16 [E, D]
    suh_u: u64,
    svh_g: u64, // fp16 [E, I]
    svh_u: u64,
    suh_d: u64, // fp16 [E, I]
    svh_d: u64, // fp16 [E, D]
    count: usize, // E
    dims: usize, // D
    width: usize, // I (this rank's)
    k2_gu: [2]u32, // (min, max) over gate and up
    k2_d: [2]u32,
    host_ptr: []const u64 = &.{}, // [gate E, up E, down E]
    host_k2: []const i32 = &.{},

    /// The shared experts' view: mul1, the fp32 SwiGLU (act_mode 1), no clamp.
    pub fn layer(x: Experts) sx.Layer {
        return .{ .gate = x.gate_ptr, .up = x.up_ptr, .down = x.down_ptr, .gate_k2 = x.gate_k2, .up_k2 = x.up_k2, .down_k2 = x.down_k2, .suh_g = x.suh_g, .suh_u = x.suh_u, .svh_g = x.svh_g, .svh_u = x.svh_u, .suh_d = x.suh_d, .svh_d = x.svh_d, .experts = x.count, .dims = x.dims, .width = x.width, .codebook = .mul1, .limit = 0, .act_mode = 1 };
    }
};

/// The scratch parts' bytes for `rows` rows of `slots` slots (the shared scratch's, without its alignment).
pub const Sizes = struct { xg: usize, xd: usize, z: usize, y: usize, ids: usize, members: usize };

pub fn sizes(e: Experts, rows: usize, slots: usize) !Sizes {
    const g_sk = try sx.plan(e.dims, e.width, true);
    const d_sk = try sx.plan(e.width, e.dims, false);
    const p = rows * slots;
    const maxu = @min(p, e.count);
    return .{ .xg = p * e.dims * 2, .xd = p * e.width * 2, .z = @max(2 * g_sk * e.width, d_sk * e.dims) * p * 4, .y = p * e.dims * 4, .ids = maxu * 4, .members = maxu * rows * 4 };
}

pub const Kernels = struct {
    lin: shared.Linear,
    ex: sx.Experts,
    x2: Module, // experts2.cu: muse_grouped2
    f_x2: Function,

    pub fn load(d: *const Driver) !Kernels {
        if (!cuda.kernels.available) return error.KernelsNotBuilt;
        var k: Kernels = undefined;
        k.lin = try shared.Linear.load(d);
        errdefer k.lin.unload();
        k.ex = try sx.Experts.load(d);
        errdefer k.ex.unload();
        k.x2 = try Module.load(d, cuda.kernels.muse_experts2);
        errdefer k.x2.unload();
        k.f_x2 = try k.x2.function("muse_grouped2");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.x2.unload();
        k.ex.unload();
        k.lin.unload();
    }

    /// xh (m, K) fp16 = Hadamard-rotated x * suh (the shared tf_exl3_rot_in).
    pub fn rotIn(k: *const Kernels, s: Stream, x: u64, x_dtype: DType, suh: u64, xh: u64, m: usize, kk: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(suh);
        a.add(xh);
        a.add(@as(i32, @backingInt(x_dtype)));
        a.add(@as(i32, @intCast(m)));
        a.add(@as(i32, @intCast(kk)));
        try launch_.launch(k.lin.rot_in, .{ .grid = .{ .x = u((kk / 128 + 3) / 4), .y = u(m), .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }

    /// y (m, N) = x @ W * svh (+ bias) through the shared linear; z: fp32 split partials when the plan splits K.
    pub fn linear(k: *const Kernels, s: Stream, l: Linear, x: u64, x_dtype: DType, xh: u64, y: u64, y_dtype: DType, z: u64, m: usize) !void {
        try k.lin.matmul(s, l.weight(), x, x_dtype, m, y, y_dtype, xh, z, l.counters);
    }

    /// The routed experts' per-slot fp32 outputs in sc.y [rows * slots, D] (shared runSlots; x fp16).
    pub fn routed(k: *const Kernels, s: Stream, e: Experts, x: u64, x_stride: usize, pick: u64, rows: usize, sc: Scratch) !void {
        try k.ex.runSlots(s, e.layer(), x, x_stride, .f16, pick, rows, sc);
    }

    /// routed() with the grouped GEMMs in 32-member tiles (wide windows): the same launches and bits otherwise.
    pub fn routedWide(k: *const Kernels, s: Stream, e: Experts, x: u64, x_stride: usize, pick: u64, rows: usize, sc: Scratch) !void {
        const l = e.layer();
        const d, const w, const slots = .{ l.dims, l.width, sc.slots };
        if (rows < 1 or rows > sc.rows or slots > 32) return error.WindowTooWide;
        const g_sk = try sx.plan(d, w, true);
        const d_sk = try sx.plan(w, d, false);
        const p = rows * slots;
        const maxu = @min(p, l.experts);
        {
            var a: Args = .{};
            inline for (.{ pick, sc.ids, sc.count, sc.members }) |v| a.add(v);
            inline for (.{ rows, slots, l.experts, rows }) |v| a.add(int(v));
            try launch_.launch(k.ex.group, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 }, .shared = u(rows * slots * 4) }, s, &a);
        }
        {
            var a: Args = .{};
            a.add(x);
            a.add(int(x_stride));
            a.add(@as(i32, @backingInt(DType.f16)));
            inline for (.{ pick, l.suh_g, l.suh_u, sc.xg, sc.xu }) |v| a.add(v);
            inline for (.{ d, slots, l.experts }) |v| a.add(int(v));
            try launch_.launch(k.ex.rot_in, .{ .grid = .{ .x = u(p), .y = u(d / 128), .z = 2 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
        }
        try k.grouped2(s, .{ sc.xg, sc.xu, l.gate, l.up, l.gate_k2, l.up_k2 }, sc, d, w, p, g_sk, rows, maxu, 2);
        {
            var a: Args = .{};
            inline for (.{ sc.z, pick, l.svh_g, l.svh_u, l.suh_d, sc.xd }) |v| a.add(v);
            inline for (.{ p, w, g_sk, l.experts }) |v| a.add(int(v));
            a.add(l.limit);
            a.add(l.act_mode);
            try launch_.launch(k.ex.gateup, .{ .grid = .{ .x = u(p), .y = u(w / 128), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
        }
        try k.grouped2(s, .{ sc.xd, sc.xd, l.down, l.down, l.down_k2, l.down_k2 }, sc, w, d, p, d_sk, rows, maxu, 1);
        {
            var a: Args = .{};
            inline for (.{ sc.z, pick, l.svh_d, sc.y, @as(u64, 0), @as(u64, 0) }) |v| a.add(v);
            inline for (.{ p, d, d_sk, l.experts, slots }) |v| a.add(int(v));
            try launch_.launch(k.ex.down_combine, .{ .grid = .{ .x = u(rows), .y = u(d / 128), .z = 1 }, .block = .{ .x = u(32 * slots), .y = 1, .z = 1 } }, s, &a);
        }
    }

    /// muse_grouped2 over (x0, x1, trellis tables 0 / 1, width tables 0 / 1): grid z = mats * sk * member tiles of 32.
    fn grouped2(k: *const Kernels, s: Stream, t: [6]u64, sc: Scratch, kk: usize, n: usize, p: usize, sk: usize, maxm: usize, maxu: usize, mats: usize) !void {
        if (kk % (16 * sk * 4) != 0 or n % 128 != 0) return error.UnevenSplit;
        var a: Args = .{};
        inline for (0..6) |i| a.add(t[i]);
        inline for (.{ sc.ids, sc.count, sc.members, sc.z }) |v| a.add(v);
        inline for (.{ kk, n, p, sk, maxm, sc.slots }) |v| a.add(int(v));
        try launch_.launch(k.f_x2, .{ .grid = .{ .x = u(maxu), .y = u(n / 128), .z = u(mats * sk * ((maxm + 31) / 32)) }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }
};

fn int(x: usize) i32 {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

test "the shared plans at Muse Spark's shapes" {
    const vectors = [_][4]usize{
        .{ 8192, 8192, 1, 8 }, .{ 8192, 4096, 8, 4 },  .{ 8192, 512, 16, 4 },  .{ 4096, 8192, 1, 8 },
        .{ 2048, 8192, 1, 8 }, .{ 8192, 2048, 16, 4 }, .{ 8192, 1024, 16, 4 }, .{ 8192, 101120, 1, 8 },
        .{ 4096, 6144, 4, 4 }, .{ 4096, 76288, 1, 8 }, .{ 1024, 4096, 2, 4 },
    };
    for (vectors) |v| try std.testing.expectEqual([2]usize{ v[2], v[3] }, plan(v[0], v[1]));
    try std.testing.expectEqual(@as(u32, 4), try sx.plan(4096, 2048, true));
    try std.testing.expectEqual(@as(u32, 1), try sx.plan(2048, 4096, false));
}

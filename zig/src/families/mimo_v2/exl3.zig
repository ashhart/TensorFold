//! MiMo-V2's EXL3 layers on the shared operators (cuda.exl3 Linear, cuda.exl3_experts): one arithmetic, any rows.

const std = @import("std");
const cuda = @import("cuda");
const shared = cuda.exl3;
const sx = cuda.exl3_experts;

const Driver = cuda.Driver;
const Stream = cuda.Stream;

pub const DType = shared.DType;
pub const Scratch = sx.Scratch;
pub const MAX_ROWS = 128; // rows one shared linear call takes

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

    /// The shared linear's view of this layer.
    pub fn weight(l: Linear) shared.Weight {
        const st = shared.strides(l.k, l.n, l.k2, true);
        return .{ .trellis = l.words, .suh = l.suh, .svh = l.svh, .bias = l.bias, .k = @intCast(l.k), .n = @intCast(l.n), .halves = l.k2, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    }

    /// Bytes of fp32 split partials `rows` rows need (0 when the plan does not split K).
    pub fn zBytes(l: Linear, rows: usize) usize {
        return if (l.split[0] > 1) l.split[0] * rows * l.n * 4 else 0;
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
    host_ptr: []const u64 = &.{}, // [gate E, up E, down E]
    host_k2: []const i32 = &.{},
    uniform_k2: u32 = 0, // every projection's half-bits when they all match (the prompt kernels need it), else 0

    /// The shared experts' view: mul1, the fp32 SwiGLU (act_mode 1), no clamp.
    pub fn layer(x: Experts) sx.Layer {
        return .{ .gate = x.gate_ptr, .up = x.up_ptr, .down = x.down_ptr, .gate_k2 = x.gate_k2, .up_k2 = x.up_k2, .down_k2 = x.down_k2, .suh_g = x.suh_g, .suh_u = x.suh_u, .svh_g = x.svh_g, .svh_u = x.svh_u, .suh_d = x.suh_d, .svh_d = x.svh_d, .experts = x.count, .dims = x.dims, .width = x.width, .codebook = .mul1, .uniform_k2 = x.uniform_k2, .limit = 0, .act_mode = 1 };
    }
};

/// The one half-bit width of every projection in `k2s`, or 0 when they differ.
pub fn uniform(k2s: []const i32) u32 {
    if (k2s.len == 0) return 0;
    for (k2s) |k| if (k != k2s[0]) return 0;
    return @intCast(k2s[0]);
}

/// The prompt kernels' buffers for `rows` rows: the shared prompt scratch and the grouping plan.
pub const PromptScratch = struct { sc: sx.prompt.Scratch, plan: cuda.grouped.Plan };

pub fn promptScratch(ar: anytype, e: Experts, rows: usize, slots: usize) !PromptScratch {
    const l = e.layer();
    const pairs = rows * slots;
    const wide = pairs > cuda.grouped.small;
    const items = cuda.grouped.maxItems(pairs, e.count, sx.prompt.tile);
    const groups: cuda.grouped.Plan = .{
        .members = try ar.zeros(pairs * 4),
        .items = try ar.zeros(items * 12),
        .counts = try ar.zeros(8),
        .rank = try ar.zeros((if (wide) pairs else 1) * 4),
        .hist = try ar.zeros((if (wide) (pairs + 1023) / 1024 * e.count else 1) * 4),
    };
    return .{ .sc = try sx.prompt.scratch(try ar.zeros(try sx.prompt.bytes(l, rows, slots)), l, rows, slots), .plan = groups };
}

/// The scratch parts' bytes for `rows` rows of `slots` slots (the shared scratch's, without its alignment).
pub const Sizes = struct { xg: usize, xd: usize, z: usize, y: usize, ids: usize, members: usize };

pub fn sizes(e: Experts, rows: usize, slots: usize) !Sizes {
    const g_sk = try sx.plan(e.dims, e.width, true);
    const d_sk = try sx.plan(e.width, e.dims, false);
    const p = rows * slots;
    const maxu = @min(p, e.count);
    return .{ .xg = p * e.dims * 2, .xd = p * e.width * 2, .z = @max(2 * g_sk * e.width, d_sk * e.dims) * p * 4, .y = p * e.dims * 4, .ids = maxu * 4, .members = maxu * rows * 4 };
}

/// The experts' scratch for `rows` rows from an arena (members set to -1, as the group kernel expects).
pub fn scratch(ar: anytype, d: *const Driver, e: Experts, rows: usize, slots: usize) !Scratch {
    const sz = try sizes(e, rows, slots);
    const sc: Scratch = .{ .xg = try ar.zeros(sz.xg), .xu = try ar.zeros(sz.xg), .xd = try ar.zeros(sz.xd), .z = try ar.zeros(sz.z), .y = try ar.zeros(sz.y), .ids = try ar.zeros(sz.ids), .count = try ar.zeros(4), .members = try ar.alloc(sz.members), .rows = rows, .slots = slots };
    try d.check(d.api.cuMemsetD32_v2(sc.members, 0xFFFF_FFFF, sz.members / 4), "members -1");
    return sc;
}

pub const Kernels = struct {
    lin: shared.Linear,
    ex: sx.Experts,
    plans: cuda.Module, // the experts image: the prompt kernels' grouping plan
    router: cuda.grouped.Router,

    pub fn load(d: *const Driver) !Kernels {
        if (!cuda.kernels.available) return error.KernelsNotBuilt;
        var k: Kernels = undefined;
        k.lin = try shared.Linear.load(d);
        errdefer k.lin.unload();
        k.ex = try sx.Experts.load(d);
        errdefer k.ex.unload();
        k.plans = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer k.plans.unload();
        k.router = try cuda.grouped.Router.resolve(k.plans);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.plans.unload();
        k.ex.unload();
        k.lin.unload();
    }

    /// y (m, N) = x @ W * svh (+ bias), m <= 128; xh: fp16 [m, K] scratch, z: split partials when the plan splits.
    pub fn linear(k: *const Kernels, s: Stream, l: Linear, x: u64, x_dtype: DType, xh: u64, y: u64, y_dtype: DType, z: u64, m: usize) !void {
        try k.lin.matmul(s, l.weight(), x, x_dtype, m, y, y_dtype, xh, z, l.counters);
    }

    /// linear() over any row count in calls of 128 rows: a row's bits never depend on its call's rows.
    pub fn rows(k: *const Kernels, s: Stream, l: Linear, x: u64, xh: u64, y: u64, y_dtype: DType, z: u64, m: usize) !void {
        const out_size: usize = if (y_dtype == .f32) 4 else 2;
        var r0: usize = 0;
        while (r0 < m) : (r0 += MAX_ROWS) {
            const n = @min(MAX_ROWS, m - r0);
            try k.linear(s, l, x + r0 * l.k * 2, .f16, xh, y + r0 * l.n * out_size, y_dtype, z, n);
        }
    }

    /// out [rows, D] fp32 = each row's routed experts (x fp16) weighted by wts, summed in slot order.
    pub fn routed(k: *const Kernels, s: Stream, e: Experts, x: u64, x_stride: usize, pick: u64, wts: u64, n: usize, out: u64, sc: Scratch) !void {
        try k.ex.run(s, e.layer(), x, x_stride, .f16, pick, wts, n, out, sc);
    }

    /// Prompt rows' routed experts (x fp16) on the shared prompt kernels: each pair's fp32 output into ps.sc.y.
    pub fn prompted(k: *const Kernels, s: Stream, e: Experts, x: u64, x_stride: usize, pick: u64, n: usize, ps: PromptScratch) !void {
        try k.ex.prompt.run(&k.ex, k.router, s, e.layer(), x, .f16, x_stride, pick, n, ps.sc, ps.plan);
    }
};

test "the shared plans at MiMo-V2.6-Flash's shapes" {
    // (K, N) -> (K splits, warps): q, k / v, o, the dense MLP and the head, a rank's under TP=2 then TP=1
    const vectors = [_][4]usize{
        .{ 4096, 6144, 4, 4 },   .{ 4096, 384, 8, 4 },  .{ 4096, 768, 8, 4 },   .{ 4096, 4096, 8, 4 },
        .{ 4096, 8192, 1, 8 },   .{ 8192, 4096, 8, 4 }, .{ 4096, 76288, 1, 8 }, .{ 4096, 12288, 1, 8 },
        .{ 4096, 1536, 8, 4 },   .{ 8192, 8192, 1, 8 }, .{ 4096, 16384, 1, 8 }, .{ 16384, 4096, 8, 4 },
        .{ 4096, 152576, 1, 8 },
    };
    for (vectors) |v| try std.testing.expectEqual([2]usize{ v[2], v[3] }, plan(v[0], v[1]));
    try std.testing.expectEqual(@as(u32, 4), try sx.plan(4096, 1024, true)); // a rank's experts: 2048 wide / 2
    try std.testing.expectEqual(@as(u32, 1), try sx.plan(1024, 4096, false));
}

test "the prompt kernels need one half-bit width for every projection" {
    try std.testing.expectEqual(@as(u32, 6), uniform(&.{ 6, 6, 6 }));
    try std.testing.expectEqual(@as(u32, 0), uniform(&.{ 6, 8, 6 }));
    try std.testing.expectEqual(@as(u32, 0), uniform(&.{}));
}

//! Inkling 2 Mini's routed experts: stored NVFP4/MXFP8 codes, bf16 activations and fp32 routing.
const std = @import("std");
const cuda = @import("cuda");
const experts = @import("experts.zig");
const core = @import("core");
const pack = @import("sm121a_pack.zig");

pub const Format = enum(i32) { nvfp4 = 0, mxfp8 = 1 };
pub const max_slots = 32;
const max_experts = 512;
/// The multi-block plan (rows >= 32): at most this many pair chunks of at most plan_chunk pairs, a block each.
const plan_blocks = 48;
const plan_chunk = 1024;

pub const Projection = struct {
    codes: u64,
    scales: u64,
    amax: u64 = 0,
    denominator: u64 = 0,
    n: u32,
    k: u32,
};

/// An NVFP4 layer's experts repacked for sm121a_moe.cu (TF_INKLING2_NVFP4_OMMA=1): per expert the gate|up cells
/// [N/32][K/32][2][144 u32] and the down cells [N/32][K/32][1][144 u32], each cell's codes and e4m3 scales together, and
/// the outer scales amax[e] / denominator (a pair an expert for w13: gate then up, one for w2). The checkpoint's own
/// codes and scales are not uploaded then (Projection pointers 0; n and k stay).
pub const Omma = struct {
    gu: u64,
    dn: u64,
    gu_scale2: u64,
    dn_scale2: u64,
};

/// An MXFP8 layer's experts repacked for the grouped QMMA path: per expert the gate|up|down codes in
/// fragment order ([npad/64][k/64][8][32][2][8]) and their ue8m0 block scales ([npad/64][k/64][64][2]).
/// Both N and K are exact multiples of 64, so npad == n and no padding rows exist.
pub const Qmma = struct {
    w8_g: u64,
    bs_g: u64,
    w8_u: u64,
    bs_u: u64,
    w8_d: u64,
    bs_d: u64,
};

/// One physical layer. Loop visits share these device pointers.
pub const Layer = struct {
    format: Format,
    experts: u32,
    w13: Projection,
    w2: Projection,
    omma: ?Omma = null,
    qmma: ?Qmma = null,
};

/// TF_INKLING2_NVFP4_OMMA=1 and the sm_121a image is in this build; false (with a warning) when asked for and absent.
pub fn ommaEnabled() bool {
    const want = @import("sys").environment.read(bool, "TF_INKLING2_NVFP4_OMMA", false) catch false;
    if (!want) return false;
    if (!cuda.kernels.available or cuda.kernels.inkling2_sm121a_moe.len == 0) {
        std.log.warn("TF_INKLING2_NVFP4_OMMA=1: this build has no sm_121a kernel set; the experts keep the bf16-decode kernels", .{});
        return false;
    }
    return true;
}

/// TF_INKLING2_MXFP8_QMMA defaults to ommaEnabled() so bisecting layers 20-23 is independent.
pub fn qmmaEnabled() bool {
    const want = @import("sys").environment.read(bool, "TF_INKLING2_MXFP8_QMMA", ommaEnabled()) catch ommaEnabled();
    if (!want) return false;
    return cuda.kernels.available and cuda.kernels.inkling2_sm121a_moe.len > 0;
}

/// The widths sm121a_moe.cu's activation quantizers are written for.
pub const omma_hidden = 2304;
pub const omma_inter = 1152;

/// sm121a_moe.cu's NVFP4 entry points and their persistent-grid sizes.
pub const OmmaKernels = struct {
    mod: cuda.Module,
    quant_x: cuda.Function,
    quant_h: cuda.Function,
    gu: Fn,
    gu_s2: Fn,
    dn: Fn,
    mxfp8_quant: cuda.Function,
    mxfp8_gu: Fn,
    mxfp8_dn: Fn,
    mxfp8_gu64: Fn,
    mxfp8_dn64: Fn,
    sms: usize,

    const block: u32 = 64;
    // S = 4 stages of (64 activation rows x 32 bytes, their 4-byte scales, WARPS x M cells of 72 uint4)
    const gu_shared: u32 = 4 * (2048 + 256 + 2 * 2 * 72 * 16);
    // tf_ink2_nvfp4_gu_s2: the same kernel with S = 2 stages, 4 CTAs an SM; the bits of gu, faster past wide_gu_rows
    const gu_s2_shared: u32 = 2 * (2048 + 256 + 2 * 2 * 72 * 16);
    const wide_gu_rows = 512;
    const dn_shared: u32 = 4 * (2048 + 256 + 2 * 1 * 72 * 16);
    // Grouped MXFP8 QMMA (BM=16, BN=64, BK=64, 128 threads, 2 stages): X tile 1024 B + 2x W tile 4096 B +
    // activation scales 32 B + 2x weight-scale tile 128 B (separate gate and up W buffers).
    const mxfp8_gu_shared: u32 = 2 * ((1024 + 2 * 4096 + 32 + 2 * 128 + 127) / 128 * 128);
    const mxfp8_dn_shared: u32 = 2 * ((1024 + 4096 + 32 + 128 + 127) / 128 * 128);
    // Grouped MXFP8 QMMA BM=64 prefill (256 threads, 2 stages): gu 4096 X + 2x4096 W + 128 SX + 2x128 SW,
    // dn 4096 X + 4096 W + 128 SX + 128 SW.
    const mxfp8_gu64_shared: u32 = 2 * ((4096 + 2 * 4096 + 128 + 2 * 128 + 127) / 128 * 128);
    const mxfp8_dn64_shared: u32 = 2 * ((4096 + 4096 + 128 + 128 + 127) / 128 * 128);
    const Fn = struct { f: cuda.Function, per_sm: usize, shared: u32 };

    fn fnOf(mod: cuda.Module, name: [:0]const u8, shared: u32) !Fn {
        const f = try mod.function(name);
        return .{ .f = f, .per_sm = @max(1, @as(usize, try f.occupancy(block, shared))), .shared = shared };
    }

    fn fnOf128(mod: cuda.Module, name: [:0]const u8, shared: u32) !Fn {
        const f = try mod.function(name);
        return .{ .f = f, .per_sm = @max(1, @as(usize, try f.occupancy(128, shared))), .shared = shared };
    }

    fn fnOf256(mod: cuda.Module, name: [:0]const u8, shared: u32) !Fn {
        const f = try mod.function(name);
        return .{ .f = f, .per_sm = @max(1, @as(usize, try f.occupancy(256, shared))), .shared = shared };
    }

    fn load(d: *const cuda.Driver, sms: usize) !OmmaKernels {
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_sm121a_moe);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .quant_x = try mod.function("tf_ink2_nvfp4_quant_2304"),
            .quant_h = try mod.function("tf_ink2_nvfp4_quant_1152"),
            .gu = try fnOf(mod, "tf_ink2_nvfp4_gu", gu_shared),
            .gu_s2 = try fnOf(mod, "tf_ink2_nvfp4_gu_s2", gu_s2_shared),
            .dn = try fnOf(mod, "tf_ink2_nvfp4_dn_b16", dn_shared),
            .mxfp8_quant = try mod.function("tf_ink2_mxfp8_quantize"),
            .mxfp8_gu = try fnOf128(mod, "tf_ink2_mxfp8_moe_gu", mxfp8_gu_shared),
            .mxfp8_dn = try fnOf128(mod, "tf_ink2_mxfp8_moe_dn_b16", mxfp8_dn_shared),
            .mxfp8_gu64 = try fnOf256(mod, "tf_ink2_mxfp8_moe_gu_64", mxfp8_gu64_shared),
            .mxfp8_dn64 = try fnOf256(mod, "tf_ink2_mxfp8_moe_dn64", mxfp8_dn64_shared),
            .sms = sms,
        };
    }

    fn grid(q: *const OmmaKernels, f: Fn, units: usize) u32 {
        return @intCast(@max(1, @min(units, f.per_sm * q.sms)));
    }
};

pub const Weight = extern struct {
    codes: u64,
    scales: u64,
    amax: u64,
    denominator: u64,
    experts: i64,
    n: i64,
    k: i64,
    group: i32,
    format: i32,
    row0: i32,
    row_step: i32,
};

pub const Kernels = struct {
    mod: cuda.Module,
    plan: cuda.Function,
    plan_count: cuda.Function,
    plan_scan: cuda.Function,
    plan_scatter: cuda.Function,
    decode: cuda.Function,
    prompt: cuda.Function,
    act: cuda.Function,
    combine: cuda.Function,
    combine8: cuda.Function,
    resident: usize,
    omma: ?OmmaKernels = null,

    pub fn load(d: *const cuda.Driver, sms: usize) !Kernels {
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_modelopt);
        errdefer mod.unload();
        const decode = try mod.function("tf_ink2moe_up");
        return .{
            .mod = mod,
            .plan = try mod.function("tf_ink2moe_plan"),
            .plan_count = try mod.function("tf_ink2moe_plan_count"),
            .plan_scan = try mod.function("tf_ink2moe_plan_scan"),
            .plan_scatter = try mod.function("tf_ink2moe_plan_scatter"),
            .decode = decode,
            .prompt = try mod.function("tf_ink2moe_prompt_up"),
            .act = try mod.function("tf_ink2moe_act"),
            .combine = try mod.function("tf_ink2moe_combine"),
            .combine8 = try mod.function("tf_ink2moe_combine8"),
            .resident = @max(1, try decode.occupancy(128, 0)) * sms,
        };
    }

    /// The sm_121a expert kernels, for layers that carry an `omma` repack.
    pub fn loadOmma(k: *Kernels, d: *const cuda.Driver, sms: usize) !void {
        k.omma = try OmmaKernels.load(d, sms);
    }

    pub fn unload(k: *Kernels) void {
        if (k.omma) |*o| o.mod.unload();
        k.mod.unload();
    }
};

fn aligned(n: usize) usize {
    return (n + 255) / 256 * 256;
}

pub const Layout = struct {
    gate: u64,
    up: u64,
    act: u64,
    down: u64,
    items: u64,
    item_count: u64,
    members: u64,
    plan_hist: u64, // the multi-block plan's per-chunk expert counts, then first member slots: int [plan_blocks][max_experts]
    // the OMMA path's activations: X rows (codes, ue4m3 scales, a row scale) and the SwiGLU rows' (the same, a pair each)
    xq: u64,
    xs: u64,
    xs2: u64,
    aq: u64,
    as: u64,
    as2: u64,
    // the grouped-MXFP8 path's activations: per-32 ue8m0 scales need no row scale; m_pad covers the
    // x rows and pairs_pad the pair rows (both N and K are multiples of 64, so no code padding)
    mx_xq: u64,
    mx_xs: u64,
    mx_aq: u64,
    mx_as: u64,
    bytes: usize,
};

/// Reconstructs the stable intermediate addresses inside one scratch allocation.
pub fn layout(base: u64, rows: usize, slots: usize, hidden: usize, intermediate: usize) Layout {
    const pairs = rows * slots;
    const items_max = (pairs + 15) / 16 + @min(pairs, max_experts);
    const m_pad = std.mem.alignForward(usize, rows, 16);
    const pairs_pad = std.mem.alignForward(usize, pairs, 16);
    var at = base;
    const take = struct {
        fn f(p: *u64, bytes: usize) u64 {
            const r = p.*;
            p.* = r + aligned(bytes);
            return r;
        }
    }.f;
    const gate = take(&at, pairs * intermediate * 2);
    const up = take(&at, pairs * intermediate * 2);
    const act = take(&at, pairs * intermediate * 2);
    const down = take(&at, pairs * hidden * 2);
    const items = take(&at, items_max * 12);
    const item_count = take(&at, 4);
    const members = take(&at, pairs * 4);
    const plan_hist = take(&at, plan_blocks * max_experts * 4);
    const xq = take(&at, rows * hidden / 2);
    const xs = take(&at, rows * hidden / 16);
    const xs2 = take(&at, rows * 4);
    const aq = take(&at, pairs * intermediate / 2);
    const as = take(&at, pairs * intermediate / 16);
    const as2 = take(&at, pairs * 4);
    const mx_xq = take(&at, m_pad * hidden);
    const mx_xs = take(&at, (hidden / 64) * m_pad * 2);
    const mx_aq = take(&at, pairs_pad * intermediate);
    const mx_as = take(&at, (intermediate / 64) * pairs_pad * 2);
    return .{ .gate = gate, .up = up, .act = act, .down = down, .items = items, .item_count = item_count, .members = members, .plan_hist = plan_hist, .xq = xq, .xs = xs, .xs2 = xs2, .aq = aq, .as = as, .as2 = as2, .mx_xq = mx_xq, .mx_xs = mx_xs, .mx_aq = mx_aq, .mx_as = mx_as, .bytes = at - base };
}

/// Gate, up, activation and per-expert outputs, plus the ascending-expert group plan.
pub fn scratchBytes(rows: usize, slots: usize, hidden: usize, intermediate: usize) !usize {
    if (rows == 0 or rows > 2048 or slots == 0 or slots > max_slots) return error.BadExperts;
    return layout(0, rows, slots, hidden, intermediate).bytes;
}

fn i32d(v: usize) i32 {
    return @intCast(v);
}

pub fn weight(p: Projection, l: Layer, row0: i32, row_step: i32) Weight {
    return .{
        .codes = p.codes,
        .scales = p.scales,
        .amax = p.amax,
        .denominator = p.denominator,
        .experts = l.experts,
        .n = p.n,
        .k = p.k,
        .group = if (l.format == .nvfp4) 16 else 32,
        .format = @backingInt(l.format),
        .row0 = row0,
        .row_step = row_step,
    };
}

fn projection(s: cuda.Stream, fn_: cuda.Function, prompt: bool, resident: usize, x: u64, x_stride: usize, slots: usize, w: Weight, items: u64, item_count: u64, members: u64, out: u64, item_bound: usize) !void {
    const kg: usize = @intCast(@divExact(w.k, 32));
    const outputs: usize = @intCast(if (w.row_step == 2) @divExact(w.n, 2) else w.n);
    const nb = outputs / 32;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(i32d(x_stride));
    a.add(i32d(slots));
    a.add(w);
    a.add(i32d(kg));
    a.add(i32d(nb));
    for ([_]u64{ items, item_count, members, out }) |v| a.add(v);
    a.add(i32d(outputs));
    a.add(@as(i32, -1));
    const bound = item_bound * nb;
    const grid = if (prompt) (bound + 3) / 4 else @min((bound + 3) / 4, resident);
    if (grid == 0) return;
    try cuda.launch.launch(fn_, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
}

/// Run one MoE layer. `pick` is row-major int32 expert IDs; routing `weights` are fp32.
pub fn run(k: *const Kernels, s: cuda.Stream, l: Layer, caps: experts.Caps, x: u64, x_stride: usize, pick: u64, weights: u64, rows: usize, slots: usize, out: u64, scratch: u64) !void {
    if (rows == 0 or rows > 2048 or slots == 0 or slots > max_slots or l.experts == 0 or l.experts > max_experts) return error.BadExperts;
    if (l.w13.n % 64 != 0 or l.w13.k % 32 != 0 or l.w2.n % 32 != 0 or l.w2.k % 32 != 0) return error.BadExperts;
    if (l.w13.k != l.w2.n or l.w13.n / 2 != l.w2.k or x_stride < l.w13.k) return error.BadExperts;
    const hidden: usize = l.w2.n;
    const intermediate: usize = l.w2.k;
    const pairs = rows * slots;
    const b = layout(scratch, rows, slots, hidden, intermediate);
    const listed = rows >= 32;
    const tile: usize = if (listed) 64 else 16;
    const item_bound = (pairs + tile - 1) / tile + @min(pairs, @as(usize, l.experts));

    try plan(k, s, b, pick, rows, slots, l.experts, tile);

    if (l.omma) |om| {
        const q = k.omma orelse return error.OmmaKernelsMissing;
        if (hidden != omma_hidden or intermediate != omma_inter) return error.BadExperts;
        try runOmma(&q, s, om, b, x, x_stride, rows, slots, item_bound, pairs);
        return combine(k, s, pick, weights, b.down, out, rows, hidden, slots, l.experts);
    }

    if (l.qmma) |qm| {
        const q = k.omma orelse return error.OmmaKernelsMissing;
        if (hidden != omma_hidden or intermediate != omma_inter) return error.BadExperts;
        try runQmma(&q, s, qm, b, x, x_stride, rows, slots, pairs, item_bound);
        return combine(k, s, pick, weights, b.down, out, rows, hidden, slots, l.experts);
    }

    const fn_ = if (listed) k.prompt else k.decode;
    try projection(s, fn_, listed, k.resident, x, x_stride, slots, weight(l.w13, l, 0, 2), b.items, b.item_count, b.members, b.gate, item_bound);
    try projection(s, fn_, listed, k.resident, x, x_stride, slots, weight(l.w13, l, 1, 2), b.items, b.item_count, b.members, b.up, item_bound);

    var aa: cuda.Args = .{};
    for ([_]u64{ b.gate, b.up, b.act }) |v| aa.add(v);
    aa.add(i32d(pairs * intermediate));
    aa.add(caps.gate);
    aa.add(caps.up);
    try cuda.launch.launch(k.act, .{ .grid = .{ .x = @intCast((pairs * intermediate + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &aa);
    try projection(s, fn_, listed, k.resident, b.act, intermediate, 0, weight(l.w2, l, 0, 1), b.items, b.item_count, b.members, b.down, item_bound);

    try combine(k, s, pick, weights, b.down, out, rows, hidden, slots, l.experts);
}

/// The ascending-expert group plan. Wide calls (rows >= 32) count, scan and scatter over up to plan_blocks blocks;
/// narrower ones keep the single-block tf_ink2moe_plan. Both leave the same items, members and item_count.
fn plan(k: *const Kernels, s: cuda.Stream, b: Layout, pick: u64, rows: usize, slots: usize, n_experts: usize, tile: usize) !void {
    const pairs = rows * slots;
    const chunk = (pairs + plan_blocks - 1) / plan_blocks;
    if (rows < 32 or chunk > plan_chunk) {
        var g: cuda.Args = .{};
        for ([_]u64{ pick, b.items, b.members, b.item_count }) |v| g.add(v);
        for ([_]usize{ rows, slots, n_experts, tile }) |v| g.add(i32d(v));
        return cuda.launch.launch(k.plan, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &g);
    }
    const blocks = (pairs + chunk - 1) / chunk;
    var c: cuda.Args = .{};
    for ([_]u64{ pick, b.plan_hist }) |v| c.add(v);
    for ([_]usize{ pairs, n_experts, chunk }) |v| c.add(i32d(v));
    try cuda.launch.launch(k.plan_count, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &c);
    var n: cuda.Args = .{};
    for ([_]u64{ b.plan_hist, b.items, b.item_count }) |v| n.add(v);
    for ([_]usize{ blocks, n_experts, tile }) |v| n.add(i32d(v));
    try cuda.launch.launch(k.plan_scan, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = max_experts } }, s, &n);
    var t: cuda.Args = .{};
    for ([_]u64{ pick, b.plan_hist, b.members }) |v| t.add(v);
    for ([_]usize{ pairs, n_experts, chunk }) |v| t.add(i32d(v));
    try cuda.launch.launch(k.plan_scatter, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &t);
}

/// The routed rows' weighted sum in ascending expert order. Wide calls with whole 16-byte rows take 8 columns a
/// thread (tf_ink2moe_combine8, the same bits); decode rows keep tf_ink2moe_combine.
fn combine(k: *const Kernels, s: cuda.Stream, pick: u64, weights: u64, down: u64, out: u64, rows: usize, hidden: usize, slots: usize, n_experts: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ pick, weights, down, out }) |v| a.add(v);
    for ([_]usize{ rows, hidden, slots, n_experts }) |v| a.add(i32d(v));
    const wide = rows > 16 and hidden % 8 == 0 and down % 16 == 0 and out % 16 == 0;
    const count = if (wide) rows * hidden / 8 else rows * hidden;
    try cuda.launch.launch(if (wide) k.combine8 else k.combine, .{ .grid = .{ .x = @intCast((count + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
}

/// X -> NVFP4 rows (a dynamic scale each), gate|up and the capped SwiGLU (bf16 rows into `gate`), those rows -> NVFP4,
/// then down into b.down (bf16 [pairs][hidden], the rows the stock down writes). The plan was built by the caller.
fn runOmma(q: *const OmmaKernels, s: cuda.Stream, om: Omma, b: Layout, x: u64, x_stride: usize, rows: usize, slots: usize, item_bound: usize, pairs: usize) !void {
    const skip: i32 = max_experts; // no shared-expert pseudo id here: nothing is skipped
    {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i32d(x_stride));
        a.add(i32d(rows));
        for ([_]u64{ b.xq, b.xs, b.xs2 }) |v| a.add(v);
        try cuda.launch.launch(q.quant_x, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 160 } }, s, &a);
    }
    {
        var a: cuda.Args = .{};
        for ([_]u64{ b.xq, b.xs, b.xs2 }) |v| a.add(v);
        a.add(@as(f32, 0));
        a.add(i32d(slots));
        for ([_]u64{ om.gu, om.gu_scale2, om.dn_scale2, b.items, b.item_count, b.members }) |v| a.add(v);
        a.add(skip);
        a.add(i32d(omma_hidden));
        a.add(i32d(omma_inter));
        for ([_]u64{ 0, 0, b.gate, 0 }) |v| a.add(v);
        const units = item_bound * ((omma_inter / 32 + 1) / 2);
        const gu = if (rows > OmmaKernels.wide_gu_rows) q.gu_s2 else q.gu;
        try cuda.launch.launch(gu.f, .{ .grid = .{ .x = q.grid(gu, units), .y = 1, .z = 1 }, .block = .{ .x = OmmaKernels.block }, .shared = gu.shared }, s, &a);
    }
    {
        var a: cuda.Args = .{};
        a.add(b.gate);
        a.add(i32d(omma_inter));
        a.add(i32d(pairs));
        for ([_]u64{ b.aq, b.as, b.as2 }) |v| a.add(v);
        try cuda.launch.launch(q.quant_h, .{ .grid = .{ .x = @intCast(pairs), .y = 1, .z = 1 }, .block = .{ .x = 160 } }, s, &a);
    }
    {
        var a: cuda.Args = .{};
        for ([_]u64{ b.aq, b.as, om.dn, om.dn_scale2, om.dn_scale2, b.as2, b.items, b.item_count, b.members }) |v| a.add(v);
        a.add(skip);
        a.add(i32d(omma_inter));
        a.add(i32d(omma_hidden));
        a.add(b.down);
        const units = item_bound * ((omma_hidden / 32 + 1) / 2);
        try cuda.launch.launch(q.dn.f, .{ .grid = .{ .x = q.grid(q.dn, units), .y = 1, .z = 1 }, .block = .{ .x = OmmaKernels.block }, .shared = q.dn.shared }, s, &a);
    }
}

/// X -> MXFP8 rows (ue8m0 scales absolute, no row scale), grouped gate|up with fused capped SwiGLU
/// (bf16 rows into `gate`), those rows -> MXFP8, then down into b.down (bf16 [pairs][hidden]).
/// The plan was built by the caller; all pointers are fixed device addresses (CUDA-graph safe).
fn runQmma(q: *const OmmaKernels, s: cuda.Stream, qm: Qmma, b: Layout, x: u64, x_stride: usize, rows: usize, slots: usize, pairs: usize, item_bound: usize) !void {
    const skip: i32 = max_experts; // no shared-expert pseudo id here: nothing is skipped
    const m_pad = std.mem.alignForward(usize, rows, 16);
    const pairs_pad = std.mem.alignForward(usize, pairs, 16);
    {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i32d(rows)); // M
        a.add(i32d(omma_hidden)); // K
        a.add(i32d(x_stride)); // ldx
        for ([_]u64{ b.mx_xq, b.mx_xs }) |v| a.add(v);
        a.add(i32d(m_pad));
        try cuda.launch.launch(q.mxfp8_quant, .{ .grid = .{ .x = @intCast((m_pad * (omma_hidden / 32) * 4 + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }
    {
        var a: cuda.Args = .{};
        for ([_]u64{ b.mx_xq, b.mx_xs }) |v| a.add(v);
        a.add(i32d(m_pad));
        for ([_]u64{ qm.w8_g, qm.bs_g, qm.w8_u, qm.bs_u }) |v| a.add(v);
        a.add(@as(f32, 1.0));
        a.add(b.gate);
        a.add(i32d(slots));
        a.add(i32d(omma_hidden)); // K
        a.add(i32d(omma_inter)); // N
        for ([_]u64{ b.items, b.item_count, b.members }) |v| a.add(v);
        a.add(skip);
        const units = item_bound * ((omma_inter + 63) / 64);
        // Prefill rows (>= 32, tile 64) run the BM=64 kernels; narrow rows keep the BM=16 path.
        if (rows >= 32) {
            try cuda.launch.launch(q.mxfp8_gu64.f, .{ .grid = .{ .x = q.grid(q.mxfp8_gu64, units), .y = 1, .z = 1 }, .block = .{ .x = 256 }, .shared = q.mxfp8_gu64.shared }, s, &a);
        } else {
            try cuda.launch.launch(q.mxfp8_gu.f, .{ .grid = .{ .x = q.grid(q.mxfp8_gu, units), .y = 1, .z = 1 }, .block = .{ .x = 128 }, .shared = q.mxfp8_gu.shared }, s, &a);
        }
    }
    {
        var a: cuda.Args = .{};
        a.add(b.gate);
        a.add(i32d(pairs)); // M
        a.add(i32d(omma_inter)); // K
        a.add(i32d(omma_inter)); // ldx
        for ([_]u64{ b.mx_aq, b.mx_as }) |v| a.add(v);
        a.add(i32d(pairs_pad));
        try cuda.launch.launch(q.mxfp8_quant, .{ .grid = .{ .x = @intCast((pairs_pad * (omma_inter / 32) * 4 + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }
    {
        var a: cuda.Args = .{};
        for ([_]u64{ b.mx_aq, b.mx_as }) |v| a.add(v);
        a.add(i32d(pairs_pad));
        for ([_]u64{ qm.w8_d, qm.bs_d }) |v| a.add(v);
        a.add(@as(f32, 1.0));
        a.add(b.down);
        a.add(i32d(omma_inter)); // K
        a.add(i32d(omma_hidden)); // N
        for ([_]u64{ b.items, b.item_count, b.members }) |v| a.add(v);
        a.add(skip);
        const units = item_bound * ((omma_hidden + 63) / 64);
        if (rows >= 32) {
            try cuda.launch.launch(q.mxfp8_dn64.f, .{ .grid = .{ .x = q.grid(q.mxfp8_dn64, units), .y = 1, .z = 1 }, .block = .{ .x = 256 }, .shared = q.mxfp8_dn64.shared }, s, &a);
        } else {
            try cuda.launch.launch(q.mxfp8_dn.f, .{ .grid = .{ .x = q.grid(q.mxfp8_dn, units), .y = 1, .z = 1 }, .block = .{ .x = 128 }, .shared = q.mxfp8_dn.shared }, s, &a);
        }
    }
}

fn put(e: anytype, bytes: []const u8) !u64 {
    const p = try e.alloc(bytes.len);
    try e.arena.upload(p - e.arena.ptr, bytes);
    return p;
}

fn loadProjection(e: anytype, ck: *core.Checkpoint, prefix: []const u8, name: []const u8, format: Format, count: usize, n: usize, k: usize) !Projection {
    var wb: [256]u8 = undefined;
    var sb: [256]u8 = undefined;
    const weight_name = try std.fmt.bufPrint(&wb, "{s}{s}_weight", .{ prefix, name });
    const scale_name = try std.fmt.bufPrint(&sb, "{s}{s}_weight_scale", .{ prefix, name });
    const w = switch (format) {
        .nvfp4 => try ck.expect(weight_name, .u8, &.{ count, n, k / 2 }),
        .mxfp8 => try ck.expect(weight_name, .f8_e4m3, &.{ count, n, k }),
    };
    const scales = switch (format) {
        .nvfp4 => try ck.expect(scale_name, .f8_e4m3, &.{ count, n, k / 16 }),
        .mxfp8 => try ck.expect(scale_name, .u8, &.{ count, n, k / 32 }),
    };
    var p: Projection = .{ .codes = try put(e, w.bytes), .scales = try put(e, scales.bytes), .n = @intCast(n), .k = @intCast(k) };
    if (format == .nvfp4) {
        var ab: [256]u8 = undefined;
        var db: [256]u8 = undefined;
        const amax = try ck.expect(try std.fmt.bufPrint(&ab, "{s}{s}_weight_amax", .{ prefix, name }), .f32, &.{count});
        const denominator = try ck.expect(try std.fmt.bufPrint(&db, "{s}{s}_weight_outer_scale_denominator", .{ prefix, name }), .f32, &.{1});
        p.amax = try put(e, amax.bytes);
        p.denominator = try put(e, denominator.bytes);
    }
    return p;
}

/// Upload the checkpoint's two stored projections without a repack or an activation-quantization side input.
pub fn load(e: anytype, ck: *core.Checkpoint, prefix: []const u8, format: Format, count: usize, hidden: usize, intermediate: usize, omma: bool) !Layer {
    if (count == 0 or count > max_experts or hidden % 32 != 0 or intermediate % 32 != 0) return error.BadExperts;
    if (omma and format == .nvfp4 and hidden == omma_hidden and intermediate == omma_inter) {
        const w13: Projection = .{ .codes = 0, .scales = 0, .n = @intCast(2 * intermediate), .k = @intCast(hidden) };
        const w2: Projection = .{ .codes = 0, .scales = 0, .n = @intCast(hidden), .k = @intCast(intermediate) };
        return .{ .format = format, .experts = @intCast(count), .w13 = w13, .w2 = w2, .omma = try loadOmma(e, ck, prefix, count, hidden, intermediate) };
    }
    if (qmmaEnabled() and format == .mxfp8 and hidden == omma_hidden and intermediate == omma_inter) {
        const w13: Projection = .{ .codes = 0, .scales = 0, .n = @intCast(2 * intermediate), .k = @intCast(hidden) };
        const w2: Projection = .{ .codes = 0, .scales = 0, .n = @intCast(hidden), .k = @intCast(intermediate) };
        return .{ .format = format, .experts = @intCast(count), .w13 = w13, .w2 = w2, .qmma = try loadQmma(e, ck, prefix, count, hidden, intermediate) };
    }
    return .{
        .format = format,
        .experts = @intCast(count),
        .w13 = try loadProjection(e, ck, prefix, "w13", format, count, 2 * intermediate, hidden),
        .w2 = try loadProjection(e, ck, prefix, "w2", format, count, hidden, intermediate),
    };
}

/// The two stored projections of an MXFP8 layer, repacked for the grouped QMMA path and uploaded expert by
/// expert (the same bytes as the stored ones in a different order, so the arena sized for the checkpoint
/// holds them). w13's rows interleave gate and up (gate even, up odd): gate takes row0=0/step=2, up
/// row0=1/step=2; w2 takes row0=0/step=1. Both N and K are multiples of 64, so npad == n.
fn loadQmma(e: anytype, ck: *core.Checkpoint, prefix: []const u8, count: usize, hidden: usize, intermediate: usize) !Qmma {
    var names: [4][256]u8 = undefined;
    const w13 = try ck.expect(try std.fmt.bufPrint(&names[0], "{s}w13_weight", .{prefix}), .f8_e4m3, &.{ count, 2 * intermediate, hidden });
    const s13 = try ck.expect(try std.fmt.bufPrint(&names[1], "{s}w13_weight_scale", .{prefix}), .u8, &.{ count, 2 * intermediate, hidden / 32 });
    const w2 = try ck.expect(try std.fmt.bufPrint(&names[2], "{s}w2_weight", .{prefix}), .f8_e4m3, &.{ count, hidden, intermediate });
    const s2 = try ck.expect(try std.fmt.bufPrint(&names[3], "{s}w2_weight_scale", .{prefix}), .u8, &.{ count, hidden, intermediate / 32 });

    const gpa = std.heap.page_allocator;
    const gu_codes = try gpa.alloc(u8, intermediate * hidden);
    defer gpa.free(gu_codes);
    const gu_scales = try gpa.alloc(u8, intermediate * hidden / 32);
    defer gpa.free(gu_scales);
    const dn_codes = try gpa.alloc(u8, hidden * intermediate);
    defer gpa.free(dn_codes);
    const dn_scales = try gpa.alloc(u8, hidden * intermediate / 32);
    defer gpa.free(dn_scales);
    const w8_g = try e.alloc(count * intermediate * hidden);
    const bs_g = try e.alloc(count * intermediate * hidden / 32);
    const w8_u = try e.alloc(count * intermediate * hidden);
    const bs_u = try e.alloc(count * intermediate * hidden / 32);
    const w8_d = try e.alloc(count * hidden * intermediate);
    const bs_d = try e.alloc(count * hidden * intermediate / 32);
    const n13 = 2 * intermediate;
    for (0..count) |x| {
        const c13 = w13.bytes[x * n13 * hidden ..][0 .. n13 * hidden];
        const sc13 = s13.bytes[x * n13 * (hidden / 32) ..][0 .. n13 * (hidden / 32)];
        pack.fragmentOrder(c13, intermediate, hidden, intermediate, 0, 2, gu_codes); // gate
        pack.tileExponents(sc13, intermediate, intermediate, hidden, 0, 2, gu_scales);
        try e.arena.upload(w8_g - e.arena.ptr + x * intermediate * hidden, gu_codes);
        try e.arena.upload(bs_g - e.arena.ptr + x * intermediate * hidden / 32, gu_scales);
        pack.fragmentOrder(c13, intermediate, hidden, intermediate, 1, 2, gu_codes); // up
        pack.tileExponents(sc13, intermediate, intermediate, hidden, 1, 2, gu_scales);
        try e.arena.upload(w8_u - e.arena.ptr + x * intermediate * hidden, gu_codes);
        try e.arena.upload(bs_u - e.arena.ptr + x * intermediate * hidden / 32, gu_scales);
        const c2 = w2.bytes[x * hidden * intermediate ..][0 .. hidden * intermediate];
        const sc2 = s2.bytes[x * hidden * (intermediate / 32) ..][0 .. hidden * (intermediate / 32)];
        pack.fragmentOrder(c2, hidden, intermediate, hidden, 0, 1, dn_codes);
        pack.tileExponents(sc2, hidden, hidden, intermediate, 0, 1, dn_scales);
        try e.arena.upload(w8_d - e.arena.ptr + x * hidden * intermediate, dn_codes);
        try e.arena.upload(bs_d - e.arena.ptr + x * hidden * intermediate / 32, dn_scales);
    }
    return .{ .w8_g = w8_g, .bs_g = bs_g, .w8_u = w8_u, .bs_u = bs_u, .w8_d = w8_d, .bs_d = bs_d };
}

fn f32At(bytes: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
}

/// The two stored projections of an NVFP4 layer, repacked for sm121a_moe.cu and uploaded expert by expert (the same bytes
/// as the stored ones in a different order, so the arena sized for the checkpoint holds them). w13's rows interleave
/// gate and up (gate even, up odd): the gate|up cells take both.
fn loadOmma(e: anytype, ck: *core.Checkpoint, prefix: []const u8, count: usize, hidden: usize, intermediate: usize) !Omma {
    var names: [4][256]u8 = undefined;
    const w13 = try ck.expect(try std.fmt.bufPrint(&names[0], "{s}w13_weight", .{prefix}), .u8, &.{ count, 2 * intermediate, hidden / 2 });
    const s13 = try ck.expect(try std.fmt.bufPrint(&names[1], "{s}w13_weight_scale", .{prefix}), .f8_e4m3, &.{ count, 2 * intermediate, hidden / 16 });
    const w2 = try ck.expect(try std.fmt.bufPrint(&names[2], "{s}w2_weight", .{prefix}), .u8, &.{ count, hidden, intermediate / 2 });
    const s2 = try ck.expect(try std.fmt.bufPrint(&names[3], "{s}w2_weight_scale", .{prefix}), .f8_e4m3, &.{ count, hidden, intermediate / 16 });
    var ab: [256]u8 = undefined;
    var db: [256]u8 = undefined;
    var ab2: [256]u8 = undefined;
    var db2: [256]u8 = undefined;
    const amax13 = try ck.expect(try std.fmt.bufPrint(&ab, "{s}w13_weight_amax", .{prefix}), .f32, &.{count});
    const den13 = try ck.expect(try std.fmt.bufPrint(&db, "{s}w13_weight_outer_scale_denominator", .{prefix}), .f32, &.{1});
    const amax2 = try ck.expect(try std.fmt.bufPrint(&ab2, "{s}w2_weight_amax", .{prefix}), .f32, &.{count});
    const den2 = try ck.expect(try std.fmt.bufPrint(&db2, "{s}w2_weight_outer_scale_denominator", .{prefix}), .f32, &.{1});

    const gpa = std.heap.page_allocator;
    const gu_words = (intermediate / 32) * (hidden / 32) * 2 * 144; // one expert's gate|up cells
    const dn_words = (hidden / 32) * (intermediate / 32) * 144;
    const gu_buf = try gpa.alloc(u32, gu_words);
    defer gpa.free(gu_buf);
    const dn_buf = try gpa.alloc(u32, dn_words);
    defer gpa.free(dn_buf);
    const gu_ptr = try e.alloc(count * gu_words * 4);
    const dn_ptr = try e.alloc(count * dn_words * 4);
    const scale_gu = try gpa.alloc(f32, count * 2);
    defer gpa.free(scale_gu);
    const scale_dn = try gpa.alloc(f32, count);
    defer gpa.free(scale_dn);
    const n13 = 2 * intermediate;
    for (0..count) |x| {
        const c13 = w13.bytes[x * n13 * (hidden / 2) ..][0 .. n13 * (hidden / 2)];
        const sc13 = s13.bytes[x * n13 * (hidden / 16) ..][0 .. n13 * (hidden / 16)];
        pack.packNvfp4Expert(c13, sc13, intermediate, hidden, 0, 2, gu_buf, 0, 2); // gate
        pack.packNvfp4Expert(c13, sc13, intermediate, hidden, 1, 2, gu_buf, 1, 2); // up
        const c2 = w2.bytes[x * hidden * (intermediate / 2) ..][0 .. hidden * (intermediate / 2)];
        const sc2 = s2.bytes[x * hidden * (intermediate / 16) ..][0 .. hidden * (intermediate / 16)];
        pack.packNvfp4Expert(c2, sc2, hidden, intermediate, 0, 1, dn_buf, 0, 1);
        try e.arena.upload(gu_ptr - e.arena.ptr + x * gu_words * 4, std.mem.sliceAsBytes(gu_buf));
        try e.arena.upload(dn_ptr - e.arena.ptr + x * dn_words * 4, std.mem.sliceAsBytes(dn_buf));
        const outer13 = pack.foldNvfp4Outer(f32At(amax13.bytes, x), f32At(den13.bytes, 0));
        scale_gu[2 * x] = outer13;
        scale_gu[2 * x + 1] = outer13;
        scale_dn[x] = pack.foldNvfp4Outer(f32At(amax2.bytes, x), f32At(den2.bytes, 0));
    }
    return .{
        .gu = gu_ptr,
        .dn = dn_ptr,
        .gu_scale2 = try put(e, std.mem.sliceAsBytes(scale_gu)),
        .dn_scale2 = try put(e, std.mem.sliceAsBytes(scale_dn)),
    };
}

test "Mini's expert scratch holds all tensor-core stages and the group plan" {
    const bytes = try scratchBytes(2048, 8, 2304, 1152);
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(Weight));
    try std.testing.expect(bytes > 2048 * 8 * (3 * 1152 + 2304) * 2);
    try std.testing.expectEqual(@as(i32, 0), @backingInt(Format.nvfp4));
    try std.testing.expectEqual(@as(i32, 1), @backingInt(Format.mxfp8));
    try std.testing.expectError(error.BadExperts, scratchBytes(0, 8, 2304, 1152));
}

/// One-member warp: A/B lanes explicit; exact rationals expose scalar/mma FP32 reduction diffs despite BF16 rounding.
const FragmentHostProbe = struct {
    const expert_count = 2;
    const hidden = 64; // two 32-column output blocks in the down projection
    const intermediate = 64; // two gate/up blocks and two 32-input groups in down
    const gate_cap = 4;
    const up_cap = 25;

    const Matrix = struct {
        codes: []const u8,
        scales: []const u8,
        n: usize,
        k: usize,

        /// Checkpoint NVFP4 -> BF16, with outer scale one to isolate fragment addressing.
        fn decoded(m: Matrix, e: usize, row: usize, col: usize) u16 {
            const z = (e * m.n + row) * m.k + col;
            const nib = (m.codes[z / 2] >> @as(u3, @intCast(4 * (col & 1)))) & 15;
            const magnitude = ([8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 })[nib & 7];
            const q = if ((nib & 8) != 0) -magnitude else magnitude;
            const scale_code = m.scales[z / 16];
            const se = (scale_code >> 3) & 15;
            const sm = scale_code & 7;
            const sign: u32 = if ((scale_code & 128) != 0) 0x80000000 else 0;
            const scale: f32 = if (se == 0)
                @as(f32, @floatFromInt(sm)) * 0x1p-9 * (if (sign != 0) @as(f32, -1) else 1)
            else
                @bitCast(sign | ((@as(u32, se) + 120) << 23) | (@as(u32, sm) << 20));
            return core.bf16_round.bits(q * scale);
        }
    };

    fn value(bits: u16) f32 {
        return core.bf16_round.value(bits);
    }

    fn fill(codes: []u8, scales: []u8, e: usize, n: usize, k: usize, one_per_row: bool) void {
        const order = [16]u8{ 2, 10, 4, 13, 1, 7, 14, 0, 3, 9, 5, 12, 6, 15, 11, 8 };
        const scale_codes = [4]u8{ 0x28, 0x30, 0x2c, 0x34 }; // 1/4, 1/2, 3/8, 3/4
        for (0..n) |row| for (0..k) |col| {
            const z = (e * n + row) * k + col;
            const index = (7 * row + 11 * col + (row * col) % 13 + 3 * e) & 15;
            var code = order[index];
            // Down's one nonzero rotates through all K positions over the two output tiles.
            if (one_per_row and col != (19 * row + 7 * e) % k) code = 0;
            if (one_per_row and col == (19 * row + 7 * e) % k and (code & 7) == 0) code = 2;
            if ((col & 1) == 0) codes[z / 2] = code else codes[z / 2] |= code << 4;
            if ((col & 15) == 0) scales[z / 16] = scale_codes[(3 * row + col / 16 + e) & 3];
        };
    }

    /// The arithmetic baseline: dequantize row-major weights and evaluate x @ W directly.
    fn scalarProjection(m: Matrix, e: usize, row0: usize, row_step: usize, x: []const u16, out: []u16) void {
        for (out, 0..) |*o, col| {
            var sum: f32 = 0;
            const row = row0 + row_step * col;
            for (0..m.k) |k| sum += value(x[k]) * value(m.decoded(e, row, k));
            o.* = core.bf16_round.bits(sum);
        }
    }

    /// Served RT=1 (ten max): materialize uint2/uint4 A/B; m16n8k16 D gathers PTX lanes, B column 2*t+n, not lane gq.
    fn fragmentProjection(m: Matrix, e: usize, row0: usize, row_step: usize, x: []const u16, out: []u16, members: usize) void {
        const width = out.len / members;
        for (0..width / 32) |cb| {
            var acc: [32][4][4]f32 = std.mem.zeroes([32][4][4]f32);
            for (0..m.k / 32) |g| {
                // [lane][row half][q][uint2 member][BF16 in the member]
                var a: [32][2][2][2][2]f32 = std.mem.zeroes([32][2][2][2][2]f32);
                // Four ld_w uint4s: [n8 tile][lane][eight BF16 values in x,y,z,w].
                var b: [4][32][8]f32 = undefined;
                for (0..32) |lane| {
                    const gq = lane >> 2;
                    const t = lane & 3;
                    for (0..2) |h| for (0..2) |q| for (0..2) |pair| for (0..2) |element| {
                        const i = gq + 8 * h;
                        // x = X + token*x_stride + 4*t; uint2 spans two adjacent pairs.
                        const col = 32 * g + 16 * q + 4 * t + 2 * pair + element;
                        a[lane][h][q][pair][element] = if (i < members) value(x[col]) else 0;
                    };
                    for (0..4) |j| for (0..8) |i| {
                        const row = row0 + row_step * (cb * 32 + 8 * j + gq);
                        const col = g * 32 + 4 * t + (i & 3) + 16 * (i >> 2);
                        b[j][lane][i] = value(m.decoded(e, row, col));
                    };
                }
                for (0..2) |q| for (0..4) |j| for (0..32) |lane| {
                    const gq = lane >> 2;
                    const t = lane & 3;
                    for (0..2) |h| for (0..2) |n| {
                        const output_col = 2 * t + n;
                        var dot: f32 = 0;
                        for (0..16) |kk| {
                            // uint2 A/B permute mma K as 4*t+(0,1,2,3), valid only together; gather both source lanes.
                            const a_lane = 4 * gq + kk / 4;
                            const b_lane = 4 * output_col + kk / 4;
                            const av = a[a_lane][h][q][(kk & 3) / 2][kk & 1];
                            const bv = b[j][b_lane][4 * q + (kk & 3)];
                            dot += av * bv;
                        }
                        acc[lane][j][2 * h + n] += dot;
                    };
                };
            }
            // Every valid member uses its own lane group's D row, including h=1,q=1.
            for (0..32) |lane| for (0..2) |h| {
                const member = (lane >> 2) + 8 * h;
                if (member >= members) continue;
                for (0..4) |j| for (0..2) |n| {
                    const col = cb * 32 + 8 * j + 2 * (lane & 3) + n;
                    out[member * width + col] = core.bf16_round.bits(acc[lane][j][2 * h + n]);
                };
            };
        }
    }

    fn reference(m13: Matrix, m2: Matrix, x: []const u16) [hidden]u16 {
        var gate: [expert_count * intermediate]u16 = undefined;
        var up: [expert_count * intermediate]u16 = undefined;
        var act: [expert_count * intermediate]u16 = undefined;
        var down: [expert_count * hidden]u16 = undefined;
        var out: [hidden]u16 = undefined;
        for (0..expert_count) |e| {
            scalarProjection(m13, e, 0, 2, x, gate[e * intermediate ..][0..intermediate]);
            scalarProjection(m13, e, 1, 2, x, up[e * intermediate ..][0..intermediate]);
            for (0..intermediate) |j| {
                const raw = value(gate[e * intermediate + j]);
                const gc = gate_cap * @as(f32, @floatCast(std.math.tanh(raw / gate_cap)));
                const uc = up_cap * @as(f32, @floatCast(std.math.tanh(value(up[e * intermediate + j]) / up_cap)));
                const sigmoid: f32 = 1 / (1 + @exp(-raw));
                act[e * intermediate + j] = core.bf16_round.bits(gc * sigmoid * uc);
            }
            scalarProjection(m2, e, 0, 1, act[e * intermediate ..][0..intermediate], down[e * hidden ..][0..hidden]);
        }
        // Routing is slot 0 -> expert 1 (+3/4), slot 1 -> expert 0 (-3/8).
        for (&out, 0..) |*o, col| {
            var total: f32 = 0;
            total += -0.375 * value(down[col]);
            total += 0.75 * value(down[hidden + col]);
            o.* = core.bf16_round.bits(total);
        }
        return out;
    }

    fn model(m13: Matrix, m2: Matrix, x: []const u16) [hidden]u16 {
        var gate: [expert_count * intermediate]u16 = undefined;
        var up: [expert_count * intermediate]u16 = undefined;
        var act: [expert_count * intermediate]u16 = undefined;
        var down: [expert_count * hidden]u16 = undefined;
        var out: [hidden]u16 = undefined;
        for (0..expert_count) |e| {
            fragmentProjection(m13, e, 0, 2, x, gate[e * intermediate ..][0..intermediate], 1);
            fragmentProjection(m13, e, 1, 2, x, up[e * intermediate ..][0..intermediate], 1);
            for (0..intermediate) |j| {
                const raw = value(gate[e * intermediate + j]);
                const gc = gate_cap * @as(f32, @floatCast(std.math.tanh(raw / gate_cap)));
                const uc = up_cap * @as(f32, @floatCast(std.math.tanh(value(up[e * intermediate + j]) / up_cap)));
                const sigmoid: f32 = 1 / (1 + @exp(-raw));
                act[e * intermediate + j] = core.bf16_round.bits(gc * sigmoid * uc);
            }
            fragmentProjection(m2, e, 0, 1, act[e * intermediate ..][0..intermediate], down[e * hidden ..][0..hidden], 1);
        }
        for (&out, 0..) |*o, col| {
            var total: f32 = 0;
            total += -0.375 * value(down[col]);
            total += 0.75 * value(down[hidden + col]);
            o.* = core.bf16_round.bits(total);
        }
        return out;
    }

    fn checkMembers(m13: Matrix, x: []const u16) !void {
        const members = 10;
        var tensor: [members * intermediate]u16 = undefined;
        var scalar: [intermediate]u16 = undefined;
        scalarProjection(m13, 0, 0, 2, x, &scalar);
        fragmentProjection(m13, 0, 0, 2, x, &tensor, members);
        var first_difference: ?usize = null;
        for (tensor, 0..) |got, i| if (got != scalar[i % intermediate]) {
            first_difference = i;
            break;
        };
        try std.testing.expectEqual(@as(?usize, null), first_difference);
    }

    fn check() !void {
        var codes13: [expert_count * 2 * intermediate * hidden / 2]u8 = undefined;
        var scales13: [expert_count * 2 * intermediate * hidden / 16]u8 = undefined;
        var codes2: [expert_count * hidden * intermediate / 2]u8 = undefined;
        var scales2: [expert_count * hidden * intermediate / 16]u8 = undefined;
        for (0..expert_count) |e| {
            fill(&codes13, &scales13, e, 2 * intermediate, hidden, false);
            fill(&codes2, &scales2, e, hidden, intermediate, true);
        }
        const m13: Matrix = .{ .codes = &codes13, .scales = &scales13, .n = 2 * intermediate, .k = hidden };
        const m2: Matrix = .{ .codes = &codes2, .scales = &scales2, .n = hidden, .k = intermediate };
        var x: [hidden]u16 = undefined;
        for (&x, 0..) |*v, i| v.* = core.bf16_round.bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 128);
        const got = model(m13, m2, &x);
        const want = reference(m13, m2, &x);
        var first_difference: ?usize = null;
        for (got, want, 0..) |g, w, i| if (g != w) {
            first_difference = i;
            break;
        };
        try std.testing.expectEqual(@as(?usize, null), first_difference);
        try checkMembers(m13, &x);
    }
};

test "one token: m16n8k16 fragment pipeline matches independent BF16 dequantization" {
    try FragmentHostProbe.check();
}

//! Nemotron's CUDA kernels: our .cu fatbins with the Python wrappers' launch logic, and a captured Triton set if given.

const std = @import("std");
const cuda = @import("cuda");
const torch_ops = @import("cuda_torch_ops.zig");
const glue = @import("cuda_glue.zig");
const train_ops = @import("cuda_train_ops.zig");

/// Mangled names of the instantiations the copies in zig/kernels/cuda export (cuobjdump -symbols of each fatbin).
const sym = struct {
    const pattn = "_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi8ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiif";
    const scan_rows = "_ZN12tf_scan_rows11scan_kernelEPK13__nv_bfloat16S2_PfPKfS5_S5_PS0_iiiiiiiiff";
};

pub const pattn_smem: u32 = 65536; // eight 32-key slots of 128 dims

/// The 4-bit projections' shared pieces (cuda/qlinear.zig): split-K scratch and a tiled weight.
pub const Split = cuda.qlinear.Split;
pub const QLinear = cuda.qlinear.Affine4;

/// A layer's grouped expert tables ([E, N/32, K/64, 1, 288] int32 blocks).
pub const Experts = struct { up: u64, down: u64, count: usize, width: usize, dims: usize };

/// Scratch the expert plan writes (cuda/grouped.zig, shared by the MoE families).
pub const Plan = cuda.grouped.Plan;

pub const Kernels = struct {
    d: *const cuda.Driver,
    mods: [22]cuda.Module,
    triton: ?cuda.aot.Set, // a captured Triton set for the glue (GB10's qualified one); null: our own glue kernels
    glue: glue.Fns,
    affine: cuda.qlinear.Affine4Kernels, // the 4-bit projections: qmm_group, lane_gemv and qmm_prefill
    router: cuda.grouped.Router,
    experts: cuda.experts.Affine4Experts, // the 4-bit grouped experts: experts, experts_prefill and experts_pack
    pattn: cuda.Function,
    scan_rows: cuda.Function,
    serial_feed: cuda.Function,
    plan_routed: cuda.Function,
    rest_rows: cuda.Function,
    draw: cuda.Function,
    draw_ids: cuda.Function,
    logprob_rows: cuda.Function,
    torch: torch_ops.Functions,
    train: train_ops.Fns = undefined, // Sliding Weights: the change after each output projection, and learning it
    train_mods: ?[2]cuda.Module = null, // loaded by loadTrain (--slide) after the weights, else never
    gb10: bool,
    discrete: bool, // the card has its own memory: checkpoint bytes reach it through page-locked slots

    /// Loads every module; `triton_dir`, if given, holds a captured aot.json and cubins for this GPU's glue.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, triton_dir: ?[]const u8) !Kernels {
        const d = ctx.d;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.d = d;
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.experts, kk.experts_prefill, kk.experts_pack, kk.prefill_attention, kk.scan_rows, kk.nemotron_ops, kk.torch_argmax, kk.torch_topk, kk.torch_pointwise, kk.torch_indexing, kk.torch_movement, kk.torch_nemotron_constants, kk.sample, kk.lane_gemv, kk.nemotron_norms, kk.nemotron_route, kk.nemotron_mamba, kk.nemotron_attention, kk.nemotron_keyed, kk.affine4_pack };
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        k.router = try cuda.grouped.Router.resolve(k.mods[2]);
        k.pattn = try k.mods[5].function(sym.pattn);
        k.scan_rows = try k.mods[6].function(sym.scan_rows);
        k.serial_feed = try k.mods[7].function("tf_serial_feed");
        k.plan_routed = try k.mods[7].function("tf_plan_routed");
        k.rest_rows = try k.mods[7].function("tf_rest_rows");
        k.torch = try torch_ops.Functions.resolve(k.mods[8..14]);
        k.draw = try k.mods[14].function("tf_draw");
        k.draw_ids = try k.mods[14].function("tf_draw_ids");
        k.glue = try glue.Fns.resolve(k.mods[16..21]);
        k.logprob_rows = try k.mods[14].function("tf_logprob_rows");
        k.train_mods = null;
        k.triton = if (triton_dir) |dir| try cuda.aot.Set.load(gpa, io, d, ctx.device, dir) else null;
        errdefer if (k.triton) |*t| t.deinit();
        try k.pattn.allowDynamicShared(pattn_smem);
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        k.experts = try cuda.experts.Affine4Experts.resolve(k.mods[2], k.mods[3], k.mods[4], sms);
        const major = try ctx.attribute(.compute_capability_major);
        const minor = try ctx.attribute(.compute_capability_minor);
        k.gb10 = major == 12 and minor == 1;
        k.discrete = try ctx.attribute(.integrated) == 0;
        k.affine = try cuda.qlinear.Affine4Kernels.resolve(d, k.mods[0], k.mods[1], k.mods[15], k.mods[21], sms, k.gb10);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.affine.deinit();
        if (k.triton) |*t| t.deinit();
        for (&k.mods) |*m| m.unload();
        if (k.train_mods) |*ms| for (ms) |*m| m.unload();
    }

    /// Sliding Weights' modules (--slide only), loaded after the weights so those sit where they would without them.
    pub fn loadTrain(k: *Kernels) !void {
        if (k.train_mods != null) return;
        var train = try cuda.Module.load(k.d, cuda.kernels.train);
        errdefer train.unload();
        var mixers = try cuda.Module.load(k.d, cuda.kernels.train_mixers);
        errdefer mixers.unload();
        k.train = try train_ops.Fns.resolve(train, mixers);
        k.train_mods = .{ train, mixers };
    }
};

/// qmm.split_k (cuda/qlinear.zig).
pub const splitK = cuda.qlinear.splitK;

/// experts.max_items (cuda/grouped.zig).
pub const maxItems = cuda.grouped.maxItems;

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

/// Launch helpers on one stream; each mirrors the Python wrapper it replaces.
pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,

    fn go(o: Ops, f: cuda.Function, grid: [3]usize, block: u32, shared: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = u(grid[0]), .y = u(grid[1]), .z = u(grid[2]) }, .block = .{ .x = block }, .shared = shared }, o.s, args);
    }

    /// The torch-op replacements on this stream.
    pub fn torch(o: Ops) torch_ops.Torch {
        return .{ .f = &o.k.torch, .s = o.s };
    }

    /// The training kernels (and the change's forward) on this stream.
    pub fn train(o: Ops) train_ops.Train {
        return .{ .f = &o.k.train, .s = o.s, .d = o.k.d };
    }

    /// sample.cu: row r of bf16 logits drawn at position meta[0] + r + 1 + offset, columns as `ids` token ids if given.
    pub fn draw(o: Ops, logits: u64, vocab: usize, rule: u64, meta: u64, offset: usize, out: u64, rows: usize, ids: ?u64, prob: ?u64) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(vocab)));
        a.add(rule);
        a.add(meta);
        a.add(@as(i32, @intCast(offset)));
        a.add(out);
        if (ids) |x| a.add(x);
        a.add(prob orelse 0);
        try o.go(if (ids != null) o.k.draw_ids else o.k.draw, .{ rows, 1, 1 }, 1024, 0, &a);
    }

    /// sample.cu: each bf16 logits row's lanes.logprob words at `out`, for its pick in `picks` and `k` best tokens.
    pub fn logprobRows(o: Ops, logits: u64, vocab: usize, picks: u64, k: u8, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(vocab)));
        a.add(picks);
        a.add(@as(u32, k));
        a.add(out);
        try o.go(o.k.logprob_rows, .{ rows, 1, 1 }, 1024, 0, &a);
    }

    /// The 4-bit projections on this stream (cuda/qlinear.zig's affine-4 paths, bits unchanged).
    pub fn dense(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize) !void {
        return o.k.affine.decode(o.s, x, xs, q, out, rows);
    }

    pub fn gemvSplit(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize, sp: Split) !void {
        return o.k.affine.gemvSplit(o.s, x, xs, q, out, rows, sk, sp);
    }

    pub fn gemv(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) !void {
        return o.k.affine.gemv(o.s, x, xs, q, out, rows, sk);
    }

    pub fn cluster(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) !void {
        return o.k.affine.cluster(o.s, x, xs, q, out, rows, sk);
    }

    pub fn prefillDense(o: Ops, x: u64, q: QLinear, out: u64, rows: usize) !void {
        return o.k.affine.prompt(o.s, x, q, out, rows);
    }

    /// nemotron_ops' rest: y (bf16, or fp32 when `y_f32`) += r x at rows `i * row_mul + row_add` for i < rows.
    pub fn restRows(o: Ops, x: u64, x_stride: usize, r: u64, y: u64, y_stride: usize, y_f32: bool, rows: usize, row_mul: usize, row_add: usize, n: usize, k: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        a.add(r);
        a.add(y);
        a.add(int(y_stride));
        a.add(@as(c_int, @intFromBool(y_f32)));
        for ([_]usize{ rows, row_mul, row_add, n, k }) |v| a.add(int(v));
        try o.go(o.k.rest_rows, .{ (n + 3) / 4, 1, 1 }, 128, 0, &a);
    }

    /// experts.route: pairs grouped by expert into items of at most `tile` pairs (16 decode, 64 prefill).
    pub fn plan(o: Ops, picks: u64, pairs: usize, count: usize, tile: usize, p: Plan) !void {
        return o.k.router.route(o.s, picks, pairs, count, tile, p);
    }

    /// experts.route's plan over the routed slots alone (rows <= 16, experts <= 128): the shared halves run apart.
    pub fn planRouted(o: Ops, picks: u64, rows: usize, slots: usize, routed: usize, count: usize, tile: usize, p: Plan) !void {
        if (rows > 16 or slots > 8 or count > 128) return error.PlanTooWide;
        var a: cuda.Args = .{};
        a.add(picks);
        for ([_]usize{ rows, slots, routed, count, tile }) |v| a.add(int(v));
        for ([_]u64{ p.members, p.items, p.counts }) |v| a.add(v);
        try o.go(o.k.plan_routed, .{ 1, 1, 1 }, 128, 0, &a);
    }

    /// The 4-bit grouped experts on this stream (cuda/experts.zig's affine-4 kernels, launches unchanged).
    pub fn experts(o: Ops, up: bool, x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize, max_units: usize) !void {
        return o.k.experts.run(o.s, up, x, x_stride, slots, w, kg, nb, p, out, n, max_units);
    }

    pub fn expertsPrefill(o: Ops, up: bool, x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize, max_items: usize) !void {
        return o.k.experts.runPrompt(o.s, up, x, x_stride, slots, w, kg, nb, p, out, n, max_items);
    }

    /// prefill_attention (head dim 128): q (rows, heads, 128) at positions p0.. against caches filled through them.
    pub fn prefillAttention(o: Ops, q: u64, kc: u64, vc: u64, out: u64, p0: usize, rows: usize, heads: usize, kv_heads: usize, scale: f32) !void {
        const g = heads / kv_heads;
        var a: cuda.Args = .{};
        for ([_]u64{ q, kc, vc, out }) |v| a.add(v);
        for ([_]usize{ p0, rows, heads, kv_heads, g }) |v| a.add(int(v));
        a.add(scale);
        try o.go(o.k.pattn, .{ (rows + 15) / 16, kv_heads * (g / 8), 1 }, 256, pattn_smem, &a);
    }

    /// mamba.scan_rows: a chunk's scan through scan_rows.cu; `state` ends at the chunk's last row.
    pub fn scanRows(o: Ops, proj: u64, xc: u64, state: u64, a_: u64, d_: u64, dtb: u64, y: u64, rows: usize, proj_w: usize, heads: usize, dh: usize, cd: usize, groups: usize, lo: f32, hi: f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ proj, xc, state, a_, d_, dtb, y }) |v| a.add(v);
        for ([_]usize{ rows, proj_w, heads * dh, cd, heads * dh + cd, dh, heads / groups, groups }) |v| a.add(int(v));
        a.add(lo);
        a.add(hi);
        try o.go(o.k.scan_rows, .{ heads, dh / 32, 1 }, 128, 0, &a);
    }

    pub fn copy(o: Ops, dst: u64, src: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try o.k.d.check(o.k.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, o.s.handle), "cuMemcpyDtoDAsync");
    }

    pub fn fill32(o: Ops, dst: u64, value: u32, words: usize) !void {
        try o.k.d.check(o.k.d.api.cuMemsetD32Async(dst, value, words, o.s.handle), "cuMemsetD32Async");
    }

    pub fn upload(o: Ops, dst: u64, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try o.k.d.check(o.k.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, o.s.handle), "cuMemcpyHtoDAsync");
    }

    pub fn download(o: Ops, dst: []u8, src: u64) !void {
        if (dst.len == 0) return;
        try o.k.d.check(o.k.d.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, o.s.handle), "cuMemcpyDtoHAsync");
    }

    /// One serial round's end on the device: its token feeds the next window, the window's meta advances.
    pub fn serialFeed(o: Ops, sampled: u64, ids: u64, meta: u64, history: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ sampled, ids, meta, history }) |v| a.add(v);
        try o.go(o.k.serial_feed, .{ 1, 1, 1 }, 1, 0, &a);
    }
};

test "items follow the Python shapes" {
    try std.testing.expectEqual(@as(usize, 135), maxItems(328, 130, 64));
    try std.testing.expectEqual(@as(usize, 1154), maxItems(16384, 130, 16));
    try std.testing.expectEqual(@as(usize, 8), maxItems(8, 130, 16));
}

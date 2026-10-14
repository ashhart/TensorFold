//! Nemotron's CUDA kernels: our .cu fatbins with the Python wrappers' launch logic, and a captured Triton set if given.

const std = @import("std");
const cuda = @import("cuda");
const torch_ops = @import("cuda_torch_ops.zig");
const train_ops = @import("cuda_train_ops.zig");
const glue = @import("cuda_glue.zig");

/// Mangled names of the instantiations the copies in zig/kernels/cuda export (cuobjdump -symbols of each fatbin).
const sym = struct {
    const pattn = "_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi8ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiif";
    const scan_rows = "_ZN12tf_scan_rows11scan_kernelILb0EEEvPK13__nv_bfloat16S3_PfPKfS6_S6_PS1_iiiiiiiiffNS_5SnapsE";
    const scan_snaps = "_ZN12tf_scan_rows11scan_kernelILb1EEEvPK13__nv_bfloat16S3_PfPKfS6_S6_PS1_iiiiiiiiffNS_5SnapsE";
};

pub const pattn_smem: u32 = 65536; // eight 32-key slots of 128 dims

/// The projections' shared pieces (cuda/qlinear.zig): split-K scratch, a weight in any format, the 4-bit tiles.
pub const Split = cuda.qlinear.Split;
pub const QLinear = cuda.qlinear.Weight;
pub const Affine4 = cuda.qlinear.Affine4;

/// A layer's grouped experts in their format's blocks (cuda/experts.zig), the shared expert's halves last.
pub const Experts = cuda.experts.Layer;

/// ModelOpt checkpoints' kernels: the lane matmul, the prompt GEMM, NVFP4 experts and the repack (loadModelopt).
pub const Modelopt = struct { lane: cuda.qmmf.Lane, gemm: cuda.nvfp4.Prompt, experts: cuda.experts.Nvfp4Experts, pack: cuda.modelopt.Packer };

/// Scratch the expert plan writes (cuda/grouped.zig, shared by the MoE families).
pub const Plan = cuda.grouped.Plan;

/// A pocket's neurons, [n, D] fp32 up and down; the first `used` in use, their count on the device at `count` too.
pub const Pocket = struct { up: u64, down: u64, n: usize, used: usize = 0, count: u64 = 0 };

/// The most neurons a pocket holds: the pocket kernels' activation rows are this wide.
pub const pocket_width = 64;

/// Rows inside a scan launch where it also stores its state (scan_rows.cu `Snaps`): `r[k]` rows taken in, -1 none.
pub const Snaps = extern struct {
    p: [max]u64 = @splat(0),
    r: [max]i32 = @splat(-1),
    pub const max = 16;
};

pub const Kernels = struct {
    d: *const cuda.Driver,
    mods: [22]cuda.Module,
    triton: ?cuda.aot.Set, // a captured Triton set for the glue (GB10's qualified one); null: our own glue kernels
    glue: glue.Fns,
    emb: cuda.embed_rows.EmbedRows,
    norms: cuda.norms.Norms,
    affine: cuda.qlinear.Affine4Kernels, // the 4-bit projections: qmm_group, lane_gemv and qmm_prefill
    router: cuda.grouped.Router,
    experts: cuda.experts.Affine4Experts, // the 4-bit grouped experts: experts, experts_prefill and experts_pack
    pattn: cuda.Function,
    scan_rows: cuda.Function,
    scan_snaps: cuda.Function, // the same scan, storing its state at rows inside the launch too
    serial_feed: cuda.Function,
    plan_routed: cuda.Function,
    rest_rows: cuda.Function,
    pocket_in: [2]cuda.Function, // the pocket's two halves a block row a row, then tiled for a prompt chunk's rows
    pocket_out: [2]cuda.Function,
    add_row: cuda.Function,
    draws: cuda.sample.Draws, // sample.cu: the keyed draws and logprob rows
    grammar_mask: cuda.Function, // structured output: disallowed tokens -inf before the draw
    torch: torch_ops.Functions,
    train: train_ops.Kernels, // Sliding Weights' learning step, and the learned change a forward adds while lessons run
    modelopt: ?Modelopt = null, // loaded by loadModelopt for a ModelOpt checkpoint, else never
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
        k.scan_snaps = try k.mods[6].function(sym.scan_snaps);
        k.serial_feed = try k.mods[7].function("tf_serial_feed");
        k.plan_routed = try k.mods[7].function("tf_plan_routed");
        k.rest_rows = try k.mods[7].function("tf_rest_rows");
        k.pocket_in = .{ try k.mods[7].function("tf_pocket_in"), try k.mods[7].function("tf_pocket_in_rows") };
        k.pocket_out = .{ try k.mods[7].function("tf_pocket_out"), try k.mods[7].function("tf_pocket_out_rows") };
        k.add_row = try k.mods[7].function("tf_add_row");
        k.torch = try torch_ops.Functions.resolve(k.mods[8..14]);
        k.draws = try cuda.sample.Draws.resolve(k.mods[14]);
        k.glue = try glue.Fns.resolve(k.mods[16..21]);
        k.emb = try cuda.embed_rows.EmbedRows.load(d);
        errdefer k.emb.unload();
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.grammar_mask = try k.mods[14].function("tf_grammar_mask");
        k.modelopt = null;
        k.triton = if (triton_dir) |dir| try cuda.aot.Set.load(gpa, io, d, ctx.device, dir) else null;
        errdefer if (k.triton) |*t| t.deinit();
        k.train = try train_ops.Kernels.load(d);
        errdefer k.train.deinit();
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
        k.emb.unload();
        k.norms.unload();
        if (k.modelopt) |*m| {
            m.lane.unload();
            m.gemm.unload();
            m.experts.unload();
            m.pack.unload();
        }
        k.affine.deinit();
        k.train.deinit();
        if (k.triton) |*t| t.deinit();
        for (&k.mods) |*m| m.unload();
    }

    /// The lane matmul, prompt GEMM, NVFP4 experts and repack a ModelOpt checkpoint's FP8, NVFP4 and bf16 tensors need.
    pub fn loadModelopt(k: *Kernels, ctx: *const cuda.Context) !void {
        if (k.modelopt != null) return;
        const major: u32 = @intCast(try ctx.attribute(.compute_capability_major));
        const minor: u32 = @intCast(try ctx.attribute(.compute_capability_minor));
        if (modeloptRefusal(10 * major + minor) != null) return error.ModeloptNeedsSm90;
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        var lane = try cuda.qmmf.Lane.load(k.d, major);
        errdefer lane.unload();
        var gemm = try cuda.nvfp4.Prompt.load(k.d);
        errdefer gemm.unload();
        var experts = try cuda.experts.Nvfp4Experts.load(k.d, sms);
        errdefer experts.unload();
        k.modelopt = .{ .lane = lane, .gemm = gemm, .experts = experts, .pack = try cuda.modelopt.Packer.load(k.d) };
    }
};

/// FP8 and NVFP4 paths stop below this full capability: decode's split-K sums meet in a thread-block cluster.
pub const modelopt_min: u32 = 90;

/// Why a ModelOpt checkpoint cannot load at `capability` (10 * major + minor), or null when it may.
pub fn modeloptRefusal(capability: u32) ?[]const u8 {
    if (capability >= modelopt_min) return null;
    return "FP8 and NVFP4 weights meet their split-K sums in a thread-block cluster: " ++
        "a ModelOpt checkpoint needs compute capability 9.0 or newer";
}

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
    pub fn train(o: Ops) train_ops.Ops {
        return .{ .k = &o.k.train, .s = o.s };
    }

    /// sample.cu's keyed draw on this stream (cuda/sample.zig, the launch unchanged).
    pub fn draw(o: Ops, logits: u64, vocab: usize, rule: u64, meta: u64, offset: usize, out: u64, rows: usize, ids: ?u64, prob: ?u64) !void {
        return o.k.draws.draw(o.s, logits, vocab, rule, meta, offset, out, rows, ids, prob);
    }

    /// sample.cu: `rows` bf16 logits rows' disallowed tokens -inf, their bits `words` words a row at `bits`.
    pub fn grammarMask(o: Ops, logits: u64, vocab: usize, bits: u64, words: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(vocab)));
        a.add(bits);
        a.add(@as(u32, @intCast(words)));
        try o.go(o.k.grammar_mask, .{ rows, 1, 1 }, 256, 0, &a);
    }

    /// sample.cu's logprob rows on this stream: each row's words for its pick, `k` best tokens and rank if `rank`.
    pub fn logprobRows(o: Ops, logits: u64, vocab: usize, picks: u64, k: u8, rank: bool, out: u64, rows: usize) !void {
        return o.k.draws.logprobRows(o.s, logits, vocab, picks, k, rank, out, rows);
    }

    /// Every projection format the kernels loaded (cuda/qlinear.zig).
    pub fn linear(o: Ops) cuda.qlinear.Linear {
        const m = if (o.k.modelopt) |*x| x else null;
        return .{ .affine4 = &o.k.affine, .lane = if (m) |x| &x.lane else null, .nvfp4_prompt = if (m) |x| &x.gemm else null };
    }

    /// Every grouped-expert format the kernels loaded (cuda/experts.zig).
    pub fn grouped(o: Ops) cuda.experts.Grouped {
        return .{ .affine4 = &o.k.experts, .nvfp4 = if (o.k.modelopt) |*x| &x.experts else null };
    }

    /// Decode rows of a projection on this stream (the affine-4 paths' bits unchanged); xs: x's 64-group sums.
    pub fn dense(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize) !void {
        return o.linear().decode(o.s, q, .{ .x = x, .sums = xs }, out, rows);
    }

    pub fn gemvSplit(o: Ops, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize, sp: Split) !void {
        return o.k.affine.gemvSplit(o.s, x, xs, q, out, rows, sk, sp);
    }

    pub fn gemv(o: Ops, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize) !void {
        return o.k.affine.gemv(o.s, x, xs, q, out, rows, sk);
    }

    pub fn cluster(o: Ops, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize) !void {
        return o.k.affine.cluster(o.s, x, xs, q, out, rows, sk);
    }

    /// Prompt rows of a projection on this stream, by its format's prompt GEMM.
    pub fn prefillDense(o: Ops, x: u64, q: QLinear, out: u64, rows: usize) !void {
        return o.linear().prompt(o.s, q, .{ .x = x }, out, rows);
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

    /// nemotron_ops' pocket: out (bf16, fp32 if `out_f32`) += down relu^2(up y) at rows `i * row_mul + row_add`.
    pub fn pocketRows(
        o: Ops,
        y: u64,
        d: usize,
        p: Pocket,
        act: u64,
        out: u64,
        out_f32: bool,
        rows: usize,
        row_mul: usize,
        row_add: usize,
    ) !void {
        if (p.n > pocket_width) return error.PocketTooWide;
        if (p.used == 0) return; // no neuron in use: nothing launched, so the output keeps its bits, an exact -0.0 too
        // a block row a row until the tiled forms (eight neurons by eight rows a warp, 16 rows a thread) run faster
        const tiled = [2]bool{ rows >= 256, rows >= 64 };
        var a: cuda.Args = .{};
        a.add(y);
        a.add(int(d));
        a.add(p.up);
        a.add(p.count);
        a.add(act);
        a.add(int(rows));
        const in_grid: [3]usize = if (tiled[0])
            .{ (p.n + 63) / 64, (rows + 7) / 8, 1 }
        else
            .{ (p.n + 3) / 4, rows, 1 };
        try o.go(o.k.pocket_in[@intFromBool(tiled[0])], in_grid, if (tiled[0]) 256 else 128, 0, &a);
        var b: cuda.Args = .{};
        b.add(act);
        b.add(p.down);
        b.add(p.count);
        b.add(int(d));
        b.add(out);
        b.add(@as(c_int, @intFromBool(out_f32)));
        for ([_]usize{ rows, row_mul, row_add }) |v| b.add(int(v));
        try o.go(
            o.k.pocket_out[@intFromBool(tiled[1])],
            .{ (d + 127) / 128, if (tiled[1]) (rows + 15) / 16 else rows, 1 },
            128,
            0,
            &b,
        );
    }

    /// nemotron_ops' nudge: y (bf16, or fp32 when `y_f32`) += x (fp32) over n values.
    pub fn addRow(o: Ops, y: u64, y_f32: bool, x: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(y);
        a.add(@as(c_int, @intFromBool(y_f32)));
        a.add(x);
        a.add(int(n));
        try o.go(o.k.add_row, .{ (n + 255) / 256, 1, 1 }, 256, 0, &a);
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

    /// prefill_attention (head dim 128): q (rows, heads, 128) at positions p0.. against caches filled through them.
    pub fn prefillAttention(o: Ops, q: u64, kc: u64, vc: u64, out: u64, p0: usize, rows: usize, heads: usize, kv_heads: usize, scale: f32) !void {
        const g = heads / kv_heads;
        var a: cuda.Args = .{};
        for ([_]u64{ q, kc, vc, out }) |v| a.add(v);
        for ([_]usize{ p0, rows, heads, kv_heads, g }) |v| a.add(int(v));
        a.add(scale);
        try o.go(o.k.pattn, .{ (rows + 15) / 16, kv_heads * (g / 8), 1 }, 256, pattn_smem, &a);
    }

    /// mamba.scan_rows: a chunk's scan through scan_rows.cu; `state` ends at the chunk's last row, `snaps` at theirs.
    pub fn scanRows(o: Ops, proj: u64, xc: u64, state: u64, a_: u64, d_: u64, dtb: u64, y: u64, rows: usize, proj_w: usize, heads: usize, dh: usize, cd: usize, groups: usize, lo: f32, hi: f32, snaps: Snaps) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ proj, xc, state, a_, d_, dtb, y }) |v| a.add(v);
        for ([_]usize{ rows, proj_w, heads * dh, cd, heads * dh + cd, dh, heads / groups, groups }) |v| a.add(int(v));
        a.add(lo);
        a.add(hi);
        a.add(snaps);
        try o.go(if (snaps.r[0] < 0) o.k.scan_rows else o.k.scan_snaps, .{ heads, dh / 32, 1 }, 128, 0, &a);
    }

    /// `height` rows of `width` bytes, `spitch` apart at `src`, to `dpitch` apart at `dst`, in one call.
    pub fn copyRows(o: Ops, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize) !void {
        if (width == 0 or height == 0) return;
        const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
        try o.k.d.check(o.k.d.api.cuMemcpy2DAsync_v2(&m, o.s.handle), "cuMemcpy2DAsync");
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

test "ModelOpt's floor refuses every card below sm_90, the Ampere ones included, and admits sm_90 up" {
    for ([_]u32{ 75, 80, 86, 89 }) |c| try std.testing.expect(modeloptRefusal(c) != null);
    for ([_]u32{ 90, 100, 120, 121 }) |c| try std.testing.expect(modeloptRefusal(c) == null);
}

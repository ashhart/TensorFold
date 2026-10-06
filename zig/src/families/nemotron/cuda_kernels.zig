//! Nemotron's CUDA kernels: our .cu fatbins with the Python wrappers' launch logic, and the captured Triton set.

const std = @import("std");
const cuda = @import("cuda");
const torch_ops = @import("cuda_torch_ops.zig");

/// Mangled names of the instantiations the copies in zig/kernels/cuda export (cuobjdump -symbols of each fatbin).
const sym = struct {
    const group = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    const prefill_mm = "_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi3ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";
    const expert_up = "_ZN10tf_experts13expert_kernelILi64ELi1ELi1ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
    const expert_down = "_ZN10tf_experts13expert_kernelILi64ELi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
    const plan = "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_";
    const plan_rank = "_ZN10tf_experts9plan_rankEPKiiiPiS2_";
    const plan_offsets = "_ZN10tf_experts12plan_offsetsEiiiPiS0_S0_";
    const plan_scatter = "_ZN10tf_experts12plan_scatterEPKiiiS1_S1_Pi";
    const pre_up = "_ZN18tf_experts_prefill14prefill_kernelILi64ELi1ELi1ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
    const pre_down = "_ZN18tf_experts_prefill14prefill_kernelILi64ELi1ELi3ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
    const pack_experts = "_ZN15tf_experts_pack11pack_kernelILi2EEEvPKjPKtS4_Pjiiii";
    const pattn = "_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi8ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiif";
    const scan_rows = "_ZN12tf_scan_rows11scan_kernelEPK13__nv_bfloat16S2_PfPKfS5_S5_PS0_iiiiiiiiff";
    const gemv = "_ZN12tf_lane_gemv11gemv_kernelILi64ELi64ELi8EEEvPK13__nv_bfloat16PKfNS_4PartEiii";
};

/// qmm_group.cu's Part and Parts, passed by value: four projections at most, one used here.
pub const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
pub const Parts = extern struct { p: [4]Part, count: c_int };

/// lane_gemv.cu's Part: one projection whose column tiles the CTAs share out.
pub const GemvPart = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int };

comptime {
    std.debug.assert(@sizeOf(Part) == 56 and @sizeOf(Parts) == 232 and @sizeOf(GemvPart) == 48);
}

pub const group_smem: u32 = 35328; // LaneTile<64, 16, 64, 1, 4, 8>::SMEM
pub const prefill_mm_smem: u32 = 62976; // Tile<64, 128, 128, 2, 4, 3>::SMEM
pub const pre_experts_smem: u32 = 38400; // Pre<64, 1, 2, 2, 4>: three stages of 800 uint4
pub const pattn_smem: u32 = 65536; // eight 32-key slots of 128 dims

/// A tiled 4-bit projection: packed words, (kg, npad) scales and biases, n outputs from k inputs.
pub const QLinear = struct { w: u64, s: u64, b: u64, n: usize, k: usize, npad: usize };

/// A layer's grouped expert tables ([E, N/32, K/64, 1, 288] int32 blocks).
pub const Experts = struct { up: u64, down: u64, count: usize, width: usize, dims: usize };

/// Scratch the expert plan writes: members, items (expert, first, count), counts, and the wide path's rank and hist.
pub const Plan = struct { members: u64, items: u64, counts: u64, rank: u64, hist: u64 };

pub const Kernels = struct {
    d: *const cuda.Driver,
    mods: [16]cuda.Module,
    triton: cuda.aot.Set,
    group: cuda.Function,
    gemv: cuda.Function,
    prefill_mm: cuda.Function,
    expert_up: cuda.Function,
    expert_down: cuda.Function,
    plan_small: cuda.Function,
    plan_rank: cuda.Function,
    plan_offsets: cuda.Function,
    plan_scatter: cuda.Function,
    pre_up: cuda.Function,
    pre_down: cuda.Function,
    pack_experts: cuda.Function,
    pattn: cuda.Function,
    scan_rows: cuda.Function,
    pack_dense: cuda.Function,
    transpose16: cuda.Function,
    serial_feed: cuda.Function,
    plan_routed: cuda.Function,
    draw: cuda.Function,
    draw_ids: cuda.Function,
    torch: torch_ops.Functions,
    expert_blocks: [2]usize, // resident blocks the decode expert kernels fill: per SM times SMs
    gemv_blocks: usize, // resident lane_gemv CTAs: per SM times SMs
    gb10: bool,

    /// Loads every module; `triton_dir` holds the captured aot.json and cubins for this GPU.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, triton_dir: []const u8) !Kernels {
        const d = ctx.d;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.d = d;
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.experts, kk.experts_prefill, kk.experts_pack, kk.prefill_attention, kk.scan_rows, kk.nemotron_ops, kk.torch_argmax, kk.torch_topk, kk.torch_pointwise, kk.torch_indexing, kk.torch_movement, kk.torch_nemotron_constants, kk.sample, kk.lane_gemv };
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        k.group = try k.mods[0].function(sym.group);
        k.prefill_mm = try k.mods[1].function(sym.prefill_mm);
        k.expert_up = try k.mods[2].function(sym.expert_up);
        k.expert_down = try k.mods[2].function(sym.expert_down);
        k.plan_small = try k.mods[2].function(sym.plan);
        k.plan_rank = try k.mods[2].function(sym.plan_rank);
        k.plan_offsets = try k.mods[2].function(sym.plan_offsets);
        k.plan_scatter = try k.mods[2].function(sym.plan_scatter);
        k.pre_up = try k.mods[3].function(sym.pre_up);
        k.pre_down = try k.mods[3].function(sym.pre_down);
        k.pack_experts = try k.mods[4].function(sym.pack_experts);
        k.pattn = try k.mods[5].function(sym.pattn);
        k.scan_rows = try k.mods[6].function(sym.scan_rows);
        k.pack_dense = try k.mods[7].function("tf_pack_dense");
        k.transpose16 = try k.mods[7].function("tf_transpose_pad16");
        k.serial_feed = try k.mods[7].function("tf_serial_feed");
        k.plan_routed = try k.mods[7].function("tf_plan_routed");
        k.gemv = try k.mods[15].function(sym.gemv);
        k.torch = try torch_ops.Functions.resolve(k.mods[8..14]);
        k.draw = try k.mods[14].function("tf_draw");
        k.draw_ids = try k.mods[14].function("tf_draw_ids");
        k.triton = try cuda.aot.Set.load(gpa, io, d, ctx.device, triton_dir);
        errdefer k.triton.deinit();
        try k.group.allowDynamicShared(group_smem);
        try k.gemv.allowDynamicShared(group_smem);
        try k.prefill_mm.allowDynamicShared(prefill_mm_smem);
        try k.pre_up.allowDynamicShared(pre_experts_smem);
        try k.pre_down.allowDynamicShared(pre_experts_smem);
        try k.pattn.allowDynamicShared(pattn_smem);
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        k.expert_blocks = .{ @max(1, try k.expert_up.occupancy(128, 0)) * sms, @max(1, try k.expert_down.occupancy(128, 0)) * sms };
        k.gemv_blocks = @max(1, try k.gemv.occupancy(128, group_smem)) * sms;
        const major = try ctx.attribute(.compute_capability_major);
        const minor = try ctx.attribute(.compute_capability_minor);
        k.gb10 = major == 12 and minor == 1;
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.triton.deinit();
        for (&k.mods) |*m| m.unload();
    }
};

/// qmm.split_k: K slices fixed by the weight's shape, never by the row count.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// experts.max_items: an item per used expert plus one per `tile` pairs past its first.
pub fn maxItems(pairs: usize, experts: usize, tile: usize) usize {
    return @min(pairs, experts) + pairs / tile;
}

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

    /// qmm.matmul on sm_12x: x (rows, k) bf16 with group sums xs -> out (rows, n) bf16, qmm_group's tile-2 bits.
    pub fn dense(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize) !void {
        if (rows > 16) return error.WindowTooWide;
        const sk = splitK(q.n, q.k);
        return if (sk > 1) o.gemv(x, xs, q, out, rows, sk) else o.cluster(x, xs, q, out, rows, sk);
    }

    /// lane_gemv: every K slice of a column tile in one CTA, summed in slice order, CTAs looping over the tiles.
    pub fn gemv(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) !void {
        const tiles = (q.n + 63) / 64;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(GemvPart{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles) });
        for ([_]usize{ rows, q.k, q.k }) |v| a.add(int(v));
        const cfg: cuda.Config = .{ .grid = .{ .x = u(@min(tiles, o.k.gemv_blocks)) }, .block = .{ .x = 128 }, .shared = group_smem, .pdl = o.k.gb10 };
        try cuda.launch.launch(o.k.gemv, cfg, o.s, &a);
    }

    /// qmm_group's tile 2: a cluster of `sk` CTAs a column tile, its K slices summed over distributed shared memory.
    pub fn cluster(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) !void {
        const tiles = (q.n + 63) / 64;
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
        const rows_t = (rows + 15) / 16;
        const clusters = rows_t * tiles; // one part: the cluster is its sk K slices of one tile
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(parts);
        for ([_]usize{ rows, q.k, q.k, rows_t, sk }) |v| a.add(int(v));
        const cfg: cuda.Config = .{
            .grid = .{ .x = u(clusters * sk) },
            .block = .{ .x = 128 },
            .shared = group_smem,
            .cluster = if (sk > 1) .{ .x = u(sk) } else null,
            .pdl = o.k.gb10,
        };
        try cuda.launch.launch(o.k.group, cfg, o.s, &a);
    }

    /// qmm.prefill_matmul (tile 0): weights rounded once to bf16, one fp32 chain over K; bf16 out.
    pub fn prefillDense(o: Ops, x: u64, q: QLinear, out: u64, rows: usize) !void {
        const rows_t = (rows + 127) / 128;
        const band = (12 << 20) / (128 * q.k * 2);
        const group = @max(1, @min(rows_t, band));
        var a: cuda.Args = .{};
        for ([_]u64{ x, q.w, q.s, q.b, out }) |v| a.add(v);
        for ([_]usize{ rows, q.n, q.k, q.npad, q.k, group }) |v| a.add(int(v));
        try o.go(o.k.prefill_mm, .{ rows_t * ((q.n + 127) / 128), 1, 1 }, 256, prefill_mm_smem, &a);
    }

    /// experts.route: pairs grouped by expert into items of at most `tile` pairs (16 decode, 64 prefill).
    pub fn plan(o: Ops, picks: u64, pairs: usize, count: usize, tile: usize, p: Plan) !void {
        var a: cuda.Args = .{};
        if (pairs <= 1024) {
            a.add(picks);
            for ([_]usize{ pairs, count, tile }) |v| a.add(int(v));
            for ([_]u64{ p.members, p.items, p.counts }) |v| a.add(v);
            return o.go(o.k.plan_small, .{ 1, 1, 1 }, 1024, 0, &a);
        }
        const nblk = (pairs + 1023) / 1024;
        a.add(picks);
        a.add(int(pairs));
        a.add(int(count));
        a.add(p.rank);
        a.add(p.hist);
        try o.go(o.k.plan_rank, .{ nblk, 1, 1 }, 1024, 0, &a);
        var b: cuda.Args = .{};
        for ([_]usize{ nblk, count, tile }) |v| b.add(int(v));
        for ([_]u64{ p.hist, p.items, p.counts }) |v| b.add(v);
        try o.go(o.k.plan_offsets, .{ 1, 1, 1 }, 1024, 0, &b);
        var c: cuda.Args = .{};
        c.add(picks);
        c.add(int(pairs));
        c.add(int(count));
        for ([_]u64{ p.rank, p.hist, p.members }) |v| c.add(v);
        try o.go(o.k.plan_scatter, .{ (pairs + 255) / 256, 1, 1 }, 256, 0, &c);
    }

    fn expertArgs(x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize) cuda.Args {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        a.add(int(slots));
        a.add(w);
        a.add(int(kg));
        a.add(int(nb));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(int(n));
        a.add(@as(f32, 0.0));
        return a;
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

    /// experts.run (decode form): `up` takes token rows (relu^2, bf16 out), else pair rows (fp32 out).
    pub fn experts(o: Ops, up: bool, x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize, max_units: usize) !void {
        const grid = @min((max_units + 3) / 4, o.k.expert_blocks[if (up) 0 else 1]);
        if (grid < 1) return;
        var a = expertArgs(x, x_stride, slots, w, kg, nb, p, out, n);
        try o.go(if (up) o.k.expert_up else o.k.expert_down, .{ grid, 1, 1 }, 128, 0, &a);
    }

    /// experts.prefill: 64 pairs x 128 columns a CTA; `up` relu^2, else bf16 sums (epilogue 3).
    pub fn expertsPrefill(o: Ops, up: bool, x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize, max_items: usize) !void {
        const grid = max_items * ((nb + 3) / 4);
        if (grid < 1) return;
        var a = expertArgs(x, x_stride, slots, w, kg, nb, p, out, n);
        try o.go(if (up) o.k.pre_up else o.k.pre_down, .{ grid, 1, 1 }, 256, pre_experts_smem, &a);
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

test "split_k and items follow the Python shapes" {
    try std.testing.expectEqual(@as(usize, 2), splitK(10304, 2688));
    try std.testing.expectEqual(@as(usize, 8), splitK(2688, 4096));
    try std.testing.expectEqual(@as(usize, 2), splitK(4608, 2688));
    try std.testing.expectEqual(@as(usize, 4), splitK(2688, 5376));
    try std.testing.expectEqual(@as(usize, 1), splitK(131072, 2688));
    try std.testing.expectEqual(@as(usize, 135), maxItems(328, 130, 64));
    try std.testing.expectEqual(@as(usize, 1154), maxItems(16384, 130, 16));
    try std.testing.expectEqual(@as(usize, 8), maxItems(8, 130, 16));
}

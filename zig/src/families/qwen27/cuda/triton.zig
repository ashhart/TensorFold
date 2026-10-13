//! The 27B's glue on the shared CUDA modules (norms, act_rows, gdn_prep, qk_rows, embed_rows, tree_attention).

const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const p = aot.ptr;

fn int(name: []const u8, value: usize) aot.Arg {
    return aot.int(name, @intCast(value));
}

fn ci(name: []const u8, value: usize) aot.Const {
    return aot.ci(name, @intCast(value));
}
fn u(x: usize) u32 {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) u32 {
    return u((a + b - 1) / b);
}

fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, n) catch unreachable;
}

/// attention.py's layout constants: 512-key chunks, folded four to a group, up to 128 window rows.
pub const chunk = 512;
pub const group_chunks = 4;
pub const max_nodes = 128;
const merge_columns = 64;

/// The DeltaNet shapes glue.gdn_pre takes: projection width, key and value heads, head dim, kept conv rows.
pub const Gdn = struct { c: usize, kh: usize, vh: usize, dk: usize, keep: usize };

/// The attention shapes: query and key heads, head dim, rotary half.
pub const Attn = struct { heads: usize, kv_heads: usize, dim: usize, half: usize };

/// The shared CUDA glue modules (zig/kernels/cuda) that replace the captured Triton set.
pub const Glue = struct {
    norms: cuda.norms.Norms,
    act: cuda.act_rows.ActRows,
    gdn: cuda.gdn_prep.GdnPrep,
    qk: cuda.qk_rows.QkRows,
    embed: cuda.embed_rows.EmbedRows,
    tree: cuda.tree_attention.TreeAttention,
    block: cuda.block_attention.BlockAttention,
    /// Only FP8 target attention uses a captured set; BF16 target and DFlash2 use native CUDA.
    set: ?cuda.aot.Set,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, device: anytype, triton_dir: []const u8) !Glue {
        var g: Glue = undefined;
        g.set = if (triton_dir.len > 0) try cuda.aot.Set.load(gpa, io, d, device, triton_dir) else null;
        errdefer if (g.set) |*s| s.deinit();
        g.norms = try cuda.norms.Norms.load(d);
        errdefer g.norms.unload();
        g.act = try cuda.act_rows.ActRows.load(d);
        errdefer g.act.unload();
        g.gdn = try cuda.gdn_prep.GdnPrep.load(d);
        errdefer g.gdn.unload();
        g.qk = try cuda.qk_rows.QkRows.load(d);
        errdefer g.qk.unload();
        g.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer g.embed.unload();
        g.tree = try cuda.tree_attention.TreeAttention.load(d);
        errdefer g.tree.unload();
        g.block = try cuda.block_attention.BlockAttention.load(d);
        return g;
    }

    pub fn unload(g: *Glue) void {
        if (g.set) |*s| s.deinit();
        g.norms.unload();
        g.act.unload();
        g.gdn.unload();
        g.qk.unload();
        g.embed.unload();
        g.tree.unload();
        g.block.unload();
    }
};

/// The glue calls the 27B makes, on the shared modules (same signatures as the Triton set's wrappers had).
pub const Tri = struct {
    g: *const Glue,
    s: cuda.Stream,
    /// The optional captured FP8 attention kernels.
    set: ?*const cuda.aot.Set,
    fn run(t: Tri, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        const set = t.set orelse return error.QwenNeedsCapturedKernels;
        try set.run(t.s, name, grid, args, consts);
    }

    pub fn embed(t: Tri, ids: u64, w: u64, s: u64, b: u64, out: u64, rows: usize, d: usize) !void {
        try t.g.embed.affine4(t.s, ids, w, s, b, out, rows, d);
    }

    pub fn addRmsnorm(t: Tri, x: u64, r: ?u64, w: u64, h: u64, y: u64, xs: u64, rows: usize, d: usize, eps: f32) !void {
        try t.g.norms.addRmsH(t.s, x, r, w, .bf16, h, y, xs, rows, d, eps);
    }

    pub fn gdnPre(t: Tri, qkv: u64, cs: u64, cw: u64, win: u64, sid: ?u64, a: u64, b: u64, alog: u64, dtb: u64, q: u64, k: u64, v: u64, g: u64, beta: u64, rows: usize, m: Gdn) !void {
        if (m.c != (2 * m.kh + m.vh) * m.dk) return error.GdnShape;
        const ld: i64 = @intCast(m.c);
        const vh: i64 = @intCast(m.vh);
        try t.g.gdn.run(t.s, .{ .qkv = qkv, .qkv_ld = ld, .state = cs, .conv = cw, .win = win, .sid = sid orelse 0, .a = a, .a_ld = vh, .b = b, .b_ld = vh, .a_log = alog, .dt_bias = dtb, .q = q, .k = k, .v = v, .g = g, .beta = beta }, rows, .{ .kh = m.kh, .vh = m.vh, .dk = m.dk, .keep = m.keep });
    }

    pub fn gatedNorm(t: Tri, y: u64, z: u64, w: u64, out: u64, xs: u64, rows: usize, vh: usize, dv: usize, eps: f32) !void {
        try t.g.norms.gatedRms(t.s, y, vh * dv, z, vh * dv, w, out, xs, rows, vh, dv, eps);
    }

    pub fn swiglu(t: Tri, gate: u64, up: u64, out: u64, xs: u64, rows: usize, n: usize) !void {
        try t.g.act.swiglu(t.s, gate, n, up, n, out, xs, rows, n);
    }

    pub fn attnPrep(t: Tri, qg: u64, kv: u64, qn: u64, kn: u64, pos: u64, inv: u64, qout: u64, kout: u64, rows: usize, a: Attn, eps: f32) !void {
        const d: i64 = @intCast(a.dim);
        const src: cuda.qk_rows.Src = .{ .q = qg, .q_ld = @as(i64, @intCast(a.heads)) * 2 * d, .q_hs = 2 * d, .k = kv, .k_ld = @as(i64, @intCast(a.kv_heads)) * d, .k_hs = d };
        try t.g.qk.rowsFrom(t.s, src, .{ .h = a.heads, .hk = a.kv_heads, .dim = a.dim }, .{ .norms = qn, .norm_k = kn, .norms_bf16 = true, .inv = inv, .rot = 2 * a.half, .eps = eps }, pos, null, .{ .q = qout, .k = kout, .v = 0 }, rows);
    }

    pub fn gateMul(t: Tri, o: u64, qg: u64, out: u64, xs: u64, rows: usize, heads: usize, dim: usize) !void {
        try t.g.act.outputGate(t.s, o, qg + dim * 2, heads * 2 * dim, 2 * dim, out, xs, rows, heads, dim);
    }

    pub fn groupSums(t: Tri, x: u64, xs: u64, ld: usize, rows: usize, k: usize) !void {
        try t.g.act.groupSums(t.s, x, ld, xs, rows, k);
    }

    pub fn paths(t: Tri, parents: u64, out: u64, depths: u64, rows: usize) !void {
        try t.g.tree.paths(t.s, parents, out, depths, rows);
    }

    /// The layout attention.attention reads: the plan's tables and the partial buffers it writes.
    pub const Plan = struct { streams: u64, items: u64, n_items: usize, rows: u64, paths: u64, depths: u64, width: usize, native_groups: usize, scratch_bytes: usize };
    pub const Partials = struct { o: u64, m: u64, l: u64 };

    /// Committed keys then each row's path in native 512-key groups (cuda/tree_attention), scratch in part.o.
    pub fn attention(t: Tri, q: u64, kn: u64, vn: u64, base: u64, offs: u64, plan: Plan, part: Partials, out: u64, a: Attn) !void {
        const TA = cuda.tree_attention.TreeAttention;
        const ng = plan.native_groups;
        if (ng == 0 or TA.scratchBytes(plan.width, a.heads, a.kv_heads, ng, a.dim) > plan.scratch_bytes) return error.TreeScratch;
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(a.dim), -0.5));
        const rows: cuda.tree_attention.Rows = .{ .row_stream = plan.rows, .streams = plan.streams, .offs = offs, .origin = base, .paths = plan.paths, .depths = plan.depths };
        try t.g.tree.run(t.s, q, kn, vn, rows, TA.scratch(part.o, plan.width, a.heads, a.kv_heads, ng, a.dim), out, plan.width, a.heads, a.kv_heads, a.dim, ng, scale);
    }

    /// attention8 (attention_kv8.py, captured outside the tree like the rest of the set): attention over e4m3
    /// caches, `offs` holding each stream's four offsets (key and value bytes, their scales' fp32 elements) from
    /// `base`; the window's own keys and values stay bf16.
    pub fn attention8(t: Tri, q: u64, kn: u64, vn: u64, base: u64, offs: u64, plan: Plan, part: Partials, out: u64, a: Attn) !void {
        const g = a.heads / a.kv_heads;
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(a.dim), -0.5));
        const w = plan.width;
        if (plan.n_items > 0) try t.run("_shared8", .{ u(plan.n_items * a.kv_heads), 1, 1 }, &.{
            p("Q", "*bf16", q),                p("KC", "*fp8e4nv", base),      p("VC", "*fp8e4nv", base), p("KS", "*fp32", base),   p("VS", "*fp32", base),   p("OFF", "*i64", offs),
            p("STREAM", "*i32", plan.streams), p("ITEMS", "*i32", plan.items), p("PO", "*fp32", part.o),  p("PM", "*fp32", part.m), p("PL", "*fp32", part.l), int("W", w),
        }, &.{ ci("H", a.heads), ci("HK", a.kv_heads), ci("D", a.dim), ci("G", g), ci("CH", chunk), aot.cf("SCALE", scale), ci("GR", group_chunks) });
        const tails = 1 + (max_nodes + chunk - 1) / chunk;
        try t.run("_tail8", .{ u(w), u(a.kv_heads), u(tails) }, &.{
            p("Q", "*bf16", q),                p("KN", "*bf16", kn),         p("VN", "*bf16", vn),           p("KC", "*fp8e4nv", base),        p("VC", "*fp8e4nv", base), p("KS", "*fp32", base),   p("VS", "*fp32", base),   p("OFF", "*i64", offs),
            p("STREAM", "*i32", plan.streams), p("ROWS", "*i32", plan.rows), p("PATHS", "*i32", plan.paths), p("DEPTHS", "*i32", plan.depths), p("PO", "*fp32", part.o),  p("PM", "*fp32", part.m), p("PL", "*fp32", part.l), int("W", w),
        }, &.{ ci("H", a.heads), ci("HK", a.kv_heads), ci("D", a.dim), ci("G", g), ci("CH", chunk), ci("MAXD", max_nodes), aot.cf("SCALE", scale), ci("GR", group_chunks) });
        try t.run("_merge", .{ u(w), u(a.kv_heads), u(a.dim / merge_columns) }, &.{
            p("PO", "*fp32", part.o), p("PM", "*fp32", part.m), p("PL", "*fp32", part.l), p("OUT", "*bf16", out), p("STREAM", "*i32", plan.streams), p("ROWS", "*i32", plan.rows), int("W", w),
        }, &.{ ci("H", a.heads), ci("D", a.dim), ci("G", g), ci("DS", merge_columns), ci("CH", chunk), ci("GR", group_chunks) });
    }
};

/// attention.groups: whole committed 2048-key groups fold in their own programs only when they fill the GPU.
pub fn groups(p_: usize, w: usize) usize {
    const g = p_ / (chunk * group_chunks);
    return if (g * w >= 64) g else 0;
}

/// attention.slots: one partial slot per folded group, then one per chunk through the window's last key.
pub fn slots(p_: usize, w: usize) usize {
    const g = groups(p_, w);
    return g + (p_ + w + chunk - 1) / chunk - g * group_chunks;
}

test "attention slots follow the Python plan" {
    try std.testing.expectEqual(@as(usize, 1), slots(0, 1));
    try std.testing.expectEqual(@as(usize, 2), slots(512, 1));
    try std.testing.expectEqual(@as(usize, 9), slots(4096, 1)); // 8 chunks + the window's own
    try std.testing.expectEqual(@as(usize, 13), slots(6144, 16)); // 3 groups x 16 rows < 64: every chunk its own
    try std.testing.expectEqual(@as(usize, 4), slots(6144, 32)); // 3 folded groups, then the window's chunk
}

test "native scratch formula and contiguous slice pointers bind the pinned CUDA wrapper" {
    const TA = cuda.tree_attention.TreeAttention;
    const plan = @import("native_attention_plan.zig");
    try std.testing.expectEqual(plan.span, cuda.tree_attention.span);
    try std.testing.expectEqual(plan.slice_items, cuda.tree_attention.slice_items);
    for ([_]usize{ 128, 256 }) |dim| for ([_]usize{ 1, 16, 256 }) |rows| for ([_]usize{ 1, 4, 8, 13, 65 }) |ng| {
        const bytes = TA.scratchBytes(rows, 24, 4, ng, dim);
        try std.testing.expectEqual(plan.required(rows, 24, 4, ng, dim), bytes);
        const base: u64 = 4096;
        const sc = TA.scratch(base, rows, 24, 4, ng, dim);
        try std.testing.expect(sc.o == base and sc.m < sc.l and sc.l < sc.so and sc.so < sc.sm and sc.sm < sc.sl);
        try std.testing.expectEqual(base + bytes, sc.sl + cuda.tree_attention.slice_items * 4 * 6 * 4);
    };
}

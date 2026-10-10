//! The 27B's Triton kernels from the captured cubins: each launch has the Python wrapper's grid and constexprs.

const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

const p = aot.ptr;

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
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

pub const Tri = struct {
    set: *const aot.Set,
    s: cuda.Stream,

    fn run(t: Tri, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        try t.set.run(t.s, name, grid, args, consts);
    }

    /// glue.embed: a 4-bit table row dequantized, one program a row and 64-group.
    pub fn embed(t: Tri, ids: u64, w: u64, s: u64, b: u64, out: u64, rows: usize, d: usize) !void {
        try t.run("_embed", .{ u(rows), u(d / 64), 1 }, &.{ p("IDS", "*i32", ids), p("Wt", "*i32", w), p("S", "*bf16", s), p("B", "*bf16", b), p("OUT", "*bf16", out) }, &.{ci("D", d)});
    }

    /// glue.add_rmsnorm: h = x + r (or x itself without r), y = rmsnorm(h) * w, xs = y's 64-group sums.
    pub fn addRmsnorm(t: Tri, x: u64, r: ?u64, w: u64, h: u64, y: u64, xs: u64, rows: usize, d: usize, eps: f32) !void {
        try t.run("_add_rmsnorm", .{ u(rows), 1, 1 }, &.{ p("X", "*bf16", x), p("R", "*bf16", r orelse x), p("W", "*bf16", w), p("H", "*bf16", if (r != null) h else x), p("Y", "*bf16", y), p("XS", "*fp32", xs), aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", pow2(d)), ci("HAS_R", @intFromBool(r != null)) });
    }

    /// glue.gdn_pre: conv over each row's window of [state; qkv], q/k norms, g and beta; `sid`: several streams.
    pub fn gdnPre(t: Tri, qkv: u64, cs: u64, cw: u64, win: u64, sid: ?u64, a: u64, b: u64, alog: u64, dtb: u64, q: u64, k: u64, v: u64, g: u64, beta: u64, rows: usize, m: Gdn) !void {
        try t.run("_gdn_pre", .{ u(rows), u(2 * m.kh + m.vh), 1 }, &.{
            p("QKV", "*bf16", qkv), p("CS", "*bf16", cs), p("CW", "*bf16", cw), p("WIN", "*i32", win), p("A", "*bf16", a), p("B", "*bf16", b),       p("ALOG", "*fp32", alog),
            p("DTB", "*fp32", dtb), p("Q", "*bf16", q),   p("K", "*bf16", k),   p("V", "*bf16", v),    p("G", "*fp32", g), p("BETA", "*fp32", beta), p("SID", "*i32", sid orelse win),
        }, &.{ ci("C", m.c), ci("KH", m.kh), ci("VH", m.vh), ci("DK", m.dk), ci("NKEEP", m.keep), ci("MULTI", @intFromBool(sid != null)) });
    }

    /// glue.gated_norm: silu(z) * rmsnorm(y) * w per value head, and its group sums.
    pub fn gatedNorm(t: Tri, y: u64, z: u64, w: u64, out: u64, xs: u64, rows: usize, vh: usize, dv: usize, eps: f32) !void {
        try t.run("_gated_norm", .{ u(rows), u(vh), 1 }, &.{ p("Yr", "*bf16", y), p("Z", "*bf16", z), p("W", "*bf16", w), p("OUT", "*bf16", out), p("XS", "*fp32", xs), aot.float("eps", eps) }, &.{ ci("VH", vh), ci("DV", dv) });
    }

    /// glue.swiglu: silu(gate) * up in 1024-column blocks, and its group sums.
    pub fn swiglu(t: Tri, gate: u64, up: u64, out: u64, xs: u64, rows: usize, n: usize) !void {
        try t.run("_swiglu", .{ u(rows), cdiv(n, 1024), 1 }, &.{ p("GATE", "*bf16", gate), p("UP", "*bf16", up), p("OUT", "*bf16", out), p("XS", "*fp32", xs) }, &.{ ci("N", n), ci("BLOCK", 1024) });
    }

    /// glue.attn_prep: q (from the [q | gate] rows) and k normed, then rotated at each row's position.
    pub fn attnPrep(t: Tri, qg: u64, kv: u64, qn: u64, kn: u64, pos: u64, inv: u64, qout: u64, kout: u64, rows: usize, a: Attn, eps: f32) !void {
        try t.run("_attn_prep", .{ u(rows), u(a.heads + a.kv_heads), 1 }, &.{ p("QG", "*bf16", qg), p("KV", "*bf16", kv), p("QN", "*bf16", qn), p("KN", "*bf16", kn), p("POS", "*i32", pos), p("INV", "*fp32", inv), p("QOUT", "*bf16", qout), p("KOUT", "*bf16", kout), aot.float("eps", eps) }, &.{
            ci("H", a.heads), ci("HKV", a.kv_heads), ci("D", a.dim), ci("HALF", a.half), ci("MROPE", 0), ci("ROWS", 0), ci("HSEC", 11), ci("WSEC", 10),
        });
    }

    /// glue.gate_mul: the attention output times sigmoid(gate) from the [q | gate] rows, and its group sums.
    pub fn gateMul(t: Tri, o: u64, qg: u64, out: u64, xs: u64, rows: usize, heads: usize, dim: usize) !void {
        try t.run("_gate_mul", .{ u(rows), u(heads), 1 }, &.{ p("O", "*bf16", o), p("QG", "*bf16", qg), p("OUT", "*bf16", out), p("XS", "*fp32", xs) }, &.{ ci("H", heads), ci("D", dim) });
    }

    /// qmm.group_sums: each row's 64-group input sums in fp32 (rows may be strided by `ld`).
    pub fn groupSums(t: Tri, x: u64, xs: u64, ld: usize, rows: usize, k: usize) !void {
        try t.run("_group_sums", .{ u(rows), cdiv(k / 64, 16), 1 }, &.{ p("X", "*bf16", x), p("XS", "*fp32", xs), int("ldx", ld) }, &.{ ci("KG", k / 64), ci("GS", 64), ci("GB", 16) });
    }

    /// attention.from_packed's _paths: each row's root-to-row window rows and its depth.
    pub fn paths(t: Tri, parents: u64, out: u64, depths: u64, rows: usize) !void {
        try t.run("_paths", .{ u(rows), 1, 1 }, &.{ p("PARENTS", "*i32", parents), p("PATHS", "*i32", out), p("DEPTHS", "*i32", depths) }, &.{ci("MAXD", max_nodes)});
    }

    /// The layout attention.attention reads: the plan's tables and the partial buffers it writes.
    pub const Plan = struct { streams: u64, items: u64, n_items: usize, rows: u64, paths: u64, depths: u64, width: usize };
    pub const Partials = struct { o: u64, m: u64, l: u64 };

    /// attention.attention: committed keys in 512-key chunks, the row's own path in its tail, merged in order.
    pub fn attention(t: Tri, q: u64, kn: u64, vn: u64, base: u64, offs: u64, plan: Plan, part: Partials, out: u64, a: Attn) !void {
        const g = a.heads / a.kv_heads;
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(a.dim), -0.5));
        const w = plan.width;
        if (plan.n_items > 0) try t.run("_shared", .{ u(plan.n_items * a.kv_heads), 1, 1 }, &.{
            p("Q", "*bf16", q),             p("KC", "*bf16", base),   p("VC", "*bf16", base),   p("OFF", "*i64", offs),   p("STREAM", "*i32", plan.streams),
            p("ITEMS", "*i32", plan.items), p("PO", "*fp32", part.o), p("PM", "*fp32", part.m), p("PL", "*fp32", part.l), int("W", w),
        }, &.{ ci("H", a.heads), ci("HK", a.kv_heads), ci("D", a.dim), ci("G", g), ci("CH", chunk), aot.cf("SCALE", scale), ci("GR", group_chunks) });
        const tails = 1 + (max_nodes + chunk - 1) / chunk;
        try t.run("_tail", .{ u(w), u(a.kv_heads), u(tails) }, &.{
            p("Q", "*bf16", q),           p("KN", "*bf16", kn),           p("VN", "*bf16", vn),             p("KC", "*bf16", base),   p("VC", "*bf16", base),   p("OFF", "*i64", offs),   p("STREAM", "*i32", plan.streams),
            p("ROWS", "*i32", plan.rows), p("PATHS", "*i32", plan.paths), p("DEPTHS", "*i32", plan.depths), p("PO", "*fp32", part.o), p("PM", "*fp32", part.m), p("PL", "*fp32", part.l), int("W", w),
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

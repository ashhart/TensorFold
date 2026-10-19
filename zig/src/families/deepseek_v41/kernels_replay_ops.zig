//! Replays the other kernels' extension calls: attention, top-k, mHC, router, prefill dense GEMM and L2 prefetch.

const ops = @import("kernels_ops.zig");
const replay = @import("kernels_replay.zig");

const Call = replay.Call;
const Tensor = replay.Tensor;
const Arg = replay.Arg;
const Kw = replay.Kw;
const numel = replay.numel;
const dim = replay.dim;
const is = replay.is;

/// The bindings' `tp<T>(t)`: an empty tensor is nullptr.
fn tp(t: Tensor) u64 {
    return if (numel(t) == 0) 0 else t.ptr;
}

fn stride0(t: Tensor) i64 {
    return if (numel(t) == 0 or t.stride.len == 0) 0 else t.stride[0];
}

// tf_dsv41_attn_cuda_v1 (csa2/attn_cuda.cpp): attn and topk as dsv41_attn_run_cuda / dsv41_topk_run_cuda fill Args
pub fn attnExt(o: ops.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (is(func, "attn")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "q", "cv", "csc", "tok", "cnt", "sv", "ssc", "lo", "hi", "pos", "sl", "pt", "sink", "cs", "po", "pm", "pl", "out", "ticket", "ring", "psh", "pts", "kw", "split", "qrope" } };
        const q = try c.tensor(0);
        const cv = try c.tensor(1);
        const csc = try c.tensor(2);
        const tok = try c.tensor(3);
        const sv = try c.tensor(5);
        const ssc = try c.tensor(6);
        const hi = try c.tensor(8);
        const sl = try c.tensor(10);
        const cs = try c.tensor(13);
        const a: ops.attn.Args = .{
            .q = tp(q),
            .has_comp = @intFromBool(numel(cv) != 0),
            .cv = tp(cv),
            .csc = tp(csc),
            .cvs = stride0(cv),
            .css = stride0(csc),
            .tok = tp(tok),
            .ts = stride0(tok),
            .cnt = tp(try c.tensor(4)),
            .sv = tp(sv),
            .ssc = tp(ssc),
            .svs = sv.stride[0],
            .sss = ssc.stride[0],
            .lo = tp(try c.tensor(7)),
            .hi = tp(hi),
            .pos = (try c.tensor(9)).ptr,
            .sl = tp(sl),
            .pt = tp(try c.tensor(11)),
            .pts = try c.int(21),
            .psh = @intCast(try c.int(20)),
            .sink = tp(try c.tensor(12)),
            .cs = tp(cs),
            .csst = cs.stride[0],
            .po = tp(try c.tensor(14)),
            .pm = tp(try c.tensor(15)),
            .pl = tp(try c.tensor(16)),
            .out = tp(try c.tensor(17)),
            .ticket = tp(try c.tensor(18)),
            .R = @intCast(try dim(q, 0)),
            .H = @intCast(try dim(q, 1)),
            .ring = @intCast(try c.int(19)),
        };
        const split: u32 = if (try c.opt(23)) |v| switch (v) {
            .int => |x| @intCast(x),
            else => return error.WrongArgument,
        } else 0; // captures before R1 have no split: today's launch
        var b = a;
        const qrope: i64 = if (try c.opt(24)) |v| switch (v) {
            .int => |n| n,
            else => return error.WrongArgument,
        } else 0;
        b.qrope = @intFromBool(qrope != 0 and split > 0);
        try o.attention(b, try c.u(22), numel(sl) > 0, numel(hi) > 0, split);
        return true;
    }
    if (is(func, "topk")) return topkCall(o, args, kwargs, false);
    return false;
}

/// tf_dsv41_attn_cuda_v1.topk, or with `bounded` tf_dsv41_topk_b_v1.topk (TF_DSV41_INDEX_BOUND, same arguments)
pub fn topkCall(o: ops.Ops, args: []const Arg, kwargs: []const Kw, bounded: bool) !bool {
    const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "s0", "c0", "o0", "n0", "nk0", "k0", "m0", "e0", "s1", "c1", "o1", "n1", "nk1", "k1", "m1", "e1", "jobs", "pos", "ratio", "bs", "R", "CL" } };
    const jobs = try c.usz(16);
    const pos = try c.tensor(17);
    var a: ops.topk.Args = .{ .job = undefined, .pos = pos.ptr, .pos64 = @intFromBool(pos.dtype == .i64), .ratio = @intCast(try c.int(18)), .bs = @intCast(try c.int(19)) };
    a.job[0] = try topkJob(c, 0);
    a.job[1] = if (jobs > 1) try topkJob(c, 8) else a.job[0];
    const e0 = try c.usz(7);
    const ept = if (jobs > 1) @max(e0, try c.usz(15)) else e0;
    if (bounded) try o.topKB(a, jobs, try c.usz(20), try c.usz(21), ept) else try o.topK(a, jobs, try c.usz(20), try c.usz(21), ept);
    return true;
}

fn topkJob(c: Call, at: usize) !ops.topk.Job {
    const sc = try c.tensor(at);
    const cand = try c.tensor(at + 1);
    const out = try c.tensor(at + 2);
    return .{ .s = sc.ptr, .ss = sc.stride[0], .cand = tp(cand), .cs = stride0(cand), .out = out.ptr, .os = out.stride[0], .cnt = tp(try c.tensor(at + 3)), .nk = @intCast(try c.int(at + 4)), .k = @intCast(try c.int(at + 5)), .mode = @intCast(try c.int(at + 6)), .ept = @intCast(try c.int(at + 7)) };
}

// tf_dsv41_mhc_cuda_v1.run (mhc_cuda.cpp / dsv41_mhc_run_cuda)
pub fn mhcRun(o: ops.Ops, args: []const Arg, kwargs: []const Kw) !bool {
    const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "xout", "g", "post", "comb", "pre", "fn", "base", "scale", "part", "c", "tap", "nw", "out", "opre", "opost", "ocomb", "cnt", "R", "mode", "eps", "hc_eps", "post_alpha", "iters", "spin", "defer" } };
    const x = try c.tensor(0);
    const g = try c.tensor(2);
    const f = try c.tensor(6);
    const tap = try c.tensor(11);
    const nw = try c.tensor(12);
    const a: ops.mhc.Args = .{
        .x = tp(x),
        .xs = x.stride[0],
        .xout = tp(try c.tensor(1)),
        .g = tp(g),
        .gr = stride0(g),
        .world = if (numel(g) != 0) @intCast(try dim(g, 0)) else 1,
        .gbf16 = @intFromBool(g.dtype == .bf16),
        .post = tp(try c.tensor(3)),
        .comb = tp(try c.tensor(4)),
        .pre = tp(try c.tensor(5)),
        .@"fn" = tp(f),
        .base = tp(try c.tensor(7)),
        .scale = tp(try c.tensor(8)),
        .part = tp(try c.tensor(9)),
        .c = tp(try c.tensor(10)),
        .tap = tp(tap),
        .ts = stride0(tap),
        .nw = nw.ptr,
        .nwbf16 = @intFromBool(nw.dtype == .bf16),
        .out = tp(try c.tensor(13)),
        .opre = tp(try c.tensor(14)),
        .opost = tp(try c.tensor(15)),
        .ocomb = tp(try c.tensor(16)),
        .cnt = tp(try c.tensor(17)),
        .R = @intCast(try c.int(18)),
        .eps = @floatCast(try c.float(20)),
        .hc_eps = @floatCast(try c.float(21)),
        .post_alpha = @floatCast(try c.float(22)),
        .iters = @intCast(try c.int(23)),
    };
    // R1: spin < 0 (or no spin: a capture before R1) is today's finish; defer only with the tail
    var b = a;
    b.spin = if (try c.opt(24)) |v| switch (v) {
        .int => |n| n,
        else => return error.WrongArgument,
    } else -1;
    b.@"defer" = @intFromBool(b.spin >= 0 and (if (try c.opt(25)) |v| switch (v) {
        .int => |n| n != 0,
        .boolean => |t| t,
        else => return error.WrongArgument,
    } else false));
    try o.mhcBoundary(b, try c.u(19), numel(f) != 0 and f.dtype == .f32);
    return true;
}

// tf_dsv41_mhc_cuda_v1.coef (R1: dsv41_mhc_coef_cuda): a deferred boundary's coefficients
pub fn mhcCoef(o: ops.Ops, args: []const Arg, kwargs: []const Kw) !bool {
    const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "part", "base", "scale", "opre", "opost", "ocomb", "R", "eps", "hc_eps", "post_alpha", "iters" } };
    const a: ops.mhc.Args = .{
        .part = try c.ptr(0),
        .base = try c.ptr(1),
        .scale = try c.ptr(2),
        .opre = try c.ptr(3),
        .opost = try c.ptr(4),
        .ocomb = try c.ptr(5),
        .R = @intCast(try c.int(6)),
        .eps = @floatCast(try c.float(7)),
        .hc_eps = @floatCast(try c.float(8)),
        .post_alpha = @floatCast(try c.float(9)),
        .iters = @intCast(try c.int(10)),
    };
    try o.mhcCoef(a);
    return true;
}

// tf_dsv41_mhc_pf_v1.run (mhc_pf.cpp / dsv41_mhc_pf_run_cuda)
pub fn mhcPfRun(o: ops.Ops, args: []const Arg, kwargs: []const Kw) !bool {
    const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "xout", "g", "post", "comb", "pre", "fn", "part", "c", "tap", "R", "mode" } };
    const x = try c.tensor(0);
    const g = try c.tensor(2);
    const f = try c.tensor(6);
    const tap = try c.tensor(9);
    const a: ops.mhc_pf.Args = .{
        .x = tp(x),
        .xs = x.stride[0],
        .xout = tp(try c.tensor(1)),
        .g = tp(g),
        .gr = stride0(g),
        .world = if (numel(g) != 0) @intCast(try dim(g, 0)) else 1,
        .gbf16 = @intFromBool(g.dtype == .bf16),
        .post = tp(try c.tensor(3)),
        .comb = tp(try c.tensor(4)),
        .pre = tp(try c.tensor(5)),
        .@"fn" = f.ptr,
        .part = tp(try c.tensor(7)),
        .c = tp(try c.tensor(8)),
        .tap = tp(tap),
        .ts = stride0(tap),
        .R = @intCast(try c.int(10)),
    };
    try o.mhcSite(a, try c.u(11), f.dtype == .f32, numel(tap) > 0);
    return true;
}

/// route's `prune` (R1c): a list of 7 floats, absent or empty = off.
fn pruneArg(c: Call, i: usize) !ops.rg.Prune {
    const l = (try c.opt(i)) orelse return .{};
    const items = switch (l) {
        .list => |x| x,
        else => return error.WrongArgument,
    };
    var v: [7]f64 = undefined;
    if (items.len != 0 and items.len != 7) return error.WrongArgument;
    for (items, 0..) |it, j| v[j] = switch (it) {
        .float => |f| f,
        .int => |n| @floatFromInt(n),
        else => return error.WrongArgument,
    };
    return ops.rg.pruneOf(v[0..items.len]);
}

// tf_dsv41_router_gemv_v2.route (router_gemv.cpp / dsv41_rg_route_cuda)
pub fn route(o: ops.Ops, args: []const Arg, kwargs: []const Kw) !bool {
    const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "X", "W", "B", "pick", "wts", "LG", "cnt", "K", "slots", "scale", "rt", "nw", "ew", "select", "kit", "gid", "gcnt", "gmem", "GE", "group", "prune" } };
    const x = try c.tensor(0);
    const w = try c.tensor(1);
    const lg = try c.tensor(5);
    const cnt = try c.tensor(6);
    const group = try c.boolean(19, false);
    const gmem = try c.tensor(17);
    if (try dim(x, 1) != ops.rg.D or try dim(w, 1) != ops.rg.D) return error.WrongArgument;
    const a: ops.rg.Args = .{
        .x = x.ptr,
        .xs = x.stride[0],
        .w = w.ptr,
        .bias = try c.ptr(2),
        .pick = try c.ptr(3),
        .wts = try c.ptr(4),
        .lg = lg.ptr,
        .cnt = cnt.ptr,
        .R = @intCast(try dim(x, 0)),
        .E = @intCast(try dim(w, 0)),
        .K = @intCast(try c.int(7)),
        .slots = @intCast(try c.int(8)),
        .select = @intFromBool(try c.boolean(13, false)),
        .scale = @floatCast(try c.float(9)),
        .kit = @intFromBool(try c.boolean(14, false)),
        .group = @intFromBool(group),
        .gid = if (group) try c.ptr(15) else 0,
        .gcnt = if (group) try c.ptr(16) else 0,
        .gmem = if (group) gmem.ptr else 0,
        .GE = @intCast(try c.int(18)),
        .maxm = if (group) @intCast(try dim(gmem, 1)) else 0,
        .prune = try pruneArg(c, 20),
    };
    try o.route(a, try c.u(10), try c.u(11), try c.u(12), @intCast(numel(lg)), @intCast(numel(cnt)));
    return true;
}

// tf_dsv41_pfdense_v1 (pfdense.cpp)
pub fn pfdenseExt(o: ops.Ops, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (is(func, "gemm")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "xh", "T", "stride_k", "stride_nb", "svh", "bias", "out", "K2", "lanes", "cfg", "group" } };
        const xh = try c.tensor(0);
        const T = try c.tensor(1);
        const out = try c.tensor(6);
        const ot: ops.pfd.OutType = switch (out.dtype) {
            .f32 => .f32,
            .bf16 => .bf16,
            .f16 => .f16,
            else => return error.WrongArgument,
        };
        const a: ops.pfd.Args = .{ .xh = xh.ptr, .T = T.ptr, .stride_k = try c.int(2), .stride_nb = try c.int(3), .svh = try c.ptr(4), .bias = try c.ptrOrNull(5), .out = out.ptr, .o_stride = out.stride[0], .M = @intCast(try dim(xh, 0)), .K = @intCast(try dim(xh, 1)), .N = @intCast(try dim(out, 1)), .group = @intCast(try c.int(10)), .out_type = @backingInt(ot) };
        try o.pfdGemm(a, try c.u(7), try c.boolean(8, false), try c.usz(9), @intCast(numel(T)));
        return true;
    }
    if (is(func, "dequant")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "T", "W", "stride_k", "stride_nb", "K2", "lanes" } };
        const w = try c.tensor(1);
        try o.pfdDequant(try c.ptr(0), try c.int(2), try c.int(3), w.ptr, try dim(w, 0), try dim(w, 1), try c.u(4), try c.boolean(5, false));
        return true;
    }
    return false;
}

// tf_dsv41_l2pace_v1.paced / tf_dsv41_l2pf_v1.segments: L2 prefetch only (no outputs; replayed for timing parity)
pub fn paced(o: ops.Ops, args: []const Arg, kwargs: []const Kw) !bool {
    const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "table", "ctas", "chunk", "ns_per_piece", "delay_ns" } };
    const t = try c.tensor(0);
    const n = try dim(t, 0);
    if (n == 0) return true;
    try o.l2Paced(t.ptr, n, try c.usz(1), try c.u(2), @intCast(try c.int(3)), @intCast(try c.int(4)));
    return true;
}

pub fn segments(o: ops.Ops, args: []const Arg, kwargs: []const Kw) !bool {
    const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "table", "grid", "threads", "chunk", "bulk" } };
    const t = try c.tensor(0);
    const n = try dim(t, 0);
    if (n == 0) return true;
    try o.l2Segments(t.ptr, n, try c.usz(1), try c.usz(2), try c.u(3), try c.boolean(4, false));
    return true;
}

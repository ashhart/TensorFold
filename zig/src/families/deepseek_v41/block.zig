//! A decode window as calls: every launch with its arguments and role-named tensors ("s." / "w." / "L." lifetimes).

const std = @import("std");
const dk = @import("dsv41_kernels");
const calls = @import("block_calls.zig");
const config = @import("config.zig");
const knob_types = @import("knob_types.zig");
const moe = @import("block_moe.zig");
const block_prefetch = @import("block_prefetch.zig");
const block_attn = @import("block_attn.zig");
const block_index = @import("block_index.zig");
const Config = config.Config;
const Arg = calls.Arg;
const Named = calls.Named;
const Dt = calls.Dt;
const dense = dk.dense;

/// The widths the loader picked: K2 of each dense group and each routed layer's expert K2 range (gate/up, down).
pub const Widths = struct {
    dense: std.StringHashMapUnmanaged(u32) = .empty,
    experts: std.AutoHashMapUnmanaged(u32, [2][2]u32) = .empty,
    /// each routed layer's expert K2s present (bit k2 set), gate/up and down: x3gm runs one launch a width present
    gm: std.AutoHashMapUnmanaged(u32, [2]u32) = .empty,

    pub fn k2(w: *const Widths, name: []const u8) !u32 {
        return w.dense.get(name) orelse error.NoWidth;
    }
};

/// Run-time sizes fixed at load (scratch and pools), and the L2 prefetch settings.
pub const Options = struct {
    /// TF_DSV41_IMAGES=native: this prefill segment holds image positions, so Engram fuses with a keep vector
    image_keep: bool = false,
    /// TF_DSV41_BIAS_VL: the segment's image rows (MoE call apart, gate.bias_vl); 0: none; < 0: the buffer plan's
    image_rows: i64 = 0,
    /// TP world and local rank; expert and vocabulary slices keep whole 128-column blocks.
    world: u32 = 2,
    rank: u32 = 0,
    /// KV positions a slot holds (pool rows = limit / ratio + 1)
    limit: i64 = 4096,
    /// rope table rows (cos / sin pairs of 32)
    rope_rows: i64 = 6144,
    /// rows the routed experts' scratch holds: the backbone's and a DSpark block's
    expert_rows: i64 = 1024,
    dspark_rows: i64 = 64,
    /// TF_DSV41_BRANCHES: the attention sublayer's indexer / compressor chains tagged for the side stream
    branches: bool = false,
    branch_rows: i64 = 64,
    /// TF_DSV41_MHC_DEFER (with `r1`): its coefficient launch on the deferred stream, joined by the next mHC call
    mhc_defer: bool = false,
    /// TF_DSV41_MHC_DEFER_AT=exchange: deferred launches held to the next exchange; _ROWS: only up to that many rows
    mhc_defer_hold: bool = false,
    /// TF_DSV41_COEF_LATE (with MHC_DEFER): a deferred coef launch moves to just before its sublayer's exchange
    coef_late: bool = false,
    mhc_defer_hold_rows: i64 = 0,
    /// TF_DSV41_MHC_PFDEC (0 = off): mixing sites of windows of at least this many rows on mhc_pf (block_wide.zig)
    mhc_pf_rows: i64 = 0,
    /// TF_DSV41_L2PF_MB (bytes), _CHUNK_KB, _PACE_GBPS, _PACE_CTAS; the GPU's SMs (a segments launch's grid bound)
    l2pf_budget: u64 = 12 << 20,
    l2pf_chunk: i64 = 32768,
    l2pf_pace_gbps: f64 = 150,
    l2pf_pace_ctas: i64 = 2,
    sms: i64 = 188,
    /// TF_DSV41_EXPERT_TOPP (lossy). null: no pruning; the router's tail rounds kit weights and groups experts itself
    expert_topp: ?f64 = null,
    /// after each block, its streams with the last post applied out of place, to glue step "trace" (never served)
    trace: bool = false,
    /// DSpark's taps: boundaries entering target layers write the streams' mean into "w.taps" bf16 [n, 3 x hidden]
    taps: bool = false,
    /// rows a prefill segment holds at most (forward.pchunk): sizes the SWA staging ring (block_prefill.zig)
    prefill_rows: i64 = 2048,
    /// TF_DSV41_PF_4K: prefill segments up to 4,096 rows, router in 2,048-row launches; off: 2,048 (longer refused)
    pf4k: bool = false,
    /// TF_DSV41_PF_OVERLAP: a prefill layer's two exchanges in this many row pieces on the side stream (branches, TP=2)
    pf_overlap: i64 = 0,
    /// TF_DSV41_PF_OVERLAP_SITE (with pf_overlap): all exchange pieces on the side stream, the next site in row pieces
    pf_overlap_site: bool = false,
    /// TF_DSV41_MHC_SITE_ROWS (0 off; 256-4,096, multiple of 16): a prefill boundary site in row pieces on main
    mhc_site_rows: i64 = 0,
    /// TF_DSV41_PF_TBO: CED encoder segments two at a time, the second a layer behind on the side stream (`branches`)
    pf_tbo: bool = false,
    /// TF_DSV41_STREAM_RB (0 off; 1 / 2 / 4): the long-context stream top-k as the row-blocked twin `_stream_rb<N>`
    stream_rb: i64 = 0,
    /// TF_DSV41_INDEX_BOUND: decode index scores and top-k as row-bounded twins that stop at each row's visible end
    index_bound: bool = false,
    /// decode knob set (same bits): ATTN_SPLIT, ATTN_ROPE, MHC_TAIL, MHC_DEFER, RG_PRUNE, PRUNE_KIT, MOE_BF16, RMS_FOLD
    r1: bool = false,
    /// TF_DSV41_ROUTER_GROUP_ROT: wide grouping and expert input rotation in one launch (default off).
    router_group_rot: bool = false,
    /// TF_DSV41_X3LD_EPI: decode x3ld + epilogues as two fused launches (ticket words "s.ex.epi" / "s.dx.epi")
    x3ld_epi: bool = false,
    /// KV-only RMS feeding SWA store; separately gated, default off.
    kv_norm_store: bool = false,
    /// `r1` off but a twin's bindings that always take its trailing arguments, at their off values (same launches)
    r1_sig: bool = false,
    /// MHC_TAIL_SPIN (SM cycles a CTA waits for the last arrival)
    mhc_spin: i64 = 8000,
    /// the KV families in the paged pool (kv.DevicePool) instead of the slot's contiguous buffers; null: contiguous
    pool: ?Pool = null,
    /// TF_DSV41_PF_DENSE=fused: prefill dense EXL3 GEMMs on the pfdense kernel; null: W_q unpack + `_gemm` (same bits)
    pfd: ?*const knob_types.PfDense = null,
    /// TF_DSV41_GM_V2 (gm2pf.zig): prefill's routed experts on x3gm v2; off: v1. The same bits either way.
    gm_v2: knob_types.GmMode = .off,
    /// TF_DSV41_GM_GU2 (with gm_v2 `one`): a 4,096-row segment's gate / up at the 128-member tile GU2
    gm_gu2: bool = false,
    /// TF_DSV41_INDEX_BUDGET_MIB (bytes): indexer scores + sort keys a launch; past it, row blocks (same bits)
    index_budget: i64 = 256 << 20,
    /// a row-mode program (batch.zig, rowmode.zig): row blocks score the context bucket's keys; no streamed top-k
    rows: bool = false,

    /// TF_DSV41_MHC_DEFER_AT(_ROWS): a program of `rows` rows issues its deferred launches at the next exchange.
    pub fn holdsDeferred(o: Options, rows: i64) bool {
        return o.mhc_defer_hold and (o.mhc_defer_hold_rows == 0 or rows <= o.mhc_defer_hold_rows);
    }
};

/// The paged pool as a window's calls see it: comp rows, index keys, page table and split table (TF_DSV41_KV_SPLIT).
pub const Pool = struct {
    /// pages each family tensor holds (comp: split's local pages + null + discard)
    comp_pages: i64,
    ik_pages: i64,
    /// pages a slot's table holds
    pts: i64,
    split: bool = false,
    /// TF_DSV41_KV_SPLIT_EARLY: an index layer's exchange right after its selection (side stream), beside the q group
    early: bool = false,
    /// split's union cap (TF_DSV41_KV_SPLIT_UNION_MIB): the prefill receive buffer's bytes
    union_cap: i64 = 128 << 20,
    pub const page: i64 = 256;
    pub const row_bytes: i64 = 584;
    pub const ik_bytes: i64 = 132;

    /// log2 of a family's rows a page
    pub fn shift(ratio: i64) i64 {
        return std.math.log2_int(u64, @intCast(@divExact(page, ratio)));
    }
};

pub const Emitter = struct {
    a: std.mem.Allocator,
    cfg: *const Config,
    w: *const Widths,
    o: Options,
    /// rows this window and the position of its first
    n: i64,
    start: i64,
    out: std.ArrayList(calls.Call) = .empty,
    /// which of the two stream buffers holds the streams, and which coefficient set the last boundary wrote
    cur: u1 = 0,
    coef: u1 = 0,
    begin: calls.Begin = .window,
    /// the current layer's compression ratio (picks its rope table)
    layer_ratio: u32 = 0,
    /// TF_DSV41_BRANCHES: the stream the next calls go to, and the marks the next call carries
    br_side: bool = false,
    br_fork: bool = false,
    br_join: bool = false,
    br_mark: u8 = 0,
    br_wait: u8 = 0,
    /// TF_DSV41_MHC_DEFER: the next call is the coefficient launch / joins it; a coefficient launch not joined yet
    df_side: bool = false,
    df_join: bool = false,
    df_pending: bool = false,
    /// TF_DSV41_COEF_LATE: the index of a deferred coefficient launch not joined yet (moved to the next exchange)
    df_at: ?usize = null,
    /// a multi-segment prefill: the current segment's first row in the run's per-row roles; 0 elsewhere
    row0: i64 = 0,
    /// TF_DSV41_PF_TBO (block_prefill.emitTbo): the call indexes where each layer's attention part and MoE part start
    marks: ?*std.ArrayList(usize) = null,
    /// its first row in the run's narrow per-row roles, a multiple of 4 (16-byte aligned slices); 0 elsewhere
    cnt0: i64 = 0,

    pub fn fmt(e: *Emitter, comptime f: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(e.a, f, args);
    }

    /// A contiguous tensor of `role`.
    pub fn t(e: *Emitter, role: calls.Role, dt: Dt, shape: []const i64) !Arg {
        try @import("block_admission.zig").tensor(shape, @import("buffers.zig").dtSize(dt));
        const s = try e.a.dupe(i64, shape);
        return .{ .t = .{ .role = role, .dt = dt, .shape = s, .stride = try calls.contiguous(e.a, s) } };
    }

    /// A view of `role`: its own strides and a byte offset.
    pub fn view(e: *Emitter, role: calls.Role, dt: Dt, shape: []const i64, stride: []const i64, offset: i64) !Arg {
        try @import("block_admission.zig").view(shape, stride, offset, @import("buffers.zig").dtSize(dt));
        return .{ .t = .{ .role = role, .dt = dt, .shape = try e.a.dupe(i64, shape), .stride = try e.a.dupe(i64, stride), .offset = offset } };
    }

    pub fn buf(e: *Emitter, comptime f: []const u8, args: anytype, dt: Dt, shape: []const i64) !Arg {
        return e.t(.{ .buf = try e.fmt(f, args) }, dt, shape);
    }

    pub fn weight(e: *Emitter, comptime f: []const u8, args: anytype, dt: Dt, shape: []const i64) !Arg {
        return e.t(.{ .weight = try e.fmt(f, args) }, dt, shape);
    }

    pub fn empty(e: *Emitter, dt: Dt) !Arg {
        return e.t(.empty, dt, &.{0});
    }

    pub fn ints(e: *Emitter, xs: []const i64) !Arg {
        const l = try e.a.alloc(Arg, xs.len);
        for (xs, l) |x, *y| y.* = .{ .i = x };
        return .{ .list = l };
    }

    fn push(e: *Emitter, c: calls.Call) !void {
        var x = c;
        x.begin = e.begin;
        e.begin = .none;
        x.side = e.br_side;
        x.fork = e.br_fork;
        x.join = e.br_join;
        x.mark = e.br_mark;
        x.wait = e.br_wait;
        e.br_fork = false;
        e.br_join = false;
        e.br_mark = 0;
        e.br_wait = 0;
        x.defer_side = e.df_side;
        x.defer_join = e.df_join;
        e.df_side = false;
        e.df_join = false;
        if (e.o.coef_late and e.marks == null) try e.lateCoef(&x);
        try e.out.append(e.a, x);
    }

    /// TF_DSV41_COEF_LATE: a deferred coef launch moves to just before the next exchange of its fork region; same data.
    fn lateCoef(e: *Emitter, x: *const calls.Call) !void {
        if (x.defer_join) e.df_at = null;
        if (x.defer_side) {
            e.df_at = e.out.items.len;
            return;
        }
        const at = e.df_at orelse return;
        if (!(x.glue and std.mem.startsWith(u8, x.name, "glue.exchange"))) return;
        e.df_at = null;
        const c = e.out.orderedRemove(at);
        std.debug.assert(c.defer_side and c.begin == .none);
        try e.out.append(e.a, c);
    }

    /// Appends positional arguments to the last emitted call (`r1`'s trailing arguments).
    pub fn appendLast(e: *Emitter, args: []const Arg) !void {
        const last = &e.out.items[e.out.items.len - 1];
        const more = try e.a.alloc(calls.Named, last.args.len + args.len);
        @memcpy(more[0..last.args.len], last.args);
        for (more[last.args.len..], args) |*m, x| m.* = .{ .arg = x };
        last.args = more;
    }

    pub fn ext(e: *Emitter, name: []const u8, args: []const Arg) !void {
        const l = try e.a.alloc(Named, args.len);
        for (args, l) |x, *y| y.* = .{ .arg = x };
        try e.push(.{ .triton = false, .name = name, .args = l });
    }

    /// A glue step ("glue.<name>"): the roles it reads / writes and its scalars, for the forward to run.
    pub fn glue(e: *Emitter, name: []const u8, args: []const Arg) !void {
        const l = try e.a.alloc(Named, args.len);
        for (args, l) |x, *y| y.* = .{ .arg = x };
        try e.push(.{ .triton = false, .glue = true, .name = try e.fmt("glue.{s}", .{name}), .args = l });
    }

    pub fn triton(e: *Emitter, name: []const u8, grid: [3]i64, args: []const Named) !void {
        try e.push(.{ .triton = true, .name = name, .grid = grid, .args = try e.a.dupe(Named, args) });
    }

    /// A config value's float as the JSON number (f64), not the f32 we keep.
    pub fn dec(x: f32) f64 {
        var b: [64]u8 = undefined;
        const s = std.fmt.bufPrint(&b, "{e}", .{x}) catch return x;
        return std.fmt.parseFloat(f64, s) catch x;
    }

    /// Positions after this window: the last row's + 1.
    pub fn end(e: *const Emitter) i64 {
        return e.start + e.n;
    }

    pub fn streams(e: *Emitter) !Arg {
        return e.buf("w.x{d}", .{e.cur}, .bf16, &.{ e.n, 4 * @as(i64, e.cfg.hidden) });
    }

    pub fn spare(e: *Emitter) !Arg {
        return e.buf("w.x{d}", .{1 - e.cur}, .bf16, &.{ e.n, 4 * @as(i64, e.cfg.hidden) });
    }

    // dense EXL3 projections ---------------------------------------------------------------------------------------

    /// A trellis's run-time words (int32 [N/128, K/16, 8, 4 K2]): strips "s.<w>.T" or dense3 lanes "s.<w>.lanes".
    pub fn strips(e: *Emitter, w: []const u8, k: i64, n: i64) !Arg {
        return e.words(w, k, n, "T");
    }

    fn words(e: *Emitter, w: []const u8, k: i64, n: i64, form: []const u8) !Arg {
        const k2: i64 = try e.w.k2(w);
        return e.t(.{ .buf = try e.fmt("s.{s}.{s}", .{ w, form }) }, .i32, &.{ @divExact(n, 128), @divExact(k, 16), 8, 4 * k2 });
    }

    /// Bytes of a group's trellis (= its strips).
    pub fn trellisBytes(e: *Emitter, w: []const u8, k: i64, n: i64) !u64 {
        const k2: u64 = try e.w.k2(w);
        return @as(u64, @intCast(@divExact(k, 16) * @divExact(n, 16))) * 8 * k2 * 2;
    }

    /// One projection of a fused group: weight prefix, input, output (a whole buffer; `col` its first column), K x N.
    pub const Proj = struct { w: []const u8, x: Arg, y: Arg, col: usize = 0, k: usize, n: usize };

    /// `r1`'s RMS_FOLD: rot_in norms its rows itself (rms(x, w, eps), rmsnorm's order) before the rotation.
    pub const Norm = struct { w: Arg, eps: f64 };

    /// x3seg rot_in, then dense3 a width: the group's projections in one rot_in launch, one linear launch a K2.
    pub fn fused(e: *Emitter, tag: []const u8, ps: []const Proj) !void {
        return e.fusedNorm(tag, ps, null);
    }

    pub fn fusedNorm(e: *Emitter, tag: []const u8, ps: []const Proj, norm: ?Norm) !void {
        const m: usize = @intCast(e.n);
        const segs = try e.a.alloc(dense.Seg, ps.len);
        for (ps, segs, 0..) |p, *s, i| {
            const pl = dense.plan(p.k, p.n);
            s.* = .{ .k = p.k, .n = p.n, .k2 = try e.w.k2(p.w), .sk = pl[0], .wk = pl[1], .out = i, .col = p.col };
        }
        const lay = try dense.layout(segs, m);
        const xs = try e.a.alloc(Arg, ps.len);
        const suh = try e.a.alloc(Arg, ps.len);
        const offs = try e.a.alloc(i64, ps.len);
        for (ps, xs, suh, offs, 0..) |p, *x, *s, *o, i| {
            x.* = p.x;
            s.* = try e.weight("{s}.suh", .{p.w}, .f16, &.{@intCast(p.k)});
            o.* = @intCast(lay.xh_off[i]);
        }
        const xh = try e.buf("L.{s}.xh", .{tag}, .f16, &.{@intCast(lay.xh_total)});
        if (norm) |nm| {
            const ws = try e.a.alloc(Arg, ps.len);
            @memset(ws, nm.w);
            try e.ext("tf_dsv41_x3seg_v3.rot_in", &.{ .{ .list = xs }, .{ .list = suh }, xh, try e.ints(offs), .{ .b = false }, .{ .list = ws }, .{ .f = nm.eps } });
        } else try e.ext("tf_dsv41_x3seg_v3.rot_in", &.{ .{ .list = xs }, .{ .list = suh }, xh, try e.ints(offs), .{ .b = false } });
        const z: Arg = if (lay.z_total > 0) try e.buf("L.{s}.z", .{tag}, .f32, &.{@intCast(lay.z_total)}) else .none;
        const counters = try e.buf("L.{s}.cnt", .{tag}, .i32, &.{@intCast(lay.c_total)});
        for (lay.launches[0..lay.n_launches]) |l| {
            const T = try e.a.alloc(Arg, l.count);
            const svh = try e.a.alloc(Arg, l.count);
            const ys = try e.a.alloc(Arg, l.count);
            var meta: std.ArrayList(i64) = .empty;
            for (l.segs[0..l.count], 0..) |si, j| {
                const s = segs[si];
                const p = ps[si];
                T[j] = try e.words(p.w, @intCast(p.k), @intCast(p.n), "lanes");
                svh[j] = try e.weight("{s}.svh", .{p.w}, .f16, &.{@intCast(p.n)});
                ys[j] = p.y;
                const st = dense.strides(s.k, s.k2);
                try meta.appendSlice(e.a, &.{ @intCast(s.k), @intCast(s.n), @intCast(s.sk), @intCast(s.wk), st[0], st[1], @intCast(lay.xh_off[si]), @intCast(lay.z_off[si]), @intCast(lay.c_off[si]), @intCast(s.col), @intCast(l.p0[j]) });
            }
            try e.ext("tf_dsv41_dense3_v1.linear", &.{
                xh,             .{ .list = T },           .{ .list = svh },                  .{ .list = ys },
                z,              counters,                 try e.ints(meta.items),            .{ .i = @intCast(l.programs) },
                .{ .i = l.k2 }, .{ .i = @intCast(l.wk) }, .{ .i = if (l.wk == 8) 1 else 3 }, .{ .b = false },
            });
        }
    }

    /// Upstream's rot_in + linear: one projection, its trellis as strips unless one 128-column block.
    pub fn upstream(e: *Emitter, w: []const u8, x: Arg, k: i64, n: i64, y: Arg) !void {
        const k2: i64 = try e.w.k2(w);
        const xh = try e.buf("L.{s}.xh", .{w}, .f16, &.{ e.n, k });
        try e.ext("tensorfold_exl3_linear_v4.rot_in", &.{ x, try e.weight("{s}.suh", .{w}, .f16, &.{k}), xh });
        const pl = dense.plan(@intCast(k), @intCast(n));
        const st = dense.strides(@intCast(k), @intCast(k2));
        const T = if (n == 128)
            try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ 1, @divExact(k, 16), 8, 4 * k2 }, &.{ 32 * k2, 32 * k2, 4 * k2, 1 }, 0)
        else
            try e.strips(w, k, n);
        if (pl[0] != 1) return error.Unsupported; // no upstream linear on the decode path splits K
        try e.ext("tensorfold_exl3_linear_v4.linear", &.{
            xh,                                                            T,            .{ .i = st[0] }, .{ .i = st[1] },           try e.weight("{s}.svh", .{w}, .f16, &.{n}), .none, y, .none,
            try e.buf("L.{s}.cnt", .{w}, .i32, &.{8 * @divExact(n, 128)}), .{ .i = k2 }, .{ .i = 2 },     .{ .i = @intCast(pl[0]) }, .{ .i = @intCast(pl[1]) },
        });
    }

    // L2 prefetch (TF_DSV41_L2PF): table rows filled by l2pf.zig at boot, issued on the runner's side stream ------

    /// What a site prefetches, in read order: a weight, a projection's trellis words, or an expert entry's trellis.
    pub const L2Src = struct { kind: enum { weight, words, expert }, name: []const u8, index: u32 = 0, bytes: u64 };
    pub const L2Site = enum { o, f, x };

    pub const linSrcs = block_prefetch.linSrcs;
    pub const l2pfSources = block_prefetch.l2pfSources;
    pub const srcSizes = block_prefetch.srcSizes;
    pub const take = block_prefetch.take;
    pub const prefetchO = block_prefetch.prefetchO;
    pub const paced = block_prefetch.paced;
    pub const prefetchF = block_prefetch.prefetchF;
    pub const prefetchX = block_prefetch.prefetchX;

    const Group = struct { ws: []const []const u8, ks: []const i64, ns: []const i64 };

    /// The x group (fused_proj): wq_a, wkv, then a KV source's compressor projections.
    pub fn xGroup(e: *Emitter, L: u32) !Group {
        const D: i64 = e.cfg.hidden;
        const ratio = e.cfg.compressRatio(L);
        const comp: usize = if (e.cfg.isKvSource(L)) (if (ratio == 2) 2 else 1) else 0;
        const ws = try e.a.alloc([]const u8, 2 + comp);
        const ns = try e.a.alloc(i64, 2 + comp);
        const names = [_][]const u8{ "wq_a", "wkv", "comp_wkv", "comp_wgate" };
        for (ws, ns, 0..) |*w, *n, i| {
            w.* = try e.fmt("L{d}.attn.{s}", .{ L, names[i] });
            n.* = if (i == 0) e.cfg.q_lora else e.cfg.head_dim;
        }
        const ks = try e.a.alloc(i64, ws.len);
        @memset(ks, D);
        return .{ .ws = ws, .ks = ks, .ns = ns };
    }

    /// The q group: wq_b, and an indexer's ix_wq_b.
    pub fn qGroup(e: *Emitter, L: u32) !Group {
        const ix = switch (e.cfg.mode(L)) {
            .full, .reindex => true,
            else => false,
        };
        const ws = try e.a.alloc([]const u8, if (ix) 2 else 1);
        const ns = try e.a.alloc(i64, ws.len);
        ws[0] = try e.fmt("L{d}.attn.wq_b", .{L});
        ns[0] = @as(i64, e.cfg.heads / e.o.world) * e.cfg.head_dim;
        if (ix) {
            ws[1] = try e.fmt("L{d}.attn.ix_wq_b", .{L});
            ns[1] = @as(i64, e.cfg.index_heads) * e.cfg.index_dim;
        }
        const ks = try e.a.alloc(i64, ws.len);
        @memset(ks, e.cfg.q_lora);
        return .{ .ws = ws, .ks = ks, .ns = ns };
    }

    pub const rope = block_attn.rope;
    pub const ropeTable = block_attn.ropeTable;
    pub const pos = block_attn.pos;
    pub const kvStore = block_attn.kvStore;

    /// A family's paging: the slot's table, rows-a-page shift and row strides (pool mode); none for ring / contiguous.
    pub const Paging = struct { pt: Arg, psh: i64, vs: i64, ss: i64 };

    pub const paging = block_attn.paging;
    pub const pool = block_attn.pool;
    pub const indexKeys = block_attn.indexKeys;

    /// Split KV: source `src`'s selected rows from their owners (dense up to 16 rows, else a union), read unpaged.
    pub const Received = struct { cv: Arg, cs: Arg, tok: Arg };

    pub const received = block_attn.received;
    pub const exchange = block_attn.exchange;
    pub const selection = block_index.selection;
    pub const ticketLen = block_index.ticketLen;
    pub const attnSplit = block_index.attnSplit;
    pub const rms = block_attn.rms;
    pub const kvNormStoreOn = block_attn.kvNormStoreOn;
    pub const normStore = block_attn.normStore;
    pub const swaStore = block_attn.swaStore;
    pub const branching = block_attn.branching;
    pub const fork = block_attn.fork;
    pub const onMain = block_attn.onMain;
    pub const join = block_attn.join;
    pub const attention = block_attn.attention;
    pub const index = block_index.index;
    pub const scoreRows = block_index.scoreRows;
    pub const blocked = block_index.blocked;
    pub const blockPos = block_index.blockPos;
    pub const topk = block_index.topk;
    pub const topkOf = block_index.topkOf;

    // the window -------------------------------------------------------------------------------------------------

    /// Every launch of a decode window of `n` rows from `start` over `layers` (ascending), and the head when `head`.
    pub fn window(e: *Emitter, layers: []const u32, head: bool) !void {
        // the window's positions and streams (this rank's embedding rows, their all-sum: the first exchange)
        try e.glue("positions", &.{ try e.pos(), try e.buf("w.pos64", .{}, .i64, &.{e.n}), try e.buf("w.lo", .{}, .i32, &.{e.n}) });
        try e.glue("embed", &.{ try e.streams(), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, e.n, e.cfg.hidden }) });
        for (layers, 0..) |L, i| {
            if (i > 0) e.begin = .layer;
            e.layer_ratio = e.cfg.compressRatio(L);
            if (e.cfg.isEngram(L)) {
                // an Engram block after the first: the last exchange's post alone, Engram, then a site2 collapse
                if (i > 0) try moe.postOnly(e, null);
                try moe.engram(e, L);
                try moe.mhc(e, L, "hc_attn", if (i == 0) .site1 else .site2);
            } else try moe.mhc(e, L, "hc_attn", if (i == 0) .site1 else .boundary);
            try e.attention(L);
            try e.prefetchF(L);
            const D: i64 = e.cfg.hidden;
            try e.glue("exchange", &.{ try e.buf("L.part", .{}, .bf16, &.{ e.n, D }), try e.buf("w.recv.attn", .{}, .bf16, &.{ e.o.world, e.n, D }) });
            try moe.mhc(e, L, "hc_ffn", .boundary);
            try moe.moe(e, L);
            if (i + 1 < layers.len) try e.prefetchX(layers[i + 1]);
            // `r1` (MOE_BF16): down_combine stores the partial as bf16, the exchange takes it as is
            if (e.o.r1)
                try e.glue("exchange", &.{ try e.buf("L.moe", .{}, .bf16, &.{ e.n, D }), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, e.n, D }) })
            else
                try e.glue("exchange_f32", &.{ try e.buf("L.moe", .{}, .f32, &.{ e.n, D }), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, e.n, D }) });
            if (e.o.trace) {
                const y = try e.buf("w.trace", .{}, .bf16, &.{ e.n, 4 * D });
                try moe.postOnly(e, y);
                try e.glue("trace", &.{ y, .{ .i = L } });
            }
        }
        if (head) try moe.finish(e);
    }
};

/// The window's calls (arena-owned by `a`).
pub fn emit(a: std.mem.Allocator, cfg: *const Config, w: *const Widths, o: Options, layers: []const u32, n: i64, start: i64, head: bool) ![]calls.Call {
    try cfg.validate();
    if (o.world == 0 or o.world > 8 or cfg.heads % o.world != 0 or cfg.o_groups % o.world != 0 or
        n < 1 or n > 64 or start < 0 or o.limit < n or start > o.limit - n or
        o.rope_rows < n or start > o.rope_rows - n) return error.InvalidBlock;
    try @import("block_admission.zig").options(cfg, o, layers, n);
    var e: Emitter = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = n, .start = start };
    try e.window(layers, head);
    if (o.holdsDeferred(n)) calls.holdDeferred(e.out.items);
    return e.out.items;
}

/// exl3 expert_loads.k2_range: the instance covering a layer's half-bit widths [lo, hi].
pub fn k2Range(lo: u32, hi: u32) ![2]i64 {
    if (lo < 2 or lo > hi) return error.NoInstance;
    const ranges = [_][2]u32{ .{ 8, 8 }, .{ 2, 10 }, .{ 2, 12 } };
    for (ranges) |r| if (r[0] <= lo and hi <= r[1]) return .{ r[0], r[1] };
    return error.NoInstance;
}

//! Nemotron-H multi-row windows: 1..16 tokens in one pass, rows bit-identical to the one-token path; else commit(c).
const std = @import("std");
const rt = @import("xpu").rt;
const model = @import("xpu_model.zig");

const npf = @import("xpu_pf.zig");
const stop = @import("xpu").stop;

const spv = @import("xpu").kernels.nem_rows;
const spv_attn_pf = @import("xpu").kernels.nem_attn_pfs;

/// Debug hook (nem_fp64): called with the activations a projection class is about to read: xn, yn or att.
pub const Stage = enum { xn, yn, att };
pub var capture: ?*const fn (w: *Win, m: *model.Model, li: usize, st: Stage, n: u32) anyerror!void = null;

/// Row capacity of the next Win created (windows above max_rows are prompt chunks: in place, prefill GEMMs).
pub var default_rows: u32 = 16;
const Buf = rt.Buffer;
pub const max_rows: u32 = 16;
const top_k: u32 = 6;

fn at(b: Buf, off: usize) Buf {
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = b.len - off };
}

var trace: ?bool = null;
var launches: u32 = 0;
/// Per-kernel time of windows (env NEM_PROF: every launch synced and timed, so entries include the round trip).
pub var prof_on: bool = false;
var prof: [48]struct { h: ?*const anyopaque = null, ns: u64 = 0, calls: u64 = 0 } = @splat(.{});

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn printProf(w: *Win, m: *model.Model, windows: usize) void {
    inline for (.{ w.k, m.k }) |ks| inline for (@typeInfo(@TypeOf(ks)).@"struct".field_names) |fname| {
        if (@TypeOf(@field(ks, fname)) == rt.Kernel) {
            const h: ?*const anyopaque = @ptrCast(@field(ks, fname).handle);
            for (prof) |e| if (e.h != null and e.h == h) std.debug.print("  kernel {s:<10} {d:>6} calls {d:>8.1} us avg {d:>8.2} ms a window\n", .{ fname, e.calls, @as(f64, @floatFromInt(e.ns)) / 1e3 / @as(f64, @floatFromInt(e.calls)), @as(f64, @floatFromInt(e.ns)) / 1e6 / @as(f64, @floatFromInt(windows)) });
        }
    };
}

fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    if (trace == null) trace = std.c.getenv("NEM_TRACE") != null;
    if (trace.?) {
        launches += 1;
        std.debug.print("launch {d} groups {d},{d},{d}\n", .{ launches, groups[0], groups[1], groups[2] });
    }
    defer if (trace.?) k.rt.sync() catch {};
    const t0: u64 = if (prof_on) blk: {
        k.rt.sync() catch {};
        break :blk nowNs();
    } else 0;
    defer if (prof_on) {
        k.rt.sync() catch {};
        const dt = nowNs() - t0;
        for (&prof) |*e| {
            if (e.h == null) e.h = @ptrCast(k.handle);
            if (e.h == @as(?*const anyopaque, @ptrCast(k.handle))) {
                e.ns += dt;
                e.calls += 1;
                break;
            }
        }
    };
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) try k.setBuffer(i, v) else if (T == f32) try k.setF32(i, v) else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

const Kernels = struct {
    qmv_bf16: rt.Kernel,
    qmv_bf: rt.Kernel,
    qmv_f32: rt.Kernel,
    conv: rt.Kernel,
    ssm: rt.Kernel,
    gnorm: rt.Kernel,
    groups: rt.Kernel,
    up_g: rt.Kernel,
    down_g: rt.Kernel,
    dup: rt.Kernel,
    ddown: rt.Kernel,
    comb: rt.Kernel,
    a_part: rt.Kernel,
    a_merge: rt.Kernel,
    /// Prompt windows above max_rows run on the matrix engine; NEM_OLD_ATTN=1 keeps the per-row kernels.
    a_pf: ?rt.Kernel = null,
    a_prep: ?rt.Kernel = null,
};

/// Per Mamba layer: the window's saved inputs (replay), the second state buffers.
const MState = struct { proj: Buf, xc: Buf, cstate2: Buf, sstate2: Buf };

pub const Win = struct {
    k: Kernels,
    ids: Buf, // token ids
    host_ids: []u32, // staging of the ids: the upload reads it when it runs, so it is owned here and rewritten only after a sync (a caller's slice may change at once)
    x: Buf,
    xn: Buf,
    delta: Buf,
    y: Buf,
    yn: Buf,
    r_logits: Buf,
    eids: Buf,
    elist: Buf, // [1 + experts]: count, then the chosen experts ascending
    wts: Buf,
    act: Buf,
    ey: Buf,
    sact: Buf,
    sy: Buf,
    q: Buf,
    qb: Buf, // query tiles of the matrix-engine attention
    att: Buf,
    po: Buf,
    pm: Buf,
    pl: Buf,
    logits: Buf,
    am_scratch: Buf,
    am_out: Buf,
    scratch_xc: Buf,
    ms: []MState,
    rcap: u32 = max_rows,
    scalar: bool = false, // NEM_SCALAR: the row kernels of nem_rows.cl for windows of up to 16 rows instead of the matrix-engine GEMMs
    mpf: ?npf.MoePf = null,
    rows: u32 = 0,
    inplace: bool = true,
    head_rows: u32 = 0,
    pos0: u32 = 0,
    cur: usize = 0, // the layer being run (for `capture`)

    pub fn init(gpa: std.mem.Allocator, m: *model.Model) !Win {
        const r = m.r;
        const c = m.cfg;
        var mod = try r.module(spv);
        const rc: u32 = @max(default_rows, max_rows);
        const R: usize = rc;
        const RA: usize = max_rows; // attention row blocks and logits
        const h: usize = c.hidden_size;
        const kq: Kernels = .{
            .qmv_bf16 = try mod.kernel("qmv4_bf16_r", .{ 16, 1, 1 }),
            .qmv_bf = try mod.kernel("qmv4_bf_r", .{ 64, 1, 1 }),
            .qmv_f32 = try mod.kernel("qmv4_f32_r", .{ 64, 1, 1 }),
            .conv = try mod.kernel("conv1d_rows", .{ 64, 1, 1 }),
            .ssm = try mod.kernel("ssm_rows", .{ 64, 1, 1 }),
            .gnorm = try mod.kernel("group_rmsnorm_r", .{ 64, 1, 1 }),
            .groups = try mod.kernel("moe_groups", .{ 1, 1, 1 }),
            .up_g = try mod.kernel("expert_up_relu2_g", .{ 16, 1, 1 }),
            .down_g = try mod.kernel("expert_down_f32_g", .{ 16, 1, 1 }),
            .dup = try mod.kernel("dense_up_relu2_r", .{ 16, 1, 1 }),
            .ddown = try mod.kernel("dense_down_f32_r", .{ 16, 1, 1 }),
            .comb = try mod.kernel("moe_combine_r", .{ 64, 1, 1 }),
            .a_part = try mod.kernel("attn_partial_r", .{ 256, 1, 1 }),
            .a_merge = try mod.kernel("attn_merge_r", .{ 16, 1, 1 }),
            .a_pf = if (rc > max_rows and c.head_dim == 128 and c.num_attention_heads == 32 and c.num_key_value_heads == 2 and std.c.getenv("NEM_OLD_ATTN") == null) blk: {
                var pm = try r.moduleWith(spv_attn_pf, "-cl-intel-256-GRF-per-thread");
                break :blk try pm.kernel("nem_attn_prefill_s", .{ 128, 1, 1 });
            } else null,
            .a_prep = if (rc > max_rows and c.head_dim == 128 and c.num_attention_heads == 32 and c.num_key_value_heads == 2 and std.c.getenv("NEM_OLD_ATTN") == null) blk: {
                var pm = try r.module(spv_attn_pf);
                break :blk try pm.kernel("nem_attn_pfs_prep", .{ 64, 1, 1 });
            } else null,
        };
        const nm: usize = blk: {
            var n: usize = 0;
            for (0..c.num_hidden_layers) |i| n += @intFromBool(c.kind(i) == .mamba);
            break :blk n;
        };
        const ms = try gpa.alloc(MState, nm);
        for (ms) |*s| s.* = .{
            .proj = try r.alloc(R * c.projDim() * 2),
            .xc = try r.alloc(R * c.convDim() * 2),
            .cstate2 = try r.alloc((c.conv_kernel - 1) * c.convDim() * 2),
            .sstate2 = try r.alloc(@as(usize, c.mamba_num_heads) * c.mamba_head_dim * c.ssm_state_size * 4),
        };
        const qd: usize = c.num_attention_heads * c.head_dim;
        const parts: usize = 64;
        return .{
            .k = kq,
            .ids = try r.alloc(R * 4),
            .host_ids = try gpa.alloc(u32, R),
            .x = try r.alloc(R * h * 2),
            .xn = try r.alloc(R * h * 2),
            .delta = try r.alloc(R * h * 2),
            .y = try r.alloc(R * c.xd() * 2),
            .yn = try r.alloc(R * c.xd() * 2),
            .r_logits = try r.alloc(R * c.n_routed_experts * 2),
            .eids = try r.alloc(R * top_k * 4),
            .elist = try r.alloc(260 * 4),
            .wts = try r.alloc(R * top_k * 4),
            .act = try r.alloc(R * top_k * c.moe_intermediate_size * 2),
            .ey = try r.alloc(R * top_k * h * 4),
            .sact = try r.alloc(R * c.moe_shared_expert_intermediate_size * 2),
            .sy = try r.alloc(R * h * 4),
            .q = try r.alloc(R * qd * 2),
            .qb = try r.alloc(R / 8 * 4 * 4 * 1024 * 4 + 4096),
            .att = try r.alloc(R * qd * 2),
            .po = try r.alloc(RA * m.max_chunks * qd * 4),
            .pm = try r.alloc(RA * m.max_chunks * c.num_attention_heads * 4),
            .pl = try r.alloc(RA * m.max_chunks * c.num_attention_heads * 4),
            .logits = try r.alloc(RA * @as(usize, c.vocab_size) * 4),
            .am_scratch = try r.alloc(RA * parts * 8),
            .am_out = try r.alloc(RA * 4),
            .scratch_xc = try r.alloc(R * c.convDim() * 2),
            .ms = ms,
            .rcap = rc,
            .scalar = std.c.getenv("NEM_SCALAR") != null,
            .mpf = try npf.MoePf.init(r, rc, c.n_routed_experts, @max(c.hidden_size, c.moe_intermediate_size)),
        };
    }

    pub fn profClear(w: *Win) !void {
        _ = w;
        prof = @splat(.{});
    }

    /// Back to an empty context: position 0, conv and SSM states zeroed (KV cache read only up to the position). Syncs.
    pub fn reset(w: *Win, m: *model.Model) !void {
        _ = w;
        const c = m.cfg;
        const zeros = try std.heap.page_allocator.alloc(u8, @as(usize, c.mamba_num_heads) * c.mamba_head_dim * c.ssm_state_size * 4);
        defer std.heap.page_allocator.free(zeros);
        @memset(zeros, 0);
        for (m.layers) |ly| switch (ly.mixer) {
            .mamba => |mw| {
                try m.r.upload(mw.cstate, zeros[0 .. (c.conv_kernel - 1) * c.convDim() * 2]);
                try m.r.upload(mw.sstate, zeros);
            },
            else => {},
        };
        try m.r.sync();
        m.pos = 0;
    }

    /// Runs `tokens` (1..16) at m.pos; last head_rows rows get logits. inplace advances states and pos, else commit(c).
    pub fn uploadIds(w: *Win, m: *model.Model, tokens: []const u32) !void {
        try m.r.sync();
        @memcpy(w.host_ids[0..tokens.len], tokens);
        try m.r.upload(w.ids, std.mem.sliceAsBytes(w.host_ids[0..tokens.len]));
    }

    /// Copies tokens to the owned staging and queues the upload; waits for the queue first (staging is read at copy).
    pub fn forward(w: *Win, m: *model.Model, tokens: []const u32, inplace: bool, head_rows: u32) !void {
        const n: u32 = @intCast(tokens.len);
        if (n == 0 or n > w.rcap or head_rows > n) return error.Invalid;
        if (m.pos + n > m.cap) return error.ContextFull;
        if (n > max_rows and (!inplace or head_rows > 1)) return error.Invalid;
        const c = m.cfg;
        const h = c.hidden_size;
        const pos0 = m.pos;
        try w.uploadIds(m, tokens);
        try run(&m.k.embed, .{ n, 1, 1 }, .{ m.emb.w, m.emb.s, m.emb.b, w.ids, w.x, h });
        var mi: usize = 0;
        const limit: usize = if (std.c.getenv("NEM_LAYERS")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch m.layers.len) else m.layers.len; // debug: first layers only
        const Pend = enum { none, add, comb }; // the last layer's residual add (and a MoE block's combine) is still to do: fused into the next norm
        var pend: Pend = .none;
        for (m.layers, 0..) |*ly, li| {
            if (li >= limit) break;
            switch (pend) {
                .none => try run(&m.k.rms, .{ n, 1, 1 }, .{ w.x, ly.norm, w.xn, h, c.layer_norm_epsilon }),
                .add => try run(&m.k.add_rms, .{ n, 1, 1 }, .{ w.x, w.delta, ly.norm, w.xn, h, c.layer_norm_epsilon }),
                .comb => try run(&m.k.e_comb_rms, .{ n, 1, 1 }, .{ w.x, w.ey, w.wts, w.sy, ly.norm, w.xn, h, top_k, c.layer_norm_epsilon }),
            }
            pend = .add;
            w.cur = li;
            if (capture) |f| try f(w, m, li, .xn, n);
            switch (ly.mixer) {
                .mamba => |*mw| {
                    try w.mamba(m, mw, &w.ms[mi], n, inplace);
                    mi += 1;
                },
                .moe => |mw| {
                    try w.moeBody(m, mw, n);
                    pend = .comb;
                },
                .attention => |aw| try w.attention(m, aw, n, pos0),
            }
        }
        w.rows = n;
        w.inplace = inplace;
        w.pos0 = pos0;
        w.head_rows = head_rows;
        if (inplace) m.pos += n;
        if (head_rows == n and pend != .none) switch (pend) {
            .add => try run(&m.k.add_rms, .{ n, 1, 1 }, .{ w.x, w.delta, m.norm_f, w.xn, h, c.layer_norm_epsilon }),
            .comb => try run(&m.k.e_comb_rms, .{ n, 1, 1 }, .{ w.x, w.ey, w.wts, w.sy, m.norm_f, w.xn, h, top_k, c.layer_norm_epsilon }),
            .none => unreachable,
        } else {
            if (pend == .comb) try run(&w.k.comb, .{ (h + 63) / 64, n, 1 }, .{ w.ey, w.wts, w.sy, w.delta, h, top_k });
            if (pend != .none) try run(&m.k.add, .{ (n * h + 63) / 64, 1, 1 }, .{ w.x, w.delta, n * h });
            if (head_rows == 0) return;
            const off = @as(usize, n - head_rows) * h * 2;
            try run(&m.k.rms, .{ head_rows, 1, 1 }, .{ at(w.x, off), m.norm_f, w.xn, h, c.layer_norm_epsilon });
        }
        try w.headRows(m, w.xn, head_rows);
    }

    /// lm_head + bf16 rounding + argmax over n normalised rows x [n][h]: fp32 logits in w.logits, argmax in w.am_out.
    fn headRows(w: *Win, m: *model.Model, x: Buf, n: u32) !void {
        const c = m.cfg;
        if (w.scalar) try run(&w.k.qmv_f32, .{ c.vocab_size / 4, 1, 1 }, .{ m.head.w, m.head.s, m.head.b, x, w.logits, c.hidden_size, c.vocab_size, n }) else try w.mpf.?.denseS(m.head.w, m.head.s, m.head.b, x, n, w.logits, c.hidden_size, 0, c.vocab_size, true);
        if (m.bf16_logits) try run(&m.k.round, .{ (n * c.vocab_size + 63) / 64, 1, 1 }, .{ w.logits, n * c.vocab_size });
        const per = (c.vocab_size + 63) / 64;
        try run(&m.k.am_part, .{ 64, n, 1 }, .{ w.logits, w.am_scratch, c.vocab_size, per });
        try run(&m.k.am_fin, .{ n, 1, 1 }, .{ w.am_scratch, w.am_out, @as(u32, 64) });
    }

    /// The head on one normalised row (the MTP drafter's output row).
    pub fn headOn(w: *Win, m: *model.Model, x: Buf) !void {
        w.head_rows = 1;
        try w.headRows(m, x, 1);
    }

    /// norm_f of the first n rows of residual stream x after the last window (hidden rows for MTP) into out [n][h].
    pub fn normRows(w: *Win, m: *model.Model, n: u32, out: Buf) !void {
        try run(&m.k.rms, .{ n, 1, 1 }, .{ w.x, m.norm_f, out, m.cfg.hidden_size, m.cfg.layer_norm_epsilon });
    }

    /// Projection of n rows: n > 16 dense GEMM, else split-K matrix-engine GEMM (decode n = 1); `scalar`: row kernels.
    fn lin(w: *Win, m: *model.Model, t: model.Table, x: Buf, n: u32, y: Buf, in: u32, y_off: u32, rows: u32, f32out: bool) !void {
        if (n > max_rows) return w.mpf.?.dense(m.r, t.w, t.s, t.b, x, n, y, in, y_off, rows, f32out);
        return w.mpf.?.denseS(t.w, t.s, t.b, x, n, y, in, y_off, rows, f32out);
    }

    fn mamba(w: *Win, m: *model.Model, mw: anytype, s: *MState, n: u32, inplace: bool) !void {
        const c = m.cfg;
        const pd = c.projDim();
        const cd = c.convDim();
        const xd = c.xd();
        if (w.scalar) try run(&w.k.qmv_bf16, .{ pd, 1, 1 }, .{ mw.in.w, mw.in.s, mw.in.b, w.xn, s.proj, c.hidden_size, pd, pd, @as(u32, 0), n }) else try w.lin(m, mw.in, w.xn, n, s.proj, c.hidden_size, 0, pd, false);
        const cout = if (inplace) mw.cstate else s.cstate2;
        const sout = if (inplace) mw.sstate else s.sstate2;
        try run(&w.k.conv, .{ cd / 64, 1, 1 }, .{ s.proj, pd, xd, mw.cstate, cout, mw.conv_w, mw.conv_b, s.xc, cd, n });
        const per_group = c.mamba_num_heads / c.n_groups;
        try run(&w.k.ssm, .{ c.mamba_num_heads, c.mamba_head_dim / 4, 1 }, .{
            s.proj, pd, xd + cd, s.xc, cd, mw.sstate, sout, mw.a_log, mw.d, mw.dtb, w.y, c.mamba_head_dim, xd, c.n_groups, per_group, @as(f32, 0.0), std.math.inf(f32), n,
        });
        try run(&w.k.gnorm, .{ c.n_groups, n, 1 }, .{ w.y, mw.gnorm, w.yn, xd / c.n_groups, xd, c.layer_norm_epsilon });
        if (capture) |f| try f(w, m, w.cur, .yn, n);
        if (w.scalar) try run(&w.k.qmv_bf16, .{ c.hidden_size, 1, 1 }, .{ mw.out.w, mw.out.s, mw.out.b, w.yn, w.delta, xd, c.hidden_size, c.hidden_size, @as(u32, 0), n }) else try w.lin(m, mw.out, w.yn, n, w.delta, xd, 0, c.hidden_size, false);
    }

    /// A MoE block of n rows: delta = block output (experts weighted + shared expert), the combine included.
    pub fn moe(w: *Win, m: *model.Model, mw: anytype, n: u32) !void {
        try w.moeBody(m, mw, n);
        try run(&w.k.comb, .{ (m.cfg.hidden_size + 63) / 64, n, 1 }, .{ w.ey, w.wts, w.sy, w.delta, m.cfg.hidden_size, top_k });
    }

    /// Router, routed experts and shared expert of n rows: fills ey, wts, sy; the combine into delta is the caller's.
    fn moeBody(w: *Win, m: *model.Model, mw: anytype, n: u32) !void {
        const c = m.cfg;
        const h = c.hidden_size;
        const wd = c.moe_intermediate_size;
        const sw = c.moe_shared_expert_intermediate_size;
        try run(&m.k.e_logits, .{ c.n_routed_experts, n, 1 }, .{ w.xn, mw.gate, w.r_logits, h, c.n_routed_experts });
        try run(&m.k.e_route, .{ n, 1, 1 }, .{ w.r_logits, mw.bias, w.eids, w.wts, c.n_routed_experts, c.num_experts_per_tok, @as(u32, @bitCast(c.routed_scaling_factor)) });
        if (n > max_rows) try w.mpf.?.runExperts(m, mw, w.xn, w.eids, w.act, w.ey, n) else if (!w.scalar) try w.mpf.?.runExpertsS(m, mw, w.xn, w.eids, w.act, w.ey, n) else {
            const ng = @min(c.n_routed_experts, n * top_k);
            try run(&w.k.groups, .{ 1, 1, 1 }, .{ w.eids, w.elist, n * top_k, c.n_routed_experts });
            try run(&w.k.up_g, .{ wd, ng, 1 }, .{ mw.fc1.w, mw.fc1.s, mw.fc1.b, w.xn, w.eids, w.act, h, wd, n * top_k, top_k, w.elist });
            try run(&w.k.down_g, .{ h, ng, 1 }, .{ mw.fc2.w, mw.fc2.s, mw.fc2.b, w.act, w.eids, w.ey, wd, h, n * top_k, w.elist });
        }
        if (w.scalar) {
            try run(&w.k.dup, .{ sw, 1, 1 }, .{ mw.shup.w, mw.shup.s, mw.shup.b, w.xn, w.sact, h, sw, n });
            try run(&w.k.ddown, .{ h, 1, 1 }, .{ mw.shdn.w, mw.shdn.s, mw.shdn.b, w.sact, w.sy, sw, h, n });
        } else {
            if (n > max_rows) {
                try w.lin(m, mw.shup, w.xn, n, w.sact, h, 0, sw, false);
                try run(&w.mpf.?.relu2, .{ (n * sw + 63) / 64, 1, 1 }, .{ w.sact, n * sw });
            } else try w.mpf.?.denseSM(mw.shup.w, mw.shup.s, mw.shup.b, w.xn, n, w.sact, h, 0, sw, 2, true); // relu2 in the finish
            try w.lin(m, mw.shdn, w.sact, n, w.sy, sw, 0, h, true);
        }
    }

    pub fn attention(w: *Win, m: *model.Model, aw: anytype, n: u32, pos0: u32) !void {
        const c = m.cfg;
        const h = c.hidden_size;
        const q_dim = c.num_attention_heads * c.head_dim;
        const kv_dim = c.num_key_value_heads * c.head_dim;
        const chunk: u32 = 512;
        if (w.scalar) {
            try run(&w.k.qmv_bf, .{ q_dim / 4, 1, 1 }, .{ aw.q.w, aw.q.s, aw.q.b, w.xn, w.q, h, @as(u32, 0), q_dim, q_dim, n });
            try run(&w.k.qmv_bf, .{ kv_dim / 4, 1, 1 }, .{ aw.k.w, aw.k.s, aw.k.b, w.xn, aw.kc, h, pos0 * kv_dim, kv_dim, kv_dim, n });
            try run(&w.k.qmv_bf, .{ kv_dim / 4, 1, 1 }, .{ aw.v.w, aw.v.s, aw.v.b, w.xn, aw.vc, h, pos0 * kv_dim, kv_dim, kv_dim, n });
        } else {
            if (n > max_rows) {
                try w.lin(m, aw.q, w.xn, n, w.q, h, 0, q_dim, false);
                try w.lin(m, aw.k, w.xn, n, aw.kc, h, pos0 * kv_dim, kv_dim, false);
                try w.lin(m, aw.v, w.xn, n, aw.vc, h, pos0 * kv_dim, kv_dim, false);
            } else { // one activation tile for the three projections
                const mp = &w.mpf.?;
                try mp.denseSM(aw.q.w, aw.q.s, aw.q.b, w.xn, n, w.q, h, 0, q_dim, 0, true);
                try mp.denseSM(aw.k.w, aw.k.s, aw.k.b, w.xn, n, aw.kc, h, pos0 * kv_dim, kv_dim, 0, false);
                try mp.denseSM(aw.v.w, aw.v.s, aw.v.b, w.xn, n, aw.vc, h, pos0 * kv_dim, kv_dim, 0, false);
            }
        }
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
        var off: u32 = 0;
        if (n > max_rows and !w.scalar and w.k.a_pf != null) { // the whole window in one launch: no per-row scratch, so no row blocks
            try run(&w.k.a_prep.?, .{ (n + 7) / 8 * 4 * 4 * 1024 / 64, 1, 1 }, .{ w.q, w.qb, n }); // 2 kv heads x 2 head groups, 4 N tiles of 1024 dwords a row tile
            try run(&w.k.a_pf.?, .{ (n + 7) / 8, 4, 1 }, .{ w.qb, aw.kc, aw.vc, w.att, pos0, n });
            off = n;
        } else if (n <= max_rows and m.k.a_dec != null) { // a window of up to 16 rows: the decode kernel per row, the same bits as the token decoded alone
            const nch1 = @max(1, (pos0 + n + chunk - 1) / chunk);
            try run(&m.k.a_dec.?, .{ c.num_key_value_heads * 2, nch1, n }, .{ w.q, aw.kc, aw.vc, w.po, w.pm, w.pl, pos0 + 1, nch1, scale });
            try run(&m.k.a_dmerge.?, .{ c.num_attention_heads, n, 4 }, .{ w.po, w.pm, w.pl, w.att, pos0 + 1, nch1 });
            off = n;
        }
        // Row blocks bound the split-K scratch (po holds max_rows * 64 chunks a row); rows are independent.
        const nch_all = @max(1, (pos0 + n + chunk - 1) / chunk);
        const blk: u32 = @max(max_rows, @min(n, (max_rows * m.max_chunks) / nch_all));
        while (off < n) : (off += blk) {
            const rb = @min(blk, n - off);
            const nch = (pos0 + off + rb + chunk - 1) / chunk;
            const qo = @as(usize, off) * q_dim * 2;
            try run(&w.k.a_part, .{ c.num_key_value_heads, nch, rb }, .{ at(w.q, qo), aw.kc, aw.vc, w.po, w.pm, w.pl, pos0 + off + 1, chunk, c.num_key_value_heads, scale, nch_all });
            try run(&w.k.a_merge, .{ c.num_attention_heads, rb, 1 }, .{ w.po, w.pm, w.pl, at(w.att, qo), pos0 + off + 1, chunk, c.num_attention_heads, nch_all });
        }
        if (capture) |f| try f(w, m, w.cur, .att, n);
        if (w.scalar) try run(&w.k.qmv_bf, .{ h / 4, 1, 1 }, .{ aw.o.w, aw.o.s, aw.o.b, w.att, w.delta, q_dim, @as(u32, 0), h, h, n }) else try w.lin(m, aw.o, w.att, n, w.delta, q_dim, 0, h, false);
    }

    /// After a non-inplace window: keep the first c rows (all: swap pending states in; fewer: replay). Advances m.pos.
    pub fn commit(w: *Win, m: *model.Model, c: u32) !void {
        if (w.inplace or c > w.rows) return error.Invalid;
        if (c == 0) return;
        const cfg = m.cfg;
        const pd = cfg.projDim();
        const cd = cfg.convDim();
        const xd = cfg.xd();
        var mi: usize = 0;
        for (m.layers) |*ly| switch (ly.mixer) {
            .mamba => |*mw| {
                const s = &w.ms[mi];
                mi += 1;
                if (c == w.rows) {
                    std.mem.swap(Buf, &mw.cstate, &s.cstate2);
                    std.mem.swap(Buf, &mw.sstate, &s.sstate2);
                } else {
                    try run(&w.k.conv, .{ cd / 64, 1, 1 }, .{ s.proj, pd, xd, mw.cstate, mw.cstate, mw.conv_w, mw.conv_b, w.scratch_xc, cd, c });
                    const per_group = cfg.mamba_num_heads / cfg.n_groups;
                    try run(&w.k.ssm, .{ cfg.mamba_num_heads, cfg.mamba_head_dim / 4, 1 }, .{
                        s.proj, pd, xd + cd, s.xc, cd, mw.sstate, mw.sstate, mw.a_log, mw.d, mw.dtb, w.y, cfg.mamba_head_dim, xd, cfg.n_groups, per_group, @as(f32, 0.0), std.math.inf(f32), c,
                    });
                }
            },
            else => {},
        };
        m.pos += c;
    }

    /// Greedy token of each head row of the last window (syncs).
    pub fn rowArgmax(w: *Win, m: *model.Model, out: []i32) !void {
        try m.r.download(std.mem.sliceAsBytes(out[0..w.head_rows]), w.am_out);
        try m.r.sync();
    }

    /// fp32 logits [head_rows][vocab] of the last window (syncs).
    pub fn fetchRowLogits(w: *Win, m: *model.Model, out: []f32) !void {
        try m.r.download(std.mem.sliceAsBytes(out[0 .. w.head_rows * m.cfg.vocab_size]), w.logits);
        try m.r.sync();
    }
};

/// Between launches: on a graceful stop request (stop.zig), drain the queue and return error.Interrupted.
pub fn stopCheck(r: *rt.Runtime) !void {
    if (stop.requested()) {
        r.sync() catch {};
        return error.Interrupted;
    }
}

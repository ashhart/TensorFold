//! Muse Spark's DFlash2 drafter (RadixArk/Muse-Spark-Open-DFlash2) as an EXL3 pack, fed from the target's run.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const fwd = @import("forward.zig");
const load_ = @import("load.zig");
const dc = @import("dflash2_config.zig");
const dl = @import("dflash2_load.zig");
const exl3 = @import("exl3.zig");
const Stream = cuda.Stream;

pub const MAX_STREAMS = 16; // a draft pass's streams: 16 x block 8 = the EXL3 GEMV's 128 rows

fn u(x: usize) u32 {
    return @intCast(x);
}
fn ci(x: usize) c_int {
    return @intCast(x);
}

/// A matrix: EXL3 (the drafter pack's quantised linears) or 16-bit dense rows [N, K] (an unquantised Linear).
pub const Lin = union(enum) {
    q: exl3.Linear,
    dense: struct { w: u64, n: usize, k: usize, bf16: bool },

    pub fn n(l: Lin) usize {
        return switch (l) {
            .q => |x| x.n,
            .dense => |x| x.n,
        };
    }
};

pub const Conv = struct { proj: Lin, base: u64 }; // base fp32 [2, KT, D]

pub const DLayer = struct {
    in_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    q: exl3.Linear,
    k: exl3.Linear,
    v: exl3.Linear,
    o: exl3.Linear,
    gate: exl3.Linear,
    up: exl3.Linear,
    down: exl3.Linear,
    aconv: Conv,
    mconv: Conv,
};

pub const Drafter = struct {
    m: *fwd.Model,
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    mod_m: cuda.Module, // dflash.cu: embed, taps, per-head norm + RoPE, cache stores, block attention, row take
    mod: cuda.Module, // muse_dflash2
    f: struct { embed: cuda.Function, tap: cuda.Function, norm_rope: cuda.Function, store: cuda.Function, attn: cuda.Function, take: cuda.Function },
    g: struct { dense: cuda.Function, prep: cuda.Function, finish: cuda.Function, silu: cuda.Function, topk: cuda.Function, walk: cuda.Function },
    arena: load_.Arena,
    full: dc.DConfig,
    c: dc.DConfig, // this rank's view
    block: usize,
    taps: []usize,
    ring: usize,
    scale: f32,
    streams: usize,
    graphs_on: bool,
    fc: exl3.Linear,
    fc_norm: u64,
    layers: []DLayer,
    norm: u64,
    mask_emb: u64, // fp32 [D]: the target's embedding row of mask_token_id (raw: the reference skips embed_norm)
    sel: Lin, // candidate_selector.hidden_projection [rank, D]
    pcb: u64,
    scb: u64,
    cb_bf16: bool,
    kc: []u64,
    vc: []u64,
    cos: u64,
    sin: u64,
    b: struct { tapbuf: u64, ctx16: u64, ctxg: u64, ctx: u64, kb: u64, vb: u64, xh: u64, z: u64, x32: u64, h: u64, c16: u64, dyn: u64, q16: u64, q32: u64, a: u64, part: u64, recv: u64, gb: u64, ub: u64, mm: u64, tk: u64, head: u64, top: u64, topg: u64, sg: u64, am: u64, in_tok: u64, in_pos0: u64, in_slot: u64, rpos: u64, rslot: u64, anc: u64, anc_idx: u64 },
    h_anc: []u16 = &.{}, // the anchors' embedding rows staged here (anc bf16 [streams, D], anc_idx iota)
    drafts: usize = 0,
    graphs: [MAX_STREAMS + 1]?cuda.graph.Exec = @splat(null),

    /// The drafter pack at `dir` on this rank; `taps_override`: target layers to tap (k-layer packs only).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, m: *fwd.Model, dir: []const u8, s: Stream, graphs_on: bool, taps_override: ?[]const usize) !*Drafter {
        const d = m.d;
        const w = m.w;
        const text = try dl.readFile(gpa, io, dir, "config.json");
        defer gpa.free(text);
        const full = try dc.DConfig.parse(gpa, text);
        if (full.hidden != w.full.hidden) return error.HiddenMismatch;
        if (full.vocab != w.full.vocab or full.mask_id >= w.full.vocab) return error.VocabMismatch;
        // the target head already applies output_multiplier and the softcap (muse_softcap): the same, applied once
        if (!dl.near(full.out_mult, w.full.out_mult) or !dl.near(full.softcap, w.full.softcap)) return error.HeadScaleMismatch;
        const c = try full.rankView(w.world);
        const dr = try gpa.create(Drafter);
        errdefer gpa.destroy(dr);
        dr.* = undefined;
        dr.m = m;
        dr.d = d;
        dr.gpa = gpa;
        dr.full = full;
        dr.c = c;
        dr.drafts = 0;
        dr.graphs = @splat(null);
        dr.graphs_on = graphs_on;
        dr.block = full.block;
        dr.ring = m.ring;
        if (dr.ring < full.window + 2 * full.block) return error.RingShorterThanWindow;
        dr.scale = @floatCast(std.math.pow(f64, @floatFromInt(full.head_dim), -0.5));
        dr.streams = @min(m.slots, MAX_STREAMS);
        const tl = taps_override orelse full.tapList();
        if (tl.len != full.ntaps) return error.TapCount; // fc's input width is fixed by the checkpoint
        dr.taps = try gpa.dupe(usize, tl);
        errdefer gpa.free(dr.taps);
        for (dr.taps, 0..) |t, i| if (t >= w.layers.len or (i > 0 and t < dr.taps[i - 1])) return error.TapPastLastLayer;
        dr.mod_m = try cuda.Module.load(d, cuda.kernels.muse_dflash);
        errdefer dr.mod_m.unload();
        dr.mod = try cuda.Module.load(d, cuda.kernels.muse_dflash2);
        errdefer dr.mod.unload();
        dr.f = .{ .embed = try dr.mod_m.function("dr_block_embed"), .tap = try dr.mod_m.function("dr_tap"), .norm_rope = try dr.mod_m.function("dr_norm_rope"), .store = try dr.mod_m.function("dr_store"), .attn = try dr.mod_m.function("dr_attn"), .take = try dr.mod_m.function("dr_take") };
        dr.g = .{ .dense = try dr.mod.function("dfl_dense"), .prep = try dr.mod.function("dfl_conv_prep"), .finish = try dr.mod.function("dfl_conv_finish"), .silu = try dr.mod.function("dfl_silu_mul"), .topk = try dr.mod.function("dfl_topk"), .walk = try dr.mod.function("dfl_walk") };
        dr.arena = .{ .d = d, .gpa = gpa };
        errdefer dr.arena.deinit();
        try dl.loadWeights(dr, io, dir);
        errdefer gpa.free(dr.layers);
        try dl.allocState(dr, s);
        errdefer {
            gpa.free(dr.kc);
            gpa.free(dr.vc);
        }
        try s.synchronize();
        return dr;
    }

    pub fn deinit(dr: *Drafter) void {
        const gpa = dr.gpa;
        for (&dr.graphs) |*g| if (g.*) |*e| {
            e.deinit();
            g.* = null;
        };
        gpa.free(dr.h_anc);
        gpa.free(dr.taps);
        gpa.free(dr.layers);
        gpa.free(dr.kc);
        gpa.free(dr.vc);
        dr.mod.unload();
        dr.mod_m.unload();
        dr.arena.deinit();
        gpa.destroy(dr);
    }

    /// Device bytes the drafter holds on this rank (weights, caches, scratch).
    pub fn bytes(dr: *const Drafter) usize {
        return dr.arena.total;
    }

    pub fn zBytes(l: exl3.Linear, R: usize) usize {
        return if (l.split[0] > 1) l.split[0] * R * l.n * 4 else 0;
    }

    fn proj(dr: *Drafter, s: Stream, l: exl3.Linear, x: u64, out: u64, out_f32: bool, n: usize) !void {
        try dr.m.k3.linear(s, l, x, .f16, dr.b.xh, out, if (out_f32) .f32 else .f16, if (l.split[0] > 1) dr.b.z else 0, n);
    }

    fn projL(dr: *Drafter, s: Stream, l: Lin, x: u64, out: u64, out_f32: bool, n: usize) !void {
        switch (l) {
            .q => |q| try dr.proj(s, q, x, out, out_f32, n),
            .dense => |w| {
                var a: cuda.Args = .{};
                a.add(x);
                a.add(w.w);
                a.add(@as(c_int, @intFromBool(w.bf16)));
                a.add(out);
                a.add(@as(c_int, @intFromBool(out_f32)));
                a.add(ci(n));
                a.add(ci(w.k));
                a.add(ci(w.n));
                try launch(dr.g.dense, s, (w.n + 7) / 8, (n + 7) / 8, 256, &a);
            },
        }
    }

    /// muse_rms: out fp16 = x * rsqrt(mean x^2 + eps) * w; parts > 0: x is the ranks' gathered column slices.
    fn rms(dr: *Drafter, s: Stream, x: u64, x_half: bool, w: u64, out: u64, n: usize, parts: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_int, @intFromBool(x_half)));
        a.add(w);
        a.add(out);
        a.add(ci(dr.c.hidden));
        a.add(@as(f32, @floatCast(dr.c.eps)));
        a.add(ci(parts));
        try launch(dr.m.f_rms, s, n, 1, fwd.NORM_THREADS, &a);
    }

    fn launch(f: cuda.Function, s: Stream, gx: usize, gy: usize, threads: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = u(gx), .y = u(gy), .z = 1 }, .block = .{ .x = u(threads), .y = 1, .z = 1 } }, s, a);
    }

    /// The target's tap: residual rows after target layer `layer` (when tapped) into the tap buffer (fp16).
    pub fn tap(dr: *Drafter, s: Stream, layer: usize, res: u64, n: usize) !void {
        return dr.tapInto(s, layer, res, n, dr.b.tapbuf);
    }

    /// tap into another buffer of the same row layout ([rows, taps, D] fp16): the prompt path's chunk-wide taps.
    pub fn tapInto(dr: *Drafter, s: Stream, layer: usize, res: u64, n: usize, buf: u64) !void {
        for (dr.taps, 0..) |t, j| if (t == layer) {
            var a: cuda.Args = .{};
            a.add(res);
            a.add(buf);
            a.add(ci(dr.c.hidden));
            a.add(ci(dr.taps.len));
            a.add(ci(j));
            try launch(dr.f.tap, s, n, 1, 256, &a);
        };
    }

    /// The context update of the target's n rows: fc, hidden_norm, each layer's k (k_norm, RoPE) and v into the ring.
    pub fn update(dr: *Drafter, s: Stream, n: usize) !void {
        return dr.updateAt(s, dr.b.tapbuf, dr.m.b.pos, dr.m.b.slot, n);
    }

    /// update from explicit taps [n, taps, D] fp16 and the rows' positions / slots (n <= the target's window rows).
    pub fn updateAt(dr: *Drafter, s: Stream, taps: u64, pos: u64, slot: u64, n: usize) !void {
        const m = dr.m;
        if (n > m.rows) return error.WindowTooWide;
        try dr.proj(s, dr.fc, taps, dr.b.ctx16, false, n);
        if (m.comm) |comm| {
            try comm.allGather(dr.b.ctx16, dr.b.ctxg, n * dr.fc.n, .f16, s.handle);
            try dr.rms(s, dr.b.ctxg, true, dr.fc_norm, dr.b.ctx, n, dr.fc.n);
        } else try dr.rms(s, dr.b.ctx16, true, dr.fc_norm, dr.b.ctx, n, 0);
        for (dr.layers, 0..) |L, i| {
            try dr.proj(s, L.k, dr.b.ctx, dr.b.kb, false, n);
            try dr.normRope(s, dr.b.kb, L.k_norm, pos, slot, 0, dr.kc[i], dr.c.kv_heads, n);
            try dr.proj(s, L.v, dr.b.ctx, dr.b.vb, false, n);
            try dr.store(s, dr.b.vb, pos, slot, dr.vc[i], n);
        }
    }

    fn normRope(dr: *Drafter, s: Stream, in: u64, w: u64, pos: u64, slot: u64, out32: u64, cache: u64, nh: usize, n: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ in, w, dr.cos, dr.sin, pos, slot, out32, cache }) |p| a.add(p);
        a.add(ci(nh));
        a.add(ci(128));
        a.add(ci(dr.ring));
        a.add(@as(f32, @floatCast(dr.c.eps)));
        try launch(dr.f.norm_rope, s, n, nh, 128, &a);
    }

    fn store(dr: *Drafter, s: Stream, in: u64, pos: u64, slot: u64, cache: u64, n: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ in, pos, slot, cache }) |p| a.add(p);
        a.add(ci(dr.c.kv_heads));
        a.add(ci(dr.ring));
        try launch(dr.f.store, s, n, 1, 256, &a);
    }

    /// conv.prepare on h: the kernel projection into dyn, the first convolution into c16.
    fn convPrep(dr: *Drafter, s: Stream, cv: Conv, n: usize) !void {
        try dr.projL(s, cv.proj, dr.b.h, dr.b.dyn, false, n);
        var a: cuda.Args = .{};
        inline for (.{ dr.b.h, dr.b.dyn, cv.base, dr.b.c16 }) |p| a.add(p);
        a.add(ci(dr.block));
        a.add(ci(dr.c.hidden));
        a.add(ci(dr.c.conv_k));
        a.add(ci(dr.c.conv_g));
        try launch(dr.g.prep, s, n, 1, 256, &a);
    }

    /// The sublayer's fp32 partials (part) gathered, summed in rank order, convolved (conv.finish) into the residual.
    fn convFinish(dr: *Drafter, s: Stream, cv: Conv, n: usize) !void {
        const D = dr.c.hidden;
        var y0 = dr.b.part;
        var y1: u64 = 0;
        if (dr.m.comm) |comm| {
            try comm.allGather(dr.b.part, dr.b.recv, n * D, .f32, s.handle);
            y0 = dr.b.recv;
            y1 = dr.b.recv + n * D * 4;
        }
        var a: cuda.Args = .{};
        inline for (.{ dr.b.x32, y0, y1, dr.b.dyn, cv.base + dr.c.conv_k * D * 4 }) |p| a.add(p);
        a.add(ci(dr.block));
        a.add(ci(D));
        a.add(ci(dr.c.conv_k));
        a.add(ci(dr.c.conv_g));
        try launch(dr.g.finish, s, n, 1, 256, &a);
    }

    /// Stage a draft: each stream's pending (anchor) token, its position and slot (the draft pass's fixed inputs).
    pub fn stage(dr: *Drafter, tokens: []const i64, pos0: []const u32, slots: []const u32) !void {
        const S = tokens.len;
        if (S == 0 or S > dr.streams) return error.TooManyStreams;
        var pz: [MAX_STREAMS]i32 = undefined;
        var sz: [MAX_STREAMS]i32 = undefined;
        var rp: [MAX_STREAMS * 16]i32 = undefined;
        var rs: [MAX_STREAMS * 16]i32 = undefined;
        for (0..S) |i| {
            if (pos0[i] >= dr.m.context) return error.PastContext;
            if (slots[i] >= dr.m.slots) return error.BadSlot;
            if (tokens[i] < 0 or tokens[i] >= dr.full.vocab) return error.BadToken;
            pz[i] = @intCast(pos0[i]);
            sz[i] = @intCast(slots[i]);
            for (0..dr.block) |b| {
                rp[i * dr.block + b] = @intCast(pos0[i] + b);
                rs[i * dr.block + b] = @intCast(slots[i]);
            }
        }
        const d = dr.d;
        try d.check(d.api.cuMemcpyHtoD_v2(dr.b.in_tok, std.mem.sliceAsBytes(tokens).ptr, S * 8), "draft in");
        try dr.m.embedRows(tokens, dr.h_anc, dr.b.anc);
        try d.check(d.api.cuMemcpyHtoD_v2(dr.b.in_pos0, @ptrCast(&pz), S * 4), "draft in");
        try d.check(d.api.cuMemcpyHtoD_v2(dr.b.in_slot, @ptrCast(&sz), S * 4), "draft in");
        try d.check(d.api.cuMemcpyHtoD_v2(dr.b.rpos, @ptrCast(&rp), S * dr.block * 4), "draft in");
        try d.check(d.api.cuMemcpyHtoD_v2(dr.b.rslot, @ptrCast(&rs), S * dr.block * 4), "draft in");
    }

    /// The draft pass on staged inputs for S streams, kernels only: drafts [S, block - 1] (int32) left in `am`.
    pub fn run(dr: *Drafter, s: Stream, S: usize) !void {
        const m = dr.m;
        const c = dr.c;
        const B = dr.block;
        const n = S * B;
        const D = c.hidden;
        const bb = &dr.b;
        {
            var a: cuda.Args = .{};
            const es = m.embedSrc(bb.anc, bb.anc_idx);
            inline for (.{ es[0], dr.mask_emb, es[1], bb.x32 }) |p| a.add(p);
            a.add(ci(B));
            a.add(ci(D));
            try launch(dr.f.embed, s, n, 1, 256, &a);
        }
        for (dr.layers, 0..) |L, i| {
            try dr.rms(s, bb.x32, false, L.in_norm, bb.h, n, 0);
            try dr.convPrep(s, L.aconv, n);
            try dr.proj(s, L.q, bb.c16, bb.q16, false, n);
            try dr.normRope(s, bb.q16, L.q_norm, bb.rpos, bb.rslot, bb.q32, 0, c.heads, n);
            try dr.proj(s, L.k, bb.c16, bb.kb, false, n);
            try dr.normRope(s, bb.kb, L.k_norm, bb.rpos, bb.rslot, 0, dr.kc[i], c.kv_heads, n);
            try dr.proj(s, L.v, bb.c16, bb.vb, false, n);
            try dr.store(s, bb.vb, bb.rpos, bb.rslot, dr.vc[i], n);
            {
                var a: cuda.Args = .{};
                inline for (.{ bb.q32, dr.kc[i], dr.vc[i], @as(u64, 0), bb.in_pos0, bb.in_slot, bb.a }) |p| a.add(p);
                a.add(ci(B));
                a.add(ci(c.window));
                a.add(ci(dr.ring));
                a.add(ci(c.heads));
                a.add(ci(c.kv_heads));
                a.add(dr.scale);
                try launch(dr.f.attn, s, n, c.kv_heads, 256, &a); // a block per (row, KV head), a warp per head
            }
            try dr.proj(s, L.o, bb.a, bb.part, true, n);
            try dr.convFinish(s, L.aconv, n);
            try dr.rms(s, bb.x32, false, L.post_norm, bb.h, n, 0);
            try dr.convPrep(s, L.mconv, n);
            try dr.proj(s, L.gate, bb.c16, bb.gb, false, n);
            try dr.proj(s, L.up, bb.c16, bb.ub, false, n);
            {
                var a: cuda.Args = .{};
                inline for (.{ bb.gb, bb.ub, bb.mm }) |p| a.add(p);
                a.add(@as(i64, @intCast(n * c.inter)));
                try launch(dr.g.silu, s, @min((n * c.inter + 255) / 256, 4096), 1, 256, &a);
            }
            try dr.proj(s, L.down, bb.mm, bb.part, true, n);
            try dr.convFinish(s, L.mconv, n);
        }
        try dr.rms(s, bb.x32, false, dr.norm, bb.h, n, 0);
        const n1 = S * (B - 1);
        {
            var a: cuda.Args = .{};
            a.add(bb.h);
            a.add(bb.tk);
            a.add(ci(B));
            a.add(ci(D));
            try launch(dr.f.take, s, n1, 1, 256, &a);
        }
        try dr.headTop(s, n1);
        try dr.projL(s, dr.sel, bb.tk, bb.sg, false, n1);
        var a: cuda.Args = .{};
        a.add(if (m.comm != null) bb.topg else bb.top);
        a.add(ci(m.w.world));
        a.add(ci(n1));
        a.add(bb.sg);
        a.add(ci(c.sel_rank));
        inline for (.{ dr.pcb, dr.scb }) |p| a.add(p);
        a.add(@as(c_int, @intFromBool(dr.cb_bf16)));
        a.add(bb.in_tok);
        a.add(ci(B - 1));
        a.add(bb.am);
        try launch(dr.g.walk, s, S, 1, 512, &a);
    }

    /// The target's head on the draft rows (this rank's slice, scaled and softcapped), each rank's top 16, gathered.
    fn headTop(dr: *Drafter, s: Stream, n1: usize) !void {
        const m = dr.m;
        const bb = &dr.b;
        try dr.proj(s, m.w.head, bb.tk, bb.head, true, n1);
        const vr = m.w.head.n;
        {
            var a: cuda.Args = .{};
            a.add(bb.head);
            a.add(@as(i64, @intCast(n1 * vr)));
            a.add(@as(f32, @floatCast(m.w.full.out_mult)));
            a.add(@as(f32, @floatCast(m.w.full.softcap)));
            try launch(m.f_softcap, s, @min((n1 * vr + 255) / 256, 4096), 1, 256, &a);
        }
        var a: cuda.Args = .{};
        a.add(bb.head);
        a.add(ci(vr));
        a.add(ci(@min(vr, m.w.full.vocab - m.w.head_lo)));
        a.add(ci(m.w.head_lo));
        a.add(bb.top);
        try launch(dr.g.topk, s, n1, 1, 256, &a);
        if (m.comm) |comm| try comm.allGather(bb.top, bb.topg, n1 * 32, .f32, s.handle);
    }

    /// A draft for S streams: drafts [S, block - 1] into `out` (row-major). Graphed per stream count when enabled.
    pub fn draft(dr: *Drafter, s: Stream, tokens: []const i64, pos0: []const u32, slots: []const u32, out: []i32) !void {
        try dr.stage(tokens, pos0, slots);
        const S = tokens.len;
        if (!dr.graphs_on) {
            try dr.run(s, S);
        } else if (dr.graphs[S]) |e| {
            try e.launchOn(s);
        } else {
            try dr.run(s, S); // warm-up: writes the same cache rows the replay will
            try s.synchronize();
            try cuda.graph.beginCapture(s, .thread_local);
            dr.run(s, S) catch |err| {
                if (cuda.graph.endCapture(s)) |g| {
                    var gg = g;
                    gg.deinit();
                } else |_| {}
                return err;
            };
            var g = try cuda.graph.endCapture(s);
            defer g.deinit();
            dr.graphs[S] = try g.instantiate();
            try dr.graphs[S].?.launchOn(s);
        }
        try s.synchronize();
        const n1 = S * (dr.block - 1);
        try dr.d.check(dr.d.api.cuMemcpyDtoH_v2(@ptrCast(out.ptr), dr.b.am, n1 * 4), "drafts");
        dr.drafts += 1;
    }
};

test "drafter modules" {
    _ = dc;
    _ = dl;
}

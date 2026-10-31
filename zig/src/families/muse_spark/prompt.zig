//! Muse Spark's prompt path: a slot's prompt chunk as wide windows, with the decode rows' bits.

const std = @import("std");
const cuda = @import("cuda");
const load_ = @import("load.zig");
const fwd = @import("forward.zig");
const exl3 = @import("exl3.zig");
const Stream = cuda.Stream;
const Model = fwd.Model;

pub const CAP = 1024; // rows a chunk
pub const EWIN = 512; // rows an experts window: each expert's trellis read once a window instead of once per 128 rows
pub const MIN = 33; // the shortest prompt (from the pass's start) on this path; shorter ones run as decode windows
const W = fwd.DECODE_ROWS; // the dense and attention windows

fn u(x: usize) u32 {
    return @intCast(x);
}
fn ci(x: usize) c_int {
    return @intCast(x);
}

/// Bytes of the experts window's scratch: (xg + xu + xd, with y over them; z, also the gathers' landing).
pub fn expertBytes(e: exl3.Experts, ewin: usize, slots: usize, world: usize, chunk_d: usize) ![2]usize {
    const sz = try exl3.sizes(e, ewin, slots);
    const front = @max(2 * sz.xg + sz.xd, sz.y);
    const land = world * @max(ewin * slots * e.dims, chunk_d) * 4;
    return .{ front, @max(sz.z, if (world > 1) land else 0) };
}

pub const Prompt = struct {
    m: *Model,
    gpa: std.mem.Allocator,
    cap: usize,
    ewin: usize,
    ar: load_.Arena,
    ids: u64, // i64 [cap]
    pos: u64, // i32 [cap]
    slot: u64,
    res: u64, // f32 [cap, D]
    x: u64, // f16 [cap, D]
    xf: u64, // f32 [cap, D]
    part: u64, // f32 [cap, D]
    big: u64, // f32: the experts' split partials (z), else an all-gather's landing (never live together)
    logits: u64, // f32 [cap, E]
    pick: u64, // i32 [cap, K]
    wts: u64, // f32 [cap, K]
    lat: u64, // f16 [cap, latent / world]
    latg: u64, // f16 [world, cap, latent / world]
    latn: u64, // f16 [cap, latent]
    acc: u64, // f16 [cap, post inputs]
    sc: exl3.Scratch, // the grouped kernel's for ewin rows; y over xg / xu / xd, z = big
    taps: u64, // drafter: the chunk's taps, f16 [cap, taps, D]
    erow: u64, // bf16 [cap, D] embedding rows, read through iota (i64 [cap])
    iota: u64,
    h_erow: []u16,
    h_ids: []i64,
    h_pos: []i32,
    h_slot: []i32,

    /// Chunks of up to `cap` rows, experts in windows of up to `ewin_` rows.
    pub fn init(m: *Model, gpa: std.mem.Allocator, cap: usize, ewin_: usize, s: Stream) !*Prompt {
        const w = m.w;
        const cfg = w.cfg;
        if (cap < 1 or ewin_ < 1) return error.PromptRows;
        const ewin = @min(ewin_, cap);
        if (ewin * cfg.top_k * 4 > 96 * 1024) return error.ExpertWindow; // the group kernel's picks in shared memory
        const pr = try gpa.create(Prompt);
        errdefer gpa.destroy(pr);
        pr.* = undefined;
        pr.m = m;
        pr.gpa = gpa;
        pr.cap = cap;
        pr.ewin = ewin;
        pr.ar = .{ .d = m.d, .gpa = gpa };
        errdefer pr.ar.deinit();
        const ar = &pr.ar;
        const R = cap;
        const D = cfg.hidden;
        const L0 = w.layers[0];
        const e0 = L0.experts;
        pr.ids = try ar.zeros(R * 8);
        pr.pos = try ar.zeros(R * 4);
        pr.slot = try ar.zeros(R * 4);
        pr.res = try ar.zeros(R * D * 4);
        pr.x = try ar.zeros(R * D * 2);
        pr.xf = try ar.zeros(R * D * 4);
        pr.part = try ar.zeros(R * D * 4);
        pr.logits = try ar.zeros(R * cfg.experts * 4);
        pr.pick = try ar.zeros(R * cfg.top_k * 4);
        pr.wts = try ar.zeros(R * cfg.top_k * 4);
        pr.lat = try ar.zeros(R * L0.pre.n * 2);
        pr.latg = try ar.zeros(R * cfg.latent * 2);
        pr.latn = try ar.zeros(R * cfg.latent * 2);
        pr.acc = try ar.zeros(R * L0.post.k * 2);
        const eb = try expertBytes(e0, ewin, cfg.top_k, w.world, R * D);
        const sz = try exl3.sizes(e0, ewin, cfg.top_k);
        const front = try ar.zeros(eb[0]);
        pr.big = try ar.zeros(eb[1]);
        pr.sc = .{ .xg = front, .xu = front + sz.xg, .xd = front + 2 * sz.xg, .z = pr.big, .y = front, .ids = try ar.zeros(sz.ids), .count = try ar.zeros(4), .members = try ar.alloc(sz.members), .rows = ewin, .slots = cfg.top_k };
        try m.d.check(m.d.api.cuMemsetD32_v2(pr.sc.members, 0xFFFF_FFFF, sz.members / 4), "members -1");
        pr.taps = if (m.dr) |dr| try ar.zeros(R * dr.taps.len * D * 2) else 0;
        pr.erow = try ar.zeros(R * D * 2);
        {
            const io_ = try gpa.alloc(i64, R);
            defer gpa.free(io_);
            for (io_, 0..) |*x, j| x.* = @intCast(j);
            pr.iota = try ar.upload(std.mem.sliceAsBytes(io_));
        }
        pr.h_erow = try gpa.alloc(u16, R * D);
        errdefer gpa.free(pr.h_erow);
        pr.h_ids = try gpa.alloc(i64, R);
        errdefer gpa.free(pr.h_ids);
        pr.h_pos = try gpa.alloc(i32, R);
        errdefer gpa.free(pr.h_pos);
        pr.h_slot = try gpa.alloc(i32, R);
        try s.synchronize();
        return pr;
    }

    pub fn deinit(pr: *Prompt) void {
        pr.gpa.free(pr.h_erow);
        pr.gpa.free(pr.h_ids);
        pr.gpa.free(pr.h_pos);
        pr.gpa.free(pr.h_slot);
        pr.ar.deinit();
        pr.gpa.destroy(pr);
    }

    pub fn bytes(pr: *const Prompt) usize {
        return pr.ar.total;
    }

    /// The ranks' fp32 partials [count] gathered into big (rank-major); (rank 0's, rank 1's or 0).
    fn gather(pr: *Prompt, s: Stream, part: u64, count: usize) ![2]u64 {
        const c = pr.m.comm orelse return .{ part, 0 };
        try c.allGather(part, pr.big, count, .f32, s.handle);
        return .{ pr.big, pr.big + count * 4 };
    }

    /// An EXL3 projection of the chunk's n rows in 128-row windows: x [n, K] fp16 -> out [n, N].
    fn proj(pr: *Prompt, s: Stream, l: exl3.Linear, x: u64, out: u64, out_f32: bool, n: usize) !void {
        var w0: usize = 0;
        while (w0 < n) : (w0 += W) {
            const k = @min(W, n - w0);
            try pr.m.proj(s, l, x + w0 * l.k * 2, out + w0 * l.n * @as(usize, if (out_f32) 4 else 2), out_f32, k);
        }
    }

    /// Layer i's attention half over the chunk at [p0, p0 + n): 128-row windows of q / k / v / g and o_proj.
    fn attention(pr: *Prompt, s: Stream, L: load_.Layer, i: usize, p0: usize, n: usize) !void {
        const m = pr.m;
        const D = m.w.cfg.hidden;
        var w0: usize = 0;
        while (w0 < n) : (w0 += W) {
            const k = @min(W, n - w0);
            const x = pr.x + w0 * D * 2;
            try m.qkvg(s, L, x, m.b.q, m.b.k, m.b.v, m.b.g, k);
            var chunks: usize = 1;
            for (0..k) |j| chunks = @max(chunks, fwd.rowChunks(m.context, p0 + w0 + j));
            try m.attentionIo(s, i, k, chunks, .{ .q = m.b.q, .k = m.b.k, .v = m.b.v, .g = m.b.g, .qo = m.b.qo, .att = m.b.att, .pos = pr.pos + w0 * 4, .slot = pr.slot + w0 * 4 });
            try m.proj(s, L.o, m.b.att, pr.part + w0 * D * 4, true, k);
        }
    }

    /// MuseSparkMoE over the chunk: router and latent for every row, experts in windows of ewin rows, then post.
    fn moe(pr: *Prompt, s: Stream, L: load_.Layer, n: usize) !void {
        const m = pr.m;
        const cfg = m.w.cfg;
        const E = cfg.experts;
        const K = cfg.top_k;
        {
            var a: cuda.Args = .{};
            a.add(pr.xf);
            a.add(L.router);
            a.add(@as(c_int, @intFromBool(L.router_bf16)));
            a.add(pr.logits);
            a.add(ci(cfg.hidden));
            a.add(ci(E));
            const rw: usize = 2;
            try cuda.launch.launch(m.f_router, .{ .grid = .{ .x = u((E + rw - 1) / rw), .y = u(n), .z = 1 }, .block = .{ .x = u(32 * rw), .y = 1, .z = 1 } }, s, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(pr.logits);
            a.add(L.bias);
            a.add(ci(E));
            a.add(ci(K));
            a.add(@as(f32, @floatCast(cfg.out_mult)));
            a.add(@as(f32, @floatCast(cfg.route_scale)));
            a.add(pr.pick);
            a.add(pr.wts);
            const tn = std.math.ceilPowerOfTwo(usize, @max(E, 32)) catch unreachable;
            try cuda.launch.launch(m.f_topk, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = u(tn), .y = 1, .z = 1 } }, s, &a);
        }
        try pr.proj(s, L.pre, pr.x, pr.lat, false, n);
        if (m.comm) |c| {
            const half = L.pre.n;
            try c.allGather(pr.lat, pr.latg, n * half, .f16, s.handle);
            try m.normParts(s, pr.latg, true, L.pre_norm, pr.latn, n, cfg.latent, cfg.eps, half);
        } else try m.norm(s, pr.lat, true, L.pre_norm, pr.latn, n, cfg.latent, cfg.eps);
        const e = L.experts;
        const sc = pr.sc;
        var w0: usize = 0;
        while (w0 < n) : (w0 += pr.ewin) {
            const r = @min(pr.ewin, n - w0);
            const p = r * K;
            const pick = pr.pick + w0 * K * 4;
            try m.k3.routedWide(s, e, pr.latn + w0 * cfg.latent * 2, cfg.latent, pick, r, sc); // y over xg / xu / xd
            const y = try pr.gather(s, sc.y, p * e.dims); // z is dead too: the landing reuses it
            var a: cuda.Args = .{};
            a.add(y[0]);
            a.add(y[1]);
            a.add(pr.wts + w0 * K * 4);
            a.add(L.expert_norm);
            a.add(@as(f32, @floatCast(cfg.post_eps)));
            a.add(pr.acc + w0 * L.post.k * 2);
            a.add(ci(cfg.latent));
            a.add(ci(K));
            a.add(ci(m.w.rank * L.post.k));
            a.add(ci(L.post.k));
            try m.rowLaunch(m.f_esum, m.f_esum_c, cfg.latent, r, s, &a);
        }
        try pr.proj(s, L.post, pr.acc, pr.part, true, n);
    }

    /// A prompt chunk at [p0, p0 + n) of `slot`: every cache written, the last row's logits in the decode buffers.
    pub fn chunk(pr: *Prompt, s: Stream, slot: u32, p0: usize, tokens: []const i64) !void {
        const m = pr.m;
        const cfg = m.w.cfg;
        const D = cfg.hidden;
        const n = tokens.len;
        if (n == 0 or n > pr.cap) return error.ChunkTooWide;
        if (p0 + n > m.context) return error.PastContext;
        if (slot >= m.slots) return error.BadSlot;
        for (tokens, 0..) |t, j| {
            if (t < 0 or t >= m.w.full.vocab) return error.BadToken;
            pr.h_ids[j] = t;
            pr.h_pos[j] = @intCast(p0 + j);
            pr.h_slot[j] = @intCast(slot);
        }
        try m.d.check(m.d.api.cuMemcpyHtoD_v2(pr.ids, @ptrCast(pr.h_ids.ptr), n * 8), "ids");
        try m.d.check(m.d.api.cuMemcpyHtoD_v2(pr.pos, @ptrCast(pr.h_pos.ptr), n * 4), "pos");
        try m.d.check(m.d.api.cuMemcpyHtoD_v2(pr.slot, @ptrCast(pr.h_slot.ptr), n * 4), "slot");
        try m.embedRows(pr.h_ids[0..n], pr.h_erow, pr.erow);
        {
            var a: cuda.Args = .{};
            const es = m.embedSrc(pr.erow, pr.iota);
            a.add(es[0]);
            a.add(es[1]);
            a.add(pr.res);
            a.add(ci(D));
            a.add(@as(f32, @floatCast(cfg.eps)));
            try cuda.launch.launch(m.f_embed, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = fwd.NORM_THREADS, .y = 1, .z = 1 } }, s, &a);
        }
        try m.norm(s, pr.res, false, m.w.layers[0].in_norm, pr.x, n, D, cfg.eps);
        for (m.w.layers, 0..) |L, i| {
            try pr.attention(s, L, i, p0, n);
            try m.gateNormAt(s, pr.res, try pr.gather(s, pr.part, n * D), 0, L.attn_ab, L.ffn_norm, pr.x, pr.xf, n);
            try pr.moe(s, L, n);
            const after = if (i + 1 < m.w.layers.len) m.w.layers[i + 1].in_norm else m.w.norm;
            try m.gateNormAt(s, pr.res, try pr.gather(s, pr.part, n * D), L.post_ffn, L.mlp_ab, after, pr.x, 0, n);
            if (m.dr) |dr| try dr.tapInto(s, i, pr.res, n, pr.taps);
        }
        if (m.dr) |dr| {
            const row = dr.taps.len * D * 2;
            var w0: usize = 0;
            while (w0 < n) : (w0 += W) {
                const k = @min(W, n - w0);
                try dr.updateAt(s, pr.taps + w0 * row, pr.pos + w0 * 4, pr.slot + w0 * 4, k);
            }
        }
        // the head on the last row: the decode GEMV, softcap and gather into the decode buffers
        try m.proj(s, m.w.head, pr.x + (n - 1) * D * 2, m.b.head, true, 1);
        const cfgf = m.w.full;
        var a: cuda.Args = .{};
        a.add(m.b.head);
        a.add(@as(i64, @intCast(m.w.head.n)));
        a.add(@as(f32, @floatCast(cfgf.out_mult)));
        a.add(@as(f32, @floatCast(cfgf.softcap)));
        try cuda.launch.launch(m.f_softcap, .{ .grid = .{ .x = u(@min((m.w.head.n + 255) / 256, 4096)), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        if (m.comm) |c| try c.allGather(m.b.head, m.b.recv, m.w.head_stride, .f32, s.handle);
    }
};

/// The chunks of a prompt of `len` tokens: `cap` rows each, the rest last.
pub fn planChunks(len: usize, cap: usize, out: *std.ArrayList(usize), gpa: std.mem.Allocator) !void {
    out.clearRetainingCapacity();
    var at: usize = 0;
    while (at < len) {
        const n = @min(cap, len - at);
        try out.append(gpa, n);
        at += n;
    }
}

test "prompt chunks" {
    const g = std.testing.allocator;
    var c: std.ArrayList(usize) = .empty;
    defer c.deinit(g);
    try planChunks(40, 1024, &c, g);
    try std.testing.expectEqualSlices(usize, &.{40}, c.items);
    try planChunks(2500, 1024, &c, g);
    try std.testing.expectEqualSlices(usize, &.{ 1024, 1024, 452 }, c.items);
    try planChunks(2048, 1024, &c, g);
    try std.testing.expectEqualSlices(usize, &.{ 1024, 1024 }, c.items);
}

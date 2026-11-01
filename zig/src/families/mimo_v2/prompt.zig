//! MiMo-V2's prompt path: chunks of prompt rows on the prompt kernels, each row's bits independent of its chunk.

const std = @import("std");
const cuda = @import("cuda");
const load_ = @import("load.zig");
const fwd = @import("forward.zig");
const ops_ = @import("ops.zig");
const exl3 = @import("exl3.zig");
const Stream = cuda.Stream;
const Model = fwd.Model;

pub const EXPERTS_ROWS = 2048; // rows a prompt-experts call takes (its window; rows are independent of it)
pub const TOKENS_PER_UPDATE = 1024; // dflash.TOKENS_PER_UPDATE

pub const Prompt = struct {
    m: *Model,
    gpa: std.mem.Allocator,
    cap: usize, // rows a chunk
    ar: load_.Arena,
    // rows [cap]
    ids: u64,
    pos: u64,
    slot: u64,
    res: u64,
    x: u64,
    q: u64,
    k: u64,
    v: u64,
    qo: u64,
    att: u64,
    po: u64,
    pm: u64,
    pl: u64,
    xh: u64,
    part: u64,
    recv: u64,
    summed: u64,
    logits: u64,
    pick: u64,
    wts: u64,
    g: u64,
    up: u64,
    h: u64,
    z: u64, // the projections' split partials, 128 rows
    ps: exl3.PromptScratch, // the prompt experts', EXPERTS_ROWS rows a call
    taps: u64, // drafter: the chunk's last TOKENS_PER_UPDATE rows' taps
    h_ids: []i64,
    h_pos: []i32,
    h_slot: []i32,

    pub fn init(m: *Model, gpa: std.mem.Allocator, cap: usize, s: Stream) !*Prompt {
        const w = m.w;
        const cfg = w.cfg;
        const pr = try gpa.create(Prompt);
        errdefer gpa.destroy(pr);
        pr.* = undefined;
        pr.m = m;
        pr.gpa = gpa;
        pr.cap = cap;
        pr.ar = .{ .d = m.d, .gpa = gpa };
        errdefer pr.ar.deinit();
        const ar = &pr.ar;
        const R = cap;
        const D = cfg.hidden;
        const H = cfg.heads;
        const hk = @max(cfg.kv_full, cfg.kv_swa);
        const nd = cfg.dense_width / w.world;
        const nch = fwd.windowChunks(cfg.window);
        var max_k: usize = @max(D, H * 128);
        max_k = @max(max_k, nd);
        const e0 = for (w.layers) |L| {
            if (L.experts) |e| break e;
        } else return error.NoMoE;
        const K8 = cfg.top_k;
        pr.ids = try ar.zeros(R * 8);
        pr.pos = try ar.zeros(R * 4);
        pr.slot = try ar.zeros(R * 4);
        pr.res = try ar.zeros(R * D * 4);
        pr.x = try ar.zeros(R * D * 2);
        pr.q = try ar.zeros(R * H * 192 * 2);
        pr.k = try ar.zeros(R * hk * 192 * 2);
        pr.v = try ar.zeros(R * hk * 192 * 2);
        pr.qo = try ar.zeros(R * H * 192 * 2);
        pr.att = try ar.zeros(R * H * 128 * 2);
        pr.po = try ar.zeros(R * nch * H * 128 * 4);
        pr.pm = try ar.zeros(R * nch * H * 4);
        pr.pl = try ar.zeros(R * nch * H * 4);
        pr.xh = try ar.zeros(R * max_k * 2);
        pr.part = try ar.zeros(R * D * 4);
        pr.recv = try ar.zeros(2 * R * D * 4);
        pr.summed = try ar.zeros(R * D * 4);
        pr.logits = try ar.zeros(R * cfg.experts * 4);
        pr.pick = try ar.zeros(R * K8 * 4);
        pr.wts = try ar.zeros(R * K8 * 4);
        pr.g = try ar.zeros(R * nd * 2);
        pr.up = try ar.zeros(R * nd * 2);
        pr.h = try ar.zeros(R * nd * 2);
        pr.z = try ar.zeros(@max(fwd.splitBytes(w, exl3.MAX_ROWS), 4));
        pr.ps = try exl3.promptScratch(ar, e0, @min(EXPERTS_ROWS, R), K8);
        pr.taps = if (m.dr) |dr| try ar.zeros(@min(R, TOKENS_PER_UPDATE) * dr.taps.len * D * 2) else 0; // fp16, as the tap buffer
        pr.h_ids = try gpa.alloc(i64, R);
        pr.h_pos = try gpa.alloc(i32, R);
        pr.h_slot = try gpa.alloc(i32, R);
        try s.synchronize();
        return pr;
    }

    pub fn deinit(pr: *Prompt) void {
        pr.gpa.free(pr.h_ids);
        pr.gpa.free(pr.h_pos);
        pr.gpa.free(pr.h_slot);
        pr.ar.deinit();
        pr.gpa.destroy(pr);
    }

    pub fn bytes(pr: *const Prompt) usize {
        return pr.ar.total;
    }

    /// out [n, N] = x [n, K] fp16 through the layer, 128 rows a call: the decode rows' arithmetic.
    fn proj(pr: *Prompt, s: Stream, l: exl3.Linear, x: u64, out: u64, out_f32: bool, n: usize) !void {
        try pr.m.k3.rows(s, l, x, pr.xh, out, if (out_f32) .f32 else .f16, pr.z, n);
    }

    fn rms(pr: *Prompt, s: Stream, x: u64, w: u64, out: u64, n: usize) !void {
        try pr.m.ops.rms(s, x, 0, w, out, n, pr.m.w.cfg.hidden, pr.m.w.cfg.eps);
    }

    fn addRms(pr: *Prompt, s: Stream, x: u64, w: u64, n: usize) !void {
        try pr.m.ops.rms(s, x, pr.res, w, pr.x, n, pr.m.w.cfg.hidden, pr.m.w.cfg.eps);
    }

    /// forward._sum over the chunk's rows.
    fn sum(pr: *Prompt, s: Stream, part: u64, n: usize) !u64 {
        const m = pr.m;
        const c = m.comm orelse return part;
        const D = m.w.cfg.hidden;
        try c.allGather(part, pr.recv, n * D, .f32, s.handle);
        try m.add2(s, pr.recv, pr.recv + n * D * 4, pr.summed, n * D);
        return pr.summed;
    }

    fn launch(f: cuda.Function, s: Stream, gx: usize, threads: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(gx), .y = 1, .z = 1 }, .block = .{ .x = @intCast(threads), .y = 1, .z = 1 } }, s, a);
    }

    /// One prompt chunk at [p0, p0 + n) of `slot`: caches written, its last row's logits in the decode buffers.
    pub fn chunk(pr: *Prompt, s: Stream, slot: u32, p0: usize, tokens: []const i64, cap: ?Model.Capture) !void {
        const m = pr.m;
        const cfg = m.w.cfg;
        const D = cfg.hidden;
        const H = cfg.heads;
        const n = tokens.len;
        if (n == 0 or n > pr.cap) return error.ChunkTooWide;
        if (n > m.ring - cfg.window) return error.RingOverrun;
        if (p0 + n > m.context) return error.PastContext;
        if (slot >= m.slots) return error.BadSlot;
        for (tokens, 0..) |t, j| {
            pr.h_ids[j] = t;
            pr.h_pos[j] = @intCast(p0 + j);
            pr.h_slot[j] = @intCast(slot);
        }
        try m.d.check(m.d.api.cuMemcpyHtoD_v2(pr.ids, @ptrCast(pr.h_ids.ptr), n * 8), "ids");
        try m.d.check(m.d.api.cuMemcpyHtoD_v2(pr.pos, @ptrCast(pr.h_pos.ptr), n * 4), "pos");
        try m.d.check(m.d.api.cuMemcpyHtoD_v2(pr.slot, @ptrCast(pr.h_slot.ptr), n * 4), "slot");
        const dr = if (cap == null) m.dr else null;
        const keep = @min(n, TOKENS_PER_UPDATE);
        {
            var a: cuda.Args = .{};
            a.add(m.w.embed);
            a.add(pr.ids);
            a.add(pr.res);
            a.add(@as(c_int, @intCast(D)));
            try launch(m.f_embed, s, n, 256, &a);
        }
        try pr.rms(s, pr.res, m.w.layers[0].input_norm, pr.x, n);
        for (m.w.layers, 0..) |L, i| {
            try pr.proj(s, L.q, pr.x, pr.q, false, n);
            try pr.proj(s, L.k, pr.x, pr.k, false, n);
            try pr.proj(s, L.v, pr.x, pr.v, false, n);
            // keys and values into the caches; full layers: the prompt program, sliding: as decode rows
            const at = m.layer(i);
            const io: ops_.AttnIo = .{ .q = pr.q, .k = pr.k, .v = pr.v, .qo = pr.qo, .att = pr.att, .pos = pr.pos, .slot = pr.slot, .po = pr.po, .pm = pr.pm, .pl = pr.pl };
            try m.ops.qkv(s, io, at, H, n);
            if (at.win == 0) {
                try m.ops.prompt(s, io, at, H, n, m.scale);
            } else try m.ops.rows(s, io, at, H, n, fwd.windowChunks(at.win), m.scale);
            if (cap) |c| if (c.hook) |hf| try hf(c.ctx, .att, i, pr.att, n * H * 128 * 2);
            try pr.proj(s, L.o, pr.att, pr.part, true, n);
            try pr.addRms(s, try pr.sum(s, pr.part, n), L.post_norm, n);
            if (cap) |c| if (c.hook) |hf| try hf(c.ctx, .attn, i, pr.res, n * D * 4);
            if (L.experts) |e| {
                try m.ops.route(s, pr.x, L.router, L.bias, pr.logits, pr.pick, pr.wts, n, D, cfg.experts, cfg.top_k, L.route_scale);
                // the prompt experts in windows: rows are independent, so windows change no bits
                var w0: usize = 0;
                while (w0 < n) : (w0 += pr.ps.sc.rows) {
                    const k = @min(pr.ps.sc.rows, n - w0);
                    const wts = pr.wts + w0 * cfg.top_k * 4;
                    try m.k3.prompted(s, e, pr.x + w0 * D * 2, D, pr.pick + w0 * cfg.top_k * 4, k, pr.ps);
                    try m.ops.wsum(s, pr.ps.sc.y, wts, pr.part + w0 * D * 4, k, cfg.top_k, D);
                }
            } else {
                try pr.proj(s, L.gate.?, pr.x, pr.g, false, n);
                try pr.proj(s, L.up.?, pr.x, pr.up, false, n);
                const nd = L.gate.?.n;
                try m.ops.silu(s, pr.g, pr.up, pr.h, n, nd);
                try pr.proj(s, L.down.?, pr.h, pr.part, true, n);
            }
            const after = if (i + 1 < m.w.layers.len) m.w.layers[i + 1].input_norm else m.w.norm;
            try pr.addRms(s, try pr.sum(s, pr.part, n), after, n);
            if (cap) |c| if (c.hook) |hf| try hf(c.ctx, .mlp, i, pr.res, n * D * 4);
            if (dr) |dd| try dd.tapInto(s, i, pr.res + (n - keep) * D * 4, keep, pr.taps);
        }
        if (dr) |dd| {
            const row = dd.taps.len * D * 2;
            var w0: usize = 0;
            while (w0 < keep) : (w0 += fwd.DECODE_ROWS) {
                const k = @min(fwd.DECODE_ROWS, keep - w0);
                const at = n - keep + w0;
                try dd.updateFrom(s, pr.taps + w0 * row, pr.pos + at * 4, pr.slot + at * 4, k);
            }
        }
        // _head on the last row: the decode GEMV, gathered into the model's decode buffers
        try m.k3.linear(s, m.w.head, pr.x + (n - 1) * D * 2, .f16, m.b.xh, m.b.head, .f32, if (m.w.head.split[0] > 1) m.b.z else 0, 1);
        if (m.comm) |c| try c.allGather(m.b.head, m.b.recv, m.w.head.n, .f32, s.handle);
    }
};

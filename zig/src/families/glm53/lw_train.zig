//! Living Weights on the full GLM-5.3: the trainer behind the learner state machine (families/glm/lw_learner.zig,
//! unchanged) on N Macs. Every rank runs the same learner in lockstep (the job server hands every rank the same lines).
//!
//! A capture is one forward of an example on a SCRATCH engine (same weights, same exchange, its own small KV caches,
//! never the serving engine's: the frame is not touched) with the change off; from it every rank keeps
//! - x: the site's input on every row, gathered whole (each rank holds its 512 intermediate columns; one host sum of
//!   zero-padded rows), and
//! - h0: the residual after layer 77 on the answer rows (already the same on every rank),
//! cached for the server's life. A step is then host math (lw_tail.step): the change, the final norm, this rank's
//! vocab rows of the head, two host sums through the same exchange window (LSE parts; W_r^T g_r).
const std = @import("std");
const mtl = @import("metal");
const m = @import("model.zig");
const fwd = @import("forward.zig");
const Exchange = @import("exchange.zig").Exchange;
const Lw = @import("lw_gpu.zig").Lw;
const lwh = @import("glm53_lw");
const lm = lwh.lm;
const tail = lwh.tail;
const lw_sites = lwh.sites;
const lw_learner = lwh.learner;

pub const Learner = lw_learner.Learner(Trainer);
pub const Lesson = lw_learner.Lesson;
pub const Example = lw_learner.Example;
const Mode = lw_learner.Mode;
const max_rows = lw_learner.max_rows;

/// The scratch engine's KV positions (an example is at most max_rows + 1 tokens) and its block rows.
pub const scratch_cap: u32 = 512;
pub const scratch_rows_max: u32 = 64;
const cache_cap: usize = 2 << 30;

/// What the trainer needs from the server: the serving engine (its weights, exchange and seq), how to make a scratch one.
pub const Host = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    serve: *fwd.Engine,
    lw: *Lw,
    kv16: bool,
};

pub const Trainer = struct {
    pub const Ctx = *Host;

    host: *Host,
    scratch: *fwd.Engine,
    sites: lw_sites.Sites,
    sketched: usize = 0,
    cache: std.AutoHashMapUnmanaged(u64, tail.Captured) = .empty,
    cache_bytes: usize = 0,
    proj: []f32,
    norm: []f32,
    head: tail.Head,
    coll: tail.Coll,
    captures: u64 = 0,
    s_capture: f64 = 0,
    s_steps: f64 = 0,

    pub fn init(gpa: std.mem.Allocator, h: *Host) !*Trainer {
        const e = h.serve;
        const w = e.w;
        if (w.first != 0 or w.last != m.LAYERS) return error.LivingNeedsAllLayers;
        const t = try gpa.create(Trainer);
        errdefer gpa.destroy(t);
        const rows = @min(e.max_rows, scratch_rows_max);
        const scratch = try fwd.Engine.init(gpa, h.device, w, e.plan, e.xc, scratch_cap, rows, h.kv16);
        const vr = w.share.vocab[1] - w.share.vocab[0];
        t.* = .{ .host = h, .scratch = scratch, .sites = undefined, .proj = try gpa.alloc(f32, max_rows * lm.candidates), .norm = try gpa.alloc(f32, m.D), .head = .{ .w = w.head.buf.slice(u16, w.head.off / 2 + vr * m.D)[w.head.off / 2 ..], .v0 = w.share.vocab[0], .n = vr }, .coll = .{} };
        const nw: [*]const u16 = @ptrCast(@alignCast(w.norm.addr()));
        for (t.norm, nw[0..m.D]) |*o, x| o.* = lm.bfVal(x);
        if (e.xc) |x| t.coll = .{ .ptr = t, .sumFn = sumFn, .rank = x.rank, .ranks = x.ranks };
        // the committed blocks loaded from the sidecar carried over; the Sites then own the change's memory
        const l = h.lw;
        const keep = l.served;
        const a0 = try gpa.dupe(f32, l.aSlice()[0 .. keep * tail.IN]);
        defer gpa.free(a0);
        const b0 = try gpa.dupe(f32, l.bSlice()[0 .. keep * tail.D]);
        defer gpa.free(b0);
        t.sites = try lw_sites.Sites.init(gpa, tail.IN, tail.D, .{ .a = l.aSlice(), .b = l.bSlice(), .tau = l.tau });
        if (keep > 0) try t.sites.adopt(a0, b0, keep);
        l.publish(t.sites.rank);
        std.debug.print("glm53 slide: learns at layers.77 shared-expert down_proj, rank {d}/{d}, vocab rows {d}, {d} committed ranks carried over\n", .{ t.coll.rank, t.coll.ranks, vr, keep });
        return t;
    }

    pub fn deinit(t: *Trainer, gpa: std.mem.Allocator) void {
        var it = t.cache.valueIterator();
        while (it.next()) |c| c.free(gpa);
        t.cache.deinit(gpa);
        t.sites.deinit();
        gpa.free(t.proj);
        gpa.free(t.norm);
        gpa.destroy(t);
    }

    fn sumFn(p: ?*anyopaque, v: []f32) anyerror!void {
        const t: *Trainer = @ptrCast(@alignCast(p.?));
        const e = t.host.serve;
        return e.xc.?.hostSum(&e.seq, v);
    }

    /// The served change follows the Sites: the leading always-on blocks (the open gated block is never served).
    pub fn attach(t: *Trainer, on: bool) void {
        t.host.lw.on = on;
        t.host.lw.publish(t.sites.rank);
    }

    /// Ranks [0, ranks) into this rank's sidecar file (every rank writes its own copy: the same bytes).
    pub fn write(t: *Trainer, ranks: usize) !usize {
        const n = try t.host.lw.save(t.host.gpa, ranks);
        t.host.lw.publish(t.sites.rank);
        return n;
    }

    pub fn projected(t: *const Trainer, k: usize, rows: usize) []const f32 {
        _ = k;
        return t.proj[0 .. rows * lm.candidates];
    }

    pub fn step(t: *Trainer, ids: []const u32, start: usize, mode: Mode) !lm.Result {
        if (ids.len < 2 or start < 1 or start >= ids.len or ids.len - 1 > max_rows) return error.BadExample;
        if ((mode == .grad or mode == .learn) and t.sites.rank == 0) return error.NoOpenBlock;
        const cap = try t.captured(ids, start);
        const site = &t.sites.list[0];
        const t0 = mtl.clock.seconds();
        defer {
            t.s_steps += mtl.clock.seconds() - t0;
            t.host.lw.publish(t.sites.rank);
        }
        const gpa = t.host.gpa;
        switch (mode) {
            .avoid, .seek => {
                const y = if (mode == .avoid) site.avoid else site.seek;
                const k: usize = if (mode == .avoid) lm.avoid_dims else lm.candidates;
                const seed: u32 = if (mode == .avoid) 1 else 2;
                const first = start - 1;
                tail.sketch(cap.x, y, cap.rows, tail.IN, k, @intCast(t.sketched), seed, 1);
                tail.sketch(cap.x[first * tail.IN ..], y, 1, tail.IN, k, @intCast((1 << 30) + t.sketched + first), seed, @floatFromInt(cap.rows - first));
                t.sketched += cap.rows;
                return .{ .loss = 0, .recalled = false };
            },
            .project => {
                tail.projectRows(cap.x, site.seek, t.proj, cap.rows, tail.IN, lm.candidates);
                return .{ .loss = 0, .recalled = false };
            },
            .loss => return tail.step(gpa, t.head, t.norm, cap, t.sites.lora(), null, 0, t.coll),
            .grad, .learn => {
                const r = try tail.step(gpa, t.head, t.norm, cap, t.sites.lora(), site.gb, t.sites.first(), t.coll);
                if (mode == .learn) t.sites.adam();
                return r;
            },
        }
    }

    fn key(ids: []const u32, start: usize) u64 {
        var h = std.hash.Wyhash.init(0x4c57_3533);
        h.update(std.mem.sliceAsBytes(ids));
        h.update(std.mem.asBytes(&start));
        return h.final();
    }

    /// The example's capture: from the cache, or one scratch forward + the x gather (every rank, the same order).
    pub fn captured(t: *Trainer, ids: []const u32, start: usize) !*const tail.Captured {
        const k = key(ids, start);
        if (t.cache.getPtr(k)) |c| return c;
        const gpa = t.host.gpa;
        if (t.cache_bytes > cache_cap) {
            var it = t.cache.valueIterator();
            while (it.next()) |c| c.free(gpa);
            t.cache.clearRetainingCapacity();
            t.cache_bytes = 0;
        }
        var c = try tail.Captured.alloc(gpa, ids.len - 1, start);
        errdefer c.free(gpa);
        const t0 = mtl.clock.seconds();
        try t.forward(ids, &c);
        for (0..c.answers()) |i| c.targets[i] = ids[start + i];
        t.s_capture += mtl.clock.seconds() - t0;
        t.captures += 1;
        try t.cache.put(gpa, k, c);
        t.cache_bytes += c.bytes();
        return t.cache.getPtr(k).?;
    }

    /// The example's rows through the scratch engine (positions 0.., change off: the scratch has no Lw), blocks of its
    /// max_rows; after each block layer 77's shared activations (this rank's columns) and the residual are still there.
    fn forward(t: *Trainer, ids: []const u32, c: *tail.Captured) !void {
        const e = t.host.serve;
        const x = t.scratch;
        const w = e.w;
        const lo = w.share.moe[0];
        const own = w.share.moe[1] - lo; // this rank's site columns [lo, lo + own)
        const ir = own; // a MoE layer's act rows are its share's width (forward.zig mlp: ir = moe[1] - moe[0])
        const rows = c.rows;
        @memset(c.x, 0);
        x.seq = e.seq; // one exchange sequence: the scratch's exchanges follow the server's
        errdefer e.seq = x.seq;
        var at: usize = 0;
        while (at < rows) {
            const n = @min(rows - at, x.max_rows);
            @memcpy(x.tok.buf.slice(u32, n), ids[at..][0..n]);
            _ = try x.runRows(@intCast(at), n, null, false);
            const act: [*]const f32 = @ptrCast(@alignCast(x.act.addr()));
            const sh = act + n * m.TOP * ir; // [n][ir]: the shared expert's silu(gate) up, layer 77 (the last one run)
            for (0..n) |j| @memcpy(c.x[(at + j) * tail.IN + lo ..][0..own], sh[j * ir ..][0..own]);
            const hh: [*]const f32 = @ptrCast(@alignCast(x.h.addr()));
            for (0..n) |j| {
                const r = at + j;
                if (r + 1 < c.start) continue;
                @memcpy(c.h0[(r + 1 - c.start) * tail.D ..][0..tail.D], hh[j * m.D ..][0..m.D]);
            }
            at += n;
        }
        // hand the sequence back BEFORE the gather: the host sum takes the next exchange number from the server engine
        // (bench 10/9 20:04: with the hand-back deferred past this line, the gather reused a number the scratch's last block
        // had already used, saw that block's flags, and summed stale slots: every rank chose a different block)
        e.seq = x.seq;
        // every rank's columns into every rank's x (zero elsewhere: the slot-ordered sum is the gather, bit for bit)
        if (t.coll.ranks > 1) try t.coll.sum(c.x);
    }
};

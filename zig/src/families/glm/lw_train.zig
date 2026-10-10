//! GLM-5.3-Flash's Living Weights trainer: one GPU forward per example (the change off) captures what layer 44
//! leaves (the site's input on every row; the shared expert's output, the MLP branch, the last expand's weights and old
//! streams on the answer rows), cached for the server's life (nothing before the site ever changes). Every learning step
//! then runs on the CPU past the site (lw_math.step): the change, the expand, the stream mean, the final norm, the 4-bit
//! head, the loss and its gradient to b. The change itself lives in the engine's Living buffers (living.zig), which
//! the served forward reads, so a step's b is what the next chat reply uses.
const std = @import("std");
const mtl = @import("metal");
const ge = @import("engine.zig");
const fwd = @import("forward.zig");
const prompt_mod = @import("prompt.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const living = @import("living.zig");
const lm = @import("lw_math.zig");
const lw_sites = @import("lw_sites.zig");
const lw_learner = @import("lw_learner.zig");
const Ref = wts.Ref;

const max_rows = lw_learner.max_rows;
const Mode = lw_learner.Mode;

/// The cache's room for captured examples; past it the cache starts over.
const cache_cap: usize = 3 << 30;

pub const Learner = lw_learner.Learner(Trainer);
pub const lm_math = lm;

pub const Trainer = struct {
    pub const Ctx = *ge.Engine;

    gpa: std.mem.Allocator,
    e: *ge.Engine,
    lw: *living.Living,
    sites: lw_sites.Sites,
    sketched: usize = 0,
    state: st.State, // the trainer's own caches: no slot's state is touched
    ids: Ref, // u32 [max_rows]
    cache: std.AutoHashMapUnmanaged(u64, lm.Captured) = .empty,
    cache_bytes: usize = 0,
    proj: []f32, // [max_rows, candidates]
    tmp: []f32,
    norm: []f32, // the final norm's weight, f32
    head: lm.Q4,
    saved: usize = 0, // ranks already in the folder (loaded or written): a v2 save spawns a shard of the ranks after them
    captures: u64 = 0, // GPU forwards run
    seconds_gpu: f64 = 0,
    seconds_cpu: f64 = 0,

    pub fn init(gpa: std.mem.Allocator, e: *ge.Engine) !*Trainer {
        if (e.ep != null or e.c.tp > 1) return error.SlideNeedsOneMac;
        if (e.c.run != e.c.layers) return error.LivingNeedsAllLayers;
        e.sync();
        const l = try e.livingOn();
        const c = &e.c;
        const t = try gpa.create(Trainer);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .e = e, .lw = l, .sites = undefined, .state = undefined, .ids = undefined, .proj = &.{}, .tmp = &.{}, .norm = &.{}, .head = undefined };
        t.proj = try gpa.alloc(f32, max_rows * lm.candidates);
        errdefer gpa.free(t.proj);
        t.tmp = try gpa.alloc(f32, living.in);
        errdefer gpa.free(t.tmp);
        t.norm = try gpa.alloc(f32, c.hidden);
        errdefer gpa.free(t.norm);
        const nw: [*]const u16 = @ptrCast(@alignCast(e.w.norm.addr()));
        for (t.norm, nw[0..c.hidden]) |*o, h| o.* = lm.bfVal(h);
        const q = e.w.head;
        if (q.n != c.vocab or q.k != c.hidden) return error.UnsupportedShape;
        const n: usize = q.n;
        const k: usize = q.k;
        t.head = .{
            .w = @as([*]const u32, @ptrCast(@alignCast(q.w.addr())))[0 .. n * k / 8],
            .s = @as([*]const u16, @ptrCast(@alignCast(q.s.addr())))[0 .. n * k / 64],
            .b = @as([*]const u16, @ptrCast(@alignCast(q.b.addr())))[0 .. n * k / 64],
            .n = n,
            .k = k,
        };
        t.state = try st.initState(&e.arena, c, max_rows + 64, null);
        t.ids = try e.arena.buffer(max_rows * 4);
        // a learned folder's committed blocks (loaded by the engine) carried over, then the learner's Sites own the buffers
        const keep = l.committed;
        const a0 = try gpa.dupe(f32, l.aRows()[0 .. keep * living.in]);
        defer gpa.free(a0);
        const b0 = try gpa.dupe(f32, l.bRows()[0 .. keep * living.out]);
        defer gpa.free(b0);
        t.sites = try lw_sites.Sites.init(gpa, living.in, living.out, .{ .a = l.aRows(), .b = l.bRows(), .tau = l.tau.slice(f32, living.max_blocks) });
        errdefer t.sites.deinit();
        if (keep > 0) try t.sites.adopt(a0, b0, keep);
        t.saved = keep;
        t.publish();
        std.log.info("slide: GLM-5.3-Flash learns at layers.44 shared-expert down_proj: {d} committed ranks carried over", .{keep});
        return t;
    }

    pub fn deinit(t: *Trainer, gpa: std.mem.Allocator) void {
        var it = t.cache.valueIterator();
        while (it.next()) |c| c.free(gpa);
        t.cache.deinit(gpa);
        t.sites.deinit();
        gpa.free(t.proj);
        gpa.free(t.tmp);
        gpa.free(t.norm);
        gpa.destroy(t);
    }

    /// The Sites' rank and committed ranks told to the engine's change (its forward reads l.rank).
    fn publish(t: *Trainer) void {
        t.lw.rank = t.sites.rank;
        var plain: usize = 0;
        while (plain < t.sites.rank and t.lw.gate(plain / lm.block).* == -std.math.inf(f32)) plain += lm.block;
        t.lw.committed = plain;
    }

    /// The change in the served forward from here on (off: the stock forward).
    pub fn attach(t: *Trainer, on: bool) void {
        t.e.sync();
        t.publish();
        t.e.livingAttach(on);
    }

    /// Ranks [0, ranks) kept in the model folder. v2 (when the engine has `living.spawnShard`): only the
    /// ranks after those already saved, as a NEW append-only living-NNNN.safetensors listed in the index; earlier shards
    /// are never rewritten. v1 fallback: the single living_weights.safetensors with every rank.
    pub fn write(t: *Trainer, ranks: usize) !usize {
        t.e.sync();
        defer t.publish();
        if (comptime @hasDecl(living, "spawnShard")) {
            if (ranks <= t.saved) return 0;
            const n = ranks - t.saved;
            const topic: []const u8 = if (std.c.getenv("LW_TOPIC")) |v| std.mem.span(v) else "chat";
            const bytes = try living.spawnShard(t.gpa, t.e.dir, t.lw.aRows()[t.saved * living.in ..][0 .. n * living.in], t.lw.bRows()[t.saved * living.out ..][0 .. n * living.out], n, topic);
            t.saved = ranks;
            return bytes;
        }
        const bytes = try living.saveRows(t.gpa, t.e.dir, t.lw.aRows(), t.lw.bRows(), ranks);
        t.saved = ranks;
        return bytes;
    }

    pub fn projected(t: *const Trainer, k: usize, rows: usize) []const f32 {
        _ = k;
        return t.proj[0 .. rows * lm.candidates];
    }

    fn tail(t: *const Trainer) lm.Tail {
        return .{ .norm = t.norm, .head = t.head, .eps = t.e.c.eps };
    }

    pub fn step(t: *Trainer, ids: []const u32, start: usize, mode: Mode) !lm.Result {
        if (ids.len < 2 or start < 1 or start >= ids.len or ids.len - 1 > max_rows) return error.BadExample;
        if ((mode == .grad or mode == .learn) and t.sites.rank == 0) return error.NoOpenBlock;
        const cap = try t.captured(ids, start);
        const site = &t.sites.list[0];
        const t0 = std.c.mach_absolute_time();
        defer t.seconds_cpu += @as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) / 24e6;
        switch (mode) {
            .avoid, .seek => {
                const y = if (mode == .avoid) site.avoid else site.seek;
                const k: usize = if (mode == .avoid) lm.avoid_dims else lm.candidates;
                const seed: u32 = if (mode == .avoid) 1 else 2;
                const first = start - 1;
                lm.sketch(cap.site, y, cap.rows, site.in, k, @intCast(t.sketched), seed, 1);
                lm.sketch(cap.site[first * site.in ..], y, 1, site.in, k, @intCast((1 << 30) + t.sketched + first), seed, @floatFromInt(cap.rows - first));
                t.sketched += cap.rows;
                return .{ .loss = 0, .recalled = false };
            },
            .project => {
                lm.projectRows(cap.site, site.seek, t.proj, cap.rows, site.in, lm.candidates, t.tmp);
                return .{ .loss = 0, .recalled = false };
            },
            .loss => return lm.step(t.gpa, t.tail(), cap, t.sites.lora(), null, 0),
            .grad, .learn => {
                const r = try lm.step(t.gpa, t.tail(), cap, t.sites.lora(), site.gb, t.sites.first());
                if (mode == .learn) t.sites.adam();
                return r;
            },
        }
    }

    fn key(ids: []const u32, start: usize) u64 {
        var h = std.hash.Wyhash.init(0x4c57);
        h.update(std.mem.sliceAsBytes(ids));
        h.update(std.mem.asBytes(&start));
        return h.final();
    }

    /// The example's capture, from the cache or one GPU forward.
    pub fn captured(t: *Trainer, ids: []const u32, start: usize) !*const lm.Captured {
        const k = key(ids, start);
        if (t.cache.getPtr(k)) |c| return c;
        if (t.cache_bytes > cache_cap) {
            var it = t.cache.valueIterator();
            while (it.next()) |c| c.free(t.gpa);
            t.cache.clearRetainingCapacity();
            t.cache_bytes = 0;
        }
        var c = try lm.Captured.alloc(t.gpa, ids.len - 1, start, living.in, t.e.c.hidden);
        errdefer c.free(t.gpa);
        const t0 = std.c.mach_absolute_time();
        try t.forward(ids, &c);
        targets(&c, ids);
        t.seconds_gpu += @as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) / 24e6;
        t.captures += 1;
        try t.cache.put(t.gpa, k, c);
        t.cache_bytes += c.bytes();
        return t.cache.getPtr(k).?;
    }

    /// Where layer 44's values sit after a backbone pass (prompt chunk or decode window scratch).
    const Places = struct { site: Ref, ys: Ref, branch: Ref, post: Ref, comb: Ref, old: Ref };

    /// One forward of `ids` (the change off) through the backbone into `c`: a prompt chunk, or 16-row decode windows.
    fn forward(t: *Trainer, ids: []const u32, c: *lm.Captured) !void {
        const e = t.e;
        const rows: u32 = @intCast(ids.len - 1);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        e.sync();
        e.livingAttach(false);
        defer e.livingAttach(true);
        t.state.reset();
        @memcpy(ge.Engine.u32s(t.ids, rows), ids[0..rows]);
        var x = e.ctx();
        x.s = &t.state;
        if (e.pr) |*pr| {
            const b = e.begin();
            var px = x;
            px.sc = &pr.streams;
            prompt_mod.backbone(pr, &px, b.enc, t.ids, rows, 0);
            try e.finish(b.cb, b.enc);
            t.take(c, .{ .site = pr.sact, .ys = pr.ys, .branch = pr.streams.branch, .post = pr.streams.post, .comb = pr.streams.comb, .old = pr.streams.x[1 - px.xi] }, 0, rows);
            return;
        }
        var at: u32 = 0;
        while (at < rows) {
            const n = @min(st.max_rows, rows - at);
            const b = e.begin();
            fwd.backbone(&x, b.enc, t.ids.at(@as(usize, at) * 4), n, at);
            fwd.flipKda(&x);
            try e.finish(b.cb, b.enc);
            t.take(c, .{ .site = e.sc.acts, .ys = e.sc.ys, .branch = e.sc.branch, .post = e.sc.post, .comb = e.sc.comb, .old = e.sc.x[1 - x.xi] }, at, n);
            at += n;
        }
    }

    /// Rows [at, at + n) of the pass into the capture: every row's site input, the answer rows' tail inputs.
    fn take(t: *Trainer, c: *lm.Captured, p: Places, at: u32, n: u32) void {
        const D = c.d;
        const in = c.in;
        const site: [*]const u16 = @ptrCast(@alignCast(p.site.addr()));
        @memcpy(c.site[@as(usize, at) * in ..][0 .. @as(usize, n) * in], site[0 .. @as(usize, n) * in]);
        const ys: [*]const u16 = @ptrCast(@alignCast(p.ys.addr()));
        const br: [*]const u16 = @ptrCast(@alignCast(p.branch.addr()));
        const post: [*]const f32 = @ptrCast(@alignCast(p.post.addr()));
        const comb: [*]const f32 = @ptrCast(@alignCast(p.comb.addr()));
        const old: [*]const u16 = @ptrCast(@alignCast(p.old.addr()));
        _ = t;
        for (0..n) |j| {
            const r = at + j; // the row in the example
            if (r + 1 < c.start) continue; // a question row: no loss there
            const i = r + 1 - c.start; // its answer row
            for (0..D) |d| {
                c.ys0[i * D + d] = lm.bfVal(ys[j * D + d]);
                c.branch0[i * D + d] = lm.bfVal(br[j * D + d]);
            }
            @memcpy(c.post[i * 4 ..][0..4], post[j * 4 ..][0..4]);
            const cm = comb[j * 16 ..][0..16];
            for (0..4) |s| for (0..D) |d| {
                // the expand's mix of the old streams, in its order: c[0 s] x0, then fma c[t s] xt
                var mm = cm[0 * 4 + s] * lm.bfVal(old[(j * 4 + 0) * D + d]);
                inline for (1..4) |tt| mm = @mulAdd(f32, cm[tt * 4 + s], lm.bfVal(old[(j * 4 + tt) * D + d]), mm);
                c.cold[(i * 4 + s) * D + d] = mm;
            };
        }
    }

    /// The answer rows' targets (kept apart from `take`, which sees only the pass's buffers).
    fn targets(c: *lm.Captured, ids: []const u32) void {
        for (0..c.answers()) |i| c.targets[i] = ids[c.start + i];
    }

    /// A check for the bench (tf-glm-lw-check): the GPU's final hidden rows with the change ON against the CPU tail's,
    /// on the answer rows of `ids`; returns the largest absolute difference and the largest |value|.
    pub fn checkTail(t: *Trainer, ids: []const u32, start: usize) ![2]f32 {
        const e = t.e;
        const cap = try t.captured(ids, start);
        const A = cap.answers();
        const D = cap.d;
        const cpu = try t.gpa.alloc(f32, A * D);
        defer t.gpa.free(cpu);
        try lm.hiddenRows(t.gpa, t.tail(), cap, t.sites.lora(), cpu);
        const rows: u32 = @intCast(ids.len - 1);
        e.sync();
        t.publish();
        e.livingAttach(true);
        t.state.reset();
        @memcpy(ge.Engine.u32s(t.ids, rows), ids[0..rows]);
        var x = e.ctx();
        x.s = &t.state;
        const pr = &(e.pr orelse return error.NeedsPromptChunks);
        const b = e.begin();
        var px = x;
        px.sc = &pr.streams;
        prompt_mod.backbone(pr, &px, b.enc, t.ids, rows, 0);
        try e.finish(b.cb, b.enc);
        const h: [*]const u16 = @ptrCast(@alignCast(pr.streams.hidden.addr()));
        var worst: f32 = 0;
        var big: f32 = 0;
        for (0..A) |i| for (0..D) |d| {
            const g = lm.bfVal(h[(start - 1 + i) * D + d]);
            worst = @max(worst, @abs(g - cpu[i * D + d]));
            big = @max(big, @abs(g));
        };
        return .{ worst, big };
    }
};

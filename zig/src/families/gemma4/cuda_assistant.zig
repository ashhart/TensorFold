//! Gemma 4's MTP assistant on CUDA (#1048 step 10): chains of drafts over the target's own last caches, as Metal's.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const cfg = @import("cuda_config.zig");
const fwd = @import("cuda_forward.zig");
const kern = @import("cuda_kernels.zig");
const be = @import("cuda_backend.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Affine4 = cuda.qlinear.Affine4;
const Pick = cuda.tree_round.Pick;
const Window = cuda.window_attention.Window;

/// Drafts a round at most (the reference's default; the head was trained for 3).
pub const depth = 6;
/// Streams one hold takes.
const max_items = 32;
const aw = @import("cuda_assistant_weights.zig");
pub const Config = aw.Config;
pub const isAssistant = aw.isAssistant;
const Weights = aw.Weights;
const load = aw.load;

/// One stream's state: the target's final-normed last kept row, its position, the token after it.
const Lane = struct {
    h: cuda.DeviceBuffer,
    position: u64 = 0,
    token: u32 = 0,
    rows: bool = false, // a row was absorbed
    held: [depth]u32 = @splat(0),
    held_len: u32 = 0,
};

/// A stream's chain in a hold: its lane, its target cache slot, the pending token, the kept row's position.
const Item = struct { lane: *Lane, slot: usize, pending: u32, position: u64, depth: u32 };

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    m: *fwd.Model,
    k: *const kern.Kernels,
    gb: *be.Gemma,
    s: cuda.Stream,
    c: Config,
    w: Weights,
    lanes: std.AutoHashMapUnmanaged(*Stream, *Lane) = .empty,
    /// The target's layers each kind reads: its last sliding and its last global layer.
    sources: [2]usize,
    tap: [1]u32,
    step_ms: f64 = 0,
    /// The target's forward ms by window width (1 to 16), timed at load: the depth rule prices drafts by them.
    costs: [be.max_window]lanes.config.Cost = undefined,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    x: Scratch,

    const Scratch = struct { ids: u64, pos: u64, slot: u64, picks: u64, e: u64, hin: u64, cat: u64, xs: u64, h: [2]u64, hs: u64, hn: u64, normed: u64, nmlp: u64, qraw: u64, q: u64, kj: u64, att: u64, o: u64, gate: u64, up: u64, act: u64, y: u64, part: u64, logits: u64 };

    pub fn init(gpa: std.mem.Allocator, io: std.Io, gb: *be.Gemma, dir: []const u8) !*Drafter {
        const m = gb.m;
        const geo = m.geo;
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        const c = try Config.parse(gpa, bytes, geo);
        const x = try gpa.create(Drafter);
        errdefer gpa.destroy(x);
        x.* = .{ .gpa = gpa, .io = io, .d = gb.d, .m = m, .k = gb.k, .gb = gb, .s = gb.s, .c = c, .w = undefined, .sources = .{ 0, 0 }, .tap = .{@intCast(m.layers - 1)}, .x = undefined };
        for (0..m.layers) |i| x.sources[@intFromBool(cfg.global(i))] = i;
        x.w = try load(gpa, io, x.d, x.s, x.k, dir, &c, geo);
        errdefer x.w.deinit(gpa);
        errdefer x.freeScratch();
        const n = max_items;
        const hid = c.hidden;
        const qw = c.heads * cfg.global_dim;
        const wide: Window = .{ .h = c.heads, .hk = geo.global_kv_heads, .d = cfg.global_dim, .win = 0, .ring = m.context, .scale = 1 };
        const narrow: Window = .{ .h = c.heads, .hk = geo.sliding_kv_heads, .d = cfg.sliding_dim, .win = cfg.window, .ring = fwd.ring, .scale = 1 };
        const part = @max(cuda.window_attention.WindowAttention.scratchBytes(wide, 16), cuda.window_attention.WindowAttention.scratchBytes(narrow, 16));
        x.x = .{
            .ids = try x.alloc(n * 4),
            .pos = try x.alloc(n * 4),
            .slot = try x.alloc(n * 4),
            .picks = try x.alloc(depth * n * @sizeOf(Pick)),
            .e = try x.alloc(n * geo.hidden * 2),
            .hin = try x.alloc(n * geo.hidden * 2),
            .cat = try x.alloc(n * 2 * geo.hidden * 2),
            .xs = try x.alloc(n * (@max(2 * geo.hidden, @max(c.intermediate, qw)) / 64) * 4),
            .h = .{ try x.alloc(n * hid * 2), try x.alloc(n * hid * 2) },
            .hs = try x.alloc(n * geo.hidden * 2),
            .hn = try x.alloc(n * hid * 2),
            .normed = try x.alloc(n * hid * 2),
            .nmlp = try x.alloc(n * hid * 2),
            .qraw = try x.alloc(n * qw * 2),
            .q = try x.alloc(n * qw * 2),
            .kj = try x.alloc(n * cfg.global_dim * 2),
            .att = try x.alloc(n * qw * 2),
            .o = try x.alloc(n * hid * 2),
            .gate = try x.alloc(n * c.intermediate * 2),
            .up = try x.alloc(n * c.intermediate * 2),
            .act = try x.alloc(n * c.intermediate * 2),
            .y = try x.alloc(n * hid * 2),
            .part = try x.alloc(part),
            .logits = try x.alloc(n * cfg.vocab * 2),
        };
        try x.warm();
        return x;
    }

    fn alloc(x: *Drafter, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(x.d, @max(bytes, 256));
        errdefer {
            var y = b;
            y.free();
        }
        try x.buffers.append(x.gpa, b);
        return b.ptr;
    }

    fn freeScratch(x: *Drafter) void {
        for (x.buffers.items) |*b| b.free();
        x.buffers.deinit(x.gpa);
    }

    pub fn deinit(x: *Drafter) void {
        x.s.synchronize() catch {};
        var it = x.lanes.valueIterator();
        while (it.next()) |l| dropLane(x, l.*);
        x.lanes.deinit(x.gpa);
        x.freeScratch();
        x.w.deinit(x.gpa);
        x.gpa.destroy(x);
    }

    pub fn drafter(x: *Drafter) dr.Drafter {
        return .{ .ptr = x, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    /// The target's last layer: its output before the final norm, which absorb applies.
    fn tapsFn(ptr: *anyopaque) []const u32 {
        return &cast(ptr).tap;
    }

    /// Chains of up to 6, one assistant forward a draft: the rule prices each draft and may choose none.
    fn factsFn(ptr: *anyopaque) dr.Facts {
        const x = cast(ptr);
        return .{ .depth = depth, .step_ms = x.step_ms, .batched = true, .stream_bytes = x.m.geo.hidden * 2, .window_costs = &x.costs };
    }

    fn newLane(x: *Drafter) !*Lane {
        const l = try x.gpa.create(Lane);
        errdefer x.gpa.destroy(l);
        l.* = .{ .h = try cuda.DeviceBuffer.alloc(x.d, x.m.geo.hidden * 2) };
        return l;
    }

    fn dropLane(x: *Drafter, l: *Lane) void {
        l.h.free();
        x.gpa.destroy(l);
    }

    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = cast(ptr);
        if (x.lanes.contains(s)) return error.DraftStreamOpen;
        const l = try x.newLane();
        errdefer x.dropLane(l);
        try x.lanes.put(x.gpa, s, l);
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const x = cast(ptr);
        if (x.lanes.fetchRemove(s)) |e| {
            x.s.synchronize() catch {}; // a queued chain may still read the lane's row
            x.dropLane(e.value);
        }
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const x = cast(ptr);
        for (streams, out) |s, o| {
            const l = x.lanes.get(s) orelse return error.DraftStreamUnknown;
            if (o.len > l.held_len) return error.DraftHeldFewer;
            @memcpy(o, l.held[0..o.len]);
        }
    }

    /// Each stream's last absorbed row through the target's final norm into its lane, queued on the target's stream.
    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const x = cast(ptr);
        const width = x.m.geo.hidden * 2;
        for (items) |it| {
            const l = x.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != width or f.rows == 0 or it.follow.len < f.rows) return error.DraftFeaturesUnsupported;
            const last = f.rows - 1;
            try x.k.norms.addRmsTail(x.s, f.buffer + f.offset + last * f.row_bytes, null, x.m.w.final_norm, x.x.hs, l.h.ptr, null, 1, x.m.geo.hidden, cfg.eps, .{ .round = true });
            l.position = it.start + last;
            l.token = it.follow[last];
            l.rows = true;
            l.held_len = 0;
        }
    }

    /// x [rows, l.k] (its group sums at xs) through `l` into out [rows, l.n], in verify tiles.
    fn project(x: *Drafter, l: Affine4, in: u64, xs: u64, out: u64, rows: usize) !void {
        const tile = x.k.affine.verifyRows();
        var at: usize = 0;
        while (at < rows) : (at += tile) {
            const n = @min(tile, rows - at);
            try x.k.affine.verify(x.s, in + at * l.k * 2, xs + at * (l.k / 64) * 4, l, out + at * l.n * 2, n);
        }
    }

    fn copy2d(x: *Drafter, dst: u64, dst_pitch: usize, src: u64, src_pitch: usize, width: usize, rows: usize) !void {
        const p: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = src_pitch, .dst = dst, .dst_pitch = dst_pitch, .width = width, .height = rows };
        try x.d.check(x.d.api.cuMemcpy2DAsync_v2(&p, x.s.handle), "cuMemcpy2DAsync");
    }

    /// Assistant layer i on the rows: input normed in normed (sums in xs), residual h[cur] into h[1 - cur].
    fn layer(x: *Drafter, i: usize, cur: usize, rows: usize) !void {
        const c = &x.c;
        const geo = x.m.geo;
        const l = &x.w.layers[i];
        const s = x.x;
        const global = !c.sliding[i];
        const dh = c.headDim(i);
        const qw = c.heads * dh;
        try x.project(l.q, s.normed, s.xs, s.qraw, rows);
        // q alone: k reads q's first head into scratch (no cache written), v none
        const src: cuda.qk_rows.Src = .{ .q = s.qraw, .q_ld = @intCast(qw), .q_hs = @intCast(dh), .k = s.qraw, .k_ld = @intCast(qw), .k_hs = @intCast(dh) };
        const shape: cuda.qk_rows.Shape = .{ .norms = l.q_norm, .norm_k = l.q_norm, .norms_bf16 = true, .inv = if (global) x.m.inv_global else x.m.inv_sliding, .rot = dh, .eps = cfg.eps, .round = true };
        try x.k.qk.rowsFrom(x.s, src, .{ .h = c.heads, .hk = 1, .dim = dh }, shape, s.pos, null, .{ .q = s.q, .k = s.kj, .v = 0 }, rows);
        const t = x.sources[@intFromBool(global)];
        const len = x.m.cacheLen(t);
        const w: Window = .{ .h = c.heads, .hk = geo.kvHeads(t), .d = dh, .win = if (global) 0 else cfg.window, .ring = len, .scale = 1 };
        var r: usize = 0;
        while (r < rows) : (r += 16) {
            const n = @min(16, rows - r);
            try x.k.win.decode(x.s, s.q + r * qw * 2, x.m.kc[t], x.m.vc[t], s.att + r * qw * 2, s.pos + r * 4, s.slot + r * 4, w, s.part, n);
        }
        try x.k.act.groupSums(x.s, s.att, qw, s.xs, rows, qw);
        try x.project(l.o, s.att, s.xs, s.o, rows);
        try x.k.norms.addRmsTail(x.s, s.o, s.h[cur], l.pre_feedforward, s.hn, s.nmlp, s.xs, rows, c.hidden, cfg.eps, .{ .round = true, .branch = l.post_attention });
        try x.project(l.gate, s.nmlp, s.xs, s.gate, rows);
        try x.project(l.up, s.nmlp, s.xs, s.up, rows);
        try x.k.act.geglu(x.s, s.gate, c.intermediate, s.up, c.intermediate, s.act, s.xs, rows, c.intermediate);
        try x.project(l.down, s.act, s.xs, s.y, rows);
        const next = if (i + 1 < c.layers) x.w.layers[i + 1].input_norm else x.w.norm;
        try x.k.norms.addRmsTail(x.s, s.y, s.hn, next, s.h[1 - cur], s.normed, s.xs, rows, c.hidden, cfg.eps, .{ .round = true, .branch = l.post_feedforward, .scalar = l.scalar });
    }

    /// Every item's chain of `steps` drafts: step s's picks at picks[s * max_items + row], queued on the stream.
    fn chain(x: *Drafter, items: []const Item, steps: u32) !void {
        const geo = x.m.geo;
        const s = x.x;
        const rows = items.len;
        const hid = geo.hidden;
        var host: [3 * max_items]i32 = undefined;
        for (items, 0..) |it, r| {
            host[r] = @intCast(it.pending);
            host[max_items + r] = @intCast(it.position);
            host[2 * max_items + r] = @intCast(it.slot);
        }
        for ([_]u64{ s.ids, s.pos, s.slot }, 0..) |dst, j| try x.d.check(x.d.api.cuMemcpyHtoDAsync_v2(dst, &host[j * max_items], rows * 4, x.s.handle), "cuMemcpyHtoDAsync");
        for (items, 0..) |it, r| try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(s.hin + r * hid * 2, it.lane.h.ptr, hid * 2, x.s.handle), "cuMemcpyDtoDAsync");
        for (0..steps) |step| {
            if (step > 0) try x.nextIds(step - 1, rows);
            try x.k.embed.affine(x.s, s.ids, x.m.w.embed_w, x.m.w.embed_s, x.m.w.embed_b, s.e, rows, hid, .{ .bits = 4, .group = cfg.group, .scale = fwd.embedScale(hid) });
            try x.copy2d(s.cat, 4 * hid, s.e, 2 * hid, 2 * hid, rows);
            try x.copy2d(s.cat + 2 * hid, 4 * hid, s.hin, 2 * hid, 2 * hid, rows);
            try x.k.act.groupSums(x.s, s.cat, 2 * hid, s.xs, rows, 2 * hid);
            try x.project(x.w.pre, s.cat, s.xs, s.h[0], rows);
            try x.k.norms.addRmsTail(x.s, s.h[0], null, x.w.layers[0].input_norm, s.hn, s.normed, s.xs, rows, x.c.hidden, cfg.eps, .{ .round = true });
            var cur: usize = 0;
            for (0..x.c.layers) |i| {
                try x.layer(i, cur, rows);
                cur = 1 - cur;
            }
            try x.project(x.w.head, s.normed, s.xs, s.logits, rows);
            try x.k.round.argmax(x.s, s.logits, s.picks + step * max_items * @sizeOf(Pick), rows, cfg.vocab, cfg.vocab);
            if (step + 1 < steps) try x.project(x.w.post, s.normed, s.xs, s.hin, rows);
        }
    }

    /// Step `step`'s picks as the next step's ids, read on the host first: a row with no finite logit stops the hold.
    fn nextIds(x: *Drafter, step: usize, rows: usize) !void {
        var picks: [max_items]Pick = undefined;
        try x.s.synchronize();
        try x.d.check(x.d.api.cuMemcpyDtoH_v2(&picks, x.x.picks + step * max_items * @sizeOf(Pick), rows * @sizeOf(Pick)), "cuMemcpyDtoH");
        for (picks[0..rows]) |p| if (p.token >= cfg.vocab) return error.DraftNonfinite;
        try x.copy2d(x.x.ids, 4, x.x.picks + step * max_items * @sizeOf(Pick), @sizeOf(Pick), 4, rows);
    }

    /// Each stream's chain of `depth` drafts after `pending`, all streams' rows through each step together.
    fn holdFn(ptr: *anyopaque, holds: []const dr.Hold) anyerror!void {
        const x = cast(ptr);
        if (holds.len > max_items) return error.DraftTooManyStreams;
        var items: [max_items]Item = undefined;
        var steps: u32 = 0;
        for (holds, items[0..holds.len]) |h, *it| {
            const l = x.lanes.get(h.stream) orelse return error.DraftStreamUnknown;
            const t = x.gb.slotOf(h.stream) orelse return error.DraftStreamUnknown;
            if (h.depth == 0 or h.depth > depth) return error.DraftDepth;
            if (h.lanes != null) return error.DraftTreesUnsupported;
            if (!l.rows or l.position + 2 != h.position or t.len < h.position - 1) return error.DraftContextGap;
            it.* = .{ .lane = l, .slot = t.slot, .pending = h.pending, .position = l.position, .depth = h.depth };
            steps = @max(steps, h.depth);
        }
        try x.chain(items[0..holds.len], steps);
        var picks: [depth * max_items]Pick = undefined;
        try x.s.synchronize();
        try x.d.check(x.d.api.cuMemcpyDtoH_v2(&picks, x.x.picks, steps * max_items * @sizeOf(Pick)), "cuMemcpyDtoH");
        for (items[0..holds.len], 0..) |it, r| {
            for (it.lane.held[0..it.depth], 0..) |*tok, step| {
                const p = picks[step * max_items + r];
                if (p.nonfinite != 0 or p.token >= cfg.vocab) return error.DraftNonfinite;
                tok.* = p.token;
            }
            it.lane.held_len = it.depth;
        }
    }

    /// Chains of 1 and 6 drafts over slot 0's caches (no stream holds them yet), timed: one draft's cost.
    fn warm(x: *Drafter) !void {
        const l = try x.newLane();
        defer x.dropLane(l);
        var ms: [2]f64 = undefined;
        for ([_]u32{ depth, 1, depth, 1, depth }, 0..) |steps, i| {
            const t0 = std.Io.Clock.awake.now(x.io);
            try x.chain(&.{.{ .lane = l, .slot = 0, .pending = 2, .position = 128, .depth = steps }}, steps);
            try x.s.synchronize();
            if (i >= 3) ms[i - 3] = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(x.io)).toNanoseconds())) / 1e6;
        }
        x.step_ms = @max(ms[1] - ms[0], 0) / @as(f64, depth - 1);
        // the target's windows of 1 to 16 rows on slot 0 (no stream holds it yet), the best of three each
        const tokens: [be.max_window]u32 = @splat(2);
        for (&x.costs, 1..) |*c, n| {
            var best: f64 = std.math.inf(f64);
            for (0..4) |i| {
                const t0 = std.Io.Clock.awake.now(x.io);
                try x.gb.run(&.{.{ .slot = 0, .p0 = 128, .tokens = tokens[0..n] }}, false, 0, n);
                try x.s.synchronize();
                const t = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(x.io)).toNanoseconds())) / 1e6;
                if (i > 0) best = @min(best, t);
            }
            c.* = .{ .width = @intCast(n), .ms = best };
        }
    }
};

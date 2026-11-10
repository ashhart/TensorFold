//! The DFlash2 drafter on HIP behind core/lanes/drafter.zig: a key ring a stream, a block's chain of drafts a hold.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const engine = @import("../engine/engine.zig");
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const context = @import("context.zig");
pub const select = @import("select.zig");
pub const frame = @import("frame.zig");
const draft_prior = @import("../../../draft_prior.zig");

const E = engine.Engine;
const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ctx = context.Ctx;

/// Rows an absorb runs a launch chunk: the most a gather takes by value.
pub const absorb_rows = hip.draft_ops.most_rows;
/// Streams one frame runs at most.
pub const max_streams = 16;

/// A stream's context and the ring it owns.
const Owned = struct { buf: hip.DeviceBuffer, ctx: Ctx };

/// Wall time the drafter's calls took (each ends on the host; a hold waits for its drafts).
pub const Times = struct { absorbs: usize = 0, absorb_ns: u64 = 0, holds: usize = 0, hold_ns: u64 = 0 };

pub const Draft = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    e: *E,
    c: config.Config,
    w: weights.Weights,
    /// Streams a frame runs together.
    frame_streams: usize,
    arena: hip.Arena,
    layout: frame.Layout,
    scalars: hip.HostBuffer,
    scalars_dev: hip.DeviceBuffer,
    /// An absorb's positions went up and the stream has not been waited on since.
    queued: bool = false,
    /// One zero tap row: what the warmups absorb.
    blank: hip.DeviceBuffer,
    /// A frame's downloads widened: top logits and selector rows.
    logits: []f32,
    projected: []f32,
    streams: std.AutoHashMapUnmanaged(*Stream, *Owned) = .empty,
    pool: std.ArrayList(*Owned) = .empty,
    /// One more draft's cost: a hold's quickest of several at load over the block's drafts.
    step_ms: f64 = 1.0,
    costs: [16]lanes.config.Cost = undefined,
    n_costs: usize = 0,
    times: Times = .{},

    /// The drafter of `dir` on the target `e`, whose taps must already be the drafter's; frames of `streams` at most.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *E, dir: []const u8, streams: usize) !*Draft {
        const m = e.model();
        if (streams == 0 or streams > max_streams) return error.BadDraftFrame;
        if (m.tp != null or m.act.size() != 2) return error.UnsupportedTarget;
        const c = try config.Config.read(gpa, io, dir, config.Target.of(m.spec));
        if (!std.mem.eql(u32, c.tapLayers(), m.taps.layers[0..m.taps.n])) return error.TapsMismatch;
        if (m.head.n != c.vocab or c.block - 1 > context.most_held) return error.UnsupportedDraftConfig;
        const d = try gpa.create(Draft);
        errdefer gpa.destroy(d);
        d.* = .{
            .gpa = gpa,
            .io = io,
            .e = e,
            .c = c,
            .w = undefined,
            .frame_streams = streams,
            .arena = undefined,
            .layout = frame.Layout.of(c, streams),
            .scalars = undefined,
            .scalars_dev = undefined,
            .blank = undefined,
            .logits = &.{},
            .projected = &.{},
        };
        d.w = try weights.load(gpa, io, &e.driver, dir, weights.Shape.of(c));
        errdefer d.w.deinit();
        const drafts = streams * (c.block - 1);
        d.logits = try gpa.alloc(f32, drafts * c.topk);
        errdefer gpa.free(d.logits);
        d.projected = try gpa.alloc(f32, drafts * c.rank);
        errdefer gpa.free(d.projected);
        d.scalars = try hip.HostBuffer.alloc(&e.driver, d.layout.total);
        errdefer d.scalars.free();
        d.scalars_dev = try hip.DeviceBuffer.alloc(&e.driver, d.layout.total);
        errdefer d.scalars_dev.free();
        d.blank = try hip.DeviceBuffer.alloc(&e.driver, c.n_taps * c.hidden * 2);
        errdefer d.blank.free();
        try d.blank.fill8(0);
        d.arena = try hip.Arena.init(&e.driver, try d.scratchBytes());
        errdefer d.arena.deinit();
        // every draft row's top-k, uploaded once
        const ks = std.mem.bytesAsSlice(i32, d.scalars.bytes[d.layout.ks..][0 .. 4 * drafts]);
        @memset(@as([]align(1) i32, ks), @intCast(c.topk));
        try d.scalars_dev.upload(d.layout.ks, d.scalars.bytes[d.layout.ks..][0 .. 4 * drafts]);
        // the first context, and a hold timed on it (the quickest of several: the first ones pay the driver's stalls)
        const owned = try d.create();
        errdefer d.drop(owned);
        var quickest: f64 = std.math.inf(f64);
        for (0..8) |_| {
            const t0 = std.Io.Clock.awake.now(io);
            try d.warm(&owned.ctx);
            quickest = @min(quickest, ms(io, t0));
        }
        d.step_ms = quickest / @as(f64, @floatFromInt(c.block - 1));
        owned.ctx.clear();
        try d.pool.append(gpa, owned);
        d.times = .{};
        return d;
    }

    pub fn deinit(d: *Draft) void {
        d.e.stream.synchronize() catch {};
        var it = d.streams.valueIterator();
        while (it.next()) |o| d.drop(o.*);
        d.streams.deinit(d.gpa);
        for (d.pool.items) |o| d.drop(o);
        d.pool.deinit(d.gpa);
        d.arena.deinit();
        d.blank.free();
        d.scalars_dev.free();
        d.scalars.free();
        d.gpa.free(d.projected);
        d.gpa.free(d.logits);
        d.w.deinit();
        d.gpa.destroy(d);
    }

    fn net(d: *const Draft) frame.Net {
        return .{ .c = d.c, .w = &d.w, .m = d.e.model(), .act = d.e.model().act };
    }

    /// The arena an absorb chunk or a full frame takes at most, counted on the counting stream.
    fn scratchBytes(d: *Draft) !usize {
        var arena = hip.Arena.counting();
        var o = d.e.ops(&arena);
        o.stream = hip.counting;
        const fake = arena.base;
        try frame.absorb(o, d.net(), .{
            .taps = fake,
            .tap_rows = 1,
            .start = 0,
            .rows = null,
            .n = absorb_rows,
            .put = 0,
            .pos = fake,
            .ring = fake,
        });
        const most = arena.peak;
        arena.reset();
        _ = try frame.forward(o, d.net(), d.layout, fake, d.frame_streams);
        return @max(most, arena.peak);
    }

    /// Device bytes a context holds.
    pub fn streamBytes(d: *const Draft) usize {
        return d.c.ringBytes();
    }

    /// A new context, its ring zeroed and run once (an absorb, then a hold), so no request pays the first launches.
    fn create(d: *Draft) !*Owned {
        const o = try d.gpa.create(Owned);
        errdefer d.gpa.destroy(o);
        o.buf = try hip.DeviceBuffer.alloc(&d.e.driver, d.c.ringBytes());
        errdefer o.buf.free();
        try o.buf.fill8(0);
        o.ctx = .{ .ring = o.buf.base() };
        try d.warm(&o.ctx);
        o.ctx.clear();
        return o;
    }

    /// An absorb of the blank row at position 0 and a full hold after it.
    fn warm(d: *Draft, ctx: *Ctx) !void {
        ctx.clear();
        try d.absorbRows(ctx, d.blank.base(), 1, 0, 1, null);
        const item: dr.Hold = .{ .stream = undefined, .pending = 0, .position = 2, .depth = @intCast(d.c.block - 1) };
        try d.frameOf(&.{item}, &.{ctx});
    }

    fn drop(d: *Draft, o: *Owned) void {
        o.buf.free();
        d.gpa.destroy(o);
    }

    /// `n` contexts ready in the pool, for the streams the server admits.
    pub fn reserve(d: *Draft, n: usize) !void {
        try d.pool.ensureTotalCapacity(d.gpa, n);
        while (d.pool.items.len < n) d.pool.appendAssumeCapacity(try d.create());
    }

    /// The target's measured window cost by rows, for the round loop's prices.
    pub fn setCost(d: *Draft, rows: u32, cost_ms: f64) void {
        if (d.n_costs < d.costs.len) {
            d.costs[d.n_costs] = .{ .width = rows, .ms = cost_ms };
            d.n_costs += 1;
        }
    }

    /// Target rows from position `start` of a tap ring of `tap_rows` rows into the context (`rows`: start + rows[i]).
    pub fn absorbRows(
        d: *Draft,
        ctx: *Ctx,
        taps: u64,
        tap_rows: usize,
        start: u64,
        n: usize,
        rows: ?[]const u32,
    ) !void {
        if (rows) |rs| if (rs.len != n or n > d.c.window) return error.BadDraftRows;
        if (n == 0) return;
        const span = context.open(ctx, start, n, d.c.window);
        const skip: usize = @intCast(span.start - start);
        const l = d.layout;
        // an earlier absorb's positions may still be going up from the pinned buffer
        if (d.queued) try d.e.stream.synchronize();
        const pos = std.mem.bytesAsSlice(i32, d.scalars.bytes[l.absorb..][0 .. 4 * span.n]);
        try context.absorbPositions(span.start, @alignCast(pos));
        try hip.raw.upload(d.scalars_dev, l.absorb, d.scalars.bytes[l.absorb..][0 .. 4 * span.n], d.e.stream.handle);
        d.queued = true;
        var done: usize = 0;
        while (done < span.n) {
            const k = @min(span.n - done, absorb_rows);
            d.arena.reset();
            const chunk: frame.Chunk = .{
                .taps = taps,
                .tap_rows = tap_rows,
                .start = if (rows != null) start else span.start + done,
                .rows = if (rows) |rs| rs[skip + done ..][0..k] else null,
                .n = k,
                .put = span.start + done,
                .pos = d.scalars_dev.base() + l.absorb + 4 * done,
                .ring = ctx.ring,
            };
            try frame.absorb(d.e.ops(&d.arena), d.net(), chunk);
            done += k;
        }
        context.close(ctx, span, d.c.window);
    }

    /// Frames of the holds, `frame_streams` at a time; each context then holds its chain.
    fn frames(d: *Draft, items: []const dr.Hold, contexts: []const *Ctx) !void {
        var at: usize = 0;
        while (at < items.len) : (at += d.frame_streams) {
            const n = @min(d.frame_streams, items.len - at);
            try d.frameOf(items[at..][0..n], contexts[at..][0..n]);
        }
    }

    fn frameOf(d: *Draft, items: []const dr.Hold, contexts: []const *Ctx) !void {
        const c = d.c;
        const l = d.layout;
        const st = d.scalars.bytes;
        const B = items.len * c.block;
        for (contexts) |ctx| ctx.held_n = 0;
        // a frame's regions are its own (an absorb's positions may still be going up); it waits at its end
        const ids = std.mem.bytesAsSlice(u32, st[l.ids..][0 .. 4 * B]);
        const pos = std.mem.bytesAsSlice(i32, st[l.pos..][0 .. 4 * B]);
        const segs = std.mem.bytesAsSlice(context.Segment, st[l.segs..][0 .. @sizeOf(context.Segment) * items.len]);
        const shape: context.Shape = .{ .block = c.block, .vocab = c.vocab, .mask = c.mask };
        try context.stage(shape, items, contexts, @alignCast(ids), @alignCast(pos), @alignCast(segs));
        const s = d.e.stream.handle;
        try hip.raw.upload(d.scalars_dev, l.ids, st[l.ids..][0 .. 4 * B], s);
        try hip.raw.upload(d.scalars_dev, l.pos, st[l.pos..][0 .. 4 * B], s);
        try hip.raw.upload(d.scalars_dev, l.segs, st[l.segs..][0 .. @sizeOf(context.Segment) * items.len], s);
        d.arena.reset();
        const out = try frame.forward(d.e.ops(&d.arena), d.net(), l, d.scalars_dev.base(), items.len);
        const rows = items.len * (c.block - 1);
        try d.fetch(l.top_ids, out.top_ids, 4 * rows * c.topk);
        try d.fetch(l.top_vals, out.top_vals, 2 * rows * c.topk);
        try d.fetch(l.hp, out.hp, 2 * rows * c.rank);
        try d.e.stream.synchronize();
        d.queued = false;
        const top_ids = std.mem.bytesAsSlice(u32, st[l.top_ids..][0 .. 4 * rows * c.topk]);
        const act = d.e.model().act;
        widen(act, std.mem.bytesAsSlice(u16, st[l.top_vals..][0 .. 2 * rows * c.topk]), d.logits[0 .. rows * c.topk]);
        widen(act, std.mem.bytesAsSlice(u16, st[l.hp..][0 .. 2 * rows * c.rank]), d.projected[0 .. rows * c.rank]);
        const per = c.block - 1;
        for (items, contexts, 0..) |it, ctx, i| {
            const cand: select.Candidates = .{
                .apart = .{
                    .ids = @alignCast(top_ids[i * per * c.topk ..][0 .. per * c.topk]),
                    .logits = d.logits[i * per * c.topk ..][0 .. per * c.topk],
                    .projected = d.projected[i * per * c.rank ..][0 .. per * c.rank],
                },
                .topk = c.topk,
                .rank = c.rank,
                .predecessor = d.w.predecessor,
                .successor = d.w.successor,
                .sampling = it.sampling,
                .position = it.position,
            };
            select.chain(cand, it.pending, ctx.held[0..per]);
            ctx.held_n = @min(it.depth, per);
        }
    }

    /// `n` device bytes at `src` into the pinned scalars at `at`, queued on the stream.
    fn fetch(d: *Draft, at: usize, src: u64, n: usize) !void {
        const api = d.e.driver.api;
        const s = d.e.stream.handle;
        try hip.runtime.check(api.hipMemcpyDtoHAsync(d.scalars.bytes[at..].ptr, @ptrFromInt(src), n, s));
    }

    // the Drafter contract

    pub fn drafter(d: *Draft) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{
            .taps = tapsFn,
            .facts = factsFn,
            .open = openFn,
            .absorb = absorbFn,
            .hold = holdFn,
            .held = heldFn,
            .release = releaseFn,
        } };
    }

    fn cast(p: *anyopaque) *Draft {
        return @ptrCast(@alignCast(p));
    }

    fn tapsFn(p: *anyopaque) []const u32 {
        return cast(p).c.tapLayers();
    }

    fn factsFn(p: *anyopaque) dr.Facts {
        const d = cast(p);
        const costs = d.costs[0..d.n_costs];
        return .{
            .depth = @intCast(d.c.block - 1),
            .step_ms = d.step_ms,
            .prior = &draft_prior.prior,
            .batched = d.frame_streams > 1,
            .stream_bytes = d.streamBytes(),
            .window_costs = costs,
            .lane_costs = costs,
        };
    }

    fn openFn(p: *anyopaque, s: *Stream) anyerror!void {
        const d = cast(p);
        if (d.streams.contains(s)) return error.StreamOpen;
        const o = d.pool.pop() orelse try d.create();
        errdefer d.pool.append(d.gpa, o) catch d.drop(o);
        o.ctx.clear();
        try d.streams.put(d.gpa, s, o);
    }

    fn absorbFn(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(p);
        const t0 = std.Io.Clock.awake.now(d.io);
        defer d.timed(&d.times.absorbs, &d.times.absorb_ns, t0);
        const m = d.e.model();
        const dtype: lanes.backend.Features.Dtype = if (m.act == .bf16) .bf16 else .f16;
        for (items) |it| {
            const o = d.streams.get(it.stream) orelse return error.UnknownStream;
            const start = try absorbStart(it, d.c.n_taps * d.c.hidden * 2, dtype);
            const n = if (it.rows) |rs| rs.len else it.features.rows;
            try d.absorbRows(&o.ctx, it.features.buffer, m.taps.rows, start, n, it.rows);
        }
    }

    fn holdFn(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(p);
        const t0 = std.Io.Clock.awake.now(d.io);
        defer d.timed(&d.times.holds, &d.times.hold_ns, t0);
        var contexts: [64]*Ctx = undefined;
        var at: usize = 0;
        while (at < items.len) : (at += contexts.len) {
            const part = items[at..][0..@min(contexts.len, items.len - at)];
            for (part, contexts[0..part.len]) |it, *ctx| {
                ctx.* = &(d.streams.get(it.stream) orelse return error.UnknownStream).ctx;
            }
            try d.frames(part, contexts[0..part.len]);
        }
    }

    fn heldFn(p: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(p);
        for (streams, out) |s, o| {
            const owned = d.streams.get(s) orelse return error.UnknownStream;
            if (o.len > owned.ctx.held_n) return error.TooFewDrafts;
            @memcpy(o, owned.ctx.held[0..o.len]);
        }
    }

    fn releaseFn(p: *anyopaque, s: *Stream) void {
        const d = cast(p);
        if (d.streams.fetchRemove(s)) |kv| d.pool.append(d.gpa, kv.value) catch d.drop(kv.value);
    }

    fn timed(d: *Draft, count: *usize, total: *u64, t0: std.Io.Timestamp) void {
        const ns: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(d.io)).toNanoseconds());
        count.* += 1;
        total.* += ns;
    }
};

/// An absorb's first row's position: its `start`, which the target's features (its tap ring) name in `offset` too.
pub fn absorbStart(it: dr.Absorb, row_bytes: usize, dtype: lanes.backend.Features.Dtype) !u64 {
    const f = it.features;
    if (f.space != .device or f.dtype != dtype or f.row_bytes != row_bytes) return error.UnsupportedFeatures;
    // hip_lanes.zig's features put their first row's position in `offset`: rows of another position are refused
    if (f.offset != it.start) return error.FeaturesNotAtStart;
    return it.start;
}

fn ms(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
}

/// 16-bit patterns of `kind` as f32.
pub fn widen(kind: hip.ops.Kind, in: []align(1) const u16, out: []f32) void {
    for (in, out) |h, *o| o.* = if (kind == .f16) @floatCast(@as(f16, @bitCast(h))) else @bitCast(@as(u32, h) << 16);
}

test {
    _ = select;
    _ = context;
    _ = frame;
}

test "an absorb starts at its own start, and refuses features whose first row is another position" {
    const f: lanes.backend.Features = .{ .buffer = 0x1000, .offset = 40, .rows = 3, .row_bytes = 64, .space = .device, .dtype = .bf16 };
    var it: dr.Absorb = .{ .stream = undefined, .features = f, .start = 40, .follow = &.{} };
    try std.testing.expectEqual(@as(u64, 40), try absorbStart(it, 64, .bf16));
    it.start = 41;
    try std.testing.expectError(error.FeaturesNotAtStart, absorbStart(it, 64, .bf16));
    it.start = 40;
    try std.testing.expectError(error.UnsupportedFeatures, absorbStart(it, 64, .f16));
    try std.testing.expectError(error.UnsupportedFeatures, absorbStart(it, 128, .bf16));
}

test "16-bit logits widen by their kind" {
    var out: [2]f32 = undefined;
    widen(.bf16, &.{ 0x3f80, 0xc000 }, &out);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, -2.0 }, &out);
    widen(.f16, &.{ 0x3c00, 0xc000 }, &out);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, -2.0 }, &out);
}

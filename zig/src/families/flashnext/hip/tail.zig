//! Flash Next's one-stream HIP tail: unchanged callbacks recorded with their current pointers and scalars.
const hip = @import("hip");
const Engine = @import("engine.zig").Engine;
const forward = @import("forward.zig");
const cache = @import("cache.zig");
const head = @import("head.zig");
const be = @import("lanes").backend;

pub const Tail = @import("tail_graph.zig").Tail;

const Mark = struct { host: usize, arena: usize, pooled: usize = 0 };
fn save(e: *Engine) Mark {
    return .{ .host = e.host.used, .arena = e.arena.used };
}
fn restore(e: *Engine, m: Mark) void {
    e.host.used = m.host;
    e.arena.used = m.arena;
}

pub fn beforeRound(e: *Engine) !void {
    if (e.tail) |*t| return t.beforeRound(e.stream);
    return e.stream.synchronize();
}

pub fn beforeDraft(e: *Engine) !void {
    if (e.tail) |*t| return t.beforeDraft(e.stream);
    return e.stream.synchronize();
}

pub fn afterRead(e: *Engine) void {
    if (e.tail) |*t| t.afterRead();
}

pub fn verify(f: anytype, windows: []const be.Window, done: @import("engine.zig").Done, argmax: bool) !void {
    const Ctx = struct {
        f: @TypeOf(f),
        windows: []const be.Window,
        done: @import("engine.zig").Done,
        argmax: bool,
        pub fn save(c: @This()) Mark {
            return @import("tail.zig").save(c.f.e);
        }
        pub fn restore(c: @This(), m: Mark) void {
            @import("tail.zig").restore(c.f.e, m);
        }
        fn body(c: @This()) anyerror!void {
            const e = c.f.e;
            const m = &e.weights.model;
            const row_bytes = m.dims.wide() * m.act.size();
            var off: usize = 0;
            for (c.windows) |w| {
                const lane = c.f.lanes.get(w.stream).?;
                try hip.raw.copy(lane.streams, 0, c.done.out.streams.ptr + off * row_bytes, w.rows() * row_bytes, e.stream.handle);
                off += w.rows();
            }
            if (c.argmax) try e.ops(&e.arena).argmaxRows(c.done.logits, off, e.drawer.vocab, e.drawer.argmaxAt());
        }
    };
    const ctx: Ctx = .{ .f = f, .windows = windows, .done = done, .argmax = argmax };
    if (f.e.tail) |*t| return t.submit(void, .{ .phase = .verify, .rows = windows.len }, f.e.stream, ctx, Ctx.body);
    return Ctx.body(ctx);
}

pub fn keep(e: *Engine, c: *cache.Caches, kept: usize, window_rows: usize) !void {
    const Ctx = struct {
        e: *Engine,
        c: *cache.Caches,
        kept: usize,
        window_rows: usize,
        pub fn save(ctx: @This()) Mark {
            return @import("tail.zig").save(ctx.e);
        }
        pub fn restore(ctx: @This(), m: Mark) void {
            @import("tail.zig").restore(ctx.e, m);
        }
        fn body(ctx: @This()) anyerror!void {
            return ctx.e.keepDevice(ctx.c, ctx.kept, ctx.window_rows);
        }
    };
    const ctx: Ctx = .{ .e = e, .c = c, .kept = kept, .window_rows = window_rows };
    if (e.tail) |*t| {
        try t.submit(void, .{ .phase = .keep, .rows = kept }, e.stream, ctx, Ctx.body);
        if (e.weights.model.gdn) |plans| if (plans.planFor(c.id)) |plan| {
            if (plan.rows > 1) try t.kept(e.stream);
        };
        return;
    }
    return Ctx.body(ctx);
}

pub fn run(e: *Engine, o: hip.ops.Ops, segs: []const forward.Seg, streams: hip.ops.Tensor, ones: hip.DeviceBuffer) !head.Out {
    const Ctx = struct {
        e: *Engine,
        o: hip.ops.Ops,
        segs: []const forward.Seg,
        streams: hip.ops.Tensor,
        ones: hip.DeviceBuffer,
        pub fn save(ctx: @This()) Mark {
            var m = @import("tail.zig").save(ctx.e);
            m.pooled = ctx.segs[0].c.pooled;
            return m;
        }
        pub fn restore(ctx: @This(), m: Mark) void {
            @import("tail.zig").restore(ctx.e, m);
            ctx.segs[0].c.pooled = m.pooled;
        }
        fn body(ctx: @This()) anyerror!head.Out {
            return head.run(ctx.o, false, &ctx.e.weights.model, &ctx.e.host, ctx.segs, ctx.streams, ctx.ones);
        }
    };
    const ctx: Ctx = .{ .e = e, .o = o, .segs = segs, .streams = streams, .ones = ones };
    if (e.tail) |*t| {
        if (segs.len != 1) return error.TailSegments;
        return t.submit(head.Out, .{ .phase = .head, .rows = segs[0].tokens.len }, e.stream, ctx, Ctx.body);
    }
    return Ctx.body(ctx);
}

pub fn draws(e: *Engine, o: hip.ops.Ops, mixed: hip.ops.Tensor, ids: []const u32, out: []u32, probs: []f32) !void {
    const Ctx = struct {
        e: *Engine,
        o: hip.ops.Ops,
        mixed: hip.ops.Tensor,
        pub fn save(ctx: @This()) Mark {
            return @import("tail.zig").save(ctx.e);
        }
        pub fn restore(ctx: @This(), m: Mark) void {
            @import("tail.zig").restore(ctx.e, m);
        }
        fn body(ctx: @This()) anyerror!head.Drawn {
            return head.drawQueued(ctx.o, &ctx.e.weights.model, &ctx.e.host, ctx.e.stream, ctx.mixed, 1, false);
        }
    };
    if (e.tail) |*t| {
        const got = try t.submit(head.Drawn, .{ .phase = .draw, .rows = 1 }, e.stream, Ctx{ .e = e, .o = o, .mixed = mixed }, Ctx.body);
        try head.finishDraws(e.stream, got, ids, out, probs);
        t.afterRead();
        return;
    }
    return head.draws(o, &e.weights.model, &e.host, e.stream, mixed, 1, ids, out, probs, false);
}

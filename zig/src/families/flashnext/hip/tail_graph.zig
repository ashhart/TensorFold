//! Exact tail submissions: refresh every node's live arguments before reusing an executable graph.
const std = @import("std");
const hip = @import("hip");

pub const Key = struct { phase: enum { verify, keep, head, draw }, rows: usize };
const Graphs = hip.graph_cache.CacheOf(Key, u8, hip.graph_split.Split);

pub const Tail = struct {
    graphs: Graphs,
    fence: hip.Event,
    host_done: bool = false,
    plan_pending: bool = false,

    pub fn enabled(on: bool, graphs: bool, world: usize, streams: usize, decline: bool) bool {
        return on and graphs and world == 1 and streams == 1 and !decline;
    }

    pub fn init(gpa: std.mem.Allocator, r: *const hip.Runtime) !Tail {
        return .{ .graphs = Graphs.init(gpa), .fence = try hip.Event.init(r, false) };
    }

    pub fn deinit(t: *Tail) void {
        t.graphs.deinit();
        t.fence.deinit();
    }

    /// The caller drained the stream before releasing any lane's buffers.
    pub fn clear(t: *Tail) void {
        const gpa = t.graphs.gpa;
        t.graphs.deinit();
        t.graphs = Graphs.init(gpa);
        t.host_done = false;
        t.plan_pending = false;
    }

    /// A draw downloaded its result and synchronized: all earlier staging reads finished too.
    pub fn afterRead(t: *Tail) void {
        t.host_done = true;
        t.plan_pending = false;
    }

    pub fn beforeRound(t: *Tail, s: hip.Stream) !void {
        if (!t.host_done) {
            try s.synchronize();
        } else if (t.plan_pending) {
            try t.fence.synchronize();
        }
        t.host_done = false;
        t.plan_pending = false;
    }

    /// Keep only reads its own plan source, so it does not delay reuse of the forward's host staging.
    pub fn beforeDraft(t: *Tail, s: hip.Stream) !void {
        if (!t.host_done) try s.synchronize();
        t.host_done = false;
    }

    pub fn kept(t: *Tail, s: hip.Stream) !void {
        t.plan_pending = true;
        t.fence.record(s) catch |err| {
            t.host_done = false;
            return err;
        };
    }

    fn install(t: *Tail, g: hip.graph.Graph, s: hip.Stream) !hip.graph_split.Split {
        var x = try g.instantiate();
        errdefer x.deinit();
        try x.upload(s);
        const pieces = try t.graphs.gpa.alloc(hip.graph_split.Piece, 1);
        pieces[0] = .{ .graph = x };
        return .{ .gpa = t.graphs.gpa, .pieces = pieces, .comm = null };
    }

    /// Capturing runs host bookkeeping once, but no device work; a failed capture restores it before eager fallback.
    pub fn submit(t: *Tail, comptime Out: type, key: Key, s: hip.Stream, ctx: anytype, comptime body: fn (@TypeOf(ctx)) anyerror!Out) !Out {
        errdefer t.host_done = false;
        if (key.phase != .keep) t.host_done = false;
        const entry = try t.graphs.find(key);
        if (entry.state == .failed) return body(ctx);
        hip.graph.beginCapture(s, .thread_local) catch {
            t.graphs.revoke(entry);
            return body(ctx);
        };
        const mark = ctx.save();
        const result = body(ctx);
        var g = hip.graph.endCapture(s) catch {
            ctx.restore(mark);
            t.graphs.revoke(entry);
            return body(ctx);
        };
        defer g.deinit();
        const out = result catch |err| {
            ctx.restore(mark);
            t.graphs.revoke(entry);
            return err;
        };
        if (entry.exec) |split| {
            std.debug.assert(split.pieces.len == 1 and split.pieces[0] == .graph);
            const updated: ?hip.abi.ExecUpdateResult = split.pieces[0].graph.update(g) catch null;
            if (updated != null and updated.? == .success) {
                split.launchOn(s) catch |err| {
                    t.graphs.revoke(entry);
                    return err;
                };
                t.graphs.replayed += 1;
                return out;
            }
        }
        const split = t.install(g, s) catch {
            ctx.restore(mark);
            t.graphs.revoke(entry);
            return body(ctx);
        };
        if (entry.exec) |*old| old.deinit();
        t.graphs.hold(entry, split, 0);
        split.launchOn(s) catch |err| {
            t.graphs.revoke(entry);
            return err;
        };
        return out;
    }
};

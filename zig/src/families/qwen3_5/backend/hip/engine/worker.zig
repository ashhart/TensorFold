//! A tensor-parallel rank above 0: runs the steps rank 0 sends in the same order, so the collectives pair up.

const std = @import("std");
const hip = @import("hip");
const serve = @import("hip_serve");
const Engine = @import("engine.zig").Engine;
const Pick = @import("engine.zig").Pick;
const state = @import("../forward/state.zig");

const draw = serve.draw;
const protocol = serve.protocol;

/// Windows a round holds at most.
const max_windows = 128;

/// The rank's engine behind the shared follower: it runs rank 0's prefill, verify and keep words on its own lanes.
pub const Worker = struct {
    pub const Lane = struct {
        caches: state.Caches,
        /// Slots written and kept: the next window starts here.
        len: usize,
        /// The last verify's slot and rows, until rank 0's keep.
        pending: ?serve.backend.Pending = null,
    };

    gpa: std.mem.Allocator,
    e: *Engine,
    reqs: []draw.Request,
    out: []u32,
    /// The linear snapshots of rank 0's prefix tree, in the slots it names.
    slots: serve.snapshots.Slots,

    const Follower = protocol.Follower(Worker);

    pub fn init(gpa: std.mem.Allocator, e: *Engine) !Worker {
        const rows = e.o.batch_rows;
        const reqs = try gpa.alloc(draw.Request, rows);
        errdefer gpa.free(reqs);
        // a follower's draws are greedy and unread: the forward and its collectives are what it shares
        @memset(reqs, .{ .sampling = null, .position = 0 });
        return .{
            .gpa = gpa,
            .e = e,
            .reqs = reqs,
            .out = try gpa.alloc(u32, rows),
            .slots = serve.snapshots.Slots.init(gpa, &e.driver),
        };
    }

    pub fn deinit(w: *Worker) void {
        w.e.stream.synchronize() catch {};
        w.slots.deinit();
        w.gpa.free(w.out);
        w.gpa.free(w.reqs);
    }

    /// Runs rank 0's steps until it says stop.
    pub fn follow(w: *Worker, link: hip.collective.Link) !void {
        var f = Follower.init(w.gpa, w);
        defer f.deinit();
        try f.follow(link);
    }

    pub fn close(w: *Worker, l: *Lane) void {
        w.e.drain();
        l.caches.deinit(w.gpa);
        w.gpa.destroy(l);
    }

    /// prefill: id, positions, prompt len, resumed at, cut count, snapshot slot, page count, the pages, cuts, prompt.
    pub fn prefill(w: *Worker, f: *Follower, m: []const u32) !void {
        const held = m[8..][0..m[7]];
        // the cuts sit between the pages and the prompt
        const prompt = m[8 + m[7] + m[5] ..][0..m[3]];
        f.close(m[1]);
        const lane = try w.gpa.create(Lane);
        errdefer w.gpa.destroy(lane);
        lane.* = .{ .caches = try w.e.emptyCaches(m[2]), .len = prompt.len };
        errdefer lane.caches.deinit(w.gpa);
        try lane.caches.set(w.gpa, 0, held);
        if (m[6] != protocol.no_snapshot) try lane.caches.loadLinear(try w.slots.get(m[6]), w.e.stream.handle);
        try f.begin(m[1], lane, prompt, m[4]);
    }

    /// The prompt's rows [from, to) into the lane; the last chunk draws (unread) as rank 0 does.
    pub fn fill(w: *Worker, l: *Lane, prompt: []const u32, from: usize, to: usize) !void {
        if (to == prompt.len) {
            _ = try w.e.prefill(&l.caches, prompt, from, null, w.reqs[0], null);
            return;
        }
        try w.e.advance(&l.caches, prompt, from, to, null);
    }

    /// verify: graph pick, count, then id, rows, tokens each; every rank derives the plan shape from them.
    pub fn verify(w: *Worker, f: *Follower, m: []const u32) !void {
        const pick: Pick = @fromBackingInt(m[1]);
        const n = m[2];
        if (n > max_windows) return error.WindowTooWide;
        var ids: [max_windows]u32 = undefined;
        var rows: [max_windows]Engine.Rows = undefined;
        var at: usize = 3;
        var total: usize = 0;
        for (0..n) |i| {
            const lane = try f.lane(m[at]);
            ids[i] = m[at];
            rows[i] = .{ .caches = &lane.caches, .pos = lane.len, .tokens = m[at + 2 ..][0..m[at + 1]] };
            total += rows[i].tokens.len;
            at += 2 + m[at + 1];
        }
        _ = try w.e.choose(rows[0..n], pick);
        _ = try w.e.verify(rows[0..n], w.reqs[0..total], w.out[0..total]);
        for (ids[0..n], 0..) |id, i| f.lanes.get(id).?.pending = .{ .slot = i, .rows = rows[i].tokens.len };
    }

    /// keep: count, then id and rows each; the keeps land in one launch.
    pub fn keep(w: *Worker, f: *Follower, m: []const u32) !void {
        for (0..m[1]) |i| {
            const lane = try f.lane(m[2 + 2 * i]);
            const p = lane.pending orelse return error.NothingToKeep;
            w.e.keep(p.slot, m[3 + 2 * i]);
            lane.len += m[3 + 2 * i];
            lane.pending = null;
        }
        try w.e.flush();
    }

    pub fn pages(w: *Worker, l: *Lane, first: usize, ids: []const u32) !void {
        try l.caches.set(w.gpa, first, ids);
    }

    /// Slot `slot` keeps the stream's linear state.
    pub fn snap(w: *Worker, l: *Lane, slot: u32) !void {
        try l.caches.saveLinear(try w.slots.buffer(slot, l.caches.linearBytes()), w.e.stream.handle);
    }

    pub fn copy(w: *Worker, from: u32, to: u32) !void {
        try w.e.pool.copyPage(from, to, w.e.stream.handle);
    }
};

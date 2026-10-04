//! The lanes backend for a cluster: the leader plans each round and broadcasts it; every rank runs its shard of every row.
const std = @import("std");
const lanes = @import("lanes");
const exchange = @import("exchange.zig");
const round = @import("round.zig");

const be = lanes.backend;
const Stream = lanes.Stream;
const Allocator = std.mem.Allocator;

/// One rank's part of the model (Metal, CUDA or the toy): it runs every row of a round through its shard and the collectives.
pub const Shard = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Rank 0 gets each sampled row's logits in `logits` (rows in round order, `vocab` floats each).
        forward: *const fn (ptr: *anyopaque, a: Allocator, x: exchange.Exchange, rows: []const round.Row, logits: ?[]f32) anyerror!void,
        keep: *const fn (ptr: *anyopaque, slot: u32, len: u32) void,
        release: *const fn (ptr: *anyopaque, slot: u32) void,
        vocab: *const fn (ptr: *anyopaque) u32,
    };
};

/// The leader's draft head: `depth` guesses after `history` (whose last token sits at position - 1).
pub const Drafter = struct {
    ptr: *anyopaque,
    guess: *const fn (ptr: *anyopaque, history: []const u32, sampling: ?lanes.Sampling, position: u64, out: []u32) anyerror!void,
};

/// Apply every command the leader broadcasts until it says stop; any rank but 0 runs this.
pub fn follow(gpa: Allocator, x: exchange.Exchange, shard: Shard) !void {
    var views: [exchange.max_ranks][]const u8 = undefined;
    while (true) {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const bytes = try exchange.broadcast(x, 0, &.{}, views[0..x.size()]);
        const cmd = try round.decode(a, try a.dupe(u8, bytes));
        apply(shard, cmd);
        switch (cmd.kind) {
            .round => try shard.vtable.forward(shard.ptr, a, x, cmd.rows, null),
            .release => {},
            .stop => return,
        }
    }
}

fn apply(shard: Shard, cmd: round.Command) void {
    for (cmd.keeps) |k| shard.vtable.keep(shard.ptr, k.slot, k.len);
    for (cmd.releases) |s| shard.vtable.release(shard.ptr, s);
}

const Lane = struct {
    slot: u32,
    /// Every token fed into the slot's caches, prompt first; a verify's rows stay until `keep` trims them.
    history: std.ArrayList(u32) = .empty,
    base: usize = 0,
    rows: std.ArrayList(u32) = .empty,
    held: std.ArrayList(u32) = .empty,
    last: []f32 = &.{},
};

/// Rank 0: the lanes vtable; each forward is one broadcast command and one round on every rank.
pub const Leader = struct {
    gpa: Allocator,
    x: exchange.Exchange,
    shard: Shard,
    drafter: ?Drafter = null,
    lanes_by: std.AutoHashMapUnmanaged(*Stream, Lane) = .empty,
    drawn: std.ArrayList(u32) = .empty,
    keeps: std.ArrayList(round.Keep) = .empty,
    next_slot: u32 = 0,
    rounds: u64 = 0,
    rows_run: u64 = 0,

    pub fn deinit(l: *Leader) void {
        var it = l.lanes_by.valueIterator();
        while (it.next()) |ln| l.freeLane(ln);
        l.lanes_by.deinit(l.gpa);
        l.drawn.deinit(l.gpa);
        l.keeps.deinit(l.gpa);
    }

    fn freeLane(l: *Leader, ln: *Lane) void {
        ln.history.deinit(l.gpa);
        ln.rows.deinit(l.gpa);
        ln.held.deinit(l.gpa);
        l.gpa.free(ln.last);
    }

    pub fn backend(l: *Leader) be.Backend {
        return .{ .ptr = l, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
    }

    /// End every follower's loop.
    pub fn stop(l: *Leader) !void {
        _ = try l.send(.{ .kind = .stop }, null);
    }

    /// One round on every rank: the command (with pending rollbacks), then this rank's shard; sampled rows' logits back.
    pub fn run(l: *Leader, rows: []const round.Row) ![]f32 {
        var sampled: usize = 0;
        for (rows) |r| sampled += @intFromBool(r.sample);
        const logits = try l.gpa.alloc(f32, sampled * l.shard.vtable.vocab(l.shard.ptr));
        errdefer l.gpa.free(logits);
        l.rounds += 1;
        l.rows_run += rows.len;
        try l.send(.{ .kind = .round, .number = l.rounds, .rows = rows }, logits);
        return logits;
    }

    fn send(l: *Leader, c: round.Command, logits: ?[]f32) !void {
        var arena: std.heap.ArenaAllocator = .init(l.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var cmd = c;
        cmd.keeps = try a.dupe(round.Keep, l.keeps.items);
        l.keeps.clearRetainingCapacity();
        const bytes = try round.encode(a, cmd);
        var views: [exchange.max_ranks][]const u8 = undefined;
        _ = try exchange.broadcast(l.x, 0, bytes, views[0..l.x.size()]);
        apply(l.shard, cmd);
        if (cmd.kind == .round) try l.shard.vtable.forward(l.shard.ptr, a, l.x, cmd.rows, logits);
    }

    fn self(ptr: *anyopaque) *Leader {
        return @ptrCast(@alignCast(ptr));
    }

    fn lane(l: *Leader, s: *Stream) *Lane {
        return l.lanes_by.getPtr(s).?;
    }

    /// Greedy rows take the lowest id among the largest logits; sampled rows use the engine's keyed host rule.
    pub fn draw(l: *Leader, row: []const f32, s: ?lanes.Sampling, position: u64) !u32 {
        if (s) |st| {
            const values = try l.gpa.alloc(f64, row.len);
            defer l.gpa.free(values);
            const ids = try l.gpa.alloc(u64, row.len);
            defer l.gpa.free(ids);
            for (values, ids, row, 0..) |*v, *id, x, i| {
                v.* = x;
                id.* = i;
            }
            return @intCast(try lanes.sampling.choose(l.gpa, values, ids, position, st));
        }
        var best: usize = 0;
        for (row, 0..) |v, i| if (v > row[best]) {
            best = i;
        };
        return @intCast(best);
    }

    fn handle(l: *Leader, token: u32) !u64 {
        try l.drawn.append(l.gpa, token);
        return l.drawn.items.len - 1;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const l = self(ptr);
        const got = try l.lanes_by.getOrPut(l.gpa, s);
        if (got.found_existing) l.freeLane(got.value_ptr);
        got.value_ptr.* = .{ .slot = l.next_slot };
        l.next_slot += 1;
        const ln = got.value_ptr;
        try ln.history.appendSlice(l.gpa, s.prompt());
        const rows = try l.gpa.alloc(round.Row, s.prompt().len);
        defer l.gpa.free(rows);
        for (rows, s.prompt(), 0..) |*r, t, i| r.* = .{ .slot = ln.slot, .token = t, .index = @intCast(i), .sample = i + 1 == rows.len };
        ln.last = try l.run(rows);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const l = self(ptr);
        const ln = l.lane(s);
        if (position != ln.history.items.len) return error.PositionMismatch;
        return l.handle(try l.draw(ln.last, s.sampling, position));
    }

    fn value(l: *Leader, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| l.drawn.items[h],
            .value => |v| v,
        };
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const l = self(ptr);
        const ln = l.lane(s);
        const t = l.value(feed);
        if (position != ln.history.items.len + 1) return error.PositionMismatch;
        const logits = try l.run(&.{.{ .slot = ln.slot, .token = t, .index = @intCast(ln.history.items.len), .sample = true }});
        defer l.gpa.free(logits);
        try ln.history.append(l.gpa, t);
        return l.handle(try l.draw(logits, s.sampling, position));
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        return self(ptr).drawn.items[h];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const l = self(ptr);
        var rows: std.ArrayList(round.Row) = .empty;
        defer rows.deinit(l.gpa);
        for (windows, out) |w, o| {
            if (w.parents != null) return error.TreeWindowsUnsupported;
            const ln = l.lane(w.stream);
            ln.base = ln.history.items.len;
            ln.rows.clearRetainingCapacity();
            try ln.rows.append(l.gpa, w.pending);
            try ln.rows.appendSlice(l.gpa, if (w.held > 0) ln.held.items[0..w.held] else w.tokens);
            @memcpy(o.drafts, ln.rows.items[1..]);
            for (ln.rows.items, 0..) |t, r| {
                if (w.positions[r] != ln.base + r + 1) return error.PositionMismatch;
                try rows.append(l.gpa, .{ .slot = ln.slot, .token = t, .index = @intCast(ln.base + r), .sample = true });
            }
        }
        const logits = try l.run(rows.items);
        defer l.gpa.free(logits);
        const vocab = l.shard.vtable.vocab(l.shard.ptr);
        var k: usize = 0;
        for (windows, out) |w, o| {
            const ln = l.lane(w.stream);
            for (0..ln.rows.items.len) |r| {
                o.sampled[r] = try l.draw(logits[k * vocab ..][0..vocab], w.stream.sampling, w.positions[r]);
                k += 1;
            }
            try ln.history.appendSlice(l.gpa, ln.rows.items);
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const l = self(ptr);
        for (windows, paths) |w, path| {
            const ln = l.lane(w.stream);
            for (path, 0..) |r, i| if (r != i) return error.TreeWindowsUnsupported;
            ln.history.shrinkRetainingCapacity(ln.base + path.len);
            try l.keeps.append(l.gpa, .{ .slot = ln.slot, .len = @intCast(ln.history.items.len) });
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const l = self(ptr);
        for (requests) |r| {
            const ln = l.lane(r.stream);
            var guess: std.ArrayList(u32) = .empty;
            defer guess.deinit(l.gpa);
            if (r.rows) |path| {
                try guess.appendSlice(l.gpa, ln.history.items[0..ln.base]);
                for (path) |row| try guess.append(l.gpa, ln.rows.items[row]);
            } else try guess.appendSlice(l.gpa, ln.history.items);
            if (r.position != guess.items.len + 1) return error.PositionMismatch;
            try guess.append(l.gpa, if (r.first) |f| l.value(f) else r.follow[r.follow.len - 1]);
            try ln.held.resize(l.gpa, r.depth);
            const d = l.drafter orelse return error.NoDrafter;
            try d.guess(d.ptr, guess.items, r.stream.sampling, r.position, ln.held.items);
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const l = self(ptr);
        const kv = l.lanes_by.fetchRemove(s) orelse return;
        var ln = kv.value;
        _ = l.send(.{ .kind = .release, .releases = &.{ln.slot} }, null) catch {};
        l.freeLane(&ln);
    }
};

test {
    _ = @import("backend_test.zig");
}

//! Tensor parallelism's step protocol: rank 0 decides and sends each step, the ranks behind the link apply it.

const std = @import("std");
const hip = @import("hip");

/// A message's first word: rank 0 decides every match, eviction, page and slot, and a rank only applies it.
pub const Op = enum(u32) { stop, prefill, fill, verify, keep, release, pages, snap, copy };

/// A message's step once it holds the words that step reads; the family checks its own prefill, verify and keep.
pub fn step(m: []const u32) error{LinkFailed}!Op {
    if (m.len == 0) return error.LinkFailed;
    const op = std.enums.fromInt(Op, m[0]) orelse return error.LinkFailed;
    const need: usize = switch (op) {
        .stop, .prefill, .verify, .keep => 1,
        .release => 2,
        .fill, .snap, .copy => 3,
        .pages => 4,
    };
    if (m.len < need or (op == .pages and m[3] > m.len - 4)) return error.LinkFailed;
    return op;
}

/// Rank 0's table from page `first` on: no gap past this rank's `held` pages, inside its `room`, ids in its pool.
pub fn pagesFit(first: usize, ids: []const u32, held: usize, room: usize, pool: usize) bool {
    if (first > held or ids.len > room or first > room - ids.len) return false;
    for (ids) |id| if (id >= pool) return false;
    return true;
}

/// A page rank 0 names, against this rank's pool of `pool` pages (every rank's pool is sized from the same options).
pub fn pageFits(page: u32, pool: usize) bool {
    return page < pool;
}

/// A snapshot slot rank 0 names: one this rank holds, or the next one (rank 0 claims the lowest free slot).
pub fn slotFits(slot: u32, slots: usize) bool {
    return slot <= slots;
}

/// The prefill message's snapshot slot when the pass starts from nothing.
pub const no_snapshot: u32 = std.math.maxInt(u32);

/// Rank 0's side: the other ranks behind the link (none on a lone rank, where every send is a no-op).
pub const Sender = struct {
    gpa: std.mem.Allocator,
    link: ?hip.collective.Link = null,
    /// The family's flag that a rank could not take a round: no step is sent after it.
    stopped: ?*const bool = null,
    msg: std.ArrayList(u32) = .empty,

    pub fn deinit(s: *Sender) void {
        s.msg.deinit(s.gpa);
    }

    pub fn on(s: *const Sender) bool {
        return s.link != null;
    }

    /// A message begun: `op` and its first words; `add` extends it, `finish` sends it.
    pub fn begin(s: *Sender, op: Op, words: []const u32) !void {
        if (s.link == null) return;
        if (s.stopped) |b| if (b.*) return error.GroupStopped;
        s.msg.clearRetainingCapacity();
        try s.msg.append(s.gpa, @backingInt(op));
        try s.msg.appendSlice(s.gpa, words);
    }

    pub fn add(s: *Sender, words: []const u32) !void {
        if (s.link == null) return;
        try s.msg.appendSlice(s.gpa, words);
    }

    pub fn finish(s: *Sender) !void {
        if (s.stopped) |b| if (b.*) return error.GroupStopped;
        if (s.link) |l| try l.send(s.msg.items);
    }

    /// `op` with `parts` joined after it, as one message.
    pub fn send(s: *Sender, op: Op, parts: []const []const u32) !void {
        if (s.link == null) return;
        try s.begin(op, &.{});
        for (parts) |p| try s.add(p);
        try s.finish();
    }

    pub fn stop(s: *Sender) !void {
        try s.send(.stop, &.{});
    }

    /// The prompt pass of lane `id` runs to row `to`.
    pub fn fill(s: *Sender, id: u32, to: usize) !void {
        try s.send(.fill, &.{&.{ id, @intCast(to) }});
    }

    pub fn release(s: *Sender, id: u32) !void {
        try s.send(.release, &.{&.{id}});
    }

    /// Lane `id`'s pages from page `first` on are `ids`.
    pub fn pages(s: *Sender, id: u32, first: usize, ids: []const u32) !void {
        try s.send(.pages, &.{ &.{ id, @intCast(first), @intCast(ids.len) }, ids });
    }

    /// Slot `slot` takes lane `id`'s state.
    pub fn snap(s: *Sender, id: u32, slot: u32) !void {
        try s.send(.snap, &.{&.{ id, slot }});
    }

    /// Page `to` becomes a copy of page `from`.
    pub fn copy(s: *Sender, from: u32, to: u32) !void {
        try s.send(.copy, &.{&.{ from, to }});
    }
};

/// A rank above 0: runs rank 0's steps in order, so the collectives pair up; `F` applies each to its own `F.Lane`s.
pub fn Follower(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The prompt pass in progress on a follower: the prompt rank 0 sent and the row it has reached.
        const Fill = struct { prompt: []u32, at: usize };

        gpa: std.mem.Allocator,
        f: *F,
        lanes: std.AutoHashMapUnmanaged(u32, *F.Lane) = .empty,
        fills: std.AutoHashMapUnmanaged(u32, Fill) = .empty,

        pub fn init(gpa: std.mem.Allocator, f: *F) Self {
            return .{ .gpa = gpa, .f = f };
        }

        pub fn deinit(w: *Self) void {
            var it = w.lanes.valueIterator();
            while (it.next()) |l| F.close(w.f, l.*);
            w.lanes.deinit(w.gpa);
            var fs = w.fills.valueIterator();
            while (fs.next()) |f| w.gpa.free(f.prompt);
            w.fills.deinit(w.gpa);
        }

        pub fn lane(w: *Self, id: u32) !*F.Lane {
            return w.lanes.get(id) orelse error.UnknownStream;
        }

        fn dropFill(w: *Self, id: u32) void {
            if (w.fills.fetchRemove(id)) |kv| w.gpa.free(kv.value.prompt);
        }

        /// Lane `id`'s current lane and prompt pass, if any, are gone (before a new lane takes the id).
        pub fn close(w: *Self, id: u32) void {
            w.dropFill(id);
            const kv = w.lanes.fetchRemove(id) orelse return;
            F.close(w.f, kv.value);
        }

        /// Lane `id` is `l` from now, running `prompt` from row `at` as rank 0 says; on error `l` stays the caller's.
        pub fn begin(w: *Self, id: u32, l: *F.Lane, prompt: []const u32, at: usize) !void {
            const copy = try w.gpa.dupe(u32, prompt);
            errdefer w.gpa.free(copy);
            try w.lanes.ensureUnusedCapacity(w.gpa, 1);
            try w.fills.ensureUnusedCapacity(w.gpa, 1);
            w.close(id);
            w.lanes.putAssumeCapacity(id, l);
            w.fills.putAssumeCapacity(id, .{ .prompt = copy, .at = at });
        }

        /// The next chunk of a prompt pass, to row `to`: rank 0's cuts and chunk ends, run the same way.
        fn fill(w: *Self, id: u32, to: usize) !void {
            const l = try w.lane(id);
            const f = w.fills.getPtr(id) orelse return error.NoPromptPass;
            if (to > f.prompt.len or to <= f.at) return error.PositionMismatch;
            try F.fill(w.f, l, f.prompt, f.at, to);
            f.at = to;
            if (to == f.prompt.len) w.dropFill(id);
        }

        /// Runs rank 0's steps until it says stop.
        pub fn follow(w: *Self, link: hip.collective.Link) !void {
            var msg: std.ArrayList(u32) = .empty;
            defer msg.deinit(w.gpa);
            while (true) {
                try link.recv(w.gpa, &msg);
                const m = msg.items;
                switch (try step(m)) {
                    .stop => return,
                    // prefill, verify and keep are the family's own words: it reads the message whole
                    .prefill => try F.prefill(w.f, w, m),
                    // fill: id, end of the prompt pass's next chunk
                    .fill => try w.fill(m[1], m[2]),
                    .verify => try F.verify(w.f, w, m),
                    .keep => try F.keep(w.f, w, m),
                    .release => w.close(m[1]),
                    // pages: id, first page index, count, then the stream's table from that index on
                    .pages => try F.pages(w.f, try w.lane(m[1]), m[2], m[4..][0..m[3]]),
                    // snap: id, slot that takes the stream's state
                    .snap => try F.snap(w.f, try w.lane(m[1]), m[2]),
                    // copy: from, to; page `to` becomes a copy of page `from`
                    .copy => try F.copy(w.f, m[1], m[2]),
                }
            }
        }
    };
}

/// A link that keeps what rank 0 sent, for the tests.
const Recorder = struct {
    gpa: std.mem.Allocator,
    sent: std.ArrayList([]u32) = .empty,

    fn sendFn(ptr: *anyopaque, words: []const u32) hip.collective.LinkError!void {
        const r: *Recorder = @ptrCast(@alignCast(ptr));
        try r.sent.append(r.gpa, try r.gpa.dupe(u32, words));
    }

    fn recvFn(_: *anyopaque, _: std.mem.Allocator, _: *std.ArrayList(u32)) hip.collective.LinkError!void {
        return error.LinkFailed;
    }

    fn agreeFn(
        _: *anyopaque,
        _: hip.collective.Step,
        _: hip.collective.How,
    ) hip.collective.LinkError!hip.collective.Verdict {
        return error.LinkFailed;
    }

    const vtable: hip.collective.Link.VTable = .{ .send = sendFn, .recv = recvFn, .agree = agreeFn };
};

test "a sender joins the op and its parts, and sends nothing on a lone rank" {
    const gpa = std.testing.allocator;
    var lone: Sender = .{ .gpa = gpa };
    defer lone.deinit();
    try lone.pages(3, 1, &.{ 7, 8 });
    try std.testing.expectEqual(@as(usize, 0), lone.msg.items.len);
    var r: Recorder = .{ .gpa = gpa };
    defer {
        for (r.sent.items) |s| gpa.free(s);
        r.sent.deinit(gpa);
    }
    var s: Sender = .{ .gpa = gpa, .link = .{ .ptr = &r, .vtable = &Recorder.vtable } };
    defer s.deinit();
    try s.pages(3, 1, &.{ 7, 8 });
    try s.begin(.verify, &.{ 9, 2 });
    try s.add(&.{ 10, 11 });
    try s.finish();
    try s.stop();
    try std.testing.expectEqualSlices(u32, &.{ @backingInt(Op.pages), 3, 1, 2, 7, 8 }, r.sent.items[0]);
    try std.testing.expectEqualSlices(u32, &.{ @backingInt(Op.verify), 9, 2, 10, 11 }, r.sent.items[1]);
    try std.testing.expectEqualSlices(u32, &.{@backingInt(Op.stop)}, r.sent.items[2]);
}

test "a sender sends nothing once its family stopped the group" {
    const gpa = std.testing.allocator;
    var r: Recorder = .{ .gpa = gpa };
    defer {
        for (r.sent.items) |s| gpa.free(s);
        r.sent.deinit(gpa);
    }
    var stopped = false;
    var s: Sender = .{ .gpa = gpa, .link = .{ .ptr = &r, .vtable = &Recorder.vtable }, .stopped = &stopped };
    defer s.deinit();
    try s.fill(1, 8);
    stopped = true;
    try std.testing.expectError(error.GroupStopped, s.fill(1, 16));
    try std.testing.expectError(error.GroupStopped, s.stop());
    try std.testing.expectEqual(@as(usize, 1), r.sent.items.len);
}

test "security: a step from the link is refused unless its message holds the words it reads" {
    const t = std.testing;
    try t.expectError(error.LinkFailed, step(&.{}));
    try t.expectError(error.LinkFailed, step(&.{9}));
    try t.expectError(error.LinkFailed, step(&.{0xffff_ffff}));
    try t.expectEqual(Op.stop, try step(&.{0}));
    for ([_]Op{ .fill, .snap, .copy }) |op| {
        try t.expectError(error.LinkFailed, step(&.{ @backingInt(op), 1 }));
        try t.expectEqual(op, try step(&.{ @backingInt(op), 1, 2 }));
    }
    try t.expectError(error.LinkFailed, step(&.{@backingInt(Op.release)}));
    try t.expectEqual(Op.release, try step(&.{ @backingInt(Op.release), 1 }));
    // pages: id, first, count, then that many ids
    const pages = @backingInt(Op.pages);
    try t.expectError(error.LinkFailed, step(&.{ pages, 1, 0 }));
    try t.expectEqual(Op.pages, try step(&.{ pages, 1, 0, 0 }));
    try t.expectEqual(Op.pages, try step(&.{ pages, 1, 0, 2, 7, 8 }));
    try t.expectError(error.LinkFailed, step(&.{ pages, 1, 0, 3, 7, 8 }));
    try t.expectError(error.LinkFailed, step(&.{ pages, 1, 0, 0xffff_ffff, 7 }));
}

test "security: rank 0's page and slot indices are refused past this rank's own pool, table and slots" {
    const t = std.testing;
    // a table of 4 pages held, room for 6, a pool of 100: the largest legal sends pass
    try t.expect(pagesFit(4, &.{ 98, 99 }, 4, 6, 100));
    try t.expect(pagesFit(0, &.{}, 0, 0, 100));
    try t.expect(pagesFit(3, &.{0}, 4, 6, 100));
    try t.expect(!pagesFit(5, &.{0}, 4, 6, 100)); // a gap after the held pages
    try t.expect(!pagesFit(4, &.{ 1, 2, 3 }, 4, 6, 100)); // past the stream's room
    try t.expect(!pagesFit(0xffff_ffff, &.{1}, 4, 6, 100));
    try t.expect(!pagesFit(0, &.{ 1, 100 }, 4, 6, 100)); // a page past the pool
    try t.expect(!pagesFit(0, &.{0xffff_ffff}, 4, 6, 100));
    try t.expect(pageFits(99, 100));
    try t.expect(!pageFits(100, 100));
    try t.expect(!pageFits(0xffff_ffff, 100));
    try t.expect(slotFits(3, 3));
    try t.expect(slotFits(0, 0));
    try t.expect(!slotFits(4, 3));
    try t.expect(!slotFits(0xffff_ffff, 3));
}

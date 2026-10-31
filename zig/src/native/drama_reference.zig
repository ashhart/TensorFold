//! CUDA Drama 3's reference cache: device copies of a lane's state after a voice's invariant prompt rows, LRU bounded.
const std = @import("std");
const d3 = @import("drama3");
const Frame = d3.model.Frame;
const State = d3.kv_state.State;

/// Entries kept without TF_DRAMA_REFERENCE_CACHE.
pub const default_entries: u32 = 2;
/// The most entries TF_DRAMA_REFERENCE_CACHE may ask for.
pub const max_entries: u32 = 16;

/// The rows a prompt resumes after: Metal's boundary on this lane's chunks, so cached and uncached cuts agree.
pub fn savedRows(total: usize, reference_rows: usize, o: d3.common.Options) usize {
    return d3.reference_cache.prefillRows(total, reference_rows, o.max_rows);
}

/// Entries shared by every lane, keyed by their saved rows plus the tower's lookahead row; worker-thread only.
pub fn Cache(comptime Lane: type) type {
    return struct {
        const Self = @This();
        const Entry = struct { key: []Frame, saved: Lane.Saved, used: u64 };

        gpa: std.mem.Allocator = undefined,
        capacity: usize = 0,
        entries: std.ArrayList(Entry) = .empty,
        clock: u64 = 0,
        warned: bool = false,

        /// One prompt round: the first resumes after a cached reference or starts whole; the boundary chunk saves.
        pub fn prompt(c: *Self, lane: *Lane, frames: []const Frame, reference_rows: usize, state: *State) !void {
            const rows = if (c.capacity == 0) 0 else savedRows(frames.len, reference_rows, lane.o);
            if (!state.started) {
                try lane.startPrompt(frames);
                state.prompt = 0;
                if (c.find(frames, rows)) |i| {
                    if (lane.loadReference(c.entries.items[i].saved, rows)) {
                        state.prompt = rows;
                        std.log.info("drama3: reference cache hit ({d} rows restored)", .{rows});
                    } else |err| {
                        std.log.warn("drama3: reference cache restore failed ({s}); prefilling whole", .{@errorName(err)});
                        c.drop(i);
                        try lane.startPrompt(frames);
                    }
                }
                state.started = true;
            }
            const from = state.prompt;
            const count = lane.o.promptRows(frames.len - from);
            try lane.promptChunk(frames.len, from, count, null);
            state.prompt += count;
            if (from < rows and state.prompt == rows) c.store(lane, frames, rows);
        }

        /// The entry saved after `rows` rows whose identity is frames[0..rows + 1], marked used.
        pub fn find(c: *Self, frames: []const Frame, rows: usize) ?usize {
            if (rows == 0) return null;
            std.debug.assert(rows < frames.len);
            const identity = std.mem.sliceAsBytes(frames[0 .. rows + 1]);
            for (c.entries.items, 0..) |*e, i| if (std.mem.eql(u8, std.mem.sliceAsBytes(e.key), identity)) {
                c.clock += 1;
                e.used = c.clock;
                return i;
            };
            return null;
        }

        /// The lane's state after frames[0..rows] as an entry, evicting the least recently used; never fails a request.
        pub fn store(c: *Self, lane: *Lane, frames: []const Frame, rows: usize) void {
            if (c.capacity == 0 or c.find(frames, rows) != null) return;
            if (c.entries.items.len >= c.capacity) c.drop(c.oldest());
            c.fill(lane, frames, rows) catch |err| {
                if (!c.warned) std.log.warn("drama3: reference cache keeps no entry ({s}); misses prefill whole", .{@errorName(err)});
                c.warned = true;
                return;
            };
            std.log.info("drama3: reference cache built ({d} rows, {d} of {d} entries)", .{ rows, c.entries.items.len, c.capacity });
        }

        fn fill(c: *Self, lane: *Lane, frames: []const Frame, rows: usize) !void {
            try c.entries.ensureUnusedCapacity(c.gpa, 1);
            const key = try c.gpa.dupe(Frame, frames[0 .. rows + 1]);
            errdefer c.gpa.free(key);
            var saved = try lane.allocReference(lane.referenceBytes(rows));
            errdefer Lane.freeReference(&saved);
            try lane.saveReference(saved, rows);
            c.clock += 1;
            c.entries.appendAssumeCapacity(.{ .key = key, .saved = saved, .used = c.clock });
        }

        pub fn drop(c: *Self, i: usize) void {
            var e = c.entries.swapRemove(i);
            Lane.freeReference(&e.saved);
            c.gpa.free(e.key);
        }

        fn oldest(c: *const Self) usize {
            var at: usize = 0;
            for (c.entries.items, 0..) |e, i| if (e.used < c.entries.items[at].used) {
                at = i;
            };
            return at;
        }

        /// Frees every entry; device copies need the lanes' context current and their streams idle.
        pub fn deinit(c: *Self) void {
            while (c.entries.items.len > 0) c.drop(c.entries.items.len - 1);
            c.entries.deinit(c.gpa);
            c.* = .{};
        }
    };
}

/// A lane whose recurrent state and cache rows hash every row it ran with its lookahead row and its chunk's cut.
const FakeLane = struct {
    pub const Saved = struct { kv: []u64 }; // cache rows, then the recurrent state
    o: d3.common.Options,
    acc: u64 = 0,
    kv: [64]u64 = @splat(0),
    frames: []const Frame = &.{},
    cuts: std.ArrayList(usize) = .empty,
    fail_alloc: bool = false,
    fail_save: bool = false,
    fail_load: bool = false,

    pub fn startPrompt(l: *FakeLane, frames: []const Frame) !void {
        l.acc = 0;
        l.frames = frames;
        l.cuts.clearRetainingCapacity();
        @memset(&l.kv, 0xa5); // rows from an earlier request
    }

    pub fn promptChunk(l: *FakeLane, n: usize, c0: usize, t: usize, _: ?*const anyopaque) !void {
        try l.cuts.append(std.testing.allocator, c0);
        for (c0..c0 + t) |row| {
            const next: Frame = if (row + 1 < n) l.frames[row + 1] else @splat(-1);
            var h = std.hash.Wyhash.init(l.acc);
            h.update(std.mem.asBytes(&l.frames[row]));
            h.update(std.mem.asBytes(&next));
            h.update(std.mem.asBytes(&[2]usize{ c0, t }));
            for (l.kv[0..row]) |k| h.update(std.mem.asBytes(&k));
            l.acc = h.final();
            l.kv[row] = l.acc;
        }
    }

    pub fn referenceBytes(_: *const FakeLane, rows: usize) usize {
        return rows + 1;
    }

    pub fn allocReference(l: *FakeLane, rows: usize) !Saved {
        if (l.fail_alloc) return error.OutOfMemory;
        return .{ .kv = try std.testing.allocator.alloc(u64, rows) };
    }

    pub fn freeReference(s: *Saved) void {
        std.testing.allocator.free(s.kv);
    }

    pub fn saveReference(l: *FakeLane, s: Saved, rows: usize) !void {
        if (l.fail_save) return error.CudaFailed;
        @memcpy(s.kv[0..rows], l.kv[0..rows]);
        s.kv[rows] = l.acc;
    }

    pub fn loadReference(l: *FakeLane, s: Saved, rows: usize) !void {
        if (l.fail_load) {
            l.acc ^= 1; // a partial restore
            return error.CudaFailed;
        }
        @memcpy(l.kv[0..rows], s.kv[0..rows]);
        l.acc = s.kv[rows];
    }

    fn deinit(l: *FakeLane) void {
        l.cuts.deinit(std.testing.allocator);
    }
};

const TestCache = Cache(FakeLane);

/// Runs a whole prompt; the lane's final state, comparable across cached and uncached runs.
fn run(c: *TestCache, l: *FakeLane, frames: []const Frame, reference_rows: usize) !u64 {
    var state: State = .{};
    while (state.prompt < frames.len) try c.prompt(l, frames, reference_rows, &state);
    try std.testing.expectEqual(frames.len, state.prompt);
    return l.acc;
}

fn voice(seed: i32, text: i32) [40]Frame {
    var frames: [40]Frame = undefined;
    for (&frames, 0..) |*f, i| f.* = @splat(seed * 1000 + @as(i32, @intCast(i)));
    for (frames[30..]) |*f| f[0] = text; // request text after the reference
    return frames;
}

test "drama reference cache: a hit makes the uncached chunk cuts and ends in the uncached state" {
    const t = std.testing;
    for ([_]usize{ 1, 4, 8, 16 }) |chunk| {
        for ([_]usize{ 2, 9, 17, 30, 39 }) |reference_rows| {
            const o: d3.common.Options = .{ .max_rows = chunk, .attn_chunk = chunk };
            var plain: TestCache = .{ .gpa = t.allocator };
            var l: FakeLane = .{ .o = o };
            defer l.deinit();
            const a = voice(1, 7);
            const want = try run(&plain, &l, &a, reference_rows);
            const uncached_cuts = try t.allocator.dupe(usize, l.cuts.items);
            defer t.allocator.free(uncached_cuts);
            var c: TestCache = .{ .gpa = t.allocator, .capacity = 2 };
            defer c.deinit();
            try t.expectEqual(want, try run(&c, &l, &a, reference_rows));
            try t.expectEqualSlices(usize, uncached_cuts, l.cuts.items);
            const rows = savedRows(a.len, reference_rows, o);
            try t.expectEqual(@as(usize, @intFromBool(rows > 0)), c.entries.items.len);
            try t.expectEqual(want, try run(&c, &l, &a, reference_rows));
            const skipped = if (rows == 0) 0 else rows / chunk;
            try t.expectEqualSlices(usize, uncached_cuts[skipped..], l.cuts.items);
            try t.expectEqual(@as(usize, 0), rows % chunk);
            if (rows > 0) try t.expect(rows < reference_rows); // the tower's lookahead row stays in the reference
        }
    }
}

test "drama reference cache: resuming off a chunk boundary would change the state" {
    const t = std.testing;
    const o: d3.common.Options = .{ .max_rows = 4, .attn_chunk = 4 };
    var l: FakeLane = .{ .o = o };
    defer l.deinit();
    const a = voice(1, 7);
    var plain: TestCache = .{ .gpa = t.allocator };
    const want = try run(&plain, &l, &a, 30);
    try l.startPrompt(&a); // a cut at row 29, the last reference row before the lookahead, splits chunk 28
    var at: usize = 0;
    while (at < a.len) {
        const count = if (at < 29) @min(o.promptRows(a.len - at), 29 - at) else o.promptRows(a.len - at);
        try l.promptChunk(a.len, at, count, null);
        at += count;
    }
    try t.expect(want != l.acc);
}

test "drama reference cache: the identity is every saved row and the lookahead row, never the request text" {
    const t = std.testing;
    const o: d3.common.Options = .{ .max_rows = 4, .attn_chunk = 4 };
    var l: FakeLane = .{ .o = o };
    defer l.deinit();
    var c: TestCache = .{ .gpa = t.allocator, .capacity = 4 };
    defer c.deinit();
    var a = voice(1, 7);
    _ = try run(&c, &l, &a, 30);
    const rows = savedRows(a.len, 30, o);
    try t.expectEqual(@as(usize, 28), rows);
    var other = a;
    other[rows + 1][0] = 99;
    try t.expect(c.find(&other, rows) != null);
    a[0][0] = 99; // the key is the cache's copy
    try t.expect(c.find(&a, rows) == null);
    a[0][0] = voice(1, 7)[0][0];
    for (0..rows + 1) |row| for (0..11) |col| {
        other = a;
        other[row][col] += 1;
        try t.expect(c.find(&other, rows) == null);
    };
    try t.expect(c.find(&a, rows - 4) == null);
    try t.expect(c.find(&a, 0) == null);
}

test "drama reference cache: the least recently used entry goes first, and 0 entries keep none" {
    const t = std.testing;
    const o: d3.common.Options = .{ .max_rows = 4, .attn_chunk = 4 };
    var l: FakeLane = .{ .o = o };
    defer l.deinit();
    var c: TestCache = .{ .gpa = t.allocator, .capacity = 2 };
    defer c.deinit();
    const a = voice(1, 7);
    const b = voice(2, 7);
    const v = voice(3, 7);
    for ([_]*const [40]Frame{ &a, &b, &a, &v }) |frames| _ = try run(&c, &l, frames, 30);
    try t.expectEqual(@as(usize, 2), c.entries.items.len);
    try t.expect(c.find(&a, 28) != null);
    try t.expect(c.find(&v, 28) != null);
    try t.expect(c.find(&b, 28) == null);
    c.store(&l, &a, 28); // a second lane reaching the boundary of a stored voice adds nothing
    try t.expectEqual(@as(usize, 2), c.entries.items.len);
    var off: TestCache = .{ .gpa = t.allocator };
    defer off.deinit();
    _ = try run(&off, &l, &a, 30);
    _ = try run(&off, &l, &a, 0);
    try t.expectEqual(@as(usize, 0), off.entries.items.len);
}

test "drama reference cache: failed saves keep no entry, a failed restore drops it, and requests still finish" {
    const t = std.testing;
    const o: d3.common.Options = .{ .max_rows = 4, .attn_chunk = 4 };
    var l: FakeLane = .{ .o = o };
    defer l.deinit();
    const a = voice(1, 7);
    var plain: TestCache = .{ .gpa = t.allocator };
    const want = try run(&plain, &l, &a, 30);
    var c: TestCache = .{ .gpa = t.allocator, .capacity = 2 };
    defer c.deinit();
    for ([_]enum { alloc, save, key }{ .alloc, .save, .key }) |fault| {
        l.fail_alloc = fault == .alloc;
        l.fail_save = fault == .save;
        var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = if (fault == .key) 0 else 99 });
        c.gpa = failing.allocator();
        try t.expectEqual(want, try run(&c, &l, &a, 30));
        try t.expectEqual(@as(usize, 0), c.entries.items.len);
    }
    l.fail_alloc = false;
    l.fail_save = false;
    c.gpa = t.allocator;
    try t.expectEqual(want, try run(&c, &l, &a, 30));
    l.fail_load = true;
    try t.expectEqual(want, try run(&c, &l, &a, 30));
    try t.expectEqual(@as(usize, 10), l.cuts.items.len); // prefilled whole, from row 0
    try t.expectEqual(@as(usize, 0), l.cuts.items[0]);
    try t.expectEqual(@as(usize, 1), c.entries.items.len); // that whole pass saved again
}

//! The lane backend's slots: chosen before a stream's map entry exists, and none left behind by a failed insertion.
const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const Lanes = @import("cuda_lanes.zig").Lanes;
const fw = @import("cuda_forward.zig");
const solo = @import("cuda_prompt_solo.zig");
const prompt_cuts = @import("cuda_prompt_cuts.zig");

test "a lane's slot is chosen before its map entry exists" {
    var storage: [8192]u8 = @splat(0);
    var fixed = std.heap.FixedBufferAllocator.init(&storage);
    var e: engine.Engine = undefined;
    e.gpa = fixed.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var streams: [3]lanes.Stream = undefined;
    const Probe = struct {
        var bad = false;
        fn choose(l: *const Lanes) u32 {
            bad = l.lanes.count() != 0;
            return 0;
        }
    };
    _ = try self.claimUsing(&streams[0], Probe.choose);
    try std.testing.expect(!Probe.bad);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[0])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
    _ = self.lanes.remove(&streams[0]);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[2])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
}

test "a failed lane insertion leaves no entry, and a retry takes slot zero" {
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var e: engine.Engine = undefined;
    e.gpa = fail.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var stream: lanes.Stream = undefined;
    try std.testing.expectError(error.OutOfMemory, self.claim(&stream));
    try std.testing.expectEqual(@as(usize, 0), self.lanes.count());
    fail.fail_index = std.math.maxInt(usize);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&stream)).slot);
}

test "a prompt pass goes solo only while no other stream is past its prompt pass" {
    var e: engine.Engine = undefined;
    e.gpa = std.testing.allocator;
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var streams: [3]lanes.Stream = undefined;
    const first = try self.claim(&streams[0]); // a stream's own lane, fresh as its first chunk starts
    try std.testing.expect(!solo.othersDecode(&self.lanes, &streams[0]));
    first.fill = fw.prefill_rows;
    (try self.claim(&streams[1])).fill = 2 * fw.prefill_rows; // another prompt pass, partway
    try std.testing.expect(!solo.othersDecode(&self.lanes, &streams[0]));
    _ = try self.claim(&streams[2]); // past its prompt pass: decoding
    try std.testing.expect(solo.othersDecode(&self.lanes, &streams[0]) and solo.othersDecode(&self.lanes, &streams[1]));
    try std.testing.expect(!solo.othersDecode(&self.lanes, &streams[2])); // the other two still fill
    _ = self.lanes.remove(&streams[2]); // released
    try std.testing.expect(!solo.othersDecode(&self.lanes, &streams[0]));
}

test "glm request starts: a kept state off the new request's cold cuts, or over other chunks below, passes from 0" {
    const m = fw.prefill_rows;
    var e: engine.Engine = undefined;
    e.gpa = std.testing.allocator;
    e.snaps = .empty;
    defer e.snaps.deinit(e.gpa);
    try e.snaps.put(e.gpa, 7, undefined); // every rank holds state 7; resumeFrom reads no more of it
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    const Case = struct { kept: []const u32, cuts: []const u32, why: ?anyerror, id: u64 = 7, at: u32 = 2 * m };
    const cases = [_]Case{
        .{ .kept = &.{}, .cuts = &.{m + 300}, .why = error.SnapshotOutOfStep }, // 2m lies inside a chunk now
        .{ .kept = &.{300}, .cuts = &.{}, .why = error.SnapshotOutOfStep, .at = 300 }, // a chat's start, a grid pass
        .{ .kept = &.{300}, .cuts = &.{ 300, 2 * m }, .why = null, .at = 300 }, // the same start: it resumes
        .{ .kept = &.{}, .cuts = &.{ 300, 2 * m }, .why = error.SnapshotOutOfStep }, // a grid state, a start below
        .{ .kept = &.{ 300, 2 * m }, .cuts = &.{ 700, 2 * m }, .why = error.SnapshotOutOfStep }, // other starts below
        .{ .kept = &.{ 300, 2 * m }, .cuts = &.{}, .why = error.SnapshotOutOfStep }, // a chat's state, a grid pass
        .{ .kept = &.{}, .cuts = &.{ 2 * m, 2 * m + 500 }, .why = null }, // the starts from 2m on chunk below it alike
        .{ .kept = &.{m}, .cuts = &.{}, .why = null }, // a start on the grid chunks as the grid does
        .{ .kept = &.{ 300, 2 * m }, .cuts = &.{ 300, 2 * m, 3 * m }, .why = null },
        .{ .kept = &.{ 300, 2 * m }, .cuts = &.{ 300, 2 * m }, .why = error.SnapshotStale, .id = 8 }, // no rank holds 8
    };
    for (cases) |c| {
        var k: engine.Kept = .{ .id = c.id, .at = c.at, .below = prompt_cuts.below(c.kept, c.at) };
        var s: lanes.Stream = undefined;
        s.chunks = c.cuts;
        s.reuse = .{ .saved = &k, .at = c.at };
        s.reuse_failed = false;
        s.reuse_error = null;
        s.cached = 0;
        const from = self.resumeFrom(&s, 4 * m);
        try std.testing.expectEqual(c.why, s.reuse_error);
        try std.testing.expectEqual(c.why != null, s.reuse_failed);
        try std.testing.expectEqual(@as(usize, if (c.why == null) c.at else 0), if (from) |r| r.at else 0);
        try std.testing.expectEqual(if (c.why == null) c.at else 0, s.cached); // refused: the pass starts at 0
    }
}

const mtp_shared = @import("cuda_mtp_lanes.zig");
const draftRound = if (@hasDecl(Lanes, "draftRound")) Lanes.draftRound else mtp_shared.draftLanes;

const FakeEngine = struct {
    gpa: std.mem.Allocator,
    mtp: bool = true,
    drafter: ?*anyopaque = null,
    slot: u32 = 0,
    positions: [8]usize = @splat(0),
    pub fn bind(e: *@This(), slot: u32) !void {
        e.slot = slot;
    }
    pub fn pos(e: *const @This()) usize {
        return e.positions[e.slot];
    }
};

const FakeLeader = struct {
    e: *FakeEngine,
    lone_drafts: usize = 0,
    shared_drafts: usize = 0,
    keeps: usize = 0,
    last_parts_len: usize = 0,
    last_slots: [8]u32 = @splat(0),
    last_kept: [8]usize = @splat(0),
    last_depths: [8]usize = @splat(0),

    pub fn draftMessages(x: *const @This()) usize {
        return x.lone_drafts + x.shared_drafts;
    }
    pub fn bind(x: *@This(), slot: u32) !void {
        try x.e.bind(slot);
    }
    pub fn keep(x: *@This(), n: usize) !void {
        x.keeps += 1;
        x.e.positions[x.e.slot] += n;
    }
    pub fn draft(x: *@This(), _: bool, follow_ids: []const u32, depth: usize, out: []u32, chances: ?[]f64) !void {
        x.lone_drafts += 1;
        const base: u32 = if (follow_ids.len > 0) follow_ids[follow_ids.len - 1] else 0;
        for (0..depth) |j| {
            out[j] = base + 1 + @as(u32, @intCast(j));
            if (chances) |c| c[j] = 0.9;
        }
    }
    pub fn draftShared(x: *@This(), parts: []const mtp_shared.DraftPart) !void {
        x.shared_drafts += 1;
        x.last_parts_len = parts.len;
        for (parts, 0..) |p, i| {
            x.last_slots[i] = p.slot;
            x.last_kept[i] = p.kept;
            x.last_depths[i] = p.depth;
            x.e.positions[p.slot] += p.kept;
            const base = p.follow[p.follow.len - 1];
            for (0..p.depth) |j| {
                p.out[j] = base + 1 + @as(u32, @intCast(j));
                if (p.chances) |c| c[j] = 0.9;
            }
        }
    }
};

const FakeLane = struct {
    slot: u32,
    pending_rows: ?usize = null,
    held: [engine.max_drafts]u32 = undefined,
    chances: [engine.max_drafts]f64 = undefined,
    held_n: usize = 0,
    first: ?u64 = null,
};

const FakeBackend = struct {
    x: FakeLeader,
    alone: bool = false,
    mtp_confidence: ?f64 = 0.5,
    shared_verifies: usize = 0,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, FakeLane) = .empty,
    drawn: [64]u32 = undefined,
    next: u64 = 0,

    fn of(ptr: *anyopaque) *@This() {
        return @ptrCast(@alignCast(ptr));
    }
    fn take(self: *@This(), tok: u32) u64 {
        const h = self.next;
        self.drawn[h % self.drawn.len] = tok;
        self.next += 1;
        return h;
    }
    pub fn value(self: *const @This(), feed: lanes.backend.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % self.drawn.len],
            .value => |v| v,
        };
    }
    pub fn bind(self: *@This(), s: *const lanes.Stream, kept: ?usize) !*FakeLane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        try self.x.bind(l.slot);
        if (l.pending_rows) |rows| try self.x.keep(kept orelse rows);
        l.pending_rows = null;
        return l;
    }
    fn backend(self: *@This()) lanes.backend.Backend {
        return .{ .ptr = self, .vtable = &vtable };
    }
    const vtable: lanes.backend.Backend.VTable = .{
        .prefill = prefillFn,
        .first = firstFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .probabilities = probsFn,
        .held = heldFn,
        .release = releaseFn,
    };
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const slot: u32 = @intCast(self.lanes.count());
        try self.lanes.put(self.x.e.gpa, s, .{ .slot = slot, .first = self.take(100 + slot * 10) });
        self.x.e.positions[slot] = s.prompt_len;
    }
    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, _: u64) anyerror!u64 {
        return of(ptr).lanes.getPtr(s).?.first.?;
    }
    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: lanes.backend.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }
    fn readFn(ptr: *anyopaque, h: u64) anyerror!u32 {
        const self = of(ptr);
        return self.drawn[h % self.drawn.len];
    }
    fn verifyFn(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len > 1) self.shared_verifies += 1;
        for (windows, out) |w, *o| {
            const l = try self.bind(w.stream, null);
            const rows = w.rows();
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(o.drafts[0..drafts.len], drafts);
            o.sampled[0] = w.pending + 1;
            for (drafts, 0..) |d, i| o.sampled[1 + i] = d + 1; // accept the first draft, reject the next if any
            if (drafts.len > 0) o.sampled[0] = drafts[0];
            l.pending_rows = rows;
        }
    }
    fn keepFn(ptr: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        for (windows, paths) |w, path| _ = try self.bind(w.stream, path.len);
    }
    fn draftFn(ptr: *anyopaque, requests: []const lanes.backend.DraftRequest) anyerror!void {
        return draftRound(of(ptr), requests);
    }
    fn probsFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const l = of(ptr).lanes.getPtr(s) orelse return false;
        if (l.held_n == 0) return false;
        const n = @min(out.len, l.held_n);
        @memcpy(out[0..n], l.chances[0..n]);
        for (out[n..]) |*c| c.* = 0;
        return true;
    }
    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const l = of(ptr).lanes.getPtr(s) orelse return error.UnknownStream;
        @memcpy(out, l.held[0..out.len]);
    }
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        _ = of(ptr).lanes.remove(s);
    }
};

test "shared head costs: facts().draft_streams is true for shared MTP and false when alone or without MTP" {
    var e: engine.Engine = undefined;
    e.mtp = true;
    e.drafter = null;
    var shared_lanes: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    try std.testing.expect(shared_lanes.facts().draft_streams);
    shared_lanes.alone = true;
    try std.testing.expect(!shared_lanes.facts().draft_streams);
    shared_lanes.alone = false;
    e.mtp = false;
    try std.testing.expect(!shared_lanes.facts().draft_streams);
    e.mtp = true;
    e.drafter = @ptrFromInt(16);
    shared_lanes.alone = true;
    try std.testing.expect(shared_lanes.facts().draft_streams);
}

test "shared head costs: a 3-stream shared MTP round sends one draft message to a fake Leader; lone and alone keep serial draft" {
    const a = std.testing.allocator;
    var real_e: engine.Engine = undefined;
    real_e.mtp = true;
    real_e.drafter = null;
    var real_lanes: Lanes = .{ .x = .{ .e = &real_e, .link = undefined } };
    var cfg = try lanes.Config.init(a, real_lanes.facts(), fw.window_rows, engine.max_drafts);
    defer cfg.deinit(a);
    cfg.timed = false;
    var fe: FakeEngine = .{ .gpa = a };
    var fb: FakeBackend = .{ .x = .{ .e = &fe } };
    defer fb.lanes.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, fb.backend(), clock.clock());
    defer core.deinit();

    var s0 = try lanes.Stream.init(a, .{ .id = "s0", .prompt = &.{ 1, 2, 3, 4 }, .max_new = 16 });
    defer s0.deinit(a);
    var s1 = try lanes.Stream.init(a, .{ .id = "s1", .prompt = &.{ 5, 6, 7, 8, 9 }, .max_new = 16 });
    defer s1.deinit(a);
    var s2 = try lanes.Stream.init(a, .{ .id = "s2", .prompt = &.{ 10, 11, 12 }, .max_new = 16 });
    defer s2.deinit(a);
    try core.addStream(&s0);
    try core.addStream(&s1);
    try core.addStream(&s2);
    fb.x.lone_drafts = 0;
    fb.x.shared_drafts = 0;
    fb.x.keeps = 0;
    fb.shared_verifies = 0;
    _ = try core.step(); // one 3-stream shared verify + draft round
    try std.testing.expectEqual(@as(usize, 1), fb.shared_verifies);
    try std.testing.expectEqual(@as(usize, 1), fb.x.draftMessages());
    try std.testing.expectEqual(@as(usize, 1), fb.x.shared_drafts);
    try std.testing.expectEqual(@as(usize, 0), fb.x.lone_drafts);
    try std.testing.expectEqual(@as(usize, 0), fb.x.keeps);
    try std.testing.expectEqual(@as(usize, 3), fb.x.last_parts_len);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, fb.x.last_slots[0..3]);
    for (fb.x.last_kept[0..3], fb.x.last_depths[0..3]) |kept, depth| {
        try std.testing.expect(kept >= 1);
        try std.testing.expect(depth >= 1);
    }

    // One-stream round keeps today's lone Leader.draft path.
    fb.x.lone_drafts = 0;
    fb.x.shared_drafts = 0;
    const one_rows = [_]u32{0};
    const one_follow = [_]u32{77};
    fb.lanes.getPtr(&s0).?.pending_rows = 1;
    const pos0 = fe.positions[0];
    try draftRound(&fb, &.{.{ .stream = &s0, .follow = &one_follow, .rows = &one_rows, .start = pos0, .position = pos0 + 2, .depth = 2 }});
    try std.testing.expectEqual(@as(usize, 1), fb.x.lone_drafts);
    try std.testing.expectEqual(@as(usize, 0), fb.x.shared_drafts);

    // Alone mode (TF_GLM_SHARED=0) keeps serial Leader.draft even when given 3 requests.
    fb.alone = true;
    fb.x.lone_drafts = 0;
    fb.x.shared_drafts = 0;
    var reqs: [3]lanes.backend.DraftRequest = undefined;
    for ([_]*lanes.Stream{ &s0, &s1, &s2 }, 0..) |s, i| {
        const l = fb.lanes.getPtr(s).?;
        l.pending_rows = 1;
        const p = fe.positions[l.slot];
        reqs[i] = .{ .stream = s, .follow = &one_follow, .rows = &one_rows, .start = p, .position = p + 2, .depth = 2 };
    }
    try draftRound(&fb, &reqs);
    try std.testing.expectEqual(@as(usize, 3), fb.x.lone_drafts);
    try std.testing.expectEqual(@as(usize, 0), fb.x.shared_drafts);

    // A malformed cohort cannot send a wire step or commit a prefix before the total-row refusal.
    fb.alone = false;
    fb.x.lone_drafts = 0;
    fb.x.shared_drafts = 0;
    fb.x.keeps = 0;
    var wide_rows: [17]u32 = undefined;
    const wide_follow: [17]u32 = @splat(77);
    for (&wide_rows, 0..) |*r, i| r.* = @intCast(i);
    var wide: [2]lanes.backend.DraftRequest = undefined;
    for ([_]*lanes.Stream{ &s0, &s1 }, 0..) |s, i| {
        const l = fb.lanes.getPtr(s).?;
        l.pending_rows = 17;
        const p = fe.positions[l.slot];
        wide[i] = .{ .stream = s, .follow = &wide_follow, .rows = &wide_rows, .start = p, .position = p + 18, .depth = 2 };
    }
    try std.testing.expectError(error.WindowTooWide, draftRound(&fb, &wide));
    try std.testing.expectEqual(@as(usize, 0), fb.x.draftMessages());
    try std.testing.expectEqual(@as(usize, 0), fb.x.keeps);
}

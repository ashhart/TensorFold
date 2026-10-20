//! Shared cancellation stays local while scoring preserves every row in its original order.
const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;
const h = lanes.proposal_head;
const score = lanes.proposal_score;
const Stream = lanes.Stream;
const t = std.testing;
const gpa = t.allocator;
comptime {
    if (@hasDecl(lanes.Engine, "askDrafts")) @compileError("draft internals must stay private");
}
const Batch = struct {
    target: lanes.fake.Fake = .{ .gpa = gpa },
    service: *const h.Service,
    after: ?*bool = null,
    expected: []const be.DraftRequest = &.{},
    plans: [2]h.Kind = @splat(.paused),
    depths: [2]u32 = @splat(0),
    scored: usize = 0,
    fn self(ptr: *anyopaque) *Batch {
        const fake: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("target", fake);
    }
    fn clear(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        if (x.target.lanes.getPtr(s)) |l| l.held.clearRetainingCapacity();
    }
    fn run(ptr: *anyopaque, _: *const h.Plan, input: score.Normalized) anyerror!score.Buffer {
        const x: *Batch = @ptrCast(@alignCast(ptr));
        try t.expectEqual(x.expected.len, input.streams.len);
        for (x.expected, input.streams, input.positions) |r, s, at| {
            try t.expectEqual(r.stream, s);
            try t.expectEqual(r.position, at);
        }
        x.scored = input.streams.len;
        if (x.after) |flag| flag.* = true;
        return .{ .address = 0x9000, .rows = input.rows.rows, .stride = lanes.fake.vocab };
    }
    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        x.expected = requests;
        var streams: [2]*Stream = undefined;
        var positions: [2]u64 = undefined;
        for (requests, 0..) |r, i| {
            x.plans[i] = r.plan.?.kind;
            x.depths[i] = r.depth;
            if (r.stream.isCancelled()) try t.expect(r.plan.?.kind == .paused and r.depth == 0 and r.lanes == null and !r.ranks);
            streams[i] = r.stream;
            positions[i] = r.position;
        }
        _ = try x.service.score(&x.service.complete, .{ .rows = .{ .buffer = 0x1000, .rows = @intCast(requests.len), .row_bytes = 16, .space = .host, .dtype = .bf16 }, .streams = streams[0..requests.len], .positions = positions[0..requests.len], .queue = 123, .role = .decode });
        const b = x.target.backend();
        try b.vtable.draft(b.ptr, requests);
    }
};
const Snapshot = struct { plan: h.Kind, depth: u32, held: [2]u32 = @splat(0), count: u32, tokens: [16]u32 = @splat(0), len: usize };
fn round(which: enum { solo, before, after }, first: bool) !Snapshot {
    var service = try h.Service.full(gpa, lanes.fake.vocab, true);
    var batch: Batch = .{ .service = &service };
    defer batch.target.deinit();
    const scorer: score.Scorer = .{ .descriptor = .{ .vocab = lanes.fake.vocab, .hidden = 8, .format = "bf16", .group = 0, .scale_recipe = "none", .space = .host, .device = 0, .queue = 123 }, .generation = null, .selection = .keyed, .ptr = &batch, .run = Batch.run };
    service = try service.withScorer(&scorer);
    var b = batch.target.backend();
    var vt = b.vtable.*;
    vt.proposal_draft = Batch.draft;
    vt.clear_proposals = Batch.clear;
    b.vtable = &vt;
    b.proposal_head = &service;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .mtp = true, .speculate = true, .speculate_early = false, .hidden_rows = true, .draft_streams = true, .drafts = 2 }, 8, 2);
    defer cfg.deinit(gpa);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, b, clock.clock());
    defer e.deinit();
    var good = try Stream.init(gpa, .{ .id = "live", .prompt = &(@as([16]u32, @splat(3))), .max_new = 100, .eos = &.{} });
    defer good.deinit(gpa);
    var cancelled = try Stream.init(gpa, .{ .id = "cancelled", .prompt = &(@as([16]u32, @splat(7))), .max_new = 100, .eos = &.{} });
    defer cancelled.deinit(gpa);
    if (which != .solo and first) try e.addStream(&cancelled);
    try e.addStream(&good);
    if (which != .solo and !first) try e.addStream(&cancelled);
    var flag = which == .before;
    cancelled.cancel_check = .{ .ptr = &flag, .check = struct {
        fn check(ptr: *anyopaque) bool {
            return @as(*const bool, @ptrCast(@alignCast(ptr))).*;
        }
    }.check };
    if (which == .after) batch.after = &flag;
    try e.step();
    const at: usize = @intFromBool(which != .solo and first);
    var result: Snapshot = .{ .plan = batch.plans[at], .depth = batch.depths[at], .count = good.next.?.count, .len = good.emitted().len };
    const held = batch.target.lanes.getPtr(&good).?.held.items;
    try t.expect(held.len <= result.held.len and result.len <= result.tokens.len);
    @memcpy(result.held[0..held.len], held);
    @memcpy(result.tokens[0..result.len], good.emitted());
    try t.expectEqual(@as(usize, if (which == .solo) 1 else 2), batch.scored);
    if (which != .solo) {
        try t.expect(flag and !cancelled.finished);
        try t.expectEqual(@as(?u32, 0), cancelled.proposal_depth);
        try t.expectEqual(@as(u32, 0), cancelled.next.?.count);
        try t.expectEqual(@as(usize, 0), batch.target.lanes.getPtr(&cancelled).?.held.items.len);
        e.discard(&cancelled);
        try t.expect(cancelled.finished and cancelled.reason == .cancelled);
    }
    return result;
}
test "proposal router: step isolates before and after producer cancellation without dropping scoring rows" {
    const solo = try round(.solo, false);
    try t.expect(solo.depth > 0 and solo.count > 0);
    for ([_]bool{ false, true }) |first| {
        try t.expectEqualDeep(solo, try round(.before, first));
        try t.expectEqualDeep(solo, try round(.after, first));
    }
}

//! Actual lane-core begin/commit/done callbacks with a CPU target and recording CUDA events.
const std = @import("std");
const lanes = @import("lanes");
const cuda = @import("cuda");
const timing = @import("qwen35").native.round_timing;
const Rig = @import("attribution_fake").Rig;
const be = lanes.backend;
const t = std.testing;
const Target = struct {
    fake: lanes.fake.Fake,
    rig: Rig,
    trace: timing.Trace,
    table: be.Backend.VTable = undefined,
    fn backend(self: *Target) be.Backend {
        self.table = self.fake.backend().vtable.*;
        self.table.verify = verify;
        return .{ .ptr = &self.fake, .vtable = &self.table, .round_observer = self.trace.observer() };
    }
    fn verify(ptr: *anyopaque, windows: []const be.Window, output: []be.Verified) !void {
        const self: *Target = @fieldParentPtr("fake", @as(*lanes.fake.Fake, @ptrCast(@alignCast(ptr))));
        const scope = cuda.round_attribution.scope(.forward);
        defer scope.end();
        var rows: usize = 0;
        for (windows) |w| rows += w.rows();
        try self.rig.timing.frames.append(t.allocator, .{ .correlation = scope.correlation, .live_rows = rows, .windows = windows.len, .shape_bucket = timing.Shape.init(rows, windows.len, true, false, 1).total, .shape_key = 1, .mt2 = false });
        try self.rig.kernel();
        _ = self.rig.driver.api.cuStreamSynchronize(null);
        try self.fake.backend().verify(windows, output);
    }
};
test "round attribution core: complete, clipped and shared commits come from actual acceptance" {
    if (comptime !@import("profile").round_attribution.enabled) return error.SkipZigTest;
    for ([_]usize{ 1, 2 }) |count| for ([_]u8{ 0, 1, 2, 3 }) |mode| {
        const rig = try Rig.init();
        defer rig.deinit();
        const id: [64]u8 = @splat('a');
        rig.timing.identity = &id;
        var target: Target = .{ .fake = .{ .gpa = t.allocator }, .rig = rig, .trace = .{ .extra = rig.timing, .legacy = false } };
        defer target.fake.deinit();
        var cfg = try lanes.Config.init(t.allocator, .{ .exact_width = 16, .hidden_rows = true, .streams_exact = true, .batch_rows = 128, .max_streams = 8 }, 16, 15);
        defer cfg.deinit(t.allocator);
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(t.allocator, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        var streams: [2]lanes.Stream = undefined;
        var initialized: usize = 0;
        defer for (streams[0..initialized]) |*s| s.deinit(t.allocator);
        var before: [2]usize = undefined;
        for (streams[0..count], 0..) |*s, i| {
            s.* = try lanes.Stream.init(t.allocator, .{ .id = if (i == 0) "alpha" else "beta", .prompt = &.{ 1, 2, 3 }, .max_new = if (mode == 1) 2 else 20, .eos = &.{} });
            initialized += 1;
            try core.addStream(s);
            before[i] = s.emitted().len;
            if (mode < 2) {
                try s.force.appendSlice(t.allocator, &.{ 11, 12, 13, 14 });
            } else {
                var history: std.ArrayList(u32) = .empty;
                defer history.deinit(t.allocator);
                try history.appendSlice(t.allocator, s.prompt());
                try history.append(t.allocator, s.pending.?);
                const guesses = try t.allocator.alloc(u32, 3);
                for (guesses) |*guess| {
                    guess.* = if (mode == 2) 0 else lanes.fake.next(history.items, s.sampling, history.items.len);
                    try history.append(t.allocator, guess.*);
                }
                s.next = .{ .count = 3, .tokens = guesses };
            }
        }
        try core.step();
        try t.expect(!rig.timing.active);
        try t.expect(rig.timing.problem == null);
        try t.expectEqual(count, rig.timing.requests.items.len);
        try t.expectEqual(count, rig.timing.windows.items.len);
        for (streams[0..count], rig.timing.windows.items, 0..) |*s, w, i| {
            try t.expectEqualSlices(u32, s.emitted()[before[i]..], w.committed_ids);
            try t.expectEqual(s.cache_len, w.cached_after);
            try t.expectEqual(s.pending, w.pending_after);
            try t.expectEqual(if (mode < 2) @as(usize, 0) else w.kept_path.len - 1, w.connected_accepted_depth);
            try t.expectEqual(w.offered_rows - w.kept_path.len, w.rolled_back_rows);
            try t.expectEqual(if (mode < 2) @as(usize, 0) else w.offered_rows - 1, w.proposal_nodes);
            if (mode == 2) try t.expect(w.rolled_back_rows > 0);
            if (mode == 3) try t.expect(w.connected_accepted_depth > 0);
        }
    };
}

test "attribution review: legacy frame compiles attribution out" {
    if (comptime @import("profile").round_attribution.enabled) return error.SkipZigTest;
    const rig = try Rig.init();
    defer rig.deinit();
    try rig.timing.frames.append(t.allocator, .{ .correlation = 1, .live_rows = 123, .windows = 1, .shape_bucket = 1, .shape_key = 1, .mt2 = false });
    var legacy: timing.Trace = .{ .extra = rig.timing };
    legacy.frame(0, 1, 4, 4, 1);
    try t.expectEqual(@as(usize, 123), rig.timing.frames.items[0].live_rows);
}
test "attribution review: legacy done compiles attribution out" {
    if (comptime @import("profile").round_attribution.enabled) return error.SkipZigTest;
    const rig = try Rig.init();
    defer rig.deinit();
    var legacy: timing.Trace = .{ .extra = rig.timing };
    const waits = rig.waitCount();
    legacy.observer().done(&legacy, &.{});
    try t.expectEqual(waits, rig.waitCount());
}

test "attribution review: resolved YaRN context is the bound identity" {
    if (comptime !@import("profile").round_attribution.enabled) return error.SkipZigTest;
    const rig = try Rig.init();
    defer rig.deinit();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{}" });
    const dir = try std.fmt.allocPrint(t.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer t.allocator.free(dir);
    const Ctx = struct {
        handle: cuda.abi.Context = null,
        pub fn attribute(_: @This(), _: anytype) !c_int { return 12; }
    };
    const G = struct { hidden: usize = 5120, layers: usize = 64, experts: usize = 0, exl3: bool = false, tfq3: bool = false, kinds: struct { modelopt: bool = true } = .{}, first: usize = 0, whole: usize = 0, context: usize = 4096 };
    const Engine = struct { g: G = .{}, stream: struct { handle: cuda.abi.Stream = null } = .{}, kv8: bool = true };
    var engine: Engine = .{};
    engine.g.context = 8192;
    try rig.timing.bind(&engine, Ctx{}, dir, .{ .checkpoint_sha256 = "fixture" });
    try t.expectEqual(@as(i64, 8192), rig.timing.header.object.get("geometry").?.object.get("context").?.integer);
    const source = @embedFile("src/families/qwen3_5/cuda_native.zig");
    try t.expect(std.mem.indexOf(u8, source, "try x.bind(e, ctx,").? > std.mem.indexOf(u8, source, "cuda_rope_table.zig\").set(e, r.rope)").?);
}

test "attribution review: MTP absorption enters the proposal scope" {
    if (comptime !@import("profile").round_attribution.enabled) return error.SkipZigTest;
    const rig = try Rig.init();
    defer rig.deinit();
    rig.timing.begin();
    var mtp: @import("qwen35").mtp.Mtp = undefined;
    mtp.io = t.io;
    mtp.trees = false;
    mtp.batch = false;
    try mtp.drafter().vtable.absorb(&mtp, &.{});
    try t.expectEqual(@as(usize, 1), rig.timing.host.items.len);
    try t.expectEqual(cuda.round_attribution.model.Phase.proposal, rig.timing.host.items[0].phase);
    try t.expectEqual(cuda.round_attribution.model.Phase.other, rig.timing.phase);
}

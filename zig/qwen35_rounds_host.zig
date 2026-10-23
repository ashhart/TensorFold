//! CUDA 64-row geometry, staged parent metadata and shared register-slot schedules, without device execution.
const std = @import("std");
const q = @import("qwen35");
const core = @import("core");
const lanes = @import("lanes");
const a = std.testing.allocator;

test "Verify width calibration: the target window and verify scratch admit all 64 rows" {
    try std.testing.expectEqual(@as(usize, 64), q.engine.most_window);
    try std.testing.expect(q.engine.most_verify >= 64);
}

test "Verify width calibration: row 63 is an ancestor bit, while a 128-row prompt carries a separate chain depth" {
    try std.testing.expect(@hasDecl(q, "round"));
    if (comptime @hasDecl(q, "round")) {
        const R = q.round;
        var storage: [R.bytes]u8 align(8) = undefined;
        var parents: [64]i32 = undefined;
        for (&parents, 0..) |*p, row| p.* = if (row == 0) -1 else @intCast(row - 1);
        const parts = [_]q.engine.Part{.{ .cache = undefined, .rows = 64, .parents = &parents }};
        try std.testing.expectEqual(@as(usize, 1), try R.build(&storage, &parts, true));
        const ancestors = R.ancestors(&storage);
        try std.testing.expectEqual(std.math.maxInt(u64), ancestors[63].bits);
        try std.testing.expectEqual(@as(u32, 63), ancestors[63].depth);
        try std.testing.expectEqual(@as(u32, 0), ancestors[63].chain);
        try std.testing.expectEqual(@as(usize, 1), try R.build(&storage, &.{q.engine.Part{ .cache = undefined, .rows = 128 }}, false));
        try std.testing.expectEqual(@as(u32, 127), R.ancestors(&storage)[127].depth);
        try std.testing.expectEqual(@as(u32, 1), R.ancestors(&storage)[127].chain);
    }
}

test "Verify width calibration: shared register-slot plans retain every parent and original output row" {
    try std.testing.expect(@hasDecl(core, "tree_slots"));
    if (comptime @hasDecl(core, "tree_slots") and @hasDecl(q, "round")) {
        const R = q.round;
        var storage: [R.bytes]u8 align(8) = undefined;
        var parents: [64]i32 = undefined;
        for (&parents, 0..) |*p, row| p.* = if (row == 0) -1 else @intCast((row - 1) / 2);
        const parts = [_]q.engine.Part{.{ .cache = undefined, .rows = 64, .parents = &parents }};
        const peak = try R.build(&storage, &parts, true);
        try std.testing.expect(peak > 1 and peak <= 8);
        try core.tree_slots.validate(&parents, R.steps(&storage)[0..64], 0, 16);
        const saved = R.steps(&storage)[1];
        R.steps(&storage)[1].read_slot = 15;
        try std.testing.expectError(error.BadSlotPlan, core.tree_slots.validate(&parents, R.steps(&storage)[0..64], 0, 16));
        R.steps(&storage)[1] = saved;
    }
}

const cuda = @import("cuda");
const Capture = struct {
    var id: usize = 0;
    var input: u64 = 0;
    var threads: u32 = 0;
    var kept: q.round.Path = undefined;
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, x: c_uint, y: c_uint, z: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        id = @intFromPtr(f);
        threads = x * y * z;
        if (id >= 10) input = @as(*const u64, @ptrCast(@alignCast(args.?[7].?))).*;
        if (id == 1 or id == 3) kept = @as(*const q.round.Path, @ptrCast(@alignCast(args.?[2].?))).*;
        return cuda.abi.success;
    }
};
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}

test "Verify width calibration: actual DeltaNet dispatch selects its register variant and binds the schedule table" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Capture.launch;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };
    e.tables = .{ .d = &d, .ptr = 0x100000, .len = q.round.bytes };
    e.g.v_heads = 4;
    e.geo = .{ .hid = 512, .nk = 2, .nv = 4, .qh = 4, .kvh = 1, .nl = 1 };
    e.q = 1;
    e.k = 2;
    e.v = 3;
    e.g_ = 4;
    e.beta = 5;
    e.y = 6;
    e.fns.gdn_chain = function(&d, 11);
    e.fns.gdn_slots_1 = function(&d, 21);
    e.fns.gdn_slots_2 = function(&d, 22);
    e.fns.gdn_slots_4 = function(&d, 24);
    e.fns.gdn_slots_8 = function(&d, 28);
    e.fns.gdn_slots_16 = function(&d, 36);
    for ([_]u8{ 1, 2, 4, 8, 16 }) |slots| {
        try e.deltaScan(true, slots, 1, 0);
        try std.testing.expectEqual(@as(usize, 20 + slots), Capture.id);
        try std.testing.expectEqual(e.tables.ptr + q.round.steps_at, Capture.input);
        try std.testing.expectEqual(@as(u32, 128), Capture.threads);
    }
    try e.deltaScan(false, 1, 1, 0);
    try std.testing.expectEqual(@as(usize, 11), Capture.id);
    try std.testing.expectEqual(e.tables.ptr + q.round.rowpar_at, Capture.input);
    try std.testing.expectError(error.InvalidGdnSlots, e.deltaScan(true, 3, 1, 0));
}

test "Verify width calibration: actual keep launches carry the complete 64-row path" {
    try std.testing.expectEqual(@as(usize, 260), @sizeOf(q.round.Path));
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Capture.launch;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };
    e.g = .{ .hidden = 512, .intermediate = 256, .layers = 2, .vocab = 248320, .k_heads = 2, .v_heads = 4, .q_heads = 4, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 4096, .linear = @splat(true) };
    e.g.linear[1] = false;
    e.geo = .{ .hid = 512, .nk = 2, .nv = 4, .qh = 4, .kvh = 1, .nl = 1 };
    e.keeps = .{ .apply = function(&d, 1), .conv = function(&d, 2), .compact = function(&d, 3) };
    e.kv8 = false;
    e.n_taps = 0;
    const buffer: cuda.DeviceBuffer = .{ .d = &d, .ptr = 0x1000, .len = 1 << 20 };
    var cache: q.engine.Cache = .{ .state = buffer, .kv = buffer, .log = buffer, .logits = &.{}, .cap = 128, .len = 69, .last = .{ .start = 5, .rows = 64 } };
    var path: [64]u32 = undefined;
    for (&path, 0..) |*row, i| row.* = @intCast(i);
    try cache.keep(&e, &path);
    try std.testing.expectEqual(@as(c_int, 64), Capture.kept.n);
    try std.testing.expectEqual(@as(c_int, 63), Capture.kept.rows[63]);
    try std.testing.expectEqual(@as(usize, 69), cache.len);
    cache.last = .{ .start = 5, .rows = 64 };
    try cache.keep(&e, &.{ 0, 2, 33, 63 });
    try std.testing.expectEqual(@as(usize, 3), Capture.id);
    try std.testing.expectEqual(@as(c_int, 63), Capture.kept.rows[3]);
    try std.testing.expectEqual(@as(usize, 9), cache.len);
}

const Verify = struct {
    var widest: usize = 0;
    var cohort: usize = 0;
    fn call(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
        cohort = @max(cohort, windows.len);
        var storage: [q.round.bytes]u8 align(8) = undefined;
        var parts: [q.engine.max_segs]q.engine.Part = undefined;
        for (windows, parts[0..windows.len]) |w, *p| {
            widest = @max(widest, w.rows());
            p.* = .{ .cache = undefined, .rows = w.rows(), .parents = w.parents };
        }
        _ = try q.round.build(&storage, parts[0..windows.len], true);
        var row: usize = 0;
        for (windows) |w| for (w.positions) |position| {
            try std.testing.expectEqual(w.stream.cache_len + 1 + q.round.ancestors(&storage)[row].depth, position);
            row += 1;
        };
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return target.backend().vtable.verify(ptr, windows, out);
    }
};
pub fn reply(depth: u32, restored: u32, streams_: usize, seed: ?u64) ![96]u32 {
    var costs: [64]lanes.config.Cost = undefined;
    const prior: [63]f64 = @splat(0.99);
    for (&costs, 1..) |*cost, width| cost.* = .{ .width = @intCast(width), .ms = 5 };
    var facts = q.native.laneFacts(q.engine.most_window, q.engine.most_verify);
    facts.mtp = depth > 0;
    facts.speculate = true;
    facts.speculate_early = false;
    facts.drafts = depth;
    facts.window_costs = &costs;
    facts.draft_prior = &prior;
    var cfg = try lanes.Config.init(a, facts, q.engine.most_window, 63);
    defer cfg.deinit(a);
    cfg.timed = false;
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var backend = target.backend();
    var vtable = backend.vtable.*;
    vtable.verify = Verify.call;
    backend.vtable = &vtable;
    Verify.widest = 0;
    Verify.cohort = 0;
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, backend, clock.clock());
    defer engine.deinit();
    const prompt = [_]u32{ 7, 1, 2, 3, 4, 5, 6, 8, 9, 7, 1, 2, 3, 4 };
    var prefix: std.ArrayList(u32) = .empty;
    defer prefix.deinit(a);
    try prefix.appendSlice(a, prompt[0..restored]);
    var streams: [128]lanes.Stream = undefined;
    var proposers: [128]lanes.SuffixLookup = undefined;
    var made: usize = 0;
    defer for (streams[0..made], proposers[0..made]) |*s, *p| {
        s.deinit(a);
        p.deinit();
    };
    for (streams[0..streams_], proposers[0..streams_], 0..) |*s, *p, i| {
        p.* = try lanes.SuffixLookup.init(a, .{ .min_match = 5 });
        s.* = try lanes.Stream.init(a, .{ .id = "64 rows", .prompt = &prompt, .max_new = 96, .drafts = depth > 0, .sampling = if (seed) |v| .{ .seed = v + i, .temperature = 0.8 } else null, .proposer = p.proposer(), .loop_guard = false, .reuse = .{ .saved = if (restored > 0) &prefix else null, .at = restored } });
        made += 1;
        try engine.addStream(s);
    }
    while (engine.activeCount() > 0) try engine.step();
    var result: [96]u32 = undefined;
    @memcpy(&result, streams[0].emitted());
    try std.testing.expectEqual(restored, streams[0].cached);
    if (depth == 63 and streams_ == 1) try std.testing.expectEqual(@as(usize, 64), Verify.widest);
    if (depth == 0 and streams_ >= 16) try std.testing.expectEqual(streams_, Verify.cohort);
    return result;
}

test "Verify width calibration: 64-row CUDA metadata in real lane rounds preserves drafted plain resumed fresh concurrent solo" {
    for ([_]?u64{ null, 17, 91 }) |seed| {
        const plain = try reply(0, 0, 1, seed);
        for ([_]u32{ 0, 1, 7, 15, 63 }) |depth| for ([_]u32{ 0, 5, 13 }) |restore| for ([_]usize{ 1, 2, 4, 8 }) |streams| {
            try std.testing.expectEqual(plain, try reply(depth, restore, streams, seed));
        };
    }
}

test "Verify width calibration: staged windows isolate local ancestry and global scheduled rows and reject invalid input" {
    const R = q.round;
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(R.Ancestor));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(R.Ancestor, "depth"));
    try std.testing.expectEqual(@as(usize, 12), @offsetOf(R.Ancestor, "chain"));
    var storage: [R.bytes]u8 align(8) = undefined;
    var parents: [32]i32 = undefined;
    for (&parents, 0..) |*p, row| p.* = if (row == 0) -1 else @intCast((row - 1) / 2);
    const parts = [_]q.engine.Part{ .{ .cache = undefined, .rows = 32, .parents = &parents }, .{ .cache = undefined, .rows = 32, .parents = &parents } };
    _ = try R.build(&storage, &parts, true);
    try std.testing.expectEqual(@as(u64, 1), R.ancestors(&storage)[32].bits);
    try std.testing.expectEqual(@as(u32, 0), R.ancestors(&storage)[32].depth);
    try core.tree_slots.validate(&parents, R.steps(&storage)[32..64], 32, 16);
    try std.testing.expectError(error.InvalidQwenSegments, R.build(storage[0 .. R.bytes - 1], &parts, true));
    try std.testing.expectError(error.InvalidQwenSegments, R.build(&storage, &.{.{ .cache = undefined, .rows = 65 }}, true));
    try std.testing.expectError(error.InvalidQwenSegments, R.build(&storage, &.{.{ .cache = undefined, .rows = 0 }}, true));
    parents[1] = 1;
    try std.testing.expectError(error.InvalidQwenParents, R.build(&storage, &parts, true));
}

test "Shared verification: shared 16 32 64 128 stream rounds preserve drafted plain restored fresh and concurrent solo" {
    for ([_]?u64{ null, 17 }) |seed| {
        const plain = try reply(0, 0, 1, seed);
        for ([_]u32{ 0, 7 }) |depth| for ([_]u32{ 0, 13 }) |restore| for ([_]usize{ 16, 32, 64, 128 }) |streams| {
            try std.testing.expectEqual(plain, try reply(depth, restore, streams, seed));
        };
    }
}

test "Verify width calibration: straight metadata needs no branch scratch and matches the general schedule" {
    const R = q.round;
    var storage: [R.bytes]u8 align(8) = undefined;
    var parents: [64]i32 = undefined;
    var expected: [64]core.tree_slots.Step = undefined;
    for (1..65) |rows| {
        for (parents[0..rows], 0..) |*parent, row| parent.* = if (row == 0) -1 else @intCast(row - 1);
        _ = try core.tree_slots.build(a, parents[0..rows], expected[0..rows], 1, 16);
        for ([_]?[]const i32{ null, parents[0..rows] }) |ps| {
            const parts = [_]q.engine.Part{ .{ .cache = undefined, .rows = 1 }, .{ .cache = undefined, .rows = rows, .parents = ps } };
            try std.testing.expectEqual(@as(usize, 1), try R.buildWithScratch(&storage, &parts, true, &.{}));
            try std.testing.expectEqualSlices(core.tree_slots.Step, expected[0..rows], R.steps(&storage)[1..][0..rows]);
            try core.tree_slots.validate(parents[0..rows], R.steps(&storage)[1..][0..rows], 1, 16);
        }
    }
    var parts: [128]q.engine.Part = @splat(.{ .cache = undefined, .rows = 1 });
    try std.testing.expectEqual(@as(usize, 1), try R.buildWithScratch(&storage, &parts, true, &.{}));
    for (R.steps(&storage)[0..128], 0..) |step, row| {
        try std.testing.expectEqual(@as(u32, @intCast(row)), step.row);
        try std.testing.expectEqual(@as(i32, -1), step.read_slot);
        try std.testing.expectEqual(@as(i32, 0), step.write_slot);
    }
    const branch = [_]i32{ -1, 0, 0, 1, 2 };
    const fork = [_]q.engine.Part{.{ .cache = undefined, .rows = branch.len, .parents = &branch }};
    try std.testing.expectError(error.OutOfMemory, R.buildWithScratch(&storage, &fork, true, &.{}));
    try std.testing.expect((try R.build(&storage, &fork, true)) > 1);
    try core.tree_slots.validate(&branch, R.steps(&storage)[0..branch.len], 0, 16);
}

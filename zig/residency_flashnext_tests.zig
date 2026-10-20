//! Both native Flash Next hosts use the real runner, snapshot pool and session allocation paths.
const std = @import("std");
const mtl = @import("metal");
const fz = @import("src/families/flashnext/replay.zig");
const Engine = @import("src/families/flashnext/engine.zig").Engine;
const Session = @import("src/families/flashnext/session.zig").Session;
const residency = @import("src/native/flashnext_residency.zig");
const a = std.testing.allocator;
const Vision = struct {
    held: [3]mtl.Buffer,
    pub fn buffers(v: *const @This()) [3]mtl.Buffer {
        return v.held;
    }
};
const Back = struct { sessions: []Session, hidden: fz.Buf };
const Host = struct {
    gpa: std.mem.Allocator = a,
    io: std.Io = std.testing.io,
    host: struct { io: std.Io } = .{ .io = std.testing.io },
    eng: *Engine,
    back: *Back,
    vis: ?*Vision = null,
    load_limit: u64 = 0,
    resident: ?mtl.resident.Resident = null,
    warm: mtl.keepalive.Target = undefined,
};
fn waiting(_: *anyopaque) bool {
    return false;
}

test "residency owner Flash Next: both native inventories cover initial pools and TP buffers and follow lazy allocations" {
    for ([_]bool{ false, true }) |shared| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        {
            var run: fz.Run = .{ .arena = a, .device = record.device(), .queue = record.queue() };
            defer {
                while (run.held.items.len > 0) run.release(run.held.items[run.held.items.len - 1]);
                run.held.deinit(a);
            }
            _ = try run.buffer(64);
            var tp: fz.Tp2 = undefined;
            tp.wbuf = try record.device().buffer(64, 0);
            defer tp.wbuf.deinit();
            tp.one = try record.device().buffer(64, 0);
            defer tp.one.deinit();
            tp.pick_tmp = try record.device().buffer(64, 0);
            defer tp.pick_tmp.deinit();
            run.tp = &tp;
            var engine: Engine = undefined;
            engine.r = &run;
            engine.snap_pool = .{ .max = 1024, .last = 64 };
            defer engine.snap_pool.deinit();
            _ = engine.snap_pool.ready(record.device(), 0, .{ .ctx = &record, .check = waiting }, 1024);
            var sessions = [_]Session{.{ .m = undefined }};
            defer sessions[0].deinit(a);
            var back: Back = .{ .sessions = if (shared) &sessions else &.{}, .hidden = undefined };
            var hidden: ?mtl.Buffer = null;
            defer if (hidden) |b| b.deinit();
            if (shared) {
                _ = try sessions[0].buffer(a, &engine, 64);
                hidden = try record.device().buffer(64, 0);
                back.hidden = .{ .b = hidden.? };
            }
            var vision: Vision = undefined;
            for (&vision.held) |*buffer| buffer.* = try record.device().buffer(64, 0);
            defer for (vision.held) |buffer| buffer.deinit();
            var h: Host = .{ .eng = &engine, .back = &back, .vis = &vision };
            if (shared) try residency.holdShared(&h) else try residency.hold(&h);
            defer if (shared) residency.dropShared(&h) else residency.drop(&h);
            try record.checkInitial();
            try record.check();
            mtl.keepalive.Target.tick(&h.warm);
            try std.testing.expect(record.copies > 0 and record.uses > 0);
            const lazy = try run.buffer(128);
            h.resident.?.held.flush();
            try record.check();
            run.release(lazy);
            try record.check();
            if (shared) {
                _ = try sessions[0].buffer(a, &engine, 128);
                h.resident.?.held.flush();
                try record.check();
            }
            engine.snap_pool.trim(0);
            try record.check();
            _ = engine.snap_pool.ready(record.device(), 1, .{ .ctx = &record, .check = waiting }, 1024);
            h.resident.?.held.flush();
            try record.check();
        }
        try record.allFreed();
    }
}

test "residency owner Flash Next: failed runner publication frees its new Metal buffer" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    {
        var owner = (try mtl.resident.Resident.hold(a, std.testing.io, record.device(), record.queue(), &.{}, "flash next")).?;
        defer owner.deinit();
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
        var run: fz.Run = .{ .arena = failing.allocator(), .device = record.device(), .queue = record.queue(), .follow = owner.follow() };
        try std.testing.expectError(error.OutOfMemory, run.buffer(64));
        owner.held.flush();
        try record.check();
    }
    try record.allFreed();
}

test "Flash Next vision memory: tower startup refuses before compilation or factory work" {
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    var run: fz.Run = .{ .arena = a, .device = record.device(), .queue = record.queue(), .load_limit = @import("flashnext_memory").margin + 64 };
    defer {
        while (run.held.items.len > 0) run.release(run.held.items[run.held.items.len - 1]);
        run.held.deinit(a);
    }
    _ = try run.buffer(64);
    const Builder = struct {
        fn construct(calls: *u32) !void {
            calls.* += 1;
        }
    };
    var calls: u32 = 0;
    try std.testing.expectError(error.FlashNextMemoryShortfall, fz.vision_memory.loadTower(void, &run, 1024, &calls, Builder.construct));
    try std.testing.expectEqual(@as(u32, 0), calls);
    try std.testing.expectEqual(@as(usize, 1), record.allocations.items.len);
}

test "Flash Next vision memory: single and shared state survives refused image/video outputs and a smaller follow-up fits" {
    for ([_]bool{ false, true }) |shared| for ([_]bool{ false, true }) |video| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        const row = fz.D * 2;
        var run: fz.Run = .{ .arena = a, .device = record.device(), .queue = record.queue(), .load_limit = @import("flashnext_memory").margin + 64 + 5 * row };
        defer {
            while (run.held.items.len > 0) run.release(run.held.items[run.held.items.len - 1]);
            run.held.deinit(a);
        }
        var engine: Engine = undefined;
        engine.r = &run;
        var session: Session = .{ .m = undefined };
        defer session.deinit(a);
        const state = if (shared) (try session.buffer(a, &engine, 64)).b else try run.buffer(64);
        @memset(state.contents()[0..64], 0xa5);
        const before = record.allocations.items.len;
        const oversized: usize = if (video) 4 else 2;
        try std.testing.expectError(error.FlashNextMemoryShortfall, fz.vision_memory.outputs(mtl.Buffer, &run, oversized, 1, row, mtl.ResourceOptions.shared));
        try std.testing.expectEqual(before, record.allocations.items.len);
        // The rows can fit while their temporary cannot; their successful prefix must unwind.
        try std.testing.expectError(error.FlashNextMemoryShortfall, fz.vision_memory.outputs(mtl.Buffer, &run, 1, 2, row, mtl.ResourceOptions.shared));
        try std.testing.expectEqual(@as(usize, 64), record.device().allocated());
        const next = try fz.vision_memory.outputs(mtl.Buffer, &run, 1, 1, row, mtl.ResourceOptions.shared);
        next.temporary.deinit();
        next.rows.deinit();
        try std.testing.expectEqual(@as(usize, 64), record.device().allocated());
        try std.testing.expectEqualSlices(u8, &(@as([64]u8, @splat(0xa5))), state.contents()[0..64]);
    };
}

test "Flash Next vision memory: metadata prices tower alignment once and both image/video output planes" {
    const Plan = fz.vision_memory.Plan;
    const p: Plan = .{ .source = 100, .weights = 128, .scratch = 200, .rows = 1000, .temporary = 50 };
    try std.testing.expectEqual(@as(u64, 1378), try p.extra());
    try std.testing.expectEqual(@as(u64, 328), try p.tower());
    const settings = try fz.vision_memory.withPlan(struct { startup_extra: u64 }{ .startup_extra = 500 }, p);
    try std.testing.expectEqual(@as(u64, 1878), settings.startup_extra);
    try std.testing.expectEqual(@as(u64, 100 + 128 + 200 + 1000 + 50), 100 + try p.extra());
    const c: @import("src/core/qwen_vl/config.zig").Config = .{
        .depth = 1,
        .hidden = 4,
        .heads = 1,
        .intermediate = 8,
        .out_hidden = 2560,
        .patch = 2,
        .temporal = 2,
        .merge = 2,
        .positions = 4,
        .image_token = 1,
        .vision_start = 2,
        .vision_end = 3,
        .min_pixels = 16,
        .max_pixels = 1024,
        .rescale = 1,
        .mean = @splat(0),
        .std = @splat(1),
    };
    const tower = @import("src/core/qwen_vl/tower.zig");
    const plan = try tower.memoryPlan(a, c, 8);
    try std.testing.expectEqual(try tower.planBytes(a, c, 8), plan.weights + plan.scratch);
    try std.testing.expect(plan.weights >= plan.source and plan.scratch > 0);
}

test "Flash Next vision memory: both native entries budget metadata before loading and the builder uses guarded factories" {
    inline for (.{ @embedFile("src/native/flashnext_host.zig"), @embedFile("src/native/flashnext_batch_host.zig") }) |source| {
        const at = std.mem.indexOf(u8, source, "pub fn openConfigured(").?;
        const planned = std.mem.indexOf(u8, source[at..], "images.budgetSettings(").?;
        const loaded = std.mem.indexOf(u8, source[at..], "fx.Engine.loadConfigured(").?;
        try std.testing.expect(planned < loaded);
    }
    const source = @embedFile("src/native/flashnext_images.zig");
    try std.testing.expect(std.mem.indexOf(u8, source, "memory.loadTower(*Tower") != null);
    try std.testing.expect(std.mem.indexOf(u8, source, "memory.outputs(mtl.Buffer, eng.r") != null);
    try std.testing.expect(std.mem.indexOf(u8, source, "eng.r.device.buffer(") == null);
    try std.testing.expect(std.mem.indexOf(u8, source, "Tower.loadWith(") != null);
}

test "Flash Next cleanup: actual constructor prefix unwinds a later guard refusal and retries on the same device" {
    const Prefix = @import("src/families/flashnext/load_prefix.zig");
    var record: mtl.Record = .{ .gpa = a };
    defer record.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const Work = struct {
        fn construct(_: void, run: anytype) !void {
            _ = try run.buffer(64);
            _ = try run.buffer(64);
            _ = try run.buffer(128);
        }
    };
    var run = fz.Run{ .arena = arena.allocator(), .device = record.device(), .queue = record.queue(), .load_limit = @import("flashnext_memory").margin + 192 };
    try std.testing.expectError(error.FlashNextMemoryShortfall, Prefix.build(void, &run, {}, Work.construct));
    try std.testing.expectEqual(@as(usize, 2), record.allocations.items.len);
    try std.testing.expectEqual(@as(usize, 1), record.waits);
    try std.testing.expectEqual(@as(usize, 1), record.queue_releases);
    try record.allFreed();
    run.deinit();
    try std.testing.expectEqual(@as(usize, 1), record.queue_releases);
    var retry = fz.Run{ .arena = arena.allocator(), .device = record.device(), .queue = record.queue(), .load_limit = @import("flashnext_memory").margin + 256 };
    try Prefix.build(void, &retry, {}, Work.construct);
    retry.deinit();
    try record.allFreed();
    try std.testing.expectEqual(@as(usize, 2), record.queue_releases);
}

test {
    _ = @import("src/families/flashnext/owned_test.zig");
}

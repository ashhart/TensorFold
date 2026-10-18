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

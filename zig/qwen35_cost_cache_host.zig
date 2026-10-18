//! The actual width loop through a fake engine and workspace-local shared-cache files.
const std = @import("std");
const q = @import("qwen35");
const costs = q.native.window_costs;
const shared = @import("lanes").cost_cache;
const a = std.testing.allocator;
const io = std.testing.io;

const Clock = struct {
    threadlocal var ns: i96 = 0;
    threadlocal var step: i96 = 1_000_000;
    threadlocal var table: std.Io.VTable = undefined;
    fn now(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp {
        ns += step;
        return .{ .nanoseconds = ns };
    }
    fn clock() std.Io {
        table = io.vtable.*;
        table.now = now;
        return .{ .userdata = io.userdata, .vtable = &table };
    }
};
pub const Fake = struct {
    const Cache = struct {
        pub fn reset(_: *@This(), _: *Fake) !void {}
    };
    const Part = struct { cache: *Cache, rows: usize };
    window_rows: usize = 64,
    n_taps: usize = 0,
    forwards: usize = 0,
    runs: usize = 0,
    step: i96 = 1_000_000,
    fail_at: usize = 0,
    prices: [64]f64 = @splat(0),
    pub fn forward(f: *Fake, batch: []const Part, ids: []const u32, _: anytype) !void {
        try std.testing.expectEqual(batch[0].rows, ids.len);
        f.forwards += 1;
        Clock.step = f.step * @as(i96, @intCast(ids.len));
        if (f.fail_at == ids.len) return error.PlantedCalibrationFailure;
    }
    pub fn setCost(f: *Fake, rows: u32, ms: f64) void {
        f.prices[rows - 1] = ms;
    }
    pub fn run(f: *Fake) !costs.Measured {
        f.runs += 1;
        Clock.step = f.step;
        var warmed: q.native.warm.Seen = .{};
        for (1..65) |rows| {
            const shape = q.engine.Engine.Shape.init(rows, 1, true, false, 1);
            warmed.mark(shape);
            warmed.mark(shape);
        }
        var cache: Cache = .{};
        try q.native.warm.time(Clock.clock(), f, f, &cache, &warmed);
        return .{ .ms = f.prices };
    }
};
const parts = [_][]const u8{ "qwen35-cuda", "gpu", "driver", "shape", "dflash", "drafter", "target", "64" };

test "window cost cache: open measures all 64 widths on a miss and zero on a hit" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var engine: Fake = .{};
    const key = costs.keys("binary", &parts);
    const first = try costs.resolve(a, io, key, 64, root, &engine);
    try std.testing.expect(!first.hit);
    try std.testing.expectEqual(@as(usize, 64), engine.forwards);
    engine.forwards = 0;
    const second = try costs.resolve(a, io, key, 64, root, &engine);
    try std.testing.expect(second.hit);
    try std.testing.expectEqual(@as(usize, 0), engine.forwards);
    try std.testing.expectEqualSlices(f64, &first.measured.ms, &second.measured.ms);
    for (second.measured.ms, 1..) |ms, width| try std.testing.expectEqual(@as(f64, @floatFromInt(width)), ms);
}

test "window cost cache: executable and every identity part separate keys" {
    const first = costs.keys("binary", &parts);
    try std.testing.expect(!std.mem.eql(u8, &first.current, &costs.keys("other-binary", &parts).current));
    try std.testing.expectEqualSlices(u8, &first.reference, &costs.keys("other-binary", &parts).reference);
    for (0..parts.len) |i| {
        var changed = parts;
        changed[i] = "changed";
        const next = costs.keys("binary", &changed);
        try std.testing.expect(!std.mem.eql(u8, &first.current, &next.current));
        try std.testing.expect(!std.mem.eql(u8, &first.reference, &next.reference));
    }
}

test "window cost cache: short foreign corrupt and invalid files measure again and are replaced" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const key = costs.keys("binary", &parts);
    var engine: Fake = .{};
    _ = try costs.resolve(a, io, key, 64, root, &engine);
    const filename = try std.fmt.allocPrint(a, "{s}.bin", .{std.fmt.bytesToHex(key.current, .lower)});
    defer a.free(filename);
    for (0..11) |mode| {
        var bad = costs.record(key.current, 64, @splat(1));
        switch (mode) {
            0 => try tmp.dir.writeFile(io, .{ .sub_path = filename, .data = "TFWC2\nshort" }),
            1 => {
                bad = costs.record(@splat(3), 64, @splat(1));
                shared.saveAt(costs.Record, a, io, key.current, bad, root);
            },
            2 => {
                bad.ms[31] = 17;
                shared.saveAt(costs.Record, a, io, key.current, bad, root);
            },
            3 => shared.saveAt(costs.Record, a, io, key.current, costs.record(key.current, 63, @splat(1)), root),
            4 => {
                bad.ms[19] = std.math.nan(f64);
                shared.saveAt(costs.Record, a, io, key.current, costs.record(key.current, 64, bad.ms), root);
            },
            5 => try tmp.dir.writeFile(io, .{ .sub_path = filename, .data = "not a window table" }),
            6, 7, 8 => {
                bad.ms[12] = switch (mode) {
                    6 => std.math.inf(f64),
                    7 => 0,
                    else => -1,
                };
                shared.saveAt(costs.Record, a, io, key.current, costs.record(key.current, 64, bad.ms), root);
            },
            9 => {
                bad.schema = 2;
                shared.saveAt(costs.Record, a, io, key.current, bad, root);
            },
            else => {
                const huge = try a.alloc(u8, (1 << 20) + 1);
                defer a.free(huge);
                @memset(huge, 0);
                try tmp.dir.writeFile(io, .{ .sub_path = filename, .data = huge });
            },
        }
        engine.forwards = 0;
        const got = try costs.resolve(a, io, key, 64, root, &engine);
        try std.testing.expect(!got.hit);
        try std.testing.expectEqual(@as(usize, 64), engine.forwards);
        engine.forwards = 0;
        try std.testing.expect((try costs.resolve(a, io, key, 64, root, &engine)).hit);
        try std.testing.expectEqual(@as(usize, 0), engine.forwards);
    }
}

test "window cost cache: a changed build measures once against its reference and retries measured drift" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var engine: Fake = .{};
    _ = try costs.resolve(a, io, costs.keys("first-build", &parts), 64, root, &engine);
    engine.forwards = 0;
    const stable = try costs.resolve(a, io, costs.keys("second-build", &parts), 64, root, &engine);
    try std.testing.expect(!stable.hit and !stable.retried);
    try std.testing.expectEqual(@as(usize, 64), engine.forwards);
    engine.forwards = 0;
    engine.step = 2_000_000;
    const drift = try costs.resolve(a, io, costs.keys("third-build", &parts), 64, root, &engine);
    try std.testing.expect(!drift.hit and drift.retried);
    try std.testing.expectEqual(@as(usize, 128), engine.forwards);
    engine.forwards = 0;
    try std.testing.expect((try costs.resolve(a, io, costs.keys("third-build", &parts), 64, root, &engine)).hit);
    try std.testing.expectEqual(@as(usize, 0), engine.forwards);
}

test "window cost cache: failed or partial measurements publish no table" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const key = costs.keys("binary", &parts);
    var engine: Fake = .{ .fail_at = 64 };
    try std.testing.expectError(error.PlantedCalibrationFailure, costs.resolve(a, io, key, 64, root, &engine));
    try std.testing.expect(shared.loadAt(costs.Record, a, io, key.current, root) == null);
    engine.fail_at = 0;
    engine.forwards = 0;
    try std.testing.expect(!(try costs.resolve(a, io, key, 64, root, &engine)).hit);
    try std.testing.expectEqual(@as(usize, 64), engine.forwards);
}

test "window cost cache: live widths fit the record and smaller windows leave unused slots zero" {
    try std.testing.expectEqual(@as(usize, 64), q.engine.most_window);
    try std.testing.expectEqual(@as(usize, 584), @sizeOf(costs.Record));
    try std.testing.expectEqual(@as(usize, 552), @offsetOf(costs.Record, "digest"));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var engine: Fake = .{ .window_rows = 16 };
    const key = costs.keys("binary-16", &parts);
    const first = try costs.resolve(a, io, key, 16, root, &engine);
    try std.testing.expectEqual(@as(usize, 16), engine.forwards);
    for (first.measured.ms[16..]) |m| try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(m)));
    engine.forwards = 0;
    try std.testing.expect((try costs.resolve(a, io, key, 16, root, &engine)).hit);
    try std.testing.expectEqual(@as(usize, 0), engine.forwards);
}

test "window cost cache: safe publication preserves the old file and cleans a failed rename" {
    try @import("qwen35_cost_files_test.zig").failure();
}
test "window cost cache: concurrent writers and readers see only complete tables" {
    try @import("qwen35_cost_files_test.zig").concurrent();
}
test "window cost cache: unavailable storage still measures every width" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "not-directory", .data = "keep" });
    const root = try tmp.dir.realPathFileAlloc(io, "not-directory", a);
    defer a.free(root);
    var engine: Fake = .{};
    const key = costs.keys("no-storage", &parts);
    for (0..2) |_| {
        engine.forwards = 0;
        try std.testing.expect(!(try costs.resolve(a, io, key, 64, root, &engine)).hit);
        try std.testing.expectEqual(@as(usize, 64), engine.forwards);
    }
}

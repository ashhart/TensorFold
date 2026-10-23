//! Source selection and actual cache command encoding; Metal arithmetic remains a device gate.
const std = @import("std");
const cache = @import("rotary_cache.zig");
const f = @import("vision_contract.zig");
const ks = @import("kernel_sources");
const a = std.testing.allocator;

test "rotary cache: all six consumers retain the precise fallback and constant address space" {
    for ([_]f.Which{ .target, .mtp, .pooled, .absolute, .mrope, .positioned }) |which| {
        const original = f.source(which);
        const changed = try cache.source(a, original, true);
        defer a.free(changed);
        try std.testing.expect(std.mem.indexOf(u8, changed, "TF_ROPE_ENABLE") != null);
        const body = std.mem.indexOf(u8, ks.flashnext_rotary_scalar, ") {").? + 3;
        try std.testing.expect(std.mem.indexOf(u8, changed, ks.flashnext_rotary_scalar[body..]) != null);
        try std.testing.expect(std.mem.indexOf(u8, changed, "const constant float* LOG2BASE") != null);
    }
}

test "rotary cache: disabled paths allocate nothing and incompatible sources refuse" {
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    const text = f.source(.target);
    try std.testing.expectEqual(text.ptr, (try cache.source(failing.allocator(), text, false)).ptr);
    try std.testing.expect(!failing.has_induced_failure);
    try std.testing.expectError(error.RotarySourceMismatch, cache.source(a, "tf_rotary_scalar incompatible", true));
    try std.testing.expectError(error.OutOfMemory, cache.source(failing.allocator(), text, true));
}

const Enc = struct {
    log: *Log,
    pub fn barrier(e: Enc) void {
        e.log.order[e.log.used] = 'b';
        e.log.used += 1;
    }
    pub fn setPipeline(e: Enc, _: u8) void {
        e.log.order[e.log.used] = 'p';
        e.log.used += 1;
    }
    pub fn setBuffer(e: Enc, b: usize, off: usize, at: usize) void {
        e.log.buffers[at] = .{ b, off };
    }
    pub fn setBytes(e: Enc, data: []const u8, at: usize) void {
        std.debug.assert(at == 3);
        e.log.rows = std.mem.readInt(u32, data[0..4], .little);
        e.log.reset = std.mem.readInt(u32, data[4..8], .little) != 0;
    }
    pub fn dispatchGroups(e: Enc, grid: anytype, threads: anytype) void {
        std.debug.assert(grid.width == 1 and grid.height == 1 and grid.depth == 1);
        std.debug.assert(threads.width == 128 and threads.height == 1 and threads.depth == 1);
        e.log.order[e.log.used] = 'd';
        e.log.used += 1;
    }
};
const Log = struct {
    order: [4]u8 = undefined,
    used: usize = 0,
    buffers: [3][2]usize = undefined,
    rows: u32 = 0,
    reset: bool = false,
};
const Run = struct { rotary_pipe: ?u8, enc: Enc };

test "rotary cache: target-window dispatch fences old reads and new writes with exact offsets" {
    for ([_]usize{ 1, 4, 8, 16 }) |rows| {
        var log: Log = .{};
        var run: Run = .{ .rotary_pipe = 1, .enc = .{ .log = &log } };
        try cache.prepare(&run, .{ .b = @as(usize, 7), .off = @as(usize, 256) }, .{ .b = @as(usize, 9), .off = @as(usize, 64) }, rows);
        try std.testing.expectEqual(@as(usize, 4), log.used);
        try std.testing.expectEqualStrings("bpdb", &log.order);
        try std.testing.expectEqual(@as(u32, @intCast(rows)), log.rows);
        try std.testing.expect(log.reset);
        try std.testing.expectEqualDeep([3][2]usize{ .{ 7, 256 }, .{ 7, 256 }, .{ 9, 64 } }, log.buffers);
    }
}

test "rotary cache: MTP appends reuse the target table and optional admission never adds a refusal" {
    var log: Log = .{};
    var run: Run = .{ .rotary_pipe = 1, .enc = .{ .log = &log } };
    const b = .{ .b = @as(usize, 7), .off = @as(usize, 256) };
    try cache.append(&run, b, b, 4);
    try std.testing.expectEqual(@as(usize, 4), log.used);
    try std.testing.expect(!log.reset);
    try std.testing.expectEqualStrings("bpdb", &log.order);
    try std.testing.expect(!cache.fits(false, 0, cache.bytes));
    try std.testing.expect(!cache.fits(true, 100, 100));
    try std.testing.expect(!cache.fits(true, 100, 100 + cache.growth - 1));
    try std.testing.expect(cache.fits(true, 100, 100 + cache.growth));
    try std.testing.expect(!cache.fits(true, std.math.maxInt(u64), std.math.maxInt(u64)));
}

test "rotary cache: optional table allocation failure selects the original path before shader selection" {
    const Owner = struct {
        rotary_enabled: bool,
        rotary_buffer: ?usize = null,
        calls: usize = 0,
        fail: bool = false,
        pub fn buffer(o: *@This(), n: usize) !usize {
            o.calls += 1;
            if (o.fail) return error.OutOfMemory;
            return n;
        }
    };
    var disabled: Owner = .{ .rotary_enabled = false };
    cache.reserve(&disabled);
    try std.testing.expectEqual(@as(usize, 0), disabled.calls);
    var failed: Owner = .{ .rotary_enabled = true, .fail = true };
    cache.reserve(&failed);
    try std.testing.expect(!failed.rotary_enabled and failed.rotary_buffer == null);
    var owned: Owner = .{ .rotary_enabled = true };
    cache.reserve(&owned);
    try std.testing.expect(owned.rotary_enabled);
    try std.testing.expectEqual(@as(?usize, cache.bytes), owned.rotary_buffer);
}

test "rotary cache: disabled windows are unchanged and invalid active windows issue no commands" {
    var log: Log = .{};
    var run: Run = .{ .rotary_pipe = null, .enc = .{ .log = &log } };
    const b = .{ .b = @as(usize, 0), .off = @as(usize, 0) };
    try cache.prepare(&run, b, b, 16);
    try std.testing.expectEqual(@as(usize, 0), log.used);
    run.rotary_pipe = 1;
    for ([_]usize{ 0, 17 }) |rows| try std.testing.expectError(error.WindowOutOfStep, cache.prepare(&run, b, b, rows));
    try std.testing.expectEqual(@as(usize, 0), log.used);
    try std.testing.expectEqual(@as(usize, 0), cache.data % 4);
    try std.testing.expectEqual(@as(usize, 33168), cache.bytes);
}

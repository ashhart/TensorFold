//! A fault at the first gather observes the real absorb adapter's role without running any GPU arithmetic.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const lanes = @import("lanes");
const P = q.draft.precision;
const T = std.testing;
const Record = struct {
    var role: *P.Role = undefined;
    var observed: ?P.Role = null;
    var host: ?[]align(16) u8 = null;
    fn synchronize(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn upload(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        host = T.allocator.alignedAlloc(u8, .@"16", len) catch return cuda.abi.error_not_ready;
        out.* = host.?.ptr;
        return cuda.abi.success;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) cuda.abi.Result {
        const allocation = host orelse return cuda.abi.error_not_ready;
        if (ptr == null or @intFromPtr(ptr.?) != @intFromPtr(allocation.ptr)) return cuda.abi.error_not_ready;
        T.allocator.free(allocation);
        host = null;
        return cuda.abi.success;
    }
    fn eventCreate(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(1);
        return cuda.abi.success;
    }
    fn eventRecord(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn eventQuery(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn eventDestroy(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        observed = role.*;
        return cuda.abi.error_not_ready;
    }
};
fn function(d: *cuda.Driver) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(1) };
}
fn check(d: anytype, ctx: anytype, s: *lanes.Stream) !void {
    try d.streams.put(T.allocator, s, ctx);
    defer d.streams.deinit(T.allocator);
    Record.role = &d.projection_role;
    const follows: [65]u32 = @splat(1);
    var indices: [65]u32 = undefined;
    for (&indices, 0..) |*v, i| v.* = @intCast(i);
    for ([_]bool{ false, true }) |prompt| {
        for ([_]usize{ 16, 17, 32, 64, 65 }) |rows| for ([_]bool{ false, true }) |explicit| {
            d.projection_role = .proposal;
            Record.observed = null;
            const item: lanes.drafter.Absorb = .{
                .stream = s,
                .features = .{ .buffer = 1, .offset = 0, .rows = @intCast(rows), .row_bytes = 512, .space = .device, .dtype = .bf16 },
                .start = 0,
                .follow = follows[0..rows],
                .rows = if (explicit) indices[0..rows] else null,
                .prompt = prompt,
            };
            if (explicit and rows > 64) {
                try T.expectError(error.BadDraftRows, d.drafter().absorb(&.{item}));
                try T.expect(Record.observed == null);
            } else {
                try T.expectError(error.NotReady, d.drafter().absorb(&.{item}));
                try T.expectEqual(@as(?P.Role, if (prompt) .prompt else .live), Record.observed);
            }
            try T.expectEqual(P.Role.proposal, d.projection_role);
        };
    }
}
test "Draft precision: DFlash real absorb adapter forwards both row roles and restores them on failure" {
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Record.launch;
    driver.api.cuStreamSynchronize = Record.synchronize;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.scratch_rows = q.engine.grid_rows;
    e.d = &driver;
    e.stream = .{ .d = &driver, .handle = null };
    var d: q.draft.Draft = undefined;
    d.gpa = T.allocator;
    d.io = T.io;
    d.e = &e;
    d.streams = .empty;
    d.c = .{ .hidden = 256, .intermediate = 256, .layers = 1, .vocab = 256, .heads = 2, .kv_heads = 1, .head_dim = 128, .block = 8, .mask = 255, .window = 2048, .group = 16, .rank = 2, .topk = 2, .eps = 1e-6, .theta = 10000000, .taps = @splat(0), .n_taps = 1 };
    d.fns.gather_taps = function(&driver);
    d.fns.gather_rows = function(&driver);
    d.taps_in = 1;
    var ctx: q.draft.Ctx = .{ .ring = .{ .d = &driver, .ptr = 1, .len = 1 } };
    var s = try lanes.Stream.init(T.allocator, .{ .id = "dflash", .prompt = &.{1}, .max_new = 8 });
    defer s.deinit(T.allocator);
    try check(&d, &ctx, &s);
}
test "Draft precision: MTP real absorb adapter forwards both row roles and restores them on failure" {
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Record.launch;
    driver.api.cuMemcpyHtoDAsync_v2 = Record.upload;
    driver.api.cuStreamSynchronize = Record.synchronize;
    driver.api.cuMemHostAlloc = Record.hostAlloc;
    driver.api.cuMemFreeHost = Record.hostFree;
    driver.api.cuEventCreate = Record.eventCreate;
    driver.api.cuEventRecord = Record.eventRecord;
    driver.api.cuEventQuery = Record.eventQuery;
    driver.api.cuEventDestroy_v2 = Record.eventDestroy;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.scratch_rows = q.engine.grid_rows;
    e.d = &driver;
    e.stream = .{ .d = &driver, .handle = null };
    e.g.hidden = 256;
    var d: q.mtp.Mtp = undefined;
    d.gpa = T.allocator;
    d.io = T.io;
    d.e = &e;
    d.streams = .empty;
    d.ring_fns.gather_taps = function(&driver);
    d.ring_fns.gather_rows = function(&driver);
    d.taps_in = 1;
    d.batch = false; // the serial absorb's roles and refusals; the batch has its own GPU check
    d.ids = 0x2000;
    var staged_storage: [4096]u8 align(16) = undefined;
    d.staged = .{ .d = &driver, .bytes = &staged_storage };
    var ctx: q.mtp.Ctx = .{ .ring = .{ .d = &driver, .ptr = 1, .len = 1 }, .last = .{ .d = &driver, .ptr = 1, .len = 1 }, .step = .{ .d = &driver, .ptr = 1, .len = 1 }, .held_dev = .{ .d = &driver, .ptr = 1, .len = 1 } };
    var s = try lanes.Stream.init(T.allocator, .{ .id = "mtp", .prompt = &.{1}, .max_new = 8 });
    defer s.deinit(T.allocator);
    try check(&d, &ctx, &s);
    try e.stream.synchronize();
    try T.expect(!cuda.copy_source.keepDriver(&driver));
    try T.expect(Record.host == null);
}

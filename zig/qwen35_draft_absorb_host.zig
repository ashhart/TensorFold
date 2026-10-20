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
    var e: q.engine.Engine = undefined;
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
    var e: q.engine.Engine = undefined;
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
    var ctx: q.mtp.Ctx = .{ .ring = .{ .d = &driver, .ptr = 1, .len = 1 }, .last = .{ .d = &driver, .ptr = 1, .len = 1 } };
    var s = try lanes.Stream.init(T.allocator, .{ .id = "mtp", .prompt = &.{1}, .max_new = 8 });
    defer s.deinit(T.allocator);
    try check(&d, &ctx, &s);
}

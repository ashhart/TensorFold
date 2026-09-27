//! A deterministic GPU transition table isolates serial scheduling and ownership.
//! Full-checkpoint arithmetic/cache comparisons live in cache_checks.checkSerial.
const std = @import("std");
const mx = @import("mlx.zig");
const Cache = @import("model.zig").Cache;
pub const Fixture = struct {
    pub const SerialPass = struct {
        scope: mx.Scope = .{},
        logits: mx.Array = mx.empty,
        token: mx.Array = mx.empty,
        pub fn deinit(p: *@This()) void {
            p.scope.deinit();
        }
    };
    kernels: mx.Kernels,
    table: mx.Array,
    cache: [1]Cache = .{.{}},
    position: i32 = 0,
    forwards: usize = 0,
    pub fn init() !Fixture {
        var values: [32 * 32]f32 = @splat(-10);
        for (0..32) |i| values[i * 32 + @min(i + 1, 31)] = 10;
        var s = mx.Scope{};
        defer s.deinit();
        return .{ .kernels = mx.Kernels.init(), .table = try mx.retain(try s.data(&values, &.{ 32, 32 }, mx.f32t)) };
    }
    pub fn deinit(m: *Fixture) void {
        m.cache[0].deinit();
        mx.free(m.table);
        m.kernels.deinit();
    }
    pub fn forwardSerialArray(m: *Fixture, token: mx.Array) !SerialPass {
        var p = SerialPass{};
        errdefer p.deinit();
        p.token = try p.scope.cast(token, mx.c.MLX_UINT32);
        p.logits = try p.scope.take(m.table, token, 0);
        m.forwards += 1;
        return p;
    }
    pub fn commitSerialQueued(m: *Fixture, p: *SerialPass) !void {
        const value = if (m.cache[0].a.ctx == null) p.token else try p.scope.cat(&.{ m.cache[0].a, p.token }, 0);
        try mx.replace(&m.cache[0].a, value);
        m.position += 1;
    }
};
fn eos(id: i32) bool {
    return id == 7;
}
pub fn exercise(a: std.mem.Allocator, first: u32, limit: usize) !void {
    var m = try Fixture.init();
    defer m.deinit();
    var generated: std.ArrayList(u32) = .empty;
    defer generated.deinit(a);
    if (limit > 0) try generated.append(a, first);
    const result = try @import("serial_pipeline.zig").generate(Fixture, &m, a, &generated, limit, .{ .metal = true, .temperature = 0 }, eos, null);
    const count = if (first <= 7) @min(limit, 8 - first) else limit;
    try std.testing.expectEqual(count, generated.items.len);
    for (generated.items, 0..) |token, i| try std.testing.expectEqual(@min(first + i, 31), token);
    const rounds = count -| 1;
    try std.testing.expectEqual(rounds, result.rounds);
    try std.testing.expectEqual(@as(i32, @intCast(rounds)), m.position);
    const queued = @min(rounds, limit -| 2);
    try std.testing.expectEqual(queued, result.queued_ahead);
    try std.testing.expectEqual(if (rounds == 0) @as(usize, 0) else queued + 1, m.forwards);
    if (rounds > 0) {
        try mx.eval(m.cache[0].a);
        try std.testing.expectEqual(rounds, mx.c.mlx_array_size(m.cache[0].a));
        for (mx.c.mlx_array_data_uint32(m.cache[0].a)[0..rounds], 0..) |token, i| try std.testing.expectEqual(@min(first + i, 31), token);
    } else try std.testing.expect(m.cache[0].a.ctx == null);
}
pub fn check() !void {
    try mx.init();
    defer mx.shutdown();
    for ([_]u32{ 0, 4, 6, 7, 8 }) |first| for ([_]usize{ 0, 1, 2, 3, 8, 17, 32 }) |limit| try exercise(mx.allocator, first, limit);
    try mx.check(mx.c.mlx_synchronize(mx.stream));
    var active: usize = 0;
    try mx.check(mx.c.mlx_get_active_memory(&active));
    try std.testing.expectEqual(@as(usize, 0), active);
    std.debug.print("PASS: 35 GPU serial scheduling cases; EOS/budget output, committed cache, queued-before-read counts and cleanup exact\n", .{});
}

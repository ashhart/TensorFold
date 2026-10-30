//! Exercise CudaEngine with an actual driver-table instantiation refusal, rather than a mock capture callback.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const gc = @import("graph_cache.zig");
const health = @import("context_health.zig");
const Fake = struct {
    var captured = false;
    var attempts: usize = 0;
    var graphs: usize = 0;
    var eager: usize = 0;
    var votes: usize = 0;
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        captured = true;
        return 0;
    }
    fn end(_: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
        captured = false;
        graphs += 1;
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn instantiate(_: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
        attempts += 1;
        return 800;
    }
    fn destroy(_: abi.Graph) callconv(.c) abi.Result {
        graphs -= 1;
        return 0;
    }
    fn body(_: *anyopaque, _: Stream) !void {
        if (!captured) eager += 1;
    }
    fn agree(_: *anyopaque, ok: bool) !bool {
        votes += 1;
        return ok;
    }
};

test "CUDA graph refusal: 800 destroys capture and runs eager once; later same and different shapes do not retry" {
    Fake.captured = false;
    Fake.attempts = 0;
    Fake.graphs = 0;
    Fake.eager = 0;
    Fake.votes = 0;
    var d: Driver = undefined;
    d.api.cuStreamBeginCapture_v2 = Fake.begin;
    d.api.cuStreamEndCapture = Fake.end;
    d.api.cuGraphInstantiateWithFlags = Fake.instantiate;
    d.api.cuGraphDestroy = Fake.destroy;
    const context: abi.Context = @ptrFromInt(0x3253);
    try health.retain(&d, context);
    defer health.release(&d, context);
    const source = @import("copy_source.zig");
    const previous = source.current();
    source.setCurrent(context);
    defer source.setCurrent(previous);
    const stream: Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var engine: gc.CudaEngine = .{ .d = &d };
    var cache = gc.Cache(u32).init(std.testing.allocator, engine.engine(), .{});
    defer cache.deinit();
    var ctx: u8 = 0;
    cache.agree = .{ .ctx = &ctx, .all = Fake.agree };
    const body: gc.Body = .{ .ctx = &ctx, .run = Fake.body };
    for ([_]u32{ 7, 7, 8, 8 }) |key| try std.testing.expectEqual(gc.Outcome.eager, try cache.run(key, stream, 3, body));
    try std.testing.expectEqual(@as(usize, 1), Fake.attempts);
    try std.testing.expectEqual(@as(usize, 0), Fake.graphs);
    try std.testing.expectEqual(@as(usize, 4), Fake.eager);
    try std.testing.expectEqual(@as(usize, 2), Fake.votes);
    try std.testing.expect(!Fake.captured and cache.failed);
    try std.testing.expect(health.healthy(&d, context));
    try std.testing.expectEqual(@as(usize, 0), cache.count());
}

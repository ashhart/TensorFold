//! The generated native graph must emit the same greedy tokens through the shared server host and direct session.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const api = @import("engine_api");
const host = @import("qwen27_host");
const graph = @import("synthetic_graph.zig");
const Box = struct {
    a: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,
    message: []const u8 = "",
    fn event(ptr: *anyopaque, _: api.Id, value: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        switch (value.*) {
            .prefilled => {},
            .tokens => |ids| b.tokens.appendSlice(b.a, ids) catch {},
            .finished => |f| {
                b.done = f.reason;
                b.message = b.a.dupe(u8, f.message) catch "allocation failed";
            },
        }
    }
    fn wait(b: *Box) !api.Reason {
        for (0..6000) |_| {
            b.mutex.lockUncancelable(b.io);
            const done = b.done;
            b.mutex.unlock(b.io);
            if (done) |value| return value;
            try std.Io.sleep(b.io, .fromMilliseconds(10), .awake);
        }
        return error.NoCompletion;
    }
};
pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try graph.create(a);
    defer model.deinit();
    var runner = try q.decode_round.Runner.init(a, model, 1, 256);
    defer runner.deinit();
    const direct = q.session.Session{ .runner = &runner };
    var prompt: [130]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast((i * 17 + 7) % 96);
    try direct.prefill(&prompt, 16);
    var expected: [8]u32 = undefined;
    for (&expected, 0..) |*id, i| {
        id.* = try direct.greedy();
        if (i + 1 < expected.len) try direct.step(id.*);
    }
    var box = Box{ .a = a, .io = init.io };
    const served = try host.attach(a, init.io, model, 256, false);
    defer host.close(served);
    const request = api.Request{ .prompt = &prompt, .max_tokens = 8, .drafts = false, .chunks = &.{ 16, 32, 48, 64, 80, 96, 112, 128 } };
    try served.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    const reason = try box.wait();
    if (reason != .length) {
        std.debug.print("served failed: {s}\n", .{box.message});
        return error.ServedFailure;
    }
    if (!std.mem.eql(u32, &expected, box.tokens.items)) return error.ServedTokensDiffer;
    std.debug.print("qwen27 served synthetic: 8 greedy tokens byte-equal to direct CLI session after a 130-token chunked prompt\n", .{});
}

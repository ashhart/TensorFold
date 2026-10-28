//! The pool's change notice: once per outermost change, after nested calls and rollbacks, with counts that add up.
const std = @import("std");
const kv = @import("kv_pool.zig");

const Seen = struct {
    calls: usize = 0,
    bad: usize = 0, // notices whose counts did not add up
    fn call(ctx: *anyopaque, p: *const kv.Pool) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        s.calls += 1;
        const taken = p.total() - p.nfree;
        if (p.available() + p.promised + taken != p.total() or p.depth != 0) s.bad += 1;
    }
    fn on(s: *Seen, p: *kv.Pool) void {
        p.notify = .{ .ctx = s, .call = call };
    }
    fn expectOnce(s: *Seen) !void {
        try std.testing.expectEqual(@as(usize, 1), s.calls);
        try std.testing.expectEqual(@as(usize, 0), s.bad);
        s.calls = 0;
    }
};

test "each public change notifies once, nested ones included" {
    const gpa = std.testing.allocator;
    var p = try kv.Pool.init(gpa, 16, 64);
    defer p.deinit();
    var seen: Seen = .{};
    seen.on(&p);
    var l: kv.Lease = .{};
    try p.promise(&l, 6);
    try seen.expectOnce();
    const id = try p.take(&l);
    try seen.expectOnce();
    try p.cover(&l, 4, null); // three more takes inside one cover
    try seen.expectOnce();
    p.retain(id);
    try seen.expectOnce();
    p.release(id); // release calls drop
    try seen.expectOnce();
    const run = p.claimRun(&l, 2).?;
    try seen.expectOnce();
    p.releaseRun(run, 2);
    try seen.expectOnce();
    p.finish(&l); // finish calls unpromise
    try seen.expectOnce();
    try std.testing.expect(p.idle());
}

test "a failed cover notifies once, after its rollback, with the counts it restored" {
    const gpa = std.testing.allocator;
    var p = try kv.Pool.init(gpa, 8, 64);
    defer p.deinit();
    var l: kv.Lease = .{};
    defer p.finish(&l);
    try p.promise(&l, 4);
    var seen: Seen = .{};
    seen.on(&p);
    const Refuse = struct {
        fn write(_: *anyopaque, _: []const u32, _: usize) anyerror!void {
            return error.TableFailed;
        }
    };
    var ctx: u8 = 0;
    try std.testing.expectError(error.TableFailed, p.cover(&l, 3, .{ .ctx = &ctx, .write = Refuse.write }));
    try seen.expectOnce();
    try std.testing.expectEqual(@as(usize, 8), p.nfree); // the takes went back to the promise
    try std.testing.expectEqual(@as(usize, 4), p.promised);
}

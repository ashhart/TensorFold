const std = @import("std");
const cuda = @import("cuda");
const mock = @import("load_mock.zig");
const sg = @import("sessions_gpu.zig");
const t = std.testing;
test "Completion ownership: complete bounded restore and rope read retain owned copies when callers unwind" {
    var d = mock.init();
    var runner: @import("run.zig").Runner = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    defer runner.deinit();
    var cfg: @import("config.zig").Config = .{};
    var widths: @import("block.zig").Widths = .{};
    var f = @import("forward.zig").Forward.init(t.allocator, &cfg, &widths, .{}, &runner, undefined);
    var dst = try cuda.DeviceBuffer.alloc(&d, 16);
    defer {
        mock.fail_sync = false;
        runner.stream.synchronize() catch unreachable;
        f.uploads.close() catch unreachable;
        dst.free();
        f.arena.deinit();
    }
    try runner.external("s.L0.swa.s", dst.ptr);
    var slot: @import("kv").sessions.Slot = undefined;
    slot.id = 0;
    var k: @import("cuda_kv.zig").Kv = undefined;
    k.slot = &slot;
    k.opts = .{ .tokens = 256, .capacity = 64 };
    var session: sg.Sessions = undefined;
    session.f = &f;
    session.kx = &k;
    var pcs = [_]sg.Piece{.{ .role = "s.L0.swa.s", .len = 16 }};
    session.pieces = &pcs;
    session.measure = .{ .window = cfg.window, .rings = 1, .carries = 0, .head_dim = cfg.head_dim, .hidden = cfg.hidden, .hc = cfg.hc_mult, .replay = false };
    session.ds_rings = false;
    session.ds_set = null;
    var valid = [_]u64{0};
    session.ds_valid = &valid;
    var wire: [sg.Sessions.slot_bytes]u8 = @splat(0);
    std.mem.writeInt(u64, wire[0..8], 8, .little);
    mock.fail_sync = true;
    {
        const data = try t.allocator.dupe(u8, &@as([16]u8, @splat(0x6b)));
        defer t.allocator.free(data);
        try t.expectError(error.NotReady, session.uploadBlobs(&.{ .{ .name = @constCast("slot"), .bytes = &wire }, .{ .name = @constCast("s.L0.swa.s"), .bytes = data } }));
    }
    try t.expectEqual(@as(u64, 0), f.slot.pos);
    try t.expect(f.uploads.pending);
    try t.expect(std.mem.allEqual(u8, mock.pinned.?[0..16], 0x6b));
    mock.fail_sync = false;
    var again: [16]u8 = @splat(0x6b);
    try session.uploadBlobs(&.{ .{ .name = @constCast("slot"), .bytes = &wire }, .{ .name = @constCast("s.L0.swa.s"), .bytes = &again } });
    try t.expectEqual(@as(u64, 8), f.slot.pos);
    try f.uploads.close();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var rope: [48]u8 = @splat(0x39);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "rope.bin", .data = &rope });
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const path = try std.fs.path.join(t.allocator, &.{ dir, "rope.bin" });
    defer t.allocator.free(path);
    const Sink = struct {
        f: *@import("forward.zig").Forward,
        dst: u64,
        pub fn table(s: @This(), _: usize, bytes: []const u8) !void {
            try s.f.upload(s.dst, bytes);
        }
        pub fn done(_: @This()) !void {}
    };
    mock.fail_sync = true;
    try t.expectError(error.NotReady, @import("boot_mem.zig").uploadRope(t.allocator, t.io, path, 16, Sink{ .f = &f, .dst = dst.ptr }));
    try t.expect(f.uploads.pending);
    try t.expect(std.mem.allEqual(u8, mock.pinned.?[0..16], 0x39));
    try t.expectEqual(@as(usize, 0), mock.early_free);
}

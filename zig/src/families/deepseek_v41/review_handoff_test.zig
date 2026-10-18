const std = @import("std");
const sg = @import("sessions_gpu.zig");
const fwd = @import("forward.zig");
const t = std.testing;
fn actualStart(ctx: *anyopaque, _: u32) u64 {
    const pass: *@import("dspark_gpu.zig").GpuPass = @ptrCast(@alignCast(ctx));
    return pass.valid;
}
fn noSet(_: *anyopaque, _: u32, _: u64) void {}
test "Completion ownership: snapshots query the live drafter interval after an unavailable-taps ingest" {
    var pass: @import("dspark_gpu.zig").GpuPass = undefined;
    pass.valid = 0;
    pass.ingest_rows = 8;
    pass.begun = null;
    pass.dl = null;
    var f: fwd.Forward = undefined;
    f.link = null;
    pass.f = &f;
    try pass.pass().ingest(0, 5, .{ .address = 0, .rows = 0, .stride = 0 }, &.{0});
    var slot: @import("kv").sessions.Slot = undefined;
    slot.id = 0;
    var k: @import("cuda_kv.zig").Kv = undefined;
    k.slot = &slot;
    var s: sg.Sessions = undefined;
    s.kx = &k;
    var stale = [_]u64{0};
    s.ds_valid = &stale;
    s.ds_set = .{ .ctx = &pass, .set = noSet, .get = actualStart };
    try t.expectEqual(@as(u64, 6), s.validStart());
}
test "Completion ownership: a full-mode piece hands off drafter rings before taking its session snapshot" {
    const End = struct {
        bytes: [2]u8 = undefined,
        n: usize = 0,
        pub fn snapshot(e: *@This()) !void {
            e.bytes[e.n] = 'S';
            e.n += 1;
        }
        pub fn handoff(e: *@This()) !void {
            e.bytes[e.n] = 'H';
            e.n += 1;
        }
    };
    var end: End = .{};
    try @import("piece_handoff.zig").finish(&end, true, true);
    try t.expectEqualStrings("HS", &end.bytes);
}

fn setStart(ctx: *anyopaque, _: u32, start: u64) void {
    const pass: *@import("dspark_gpu.zig").GpuPass = @ptrCast(@alignCast(ctx));
    pass.valid = start;
}
test "Completion ownership: multiple piece snapshots restore identical synthetic ring bytes and the live valid interval" {
    const mock = @import("load_mock.zig");
    const cuda = @import("cuda");
    var d = mock.init();
    var runner: @import("run.zig").Runner = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    defer runner.deinit();
    var cfg: @import("config.zig").Config = .{ .window = 2, .dspark_block = 1 };
    var widths: @import("block.zig").Widths = .{};
    var f = fwd.Forward.init(t.allocator, &cfg, &widths, .{}, &runner, undefined);
    var dst = try cuda.DeviceBuffer.alloc(&d, 32);
    defer {
        runner.stream.synchronize() catch unreachable;
        f.uploads.close() catch unreachable;
        dst.free();
        f.arena.deinit();
    }
    try runner.external("s.L40.ring.s", dst.ptr);
    var pool_slot: @import("kv").sessions.Slot = undefined;
    pool_slot.id = 0;
    var k: @import("cuda_kv.zig").Kv = undefined;
    k.slot = &pool_slot;
    var session: sg.Sessions = undefined;
    session.gpa = t.allocator;
    session.f = &f;
    session.kx = &k;
    session.slot_set = null;
    session.bounded_bytes = 32;
    var pieces = [_]sg.Piece{.{ .role = "s.L40.ring.s", .len = 32 }};
    session.pieces = &pieces;
    session.measure = .{ .window = 2, .rings = 0, .carries = 0, .head_dim = cfg.head_dim, .hidden = cfg.hidden, .hc = cfg.hc_mult, .ds_blocks = 1, .replay = false };
    session.ds_rings = true;
    var stale = [_]u64{0};
    session.ds_valid = &stale;
    var pass: @import("dspark_gpu.zig").GpuPass = undefined;
    pass.f = &f;
    pass.valid = 0;
    pass.ingest_rows = 2;
    pass.begun = null;
    pass.dl = null;
    session.ds_set = .{ .ctx = &pass, .set = setStart, .get = actualStart };
    var reference: [32]u8 = @splat(0);
    @memset(mock.memory[1][0..32], 0);
    for (0..3) |part| {
        const start = part * 2;
        const end = start + 2;
        const have = part != 1;
        if (have) for (start..end) |row| for (0..8) |j| {
            reference[(row % 4) * 8 + j] = @truncate(row * 13 + j);
        };
        f.slot.pos = end;
        var saved_image: ?*sg.Sessions.Snap = null;
        const End = struct {
            session: *sg.Sessions,
            pass: *@import("dspark_gpu.zig").GpuPass,
            start: usize,
            end: usize,
            have: bool,
            out: *?*sg.Sessions.Snap,
            pub fn handoff(e: @This()) !void {
                if (e.have) {
                    for (e.start..e.end) |row| for (0..8) |j| {
                        mock.memory[1][(row % 4) * 8 + j] = @truncate(row * 13 + j);
                    };
                } else try e.pass.pass().ingest(0, e.start, .{ .address = 0, .rows = 0, .stride = 0 }, &.{ 0, 1 });
            }
            pub fn snapshot(e: @This()) !void {
                e.out.* = try e.session.snapshot();
            }
        };
        try @import("piece_handoff.zig").finish(End{ .session = &session, .pass = &pass, .start = start, .end = end, .have = have, .out = &saved_image }, true, true);
        const saved = saved_image.?;
        defer sg.Sessions.Snap.release(saved);
        try t.expectEqualSlices(u8, &reference, saved.bytes);
        const valid = @max(pass.valid, end -| @as(u64, 2));
        @memset(mock.memory[1][0..32], 0xff);
        pass.valid = 0;
        try session.uploadSnap(saved);
        try t.expectEqualSlices(u8, &reference, mock.memory[1][0..32]);
        try t.expectEqual(valid, pass.valid);
    }
}

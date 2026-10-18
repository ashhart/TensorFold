//! Regression witnesses call real H3 forward, host allocation, geometry and destruction code.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const host = @import("cuda_host.zig");
const mock = @import("cuda_runtime_fake.zig");
test "h3 runtime: failed forwardHost growth keeps old optional owner" {
    const m = try mock.model();
    defer mock.clean(m);
    m.inputs[0] = try cuda.DeviceBuffer.alloc(&m.driver, 2);
    const old = m.inputs[0].?.ptr;
    mock.alloc_fail = true;
    defer mock.alloc_fail = false;
    var in = mock.input();
    try std.testing.expectError(error.NotReady, host.forwardHost(m, &in));
    const was_freed = mock.last_freed == old;
    if (was_freed) m.inputs[0] = null;
    try std.testing.expect(!was_freed);
    try std.testing.expectEqual(old, m.inputs[0].?.ptr);
    try std.testing.expectEqual(@as(usize, 2), m.inputs[0].?.len);
    m.inputs[0].?.free();
    m.inputs[0] = null;
    try std.testing.expectEqual(@as(usize, 1), mock.frees);
}
test "h3 runtime: host mapped lookup unwind and normal free balance accounting" {
    const m = try mock.model();
    defer mock.clean(m);
    const before = cuda.usage(false).host;
    mock.map_fail = true;
    var dev: u64 = 0;
    defer if (mock.hosts != 0) {
        var b: cuda.HostBuffer = .{ .d = &m.driver, .bytes = mock.memory[0..64] };
        b.free();
    };
    try std.testing.expectError(error.NotReady, host.hostAlloc(m, 64, &dev));
    try std.testing.expectEqual(@as(usize, 0), mock.hosts);
    try std.testing.expectEqual(before, cuda.usage(false).host);
    mock.map_fail = false;
    const ptr = try host.hostAlloc(m, 64, &dev);
    host.hostFree(m, ptr);
    try std.testing.expectEqual(before, cuda.usage(false).host);
}
test "h3 runtime: failed forward completion retains every parent at model close" {
    const m = try mock.model();
    var in = mock.input();
    mock.wait_fail = true;
    try std.testing.expectError(error.NotReady, m.forward(&in));
    try std.testing.expect(mock.pending);
    m.deinit();
    const destroyed = mock.unloads != 0;
    defer if (!destroyed) mock.clean(m);
    try std.testing.expectEqual(@as(usize, 0), mock.unsafe_frees);
    try std.testing.expect(!destroyed);
}
test "h3 runtime: geometry resize refuses disposal after failed forward completion" {
    const m = try mock.model();
    defer mock.clean(m);
    var in = mock.input();
    mock.wait_fail = true;
    try std.testing.expectError(error.NotReady, m.forward(&in));
    const used = m.geometry.used;
    h3.geometry.release(m);
    try std.testing.expectEqual(@as(usize, 0), mock.unsafe_frees);
    try std.testing.expectEqual(used, m.geometry.used);
    in.rows = 1;
    try std.testing.expectError(error.NotReady, m.forward(&in));
    try std.testing.expectEqual(@as(usize, 0), mock.unsafe_frees);
    try std.testing.expectEqual(used, m.geometry.used);
}
test "h3 runtime: equal dimensions and same slot pointer refresh inverse map like fresh input" {
    var repeated: [2]i32 = undefined;
    for (0..2) |fresh| {
        const m = try mock.model();
        defer mock.clean(m);
        var in = mock.input();
        mock.wait_fail = true;
        if (fresh == 0) {
            mock.slot = .{ 0, 1 };
            try std.testing.expectError(error.NotReady, m.forward(&in));
            mock.pending = false;
        }
        mock.slot = .{ 1, 0 };
        defer mock.slot = .{ 0, 1 };
        try std.testing.expectError(error.NotReady, m.forward(&in));
        const got = @as([*]const i32, @ptrFromInt(m.geometry.row_of))[0..2];
        if (fresh == 0) @memcpy(&repeated, got) else try std.testing.expectEqualSlices(i32, got, &repeated);
        try std.testing.expectEqualSlices(i32, &.{ 1, 0 }, got);
    }
}

test "h3 runtime: keep exceeds actual video choices before dispatch" {
    const m = try mock.model();
    defer mock.clean(m);
    var in = mock.input();
    var slots = [_]i32{ 0, 64, 128, 192 };
    var sizes = [_]i32{ 1, 1, 1, 1 };
    var lines = [_]i32{ 0, 0, 0, 0 };
    in.rows = 4;
    in.tiles = 4;
    in.prefix_tiles = 2;
    in.keep = 3;
    in.slot = @intFromPtr(&slots);
    in.sizes = @intFromPtr(&sizes);
    in.line = @intFromPtr(&lines);
    mock.wait_fail = true;
    try std.testing.expectError(error.BadH3Keep, m.forward(&in));
    try std.testing.expectEqual(@as(usize, 0), mock.at);
}

test "h3 runtime: host optional attention output uses separate device storage and host readback" {
    const m = try mock.model();
    defer mock.clean(m);
    var out: [256]u16 = @splat(17);
    var in = mock.input();
    in.attention_out = @intFromPtr(&out);
    const p = try m.plan(&in);
    try m.drain();
    const staged = try host.stage(m, &in, p);
    try std.testing.expect(staged.attention_out != in.attention_out);
    try std.testing.expect(staged.attention_out == m.inputs[7].?.ptr);
    try std.testing.expect(std.mem.allEqual(u16, &out, 17));
    @memset(@as([*]u16, @ptrFromInt(staged.attention_out))[0..out.len], 23);
    try host.readback(m, &in, p);
    try std.testing.expect(std.mem.allEqual(u16, &out, 23));
}
test "h3 runtime: packing rejects invalid lines duplicates sizes and holes before device buffer replacement" {
    const m = try mock.model();
    defer mock.clean(m);
    var in = mock.input();
    mock.line[0] = 1;
    defer mock.line[0] = 0;
    try std.testing.expectError(error.BadH3Line, m.forward(&in));
    mock.line[0] = 0;
    mock.slot = .{ 0, 0 };
    defer mock.slot = .{ 0, 1 };
    try std.testing.expectError(error.BadH3Packing, m.forward(&in));
    mock.slot = .{ 0, 2 };
    try std.testing.expectError(error.BadH3Packing, m.forward(&in));
    mock.slot = .{ 0, 1 };
    mock.sizes[0] = 0;
    defer mock.sizes[0] = 2;
    try std.testing.expectError(error.BadH3Packing, m.forward(&in));
    mock.sizes[0] = 65;
    try std.testing.expectError(error.BadH3Packing, m.forward(&in));
    mock.sizes[0] = 1;
    try std.testing.expectError(error.BadH3Packing, m.forward(&in));
    try std.testing.expectEqual(@as(usize, 0), m.geometry.used);
}

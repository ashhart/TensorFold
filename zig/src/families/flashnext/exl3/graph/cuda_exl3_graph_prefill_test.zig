//! Host tests of the EXL3 prefill graph keys, retention and sighting history.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const memory = @import("cuda_exl3_graph_memory.zig");
const gprefill = @import("cuda_exl3_graph_prefill.zig");
const gdriver = @import("cuda_exl3_graph_test_driver.zig");
const PKey = gprefill.PKey;
const PGdn = gprefill.PGdn;
const GraphTestDriver = gdriver.GraphTestDriver;

test "exl3: GDN prefill keys bind the actual layer buffers including driver and extent" {
    var driver: cuda.Driver = undefined;
    var other_driver: cuda.Driver = undefined;
    var conv = [_]cuda.DeviceBuffer{
        .{ .d = &driver, .ptr = 100, .len = 16 },
        .{ .d = &driver, .ptr = 200, .len = 16 },
        .{ .d = &driver, .ptr = 300, .len = 16 },
    };
    var state = conv;
    for (&state) |*buffer| buffer.ptr += 1000;
    const sequence = .{ .conv = &conv, .state = &state };
    var key = std.mem.zeroes(PKey);
    key.layer = 2;
    key.seq = .{ conv[0].ptr, 0, 500, 4096, state[0].ptr };
    key.gdn = PGdn.of(sequence, key.layer);
    const original = key.gdn.?;
    try std.testing.expectEqual(@as(u64, 300), original.conv.address);
    try std.testing.expectEqual(@as(u64, 1300), original.state.address);
    for ([_]bool{ false, true }) |recurrent| {
        const buffer = if (recurrent) &state[2] else &conv[2];
        const saved = buffer.*;
        defer buffer.* = saved;
        for (0..3) |field| {
            buffer.* = saved;
            switch (field) {
                0 => buffer.ptr += 1,
                1 => buffer.len += 1,
                2 => buffer.d = &other_driver,
                else => unreachable,
            }
            var changed = key;
            changed.gdn = PGdn.of(sequence, key.layer);
            try std.testing.expectEqual(key.seq, changed.seq);
            try std.testing.expect(!std.meta.eql(gprefill.pshape(key), gprefill.pshape(changed)));
        }
    }
    try std.testing.expectEqualDeep(original, PGdn.of(sequence, key.layer));
}

test "exl3: prefill retained graphs release their graph budget ticket on retirement" {
    gprefill.clearPrefill();
    try memory.configure(.{ .reservation_bytes = 100, .headroom_bytes = 0, .charge = .{ .exec = 10, .template = 10, .transient = 10 }, .prefill_charge = .{ .exec = 5, .template = 0, .transient = 10 } });
    const exec_ticket = try memory.admit(.prefill, 0, 100);
    try memory.installed(exec_ticket);
    try std.testing.expectEqual(@as(u64, 5), memory.charged());
    var driver = GraphTestDriver.driver();
    GraphTestDriver.exec_drops = 0;
    gprefill.pentries[0] = .{ .key = std.mem.zeroes(PKey), .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 1, .admission = exec_ticket, .stream = .{ .d = &driver, .handle = @ptrFromInt(1) } };
    gprefill.clearPrefill();
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.exec_drops);
    try std.testing.expectEqual(@as(u64, 0), memory.charged());
    try std.testing.expectError(error.StaleGraphTicket, memory.release(exec_ticket));
}

test "prefill graph update candidates tolerate a dynamic position only" {
    const base: PKey = .{ .layer = 7, .rows = 4096, .sub = 256, .pos0 = 4096, .deep = 1, .base = 2, .base_cap = 4096, .seq = .{ 3, 4, 5, 4096, 6 }, .wait_count = 0, .wait_key = 0, .scores = 13, .tiles = true };
    var moved = base;
    moved.pos0 = 8192;
    try std.testing.expect(!std.meta.eql(base, moved));
    try std.testing.expect(std.meta.eql(gprefill.pshape(base), gprefill.pshape(moved)));

    moved.seq[0] += 1;
    try std.testing.expect(!std.meta.eql(gprefill.pshape(base), gprefill.pshape(moved)));
    moved = base;
    moved.rows += 1;
    try std.testing.expect(!std.meta.eql(gprefill.pshape(base), gprefill.pshape(moved)));
    moved = base;
    moved.wait_key += 1;
    try std.testing.expect(!std.meta.eql(gprefill.pshape(base), gprefill.pshape(moved)));
}

test "exl3: prefill warmup history retains two lanes and two chunk banks" {
    gprefill.pseen = @splat(null);
    gprefill.pseen_at = 0;
    defer {
        gprefill.pseen = @splat(null);
        gprefill.pseen_at = 0;
    }
    var keys: [2 * 2 * wt.layers_n]PKey = undefined;
    var count: usize = 0;
    for (0..2) |sequence| for (0..2) |bank| for (0..wt.layers_n) |layer| {
        keys[count] = .{
            .layer = layer,
            .rows = 4096,
            .sub = 256,
            .pos0 = bank * 4096,
            .deep = 100 + bank,
            .base = 200,
            .base_cap = 8192,
            .seq = .{ 1, 2, 300 + sequence, 8192, 4 },
            .wait_count = 0,
            .wait_key = 0,
            .scores = 10,
            .tiles = true,
            .positions = 11,
            .stream = 12,
        };
        count += 1;
    };
    try std.testing.expectEqual(@as(usize, 192), count);
    for (keys) |key| try std.testing.expectEqual(@as(u32, 1), gprefill.psighting(key));
    for (keys) |key| try std.testing.expectEqual(@as(u32, 2), gprefill.psighting(key));
}

test "exl3: retiring one prefill runtime preserves another runtime's graph and sightings" {
    gprefill.clearPrefill();
    defer gprefill.clearPrefill();
    var driver = GraphTestDriver.driver();
    GraphTestDriver.exec_drops = 0;
    var first = std.mem.zeroes(PKey);
    first.owner = 1;
    var second = first;
    second.owner = 2;
    gprefill.pentries[0] = .{ .key = first, .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 1 };
    gprefill.pentries[1] = .{ .key = second, .exec = .{ .d = &driver, .handle = @ptrFromInt(2) }, .used = 2 };
    gprefill.pseen[0] = .{ .key = first, .n = 3 };
    gprefill.pseen[1] = .{ .key = second, .n = 4 };
    gprefill.clearRuntime(1);
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.exec_drops);
    try std.testing.expect(gprefill.pentries[0] == null and gprefill.pseen[0] == null);
    try std.testing.expect(gprefill.pentries[1] != null);
    try std.testing.expectEqual(@as(u32, 4), gprefill.pseen[1].?.n);
    gprefill.clearRuntime(1);
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.exec_drops);
}

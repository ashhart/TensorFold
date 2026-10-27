//! Host tests of the EXL3 prefill graph keys, origin rebinding, retention and sighting history.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const memory = @import("cuda_exl3_graph_memory.zig");
const prefill_binding = @import("../prefill/cuda_exl3_prefill_binding.zig");
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

test "exl3: prefill origin candidates preserve paging generation, incarnation and allocation keys" {
    const key: PKey = .{ .layer = 7, .rows = 512, .sub = 256, .pos0 = 4096, .deep = 1, .base = 2, .base_cap = 4096, .seq = .{ 3, 4, 5, 4096, 6 }, .paging = .{ 7, 8, 9, 10, 11, 12 }, .wait_count = 0, .wait_key = 0, .scores = 13, .tiles = true, .positions = 14, .stream = 15 };
    const request: prefill_binding.Request = .{ .functions = .{ .prep = 1, .scores = 2, .chunks = 3, .ahead = 4, .plan = 5, .remap = 6 }, .rows = 512, .sub = 256, .pos0 = 4096, .positions = 14, .cap = 32768, .origin = 1024, .page_rows = 256, .shared = true, .generation = 1, .keys = 16, .values = 17 };
    var binding: prefill_binding.Binding = .{ .graph = undefined, .request = request, .records = &.{} };
    var driver: cuda.Driver = undefined;
    gprefill.pentries = @splat(null);
    defer gprefill.pentries = @splat(null);
    gprefill.pentries[0] = .{ .key = key, .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 1, .binding = &binding };
    var moved = key;
    moved.pos0 = 8192;
    moved.paging[5] += 1;
    var target = request;
    target.pos0 = 8192;
    target.origin = 4096;
    try std.testing.expect(gprefill.prefillOriginCandidate(moved, target) != null);
    for (0..5) |index| {
        var changed = moved;
        changed.paging[index] += 1;
        try std.testing.expect(gprefill.prefillOriginCandidate(changed, target) == null);
        changed = moved;
        changed.seq[index] += 1;
        try std.testing.expect(gprefill.prefillOriginCandidate(changed, target) == null);
    }
    inline for (.{ "deep", "base", "base_cap", "scores", "positions", "stream", "wait_key", "wait_count", "rows", "sub", "layer" }) |field| {
        var changed = moved;
        @field(changed, field) += 1;
        try std.testing.expect(gprefill.prefillOriginCandidate(changed, target) == null);
    }
    binding.poisoned = true;
    try std.testing.expect(gprefill.prefillOriginCandidate(moved, target) == null);
}

test "exl3: prefill retained templates release both graph budget tickets on retirement" {
    gprefill.clearPrefill();
    try memory.configure(.{ .reservation_bytes = 100, .headroom_bytes = 0, .charge = .{ .exec = 10, .template = 10, .transient = 10 }, .prefill_charge = .{ .exec = 5, .template = 0, .transient = 10 } });
    const exec_ticket = try memory.admit(.prefill, 0, 100);
    try memory.installed(exec_ticket);
    const template_ticket = try memory.admit(.prefill, 0, 100);
    try memory.installed(template_ticket);
    try std.testing.expectEqual(@as(u64, 10), memory.charged());
    var driver = GraphTestDriver.driver();
    GraphTestDriver.exec_drops = 0;
    GraphTestDriver.graph_drops = 0;
    const binding = try std.heap.c_allocator.create(prefill_binding.Binding);
    binding.* = .{ .graph = .{ .d = &driver, .handle = @ptrFromInt(1) }, .request = undefined, .records = try std.heap.c_allocator.alloc(prefill_binding.Record, 0) };
    gprefill.pentries[0] = .{ .key = std.mem.zeroes(PKey), .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 1, .admission = exec_ticket, .binding = binding, .binding_admission = template_ticket, .stream = .{ .d = &driver, .handle = @ptrFromInt(1) } };
    gprefill.clearPrefill();
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.exec_drops);
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.graph_drops);
    try std.testing.expectEqual(@as(u64, 0), memory.charged());
    try std.testing.expectError(error.StaleGraphTicket, memory.release(exec_ticket));
    try std.testing.expectError(error.StaleGraphTicket, memory.release(template_ticket));
}

test "prefill graph update candidates tolerate dynamic position and page-window origins only" {
    const base: PKey = .{ .layer = 7, .rows = 4096, .sub = 256, .pos0 = 4096, .deep = 1, .base = 2, .base_cap = 4096, .seq = .{ 3, 4, 5, 4096, 6 }, .paging = .{ 7, 8, 9, 10, 11, 12 }, .wait_count = 0, .wait_key = 0, .scores = 13, .tiles = true };
    var moved = base;
    moved.pos0 = 8192;
    moved.paging[5] = 14;
    try std.testing.expect(!std.meta.eql(base, moved));
    try std.testing.expect(std.meta.eql(gprefill.pshape(base), gprefill.pshape(moved)));

    moved.seq[0] += 1;
    try std.testing.expect(!std.meta.eql(gprefill.pshape(base), gprefill.pshape(moved)));
    moved = base;
    moved.paging[0] += 1;
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
            .paging = .{ 5, 6, 7, 8, 9, sequence + bank + 1 },
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

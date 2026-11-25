//! A call's scratch (forward.Buffers) on a fake driver: every allocation that can fail leaves nothing behind.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("forward.zig");
const cfg = @import("shape.zig");
const weights = @import("pack_weights.zig");
const shard = @import("shard.zig");

const Fake = struct {
    var made: usize = 0;
    var freed: usize = 0;
    var fail_at: usize = 0; // the allocation (1 for the first) that fails, 0 for none
    var next: u64 = 1 << 20;

    fn alloc(p: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
        if (made + 1 == fail_at) return cuda.abi.error_not_ready;
        made += 1;
        p.* = next;
        next += 1 << 20;
        return 0;
    }
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        freed += 1;
        return 0;
    }
    fn fill(_: cuda.abi.DevicePtr, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn reset(at: usize) void {
        made = 0;
        freed = 0;
        fail_at = at;
    }
};

test "forward: a call's scratch frees exactly what it made when any one of its allocations fails" {
    const gpa = std.testing.allocator;
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Fake.alloc;
    d.api.cuMemFree_v2 = Fake.free;
    d.api.cuMemsetD8_v2 = Fake.fill;
    var w: weights.Weights = .{ .gpa = gpa, .d = &d, .config = try cfg.parse(gpa, @embedFile("fixtures/config.json")), .share = undefined };
    w.share = try shard.Share.of(&w.config, 201088, 0, 2);
    var layer: weights.Layer = undefined;
    layer.experts = .{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .experts = w.config.experts, .dims = w.config.hidden, .width = 1152, .codebook = .mul1 };
    w.layers = (&layer)[0..1];
    const shape: fw.Shape = .{ .rows = 8, .context = 4096, .heads = 6, .kv = 1 };
    Fake.reset(0);
    var whole = try fw.Buffers.init(gpa, &d, &w, shape, null);
    whole.deinit();
    const total = Fake.made;
    try std.testing.expect(total > 20);
    try std.testing.expectEqual(total, Fake.freed);
    for (1..total + 1) |at| {
        Fake.reset(at);
        try std.testing.expectError(error.NotReady, fw.Buffers.init(gpa, &d, &w, shape, null));
        try std.testing.expectEqual(at - 1, Fake.made);
        try std.testing.expectEqual(Fake.made, Fake.freed);
    }
}

//! Paged MTP admission: the target tap is in the lane's run, never an additional fixed charge.
const std = @import("std");
const mtp = @import("cuda_mtp.zig");
const native = @import("cuda_native.zig");
const Cache = @import("cuda_cache.zig").Cache;
const E = @import("cuda_engine.zig").Engine;
const t = std.testing;

test "MTP on a pool lane: admission includes the target tap ring exactly once, paged fixed bytes" {
    var e: E = undefined;
    e.d = undefined;
    e.scratch_rows = 4096;
    e.g.hidden = 5120;
    e.g.intermediate = 17408;
    e.g.layers = 64;
    e.g.q_heads = 24;
    e.g.vocab = 248320;
    e.g.kv_heads = 4;
    e.g.k_heads = 1;
    e.g.v_heads = 1;
    e.g.experts = 0;
    e.g.first = 0;
    e.g.linear = @splat(false);
    var d: mtp.Mtp = undefined;
    d.e = &e;
    d.capacity = 6144;
    d.tap = .{63};
    try d.initializeMemorySizes();
    const tap_bytes = 6144 * e.g.hidden * 2;
    try t.expectEqual(@as(usize, 60 * 1024 * 1024), tap_bytes);

    e.n_taps = d.taps().len;
    e.rollback_capable = false;
    var l: native.Lanes = .{ .gpa = t.allocator, .io = undefined, .e = &e, .capacity = d.capacity, .page_tokens = 256, .target = undefined, .mtp = &d };
    e.n_taps = 0;
    const run_without_tap = Cache.runBytes(&e, l.ringBytes());
    e.n_taps = d.taps().len;
    const run_with_tap = Cache.runBytes(&e, l.ringBytes());
    try t.expectEqual(run_without_tap + tap_bytes, run_with_tap);
    const paged = native.pagedLayout(&l, d.stream_bytes).?;
    const table_bytes = ((d.capacity + native.window_rows + 255) / 256) * 4;
    try t.expectEqual(table_bytes, paged.fixed_bytes);
    try t.expectEqual(run_with_tap, paged.run_bytes);
}

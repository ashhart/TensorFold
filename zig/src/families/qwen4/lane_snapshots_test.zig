//! The real shared store calls native snapshot hooks; failure ownership remains reachable after an unsuccessful save.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const Api = @import("core").api;
const L = @import("core").lanes;
const support = @import("lanes_adapter_test.zig");
const Hooks = @import("lane_snapshots.zig").Hooks(support.A, Api.prompt_cache);
test "Qwen4 shared prompt cache: native states obey budget, restore and retain a failed save's destination" {
    var f = try support.Fixture.open();
    defer f.close();
    const backend = f.adapter.backend();
    var s = try L.Stream.init(a, .{ .id = "source", .prompt = &.{ 1, 2, 3, 4, 5, 6 }, .max_new = 8 });
    defer s.deinit(a);
    try backend.prefill(&s);
    var hooks: Hooks = .{ .adapter = f.adapter };
    defer hooks.reclaimDrained();
    const family = hooks.snapshots();
    const charged = family.vtable.bytes(family.ptr, 6);
    var store = Api.prompt_cache.Store.init(a, family, .{ .min_prompt = 1, .min_gap = 1, .chunk = 1 }, charged);
    defer store.deinit();
    try t.expect(store.keep(s.prompt(), 6, &s, &.{}, &.{}));
    try t.expectEqual(@as(usize, 1), store.entries.items.len);
    try t.expect(store.held <= store.budget);
    const snap: *support.A.Snap = @ptrCast(@alignCast(store.entries.items[0].saved));
    try t.expectEqual(charged, snap.charged);
    try t.expect(snap.last_valid);
    const before = f.b.held;
    f.b.fail_drain = true;
    try t.expectError(error.PlantedDrain, family.vtable.save(family.ptr, &s, 6));
    try t.expect(f.b.held > before);
    try t.expect(hooks.retained != null);
    try t.expect(!f.e.healthy());
    f.b.fail_drain = false;
    try f.b.drain();
    hooks.reclaimDrained();
    try t.expectEqual(before, f.b.held);
}
test "Qwen4 native serve admission: unsupported switches name their refusal before backend construction" {
    const native = @import("native_host.zig");
    const base: Api.Open = .{ .dir = "fixture", .model_type = "qwen4" };
    try t.expectEqual(@as(?[]const u8, null), native.refusal(base));
    const Case = struct { field: []const u8, value: Api.Open };
    const cases = [_]Case{
        .{ .field = "--tp", .value = blk: {
            var v = base;
            v.tp = 2;
            break :blk v;
        } },
        .{ .field = "--prompt-precision", .value = blk: {
            var v = base;
            v.prompt_int8 = true;
            break :blk v;
        } },
        .{ .field = "--precision", .value = blk: {
            var v = base;
            v.precision = .checkpoint;
            break :blk v;
        } },
        .{ .field = "--prompt-cache-gib", .value = blk: {
            var v = base;
            v.prompt_cache_gib = std.math.nan(f64);
            break :blk v;
        } },
        .{ .field = "--parallel", .value = blk: {
            var v = base;
            v.lanes = 0;
            break :blk v;
        } },
        .{ .field = "--mtp", .value = blk: {
            var v = base;
            v.mtp_drafts = 4;
            break :blk v;
        } },
    };
    for (cases) |item| try t.expect(std.mem.indexOf(u8, native.refusal(item.value).?, item.field) != null);
}

//! Actual backend tables retain immutable lifetime across owners and post-first configuration changes.
const std = @import("std");
const be = @import("lanes").backend;
const Cuda = @import("lanes.zig").Cuda;
const Cache = @import("exact_prompt_cache.zig").Cache;
const shape = @import("shape.zig");
const a = std.testing.allocator;

fn fixture(enabled: bool) !*Cuda {
    const x = try a.create(Cuda);
    x.exact_cache = Cache.init(a, true);
    x.exact_cache.post_first.enabled = enabled;
    return x;
}

noinline fn obtain(x: *Cuda) be.Backend {
    return x.backend();
}

test "captured off and on backend tables survive later calls and flag toggles" {
    const x = try fixture(false);
    defer a.destroy(x);
    const off = obtain(x);
    x.exact_cache.post_first.enabled = true;
    const on = obtain(x);
    x.exact_cache.post_first.enabled = false;
    const off_again = obtain(x);
    try std.testing.expect(off.vtable != on.vtable);
    try std.testing.expect(off.vtable == off_again.vtable);
    try std.testing.expect(off.vtable.after_first == null);
    try std.testing.expect(on.vtable.after_first != null);
    try std.testing.expect(off.first_before_draft and on.first_before_draft);
    try std.testing.expectEqual(@as(u32, 123), try off.read(123));
    try std.testing.expectEqual(@as(u32, 321), try on.read(321));
    try std.testing.expectError(error.InvalidQwenToken, off.read(shape.vocab));
}

test "two owners share immutable mode tables while retaining distinct backend pointers" {
    const first = try fixture(false);
    defer a.destroy(first);
    const second = try fixture(true);
    defer a.destroy(second);
    const off = obtain(first);
    const on = obtain(second);
    first.exact_cache.post_first.enabled = true;
    const first_on = obtain(first);
    second.exact_cache.post_first.enabled = false;
    const second_off = obtain(second);
    try std.testing.expect(off.ptr != on.ptr);
    try std.testing.expect(off.ptr == first_on.ptr);
    try std.testing.expect(on.ptr == second_off.ptr);
    try std.testing.expect(off.vtable == second_off.vtable);
    try std.testing.expect(on.vtable == first_on.vtable);
    try std.testing.expect(off.vtable != on.vtable);
    try std.testing.expect(off.vtable.after_first == null and on.vtable.after_first != null);
    try std.testing.expect(off.vtable.prefill == on.vtable.prefill);
    try std.testing.expect(off.vtable.first == on.vtable.first);
    try std.testing.expect(off.vtable.draft == on.vtable.draft);
    try std.testing.expectEqual(@as(u32, 456), try first_on.read(456));
    try std.testing.expectEqual(@as(u32, 654), try second_off.read(654));
}

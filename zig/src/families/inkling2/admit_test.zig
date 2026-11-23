//! Inkling 2's admission on a fake driver: only sm_121 at two ranks passes, refused before any image loads.
const std = @import("std");
const cuda = @import("cuda");
const admit = @import("admit.zig");
const t = std.testing;

const Fake = struct {
    var sm: u32 = 121;
    var reads: usize = 0;
    var loads: usize = 0;
    fn attribute(out: *c_int, key: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        reads += 1;
        out.* = switch (key) {
            .compute_capability_major => @intCast(sm / 10),
            .compute_capability_minor => @intCast(sm % 10),
            else => unreachable,
        };
        return cuda.abi.success;
    }
    fn load(_: *cuda.abi.Module, _: ?*const anyopaque) callconv(.c) cuda.abi.Result {
        loads += 1;
        return cuda.abi.error_not_ready;
    }
};

test "inkling2 admission: GB10 at two ranks only, read from the device before any module load" {
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetAttribute = Fake.attribute;
    driver.api.cuModuleLoadData = Fake.load;
    const ctx: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    Fake.sm = 121;
    try admit.check(&ctx, 2);
    for ([_]usize{ 1, 3, 4 }) |w| try t.expectError(error.Inkling2RequiresTwoRanks, admit.check(&ctx, w));
    for ([_]u32{ 80, 90, 100, 120, 122 }) |sm| {
        Fake.sm = sm;
        try t.expectError(error.Inkling2RequiresSm121, admit.check(&ctx, 2));
    }
    try t.expect(Fake.reads > 0);
    try t.expectEqual(@as(usize, 0), Fake.loads);
    try t.expect(admit.explain(error.Inkling2RequiresSm121) != null);
    try t.expect(admit.explain(error.Inkling2RequiresTwoRanks) != null);
}

test "inkling2 admission: the rank settings come in pairs and are read through the environment table" {
    try t.expectEqual(admit.Place{ .rank = 0, .world = 1 }, try admit.placeFrom(null, null));
    try t.expectEqual(admit.Place{ .rank = 1, .world = 2 }, try admit.placeFrom("1", "2"));
    try t.expectError(error.Inkling2NoRank, admit.placeFrom("1", null));
    try t.expectError(error.Inkling2NoRank, admit.placeFrom("2", "2"));
    try t.expectError(error.InvalidEnvironment, admit.placeFrom("one", "2"));
    // one rank parses as a place; admission is what refuses it
    try t.expectError(error.Inkling2RequiresTwoRanks, admit.admit(121, (try admit.placeFrom(null, null)).world));
}

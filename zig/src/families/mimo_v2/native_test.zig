//! MiMo-V2's admission: only what was validated opens (GB10); the rest is refused by name.

const std = @import("std");
const cuda = @import("cuda");
const native = @import("native.zig");

test "MiMo-V2 refuses every GPU but GB10 (sm_121) before reading anything" {
    const abi = cuda.abi;
    const Fake = struct {
        var sm: u32 = 0;
        fn attribute(out: *c_int, key: abi.DeviceAttribute, _: abi.Device) callconv(.c) abi.Result {
            out.* = switch (key) {
                .compute_capability_major => @intCast(sm / 10),
                .compute_capability_minor => @intCast(sm % 10),
                else => unreachable,
            };
            return abi.success;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetAttribute = Fake.attribute;
    const ctx: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    const a = std.testing.allocator;
    const o: native.Options = .{ .context = 4096, .drafts = false, .lanes = 0 };
    for ([_]u32{ 80, 86, 89, 90, 100, 103, 110, 120 }) |sm| {
        Fake.sm = sm;
        try std.testing.expectError(error.MimoV2RequiresSm121, native.open(a, std.testing.io, &ctx, "no-such-pack", null, o));
    }
    Fake.sm = 121;
    try std.testing.expectError(error.Slots, native.open(a, std.testing.io, &ctx, "no-such-pack", null, o)); // past the GPU check
}

test "the admission refusals say what was validated" {
    for ([_]anyerror{error.MimoV2RequiresSm121}) |err| {
        const text = native.explain(null, err) orelse return error.Unexplained;
        try std.testing.expect(std.mem.indexOf(u8, text, "validated") != null);
    }
}

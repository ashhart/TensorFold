//! Startup admission with a capability-only driver; reaching weights is a test failure.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("cuda_native.zig");
const t = std.testing;
const env = @import("sys").env;
var sm: u32 = 121;
fn attribute(out: *c_int, a: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
    out.* = @intCast(if (a == .compute_capability_major) sm / 10 else sm % 10);
    return 0;
}
pub fn driver(capability: u32) cuda.Driver {
    sm = capability;
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuDeviceGetAttribute = attribute;
    return d;
}

test "SM110 startup refuses sharing without pages before the generic page requirement" {
    var d = driver(110);
    const ctx: cuda.Context = .{ .d = &d, .device = 0, .handle = null };
    _ = env.setenv("TF_QWEN_KV_SHARE", "32", 1);
    defer _ = env.unsetenv("TF_QWEN_KV_SHARE");
    try t.expectError(error.QwenSm110KvShareUnqualified, native.openLanes(t.allocator, t.io, &ctx, "missing", null, .{ .context = 8192, .drafts = false }));
}

test "SM110 startup refuses slide and DSpine while admitting the DFlash2 declaration" {
    var d = driver(110);
    const ctx: cuda.Context = .{ .d = &d, .device = 0, .handle = null };
    const guard = @import("cuda_sm110_open.zig");
    try t.expectError(error.QwenSm110SlideUnqualified, guard.check(t.allocator, t.io, &ctx, .{ .context = 8192, .drafts = false, .slide = true }));
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const o: native.Options = .{ .context = 8192, .drafts = true, .drafter = dir };
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"dflash_config\":{}}" });
    try guard.check(t.allocator, t.io, &ctx, o);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"dflash_config\":{\"selected_prefix_code_bridge\":{}}}" });
    try t.expectError(error.QwenSm110DspineUnqualified, guard.check(t.allocator, t.io, &ctx, o));
    _ = env.setenv("TF_DSPINE_WINDOW", "256", 1);
    defer _ = env.unsetenv("TF_DSPINE_WINDOW");
    try t.expectError(error.QwenSm110DspineWindowUnqualified, guard.check(t.allocator, t.io, &ctx, o));
}

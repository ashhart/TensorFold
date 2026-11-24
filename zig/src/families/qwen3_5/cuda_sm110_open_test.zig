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
fn dense27B() @import("cuda_geometry.zig").Geometry {
    const geometry = @import("cuda_geometry.zig");
    var g: geometry.Geometry = .{
        .hidden = 5120,
        .intermediate = 17408,
        .layers = 64,
        .vocab = 248320,
        .q_heads = 24,
        .kv_heads = 4,
        .k_heads = 16,
        .v_heads = 48,
        .eps = 1e-6,
        .tied = false,
        .context = 8192,
        .linear = @splat(false),
    };
    for (0..g.layers) |i| g.linear[i] = i % 4 != 3;
    return g;
}

test "SM110 product admits the two dense 27B formats and refuses other geometry or weight recipes" {
    const geometry = @import("cuda_geometry.zig");
    var g = dense27B();
    try std.testing.expect(g.sm110Product());
    g.bits.head = 8;
    try std.testing.expect(!g.sm110Product());
    g.kinds = .{ .modelopt = true, .head = .nvfp4 };
    for (0..g.layers) |i| {
        for ([_]geometry.Slot{ .gate, .up, .down }) |slot| g.kinds.proj[i][@backingInt(slot)] = .nvfp4;
        for ([_]geometry.Slot{ .qkv, .z, .out, .q, .k, .v, .o }) |slot| g.kinds.proj[i][@backingInt(slot)] = .fp8;
        for ([_]geometry.Slot{ .a, .b }) |slot| g.kinds.proj[i][@backingInt(slot)] = .bf16;
    }
    try std.testing.expect(g.sm110Product());
    g.kinds.proj[0][@backingInt(geometry.Slot.down)] = .bf16;
    try std.testing.expect(!g.sm110Product());
    g.kinds.proj[0][@backingInt(geometry.Slot.down)] = .nvfp4;
    g.hidden = 4096;
    try std.testing.expect(!g.sm110Product());
}

test "SM110 product refuses a different recurrent key head count" {
    var g = dense27B();
    g.k_heads = 8;
    try t.expect(!g.sm110Product());
}

test "SM110 product refuses a different recurrent value head count" {
    var g = dense27B();
    g.v_heads = 32;
    try t.expect(!g.sm110Product());
}

test "SM110 product refuses a different normalization epsilon" {
    var g = dense27B();
    g.eps = 1e-5;
    try t.expect(!g.sm110Product());
}

test "SM110 product refuses tied embedding and output weights" {
    var g = dense27B();
    g.tied = true;
    try t.expect(!g.sm110Product());
}

test "SM110 product refuses a different attention layer schedule" {
    var g = dense27B();
    g.linear[g.layers - 1] = true;
    try t.expect(!g.sm110Product());
}

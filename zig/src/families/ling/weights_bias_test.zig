//! The loader's expert bias against route.checkBias: a group holding +inf and -inf scores NaN, and the kernel keeps it.
const std = @import("std");
const t = std.testing;
const core = @import("core");
const config = @import("config.zig");
const synth = @import("synth.zig");
const weights = @import("weights.zig");
const fake = @import("fake_driver.zig");

const inf = std.math.inf(f32);

/// Layer 1's expert bias `bias` (fp32; the tiny config's 16 experts in 4 groups) read by weights.routerBias, fake GPU.
fn load(bias: *const [16]f32) !void {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, try synth.configJson(a, .bf16, 2), &why);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const name = "model.layers.1.mlp.gate.expert_bias";
    const head = try std.fmt.allocPrint(a, "{{\"{s}\":{{\"dtype\":\"F32\",\"shape\":[16],\"data_offsets\":[0,64]}}}}", .{name});
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, head.len, .little);
    const file = try std.mem.concat(a, u8, &.{ &len, head, std.mem.sliceAsBytes(bias) });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = file });
    var ck = try core.Checkpoint.openModel(t.allocator, t.io, try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    const d = fake.driver();
    fake.reset(null, 2);
    var store: weights.Store = .{ .gpa = t.allocator, .d = &d };
    defer store.deinit();
    var r: weights.proj.Reader = .{ .gpa = t.allocator, .stream = undefined, .pack = undefined, .ck = &ck, .c = &c, .store = &store };
    _ = try weights.routerBias(&r, 1);
    try t.expectEqual(@as(usize, 1), store.buffers.items.len);
}

test "the loader refuses an expert bias with a +inf and a -inf in one routing group, before it uploads it" {
    var bias: [16]f32 = @splat(0);
    bias[5] = inf;
    bias[6] = -inf;
    try t.expectError(error.InfiniteBiasPair, load(&bias));
}

test "the loader keeps a bias whose infinities of each sign sit in different groups" {
    var bias: [16]f32 = @splat(0.5);
    bias[0] = inf;
    bias[1] = inf;
    bias[4] = -inf;
    try load(&bias);
}

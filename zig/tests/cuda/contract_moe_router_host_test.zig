//! The router fixture's cancellation and row-coupling witnesses, without a device.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const suite = @import("contract_moe_router.zig");
const g = core.contract.gates;

export fn moe_router_suite_compile(gpu: *const @import("check.zig").Gpu, s: *const cuda.Stream, m: *const cuda.Module) void {
    var cases = suite.Suite.init(gpu.*, s.*, m.*) catch return;
    defer cases.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(gpu.gpa);
    cases.append(gpu.gpa, &list) catch return;
}
test "MoE router port: independent FP64 hooks and raw live-row byte hooks stay distinct" {
    var t: suite.Case = undefined;
    t.replacement = false;
    t.live = false;
    t.pad = 0;
    t.control = false;
    const fp64 = t.kernel();
    try std.testing.expect(fp64.spec.replaces == null and fp64.vt.check == null and fp64.vt.simulate != null);
    t.live = true;
    t.replacement = true;
    const live = t.kernel();
    try std.testing.expect(live.vt.check != null and live.vt.replaced != null);
    try std.testing.expectEqual(core.contract.spec.Elem.raw, live.spec.out);
}
test "MoE router port: padded tails are NaN and the real fixture exposes coupled rows" {
    const input = try suite.Inputs.init(std.testing.allocator, .{ .rows = 3, .n = 17, .k = 257 }, 16, .random, 49);
    defer input.deinit();
    var want: [51]f64 = undefined;
    var mag: [51]f64 = undefined;
    try input.d.reference(&want, &mag);
    try std.testing.expect(!std.mem.eql(f64, want[0..17], want[17..34]));
    for (0..3) |row| for (input.x[row * input.stride + 257 ..][0..16]) |v| try std.testing.expectEqual(@as(u16, 0x7fc1), v);
    const input_nf = try suite.Inputs.init(std.testing.allocator, .{ .rows = 5, .n = 17, .k = 257 }, 0, .nonfinite, 9);
    defer input_nf.deinit();
    var nf: [85]f64 = undefined;
    var nm: [85]f64 = undefined;
    try input_nf.d.reference(&nf, &nm);
    try std.testing.expect(std.math.isNan(nf[0]));
}
test "MoE router port: production stress fixture is exact in FP32 and rejects reduced sums in every order" {
    const input = try suite.Inputs.init(std.testing.allocator, .{ .rows = 4, .n = 17, .k = 4096 }, 0, .stress, 31);
    defer input.deinit();
    var want: [68]f64 = undefined;
    var mag: [68]f64 = undefined;
    var got: [68]f64 = undefined;
    try input.d.reference(&want, &mag);
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
        try input.d.simulate(.f32, order, &got);
        try std.testing.expectEqualSlices(f64, &want, &got);
        for ([_]core.contract.spec.Accum{ .f16, .bf16 }) |acc| {
            try input.d.simulate(acc, order, &got);
            var red = false;
            for (got, want) |actual, expected| if (!std.math.isFinite(actual) or @abs(actual - expected) > 100 * @abs(expected) * 0x1p-24) {
                red = true;
            };
            try std.testing.expect(red);
        }
    }
}

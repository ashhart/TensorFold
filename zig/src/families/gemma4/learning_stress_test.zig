//! The new RMS reduction must retain the shared contract's FP32 requirement in every simulated order.
const std = @import("std");
const core = @import("core");
const data = @import("../qwen4/learning_contract_data.zig");
fn ratio(actual: []const f64, values: anytype) f64 {
    var at: usize = 0;
    var worst: f64 = 0;
    for (values) |vs| for (vs) |v| {
        const x = actual[at];
        at += 1;
        if (core.contract.spec.classOf(x) != core.contract.spec.classOf(v.value)) return std.math.inf(f64);
        if (!std.math.isFinite(v.value)) continue;
        const err = @abs(x - v.value);
        worst = @max(worst, if (v.err == 0) (if (err == 0) @as(f64, 0) else std.math.inf(f64)) else err / v.err);
    };
    return worst;
}
test "Gemma4 learning: RMS stress discriminates both low accumulators by the unchanged hundredfold bound" {
    const spec = core.living_gemma;
    var p = data.params(spec, .rms, 4, 128);
    p.f[0] = 3;
    var f = try data.Fixture(spec).init(std.testing.allocator, .rms, p, 0x556, false);
    defer f.deinit();
    @memset(std.mem.bytesAsSlice(u16, f.raw.data[0]), 0x3f80);
    @memset(std.mem.bytesAsSlice(u16, f.raw.data[1]), 0x3f80);
    @memset(f.raw.floats(2), 0);
    for (0..f.counts[2]) |i| if (i % 128 < 3) {
        f.raw.floats(2)[i] = ([_]f32{ 65537, 65536, -131072 })[i % 128];
    };
    try f.reference();
    const simulated = try std.testing.allocator.alloc(f64, f.counts[8]);
    defer std.testing.allocator.free(simulated);
    var low: [2]f64 = @splat(std.math.inf(f64));
    var full: f64 = 0;
    for (std.enums.values(core.contract.gates.Order)) |order| {
        try f.simulate(.f32, order, simulated);
        full = @max(full, ratio(simulated, f.values));
        for ([_]core.contract.spec.Accum{ .f16, .bf16 }, &low) |acc, *margin| {
            try f.simulate(acc, order, simulated);
            margin.* = @min(margin.*, ratio(simulated, f.values));
        }
    }
    std.debug.print("GEMMA_RMS_STRESS f32={d} f16={d} bf16={d}\n", .{ full, low[0], low[1] });
    try std.testing.expect(full <= 1 and low[0] >= 100 and low[1] >= 100);
}

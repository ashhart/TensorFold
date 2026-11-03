//! Counterfactual accumulators must miss the same scalar primitive reference by the contract's unchanged factor.
const std = @import("std");
const core = @import("core");
const data = @import("learning_contract_data.zig");
const c = core.contract.spec;

fn ratio(got: []const f64, values: anytype) f64 {
    var at: usize = 0;
    var worst: f64 = 0;
    for (values) |vs| for (vs) |v| {
        const g = got[at];
        at += 1;
        if (c.classOf(g) != c.classOf(v.value)) return std.math.inf(f64);
        if (c.classOf(g) != .finite) continue;
        const diff = @abs(g - v.value);
        worst = @max(worst, if (v.err == 0) (if (diff == 0) @as(f64, 0) else std.math.inf(f64)) else diff / v.err);
    };
    return worst;
}
test "Qwen4 learning stress: all reduction primitives discriminate both low accumulators in every order" {
    var failures: usize = 0;
    inline for (.{ core.living_rows, core.living_gdn, core.living_qsa }) |spec| for (std.enums.values(spec.Op)) |op| {
        if (!data.reduces(spec, op)) continue;
        var p = data.params(spec, op, 4, 128);
        if (spec == core.living_rows and op == .linear) p.u[1] = 128;
        var f = try data.Fixture(spec).init(std.testing.allocator, op, p, 0x5733, false);
        defer f.deinit();
        f.stress();
        try f.reference();
        var n: usize = 0;
        for (f.values) |vs| n += vs.len;
        const simulated = try std.testing.allocator.alloc(f64, n);
        defer std.testing.allocator.free(simulated);
        var margins: [2]f64 = @splat(std.math.inf(f64));
        var full: f64 = 0;
        for (std.enums.values(core.contract.gates.Order)) |order| {
            try f.simulate(.f32, order, simulated);
            full = @max(full, ratio(simulated, f.values));
            for ([_]c.Accum{ .f16, .bf16 }, &margins) |acc, *margin| {
                try f.simulate(acc, order, simulated);
                margin.* = @min(margin.*, ratio(simulated, f.values));
            }
        }
        std.debug.print("LIVING_STRESS op={s} f32={d} f16={d} bf16={d}\n", .{ @tagName(op), full, margins[0], margins[1] });
        if (full > 1 or margins[0] < 100 or margins[1] < 100) failures += 1;
    };
    try std.testing.expectEqual(@as(usize, 0), failures);
}

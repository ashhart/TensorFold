//! Precision-oracle sanity is separate from the independent finite differences of the derivative formulas.
const std = @import("std");
const core = @import("core");
const data = @import("learning_contract_data.zig");
test "Qwen4 learning: binary64 primitive intervals cover native scalar bodies and retain nonfinite classes" {
    inline for (.{ core.living_rows, core.living_gdn, core.living_qsa }) |spec| {
        for (std.enums.values(spec.Op)) |op| for ([_]bool{ false, true }) |nonfinite| {
            const p = data.params(spec, op, 4, 64);
            var f = try data.Fixture(spec).init(std.testing.allocator, op, p, 0x3467, nonfinite);
            defer f.deinit();
            try f.reference();
            if (nonfinite) {
                var seen_nan = false;
                for (f.values) |values| for (values) |v| {
                    seen_nan = seen_nan or std.math.isNan(v.value);
                };
                try std.testing.expect(seen_nan);
            }
            try f.raw.run();
            for (f.values, 8..) |values, i| for (values, 0..) |v, j| {
                const got = f.output(i, j);
                const g = core.contract.spec.classOf(got);
                const want = core.contract.spec.classOf(v.value);
                try std.testing.expectEqual(want, g);
                if (g == .finite) {
                    if (!std.math.isFinite(v.err) or @abs(got - v.value) > v.err) {
                        std.debug.print("LIVING_INTERVAL spec={s} op={s} out={d} cell={d} got={d} want={d} bound={d}\n", .{ @typeName(spec), @tagName(op), i, j, got, v.value, v.err });
                        return error.NativePrecisionInterval;
                    }
                }
            };
        };
    }
}

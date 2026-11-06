//! Host-only: each NVFP4 waves kind's largest served shape needs a second wave of the entry it launches, as loaded.
const std = @import("std");
const cuda = @import("cuda");
const contract = @import("core").contract;
const fact = @import("contract_nvfp4_fact.zig");

const fx = cuda.experts.Nvfp4Experts;
const Kernel = contract.gates.Kernel;
const Spec = contract.spec.Spec;

/// Loaded kernels whose waves entries hold `up` (gate-up, entry 4) and `down` (entry 5) CTAs of `warps` warps.
fn loaded(warps: usize, up: usize, down: usize) fx {
    var resident: [7]usize = @splat(1);
    resident[4] = up;
    resident[5] = down;
    return .{ .mod = undefined, .fns = undefined, .resident = resident, .prompt_fns = undefined, .fact_fns = undefined, .warps = warps };
}

/// The waves kinds, the entry each launches and whether it runs the 1-warp CTAs; each control follows its twin.
const waves = [_]struct { kind: fact.Kind, entry: usize, one: bool }{
    .{ .kind = .up_waves, .entry = 4, .one = false },
    .{ .kind = .cut_waves, .entry = 4, .one = false },
    .{ .kind = .up_waves_one, .entry = 4, .one = true },
    .{ .kind = .down_waves, .entry = 5, .one = false },
    .{ .kind = .down_waves_one, .entry = 5, .one = true },
    .{ .kind = .cut_waves_one, .entry = 5, .one = true },
};

/// Every waves kind's largest shape routes more units than its entry's resident warps hold (`rows` rows when given).
fn twoWaves(four: fx, one: fx, rows: ?usize) !void {
    var xs: [waves.len]fact.Experts = undefined;
    var specs: [waves.len]Spec = undefined;
    for (waves, &xs, &specs) |w, *x, *s| {
        x.* = .{ .rig = undefined, .kind = w.kind, .k = &four, .k1 = &one, .router = undefined };
        const k: anyerror!Kernel = x.kernel();
        s.* = (try k).spec;
        var big = s.served[0];
        for (s.served) |shape| if (shape.rows > big.rows) {
            big = shape;
        };
        // 2 pairs a row in items of 16 pairs: however the router spreads them, at least rows / 8 items
        const units = (big.rows * 2 + 15) / 16 * (big.n / fx.cols);
        const l = if (w.one) one else four;
        const held = l.resident[w.entry] * l.warps;
        if (units <= held) {
            std.debug.print("{t}: {d} rows route at least {d} units, and {d} resident warps hold them in one wave\n", .{ w.kind, big.rows, units, held });
            return error.OneWave;
        }
        if (rows) |want| try std.testing.expectEqual(want, big.rows);
    }
    try std.testing.expectEqualDeep(specs[0].served, specs[1].served);
    try std.testing.expectEqualDeep(specs[4].served, specs[5].served);
}

test "nvfp4 waves: 768 gate-up and 960 down resident warps keep the 1,024-row shape, already two waves" {
    // 48 SMs at 4 gate-up and 5 down CTAs of 4 warps; 1-warp CTAs hold as many warps
    try twoWaves(loaded(4, 192, 240), loaded(1, 768, 960), 1024);
}

test "nvfp4 waves: 2,720 gate-up and 3,400 down resident warps, 170 SMs at 16 and 20 warps" {
    try twoWaves(loaded(4, 680, 850), loaded(1, 2720, 3400), null);
}

test "nvfp4 waves: 8,160 resident warps, 170 SMs at 48 warps" {
    try twoWaves(loaded(4, 2040, 2040), loaded(1, 8160, 8160), null);
}

test "nvfp4 waves: warps that 1,024 rows' units fill exactly take one more item, 1,032 rows" {
    // 128 items: 2,560 gate-up units (20 column blocks) and 10,240 down units (80), one full wave each
    try twoWaves(loaded(4, 640, 2560), loaded(1, 2560, 10240), 1032);
}

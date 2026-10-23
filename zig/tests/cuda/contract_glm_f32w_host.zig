//! Host checks of the GLM norm contract's real input builder, mathematical controls and declared envelope.
const std = @import("std");
const core = @import("core");
const ref = @import("contract_glm_f32w_ref.zig");
const impl = @import("contract_glm_f32w.zig");
const c = core.contract.spec;
const g = core.contract.gates;

fn outputs(a: std.mem.Allocator, in: ref.In, which: impl.Which, sum: ref.Sum, want: []f64, mag: []f64) !void {
    return switch (which) {
        .norm_y, .hc_y => ref.normed(a, in, sum, want, mag),
        .norm_sums, .hc_sums => ref.sums(a, in, sum, want, mag),
        .hc_post => ref.post(in, sum, want, mag),
        .hc_comb => ref.comb(in, sum, want, mag),
    };
}

fn count(which: impl.Which, width: usize) usize {
    return switch (which) {
        .norm_y, .hc_y => width,
        .norm_sums, .hc_sums => width / 64,
        .hc_post => 4,
        .hc_comb => 16,
    };
}

test "GLM 64-sums stress separates BF16 at every registered width independently of seed" {
    const seeds = [_]u64{ 0, 1, 7, 0x6b63, 0x9e37 * 125, 0x9e37 * 134, std.math.maxInt(u64) };
    for (impl.bf16_entries) |entry| {
        if (entry.which != .norm_sums and entry.which != .hc_sums) continue;
        const spec = impl.specWidth(entry.which, entry.width);
        for (seeds) |seed| {
            var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
            defer arena.deinit();
            const a = arena.allocator();
            const in = try ref.fillWidth(a, entry.which == .hc_sums, entry.width, 4, .stress, seed, true);
            const n = 4 * entry.width / 64;
            const want = try a.alloc(f64, n);
            const got = try a.alloc(f64, n);
            const mag = try a.alloc(f64, n);
            try ref.sums(a, in, .{}, want, mag);
            for (want) |v| try std.testing.expectEqual(@as(f64, 511), v);
            var least: f64 = std.math.inf(f64);
            for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
                for ([_]c.Accum{ .f32, .f16, .bf16 }) |acc| {
                    try ref.sums(a, in, .{ .acc = acc, .order = order }, got, mag);
                    for (got, want) |v, target| {
                        const ratio = @abs(v - target) / c.boundExact(spec, target);
                        if (acc == .bf16) {
                            try std.testing.expect(ratio >= spec.sums.bf16);
                            least = @min(least, ratio);
                        } else try std.testing.expectEqual(target, v);
                    }
                }
            }
            if (seed == 0) std.debug.print("GLM 64-sums width={d} hc={} BF16 minimum={d}x\n", .{
                entry.width, entry.which == .hc_sums, least,
            });
        }
    }
}

test "GLM norm nonfinite fixture includes both infinity signs together and leaves a finite row" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |hc| for ([_]usize{ 512, 1536, 4096 }) |width| {
        if (hc and width != ref.d) continue;
        const in = try ref.fillWidth(a, hc, width, 5, .nonfinite, 7, false);
        const stride = width * @as(usize, if (hc) ref.s else 1);
        try std.testing.expectEqual(c.Class.pos_inf, c.classOf(c.fromBf16(in.x[3 * stride])));
        try std.testing.expectEqual(c.Class.neg_inf, c.classOf(c.fromBf16(in.x[3 * stride + 1])));
        const want = try a.alloc(f64, 5 * width);
        const mag = try a.alloc(f64, 5 * width);
        try ref.normed(a, in, .{}, want, mag);
        try std.testing.expect(std.math.isNan(want[3 * width]));
        for (want[4 * width ..][0..width]) |v| try std.testing.expect(std.math.isFinite(v));
    };
}

test "GLM norm narrowing control fails the unchanged two-rounding bound deterministically" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var in = try ref.fill(a, false, 1, .random, 7, false);
    ref.narrowControl(&in);
    const want = try a.alloc(f64, ref.d);
    const mag = try a.alloc(f64, ref.d);
    try ref.normed(a, in, .{}, want, mag);
    var square: f64 = 0;
    for (in.x) |v| square += @as(f64, c.fromBf16(v)) * c.fromBf16(v);
    const inv: f32 = @floatCast(1 / @sqrt(in.eps + square / ref.d));
    const x: f32 = c.fromBf16(in.x[0]);
    const bad = ref.bf(ref.bf(in.gamma[0]) * ref.bf(@as(f32, inv * x)));
    const spec = impl.spec(.norm_y);
    try std.testing.expect(@abs(bad - want[0]) / c.bound(spec, spec.served[0], want[0], mag[0]) > 1.2);
    try std.testing.expectEqual(@as(f64, 0.279296875), bad);
}

test "GLM norm contract declares prompt-capacity rows and simulates its stated accumulation limits" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (std.enums.values(impl.Which)) |which| {
        const spec = impl.spec(which);
        var largest: usize = 0;
        for (spec.served) |shape| largest = @max(largest, shape.rows);
        try std.testing.expectEqual(@as(usize, 2048), largest);
        const hc = switch (which) {
            .norm_y, .norm_sums => false,
            else => true,
        };
        const exact = which == .norm_sums or which == .hc_sums;
        const in = try ref.fill(a, hc, 1, .stress, 0x6b63, exact);
        const n = count(which, ref.d);
        const want = try a.alloc(f64, n);
        const mag = try a.alloc(f64, n);
        const got = try a.alloc(f64, n);
        try outputs(a, in, which, .{}, want, mag);
        var minimum = [2]f64{ std.math.inf(f64), std.math.inf(f64) };
        for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
            try outputs(a, in, which, .{ .acc = .f32, .order = order }, got, mag);
            for (got, want) |v, target| try std.testing.expect(@abs(v - target) <= c.boundExact(spec, target));
            for ([_]c.Accum{ .f16, .bf16 }, &minimum) |acc, *least| {
                try outputs(a, in, which, .{ .acc = acc, .order = order }, got, mag);
                var worst: f64 = 0;
                for (got, want) |v, target| worst = @max(worst, if (v == target) 0 else if (!std.math.isFinite(v)) std.math.inf(f64) else @abs(v - target) / c.boundExact(spec, target));
                least.* = @min(least.*, worst);
            }
        }
        try std.testing.expect(minimum[0] >= spec.sums.f16);
        try std.testing.expect(minimum[1] >= spec.sums.bf16);
    }
    for (impl.bf16_entries) |entry| {
        try std.testing.expect(entry.width == 512 or entry.width == 1536 or entry.width == 4096);
        try std.testing.expect(std.mem.startsWith(u8, entry.old, "tf_ctl_glm_old_"));
    }
}

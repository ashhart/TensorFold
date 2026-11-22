//! Count-checked mutations of the tensor-unit dense kernels: every entry has a genuine arithmetic control.
const std = @import("std");
const src = @import("inkling2_metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;

pub const Mutation = enum {
    none,
    dense_16,
    dense_32,
    dense_64,
    dense6_16,
    dense6_32,
    dense6_64,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .dense_16 => "control: metal inkling2 dense_units tf_ink2_dense_units_16 k-step",
            .dense_32 => "control: metal inkling2 dense_units tf_ink2_dense_units_32 k-step",
            .dense_64 => "control: metal inkling2 dense_units tf_ink2_dense_units_64 k-step",
            .dense6_16 => "control: metal inkling2 dense_units tf_ink2_dense6_units_16 k-step",
            .dense6_32 => "control: metal inkling2 dense_units tf_ink2_dense6_units_32 k-step",
            .dense6_64 => "control: metal inkling2 dense_units tf_ink2_dense6_units_64 k-step",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .dense_16 => .{ .from = "ink2_dense_core<1u, false>(", .to = "ink2_dense_core<1u, true>(" },
            .dense_32 => .{ .from = "ink2_dense_core<2u, false>(", .to = "ink2_dense_core<2u, true>(" },
            .dense_64 => .{ .from = "ink2_dense_core<4u, false>(", .to = "ink2_dense_core<4u, true>(" },
            .dense6_16 => .{ .from = "ink2_dense6<1u, false>(", .to = "ink2_dense6<1u, true>(" },
            .dense6_32 => .{ .from = "ink2_dense6<2u, false>(", .to = "ink2_dense6<2u, true>(" },
            .dense6_64 => .{ .from = "ink2_dense6<4u, false>(", .to = "ink2_dense6<4u, true>(" },
        };
    }
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.dense_units);
    const change = m.edit();
    if (std.mem.count(u8, src.dense_units, change.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.dense_units, change.from, change.to);
}

test "every dense_units control edits exactly its planted arithmetic" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 dense_units "));
    }
    const plain = try apply(a, .none);
    defer a.free(plain);
    try std.testing.expectEqualStrings(src.dense_units, plain);
}

test "dense_units source exports all six entries once each" {
    inline for ([_][]const u8{
        "tf_ink2_dense_units_16",
        "tf_ink2_dense_units_32",
        "tf_ink2_dense_units_64",
        "tf_ink2_dense6_units_16",
        "tf_ink2_dense6_units_32",
        "tf_ink2_dense6_units_64",
    }) |entry|
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src.dense_units, entry ++ "("));
}

test "dense_units stress inputs stay exact in FP32 and separate FP16/BF16 by >= 100x at K=3072" {
    const a = std.testing.allocator;
    const rows: usize = 4;
    const n: usize = 24;
    const k: usize = 3072;
    const h: usize = k / 2;
    var prng = std.Random.DefaultPrng.init(0x6b63);
    const rng = prng.random();
    const x = try a.alloc(u16, rows * k);
    defer a.free(x);
    const w = try a.alloc(u16, n * k);
    defer a.free(w);
    const terms = try a.alloc(f64, k);
    defer a.free(terms);
    for (0..rows) |r| for (0..h) |i| {
        const v: f32 = @floatFromInt(2 * (128 + rng.uintLessThan(u32, 127)));
        x[r * k + i] = c.toBf16(v);
        x[r * k + h + i] = c.toBf16(v + if (i < 8) @as(f32, 2) else 0);
    };
    for (0..n) |j| for (0..h) |i| {
        const q: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 8));
        w[j * k + i] = c.toBf16(q);
        w[j * k + h + i] = c.toBf16(-q);
    };
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
        var max_f16: f64 = 0;
        var max_bf16: f64 = 0;
        for (0..rows) |r| for (0..n) |j| {
            var exact: f64 = 0;
            for (0..k) |i| {
                terms[i] = @as(f64, c.fromBf16(x[r * k + i])) * c.fromBf16(w[j * k + i]);
                exact += terms[i];
            }
            const allow = 0x1p-8 * @abs(exact) + 0x1p-133;
            try std.testing.expectEqual(exact, contract.rows.sumIn(.f32, order, terms));
            for ([_]c.Accum{ .f16, .bf16 }, [_]*f64{ &max_f16, &max_bf16 }) |acc, mx| {
                const got = contract.rows.sumIn(acc, order, terms);
                const q = if (!std.math.isFinite(got)) std.math.inf(f64) else @abs(got - exact) / allow;
                mx.* = @max(mx.*, q);
            }
        };
        try std.testing.expect(max_f16 >= 100);
        try std.testing.expect(max_bf16 >= 100);
    }
}

//! Source-planted arithmetic and reduction-order controls for Inkling 2 Mini MLX 4-bit affine g64 Metal kernels.
const std = @import("std");
const contract = @import("contract");
const g = contract.gates;
const src = @import("inkling2_metal");

pub const Kind = enum { embed_norm, dense, logits };

pub const Mutation = enum {
    none,
    embed_skip_bias,
    embed_skip_bf16_dequant,
    dense_wrong_nibble,
    dense_wrong_pair_order,
    dense_wrong_warp_order,
    dense_bf16_accum,
    logits_skip_softcap,
    logits_skip_bf16_round,

    pub fn kind(m: Mutation) Kind {
        return switch (m) {
            .none => unreachable,
            .embed_skip_bias, .embed_skip_bf16_dequant => .embed_norm,
            .dense_wrong_nibble, .dense_wrong_pair_order, .dense_wrong_warp_order, .dense_bf16_accum => .dense,
            .logits_skip_softcap, .logits_skip_bf16_round => .logits,
        };
    }

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .embed_skip_bias => "control: metal inkling2 affine4 tf_ink2_embed_norm_q4 drops affine bias",
            .embed_skip_bf16_dequant => "control: metal inkling2 affine4 tf_ink2_embed_norm_q4 skips bf16 dequant rounding",
            .dense_wrong_nibble => "control: metal inkling2 affine4 tf_ink2_dense_q4 swaps low and high nibbles",
            .dense_wrong_pair_order => "control: metal inkling2 affine4 tf_ink2_dense_q4 reassociates pair sum",
            .dense_wrong_warp_order => "control: metal inkling2 affine4 tf_ink2_dense_q4 reverses warp tree order",
            .dense_bf16_accum => "control: metal inkling2 affine4 tf_ink2_dense_q4 accumulates in bf16",
            .logits_skip_softcap => "control: metal inkling2 affine4 tf_ink2_logits_q4 skips softcap30",
            .logits_skip_bf16_round => "control: metal inkling2 affine4 tf_ink2_logits_q4 skips bf16 dot rounding",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return switch (m) {
            .none => null,
            .embed_skip_bias,
            .embed_skip_bf16_dequant,
            .dense_wrong_nibble,
            .dense_wrong_pair_order,
            .dense_wrong_warp_order,
            .logits_skip_softcap,
            .logits_skip_bf16_round,
            => .shapes,
            .dense_bf16_accum => .sums,
        };
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .embed_skip_bias => .{
                .from = "return float(bfloat(float(q) * float(s_row[g]) + float(b_row[g])));",
                .to = "return float(bfloat(float(q) * float(s_row[g])));",
            },
            .embed_skip_bf16_dequant => .{
                .from = "return float(bfloat(float(q) * float(s_row[g]) + float(b_row[g])));",
                .to = "return float(q) * float(s_row[g]) + float(b_row[g]);",
            },
            .dense_wrong_nibble => .{
                .from = "const float dw0 = float((word >> sh) & 0xfu) * sc + bi;\n    const float dw1 = float((word >> (sh + 4u)) & 0xfu) * sc + bi;",
                .to = "const float dw0 = float((word >> (sh + 4u)) & 0xfu) * sc + bi;\n    const float dw1 = float((word >> sh) & 0xfu) * sc + bi;",
            },
            .dense_wrong_pair_order => .{
                .from = "return (acc + p0) + p1;",
                .to = "return acc + (p0 + p1);",
            },
            .dense_wrong_warp_order => .{
                .from = "  v += simd_shuffle_down(v, 16u);\n  v += simd_shuffle_down(v, 8u);\n  v += simd_shuffle_down(v, 4u);\n  v += simd_shuffle_down(v, 2u);\n  v += simd_shuffle_down(v, 1u);",
                .to = "  v += simd_shuffle_down(v, 1u);\n  v += simd_shuffle_down(v, 2u);\n  v += simd_shuffle_down(v, 4u);\n  v += simd_shuffle_down(v, 8u);\n  v += simd_shuffle_down(v, 16u);",
            },
            .dense_bf16_accum => .{
                .from = "return (acc + p0) + p1;",
                .to = "return float(bfloat(float(bfloat(acc + p0)) + p1));",
            },
            .logits_skip_softcap => .{
                .from = "return INK2_Q4_CAP * ink2_q4_tanh(precise::divide(v, INK2_Q4_CAP));",
                .to = "return v;",
            },
            .logits_skip_bf16_round => .{
                .from = "const float v = float(bfloat(dot));",
                .to = "const float v = dot;",
            },
        };
    }
};

pub fn source(a: std.mem.Allocator, production: []const u8, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, production);
    const change = m.edit();
    if (std.mem.count(u8, production, change.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, production, change.from, change.to);
}

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    return source(a, src.affine4, m);
}

test "every affine4 control hits one anchor and refuses missing or duplicated matches" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        const change = m.edit();
        try std.testing.expect(std.mem.indexOf(u8, edited, change.to) != null);
        try std.testing.expect(m.gate() != null);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 affine4 "));
        try std.testing.expectError(error.ControlEditCount, source(a, edited, m));
        try std.testing.expectError(error.ControlEditCount, source(a, "", m));
        const doubled = try std.mem.concat(a, u8, &.{ src.affine4, src.affine4 });
        defer a.free(doubled);
        try std.testing.expectError(error.ControlEditCount, source(a, doubled, m));
    }
    const plain = try apply(a, .none);
    defer a.free(plain);
    try std.testing.expectEqualStrings(src.affine4, plain);
}

const MockProbe = struct {
    deinits: *usize,
    fn kernel(_: *MockProbe) g.Kernel {
        return .{ .spec = undefined, .ctx = undefined, .vt = undefined };
    }
};

test "affine4 control apply and shared contract.register tolerate OOM without leaks" {
    const base = std.testing.allocator;
    var apply_failing = std.testing.FailingAllocator.init(base, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, apply(apply_failing.allocator(), .embed_skip_bias));
    for ([_]usize{ 0, 1, std.math.maxInt(usize) }) |fail_index| {
        var cleaned: usize = 0;
        var probes: std.ArrayList(*MockProbe) = .empty;
        var kernels: std.ArrayList(g.Kernel) = .empty;
        defer probes.deinit(base);
        defer kernels.deinit(base);
        const p = try base.create(MockProbe);
        p.* = .{ .deinits = &cleaned };
        const Helper = struct {
            fn add(a: std.mem.Allocator, list_a: std.mem.Allocator, ps: *std.ArrayList(*MockProbe), ks: *std.ArrayList(g.Kernel), probe: *MockProbe) !void {
                errdefer {
                    probe.deinits.* += 1;
                    a.destroy(probe);
                }
                try contract.register(list_a, ps, ks, probe, probe.kernel());
            }
        };
        var failing = std.testing.FailingAllocator.init(base, .{ .fail_index = fail_index });
        const result = Helper.add(base, failing.allocator(), &probes, &kernels, p);
        if (fail_index < 2) try std.testing.expectError(error.OutOfMemory, result) else try result;
        for (probes.items) |owned| {
            owned.deinits.* += 1;
            base.destroy(owned);
        }
        try std.testing.expectEqual(@as(usize, 1), cleaned);
    }
}

//! One host-checkable arithmetic mutation for every Mini DSpark Metal kernel.
const std = @import("std");
const ct = @import("contract");
const g = ct.gates;

pub const Kind = enum { init_rows, fuse_taps, block_qkv, context_kv, attention, residual_norm, swiglu, kappa, recurrent, base, logits, logits_q4 };
const Edit = struct { from: []const u8, to: []const u8 };
fn edit(kind: Kind) Edit {
    return switch (kind) {
        .init_rows => .{ .from = "rows[row * DSP_H + d] = src[d];", .to = "rows[row * DSP_H + d] = mask[d];" },
        .fuse_taps => .{ .from = "alpha[j] = float(bfloat(scores[j] / sum));", .to = "alpha[j] = scores[j] / sum;" },
        .block_qkv => .{ .from = "const float rotated = d < 64u ? x * cs - other * sn : other * sn + x * cs;", .to = "const float rotated = d < 64u ? x * cs - other * sn : x * sn + other * cs;" },
        .context_kv => .{ .from = "keys[at] = bfloat(d < 64u ? x * cs - other * sn : other * sn + x * cs);", .to = "keys[at] = bfloat(d < 64u ? x * cs - other * sn : x * sn + other * cs);" },
        .attention => .{ .from = "const uint first = max(from, a.local && committed > 1024u ? committed - 1024u : 0u);", .to = "const uint first = from;" },
        .residual_norm => .{ .from = "normalized[at] = bfloat(sigma * scale * float(weight[d]));", .to = "normalized[at] = bfloat(float(new_residual[at]) * scale * float(weight[d]));" },
        .swiglu => .{ .from = "out[i] = bfloat((gate / (1.0f + exp(-gate))) * float(u[i]));", .to = "out[i] = bfloat(gate * float(u[i]));" },
        .kappa => .{ .from = "const float wy = float(w_kappa[uint64_t(d) * DSP_RNN + ch]);", .to = "const float wy = float(w_kappa[uint64_t(DSP_H + d) * DSP_RNN + ch]);" },
        .recurrent => .{ .from = "const bfloat left = bfloat(float(gamma) * float(sh_s[i]));", .to = "const float left = float(gamma) * float(sh_s[i]);" },
        .base => .{ .from = "static inline void dspark_store_base(device bfloat* out, uint64_t i, float val) { out[i] = bfloat(val); }", .to = "static inline void dspark_store_base(device bfloat* out, uint64_t i, float val) { out[i] = bfloat(30.0f * tanh(val / 30.0f)); }" },
        .logits => .{ .from = "const bfloat bias = bfloat(bval + dots[s]);", .to = "const float bias = bval + dots[s];" },
        .logits_q4 => .{ .from = "wv[e] = fma(s, float((word >> (4u * e)) & 0xfu), b);", .to = "wv[e] = fma(s, float((word >> (4u * e)) & 0x7u), b);" },
    };
}

pub fn name(kind: Kind) []const u8 {
    return switch (kind) {
        inline else => |k| "control: metal inkling2 dspark " ++ @tagName(k),
    };
}
pub fn source(a: std.mem.Allocator, production: []const u8, kind: Kind) ![]u8 {
    const e = edit(kind);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

test "every DSpark control applies once and only mutates its arithmetic anchor" {
    const production = @import("inkling2_metal").inkling2_dspark;
    const a = std.testing.allocator;
    for (std.enums.values(Kind)) |kind| {
        const e = edit(kind);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, production, e.from));
        const changed = try source(a, production, kind);
        defer a.free(changed);
        const at = std.mem.indexOf(u8, production, e.from).?;
        try std.testing.expectEqualStrings(production[0..at], changed[0..at]);
        try std.testing.expectEqualStrings(e.to, changed[at..][0..e.to.len]);
        try std.testing.expectEqualStrings(production[at + e.from.len ..], changed[at + e.to.len ..]);
        try std.testing.expectError(error.ControlEditMissing, source(a, changed, kind));
        try std.testing.expectError(error.ControlEditMissing, source(a, "", kind));
    }
}

test "repeated DSpark control anchors are refused" {
    const production = @import("inkling2_metal").inkling2_dspark;
    const a = std.testing.allocator;
    const twice = try std.mem.concat(a, u8, &.{ production, production });
    defer a.free(twice);
    for (std.enums.values(Kind)) |kind| try std.testing.expectError(error.ControlEditMissing, source(a, twice, kind));
}

test "contract.register keeps existing owner list intact when kernel growth hits OutOfMemory" {
    const a = std.testing.allocator;
    const MockProbe = struct {
        deinits: *usize,
        fn deinit(p: *@This()) void {
            p.deinits.* += 1;
        }
        fn kernel(p: *@This()) g.Kernel {
            const H = struct {
                const vt: g.VTable = .{
                    .prepare = undefined,
                    .run = undefined,
                    .output = undefined,
                    .inputs = undefined,
                    .intact = undefined,
                    .overlap = undefined,
                    .count = undefined,
                    .reference = undefined,
                };
            };
            return .{ .spec = undefined, .ctx = p, .vt = &H.vt };
        }
    };
    const MockSuite = struct {
        a: std.mem.Allocator,
        kernels: *std.ArrayList(g.Kernel),
        probes: std.ArrayList(*MockProbe) = .empty,

        fn add(s: *@This(), list_alloc: std.mem.Allocator, deinits: *usize) !void {
            const probe = try s.a.create(MockProbe);
            errdefer s.a.destroy(probe);
            probe.* = .{ .deinits = deinits };
            errdefer probe.deinit();
            try ct.register(list_alloc, &s.probes, s.kernels, probe, probe.kernel());
        }

        fn deinit(s: *@This()) void {
            for (s.probes.items) |probe| {
                probe.deinit();
                s.a.destroy(probe);
            }
            s.probes.deinit(s.a);
        }
    };

    var first_deinits: usize = 0;
    var second_deinits: usize = 0;
    {
        var kernels: std.ArrayList(g.Kernel) = .empty;
        defer kernels.deinit(a);
        var suite: MockSuite = .{ .a = a, .kernels = &kernels };
        defer suite.deinit();

        try suite.probes.ensureTotalCapacityPrecise(a, 1);
        try kernels.ensureTotalCapacityPrecise(a, 1);
        try suite.add(a, &first_deinits);
        try std.testing.expectEqual(@as(usize, 1), suite.probes.items.len);
        try std.testing.expectEqual(@as(usize, 1), kernels.items.len);
        const first_probe = suite.probes.items[0];

        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 1, .resize_fail_index = 0 });
        try std.testing.expectError(error.OutOfMemory, suite.add(failing.allocator(), &second_deinits));
        try std.testing.expect(failing.has_induced_failure);
        try std.testing.expectEqual(@as(usize, 1), failing.allocations);
        try std.testing.expectEqual(@as(usize, 1), second_deinits);
        try std.testing.expectEqual(@as(usize, 0), first_deinits);
        try std.testing.expectEqual(@as(usize, 1), suite.probes.items.len);
        try std.testing.expectEqual(first_probe, suite.probes.items[0]);
        try std.testing.expectEqual(@as(usize, 1), kernels.items.len);
        try std.testing.expectEqual(@as(*anyopaque, first_probe), kernels.items[0].ctx);
    }
    try std.testing.expectEqual(@as(usize, 1), first_deinits);
    try std.testing.expectEqual(@as(usize, 1), second_deinits);
}

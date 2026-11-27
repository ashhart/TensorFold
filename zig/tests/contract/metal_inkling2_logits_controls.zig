//! Source-planted arithmetic controls for the coalesced multi-row Inkling 2 Mini logits kernels.
const std = @import("std");
const contract = @import("contract");
const g = contract.gates;
const src = @import("inkling2_metal");

pub const Mutation = enum {
    none,
    bf16_round,
    pair_assoc,
    chunk_order,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .bf16_round => "control: metal inkling2 logits tf_inkling2_logits_coalesced skips bf16 round before softcap30",
            .pair_assoc => "control: metal inkling2 logits tf_inkling2_logits_coalesced associates bf16 pair before sum",
            .chunk_order => "control: metal inkling2 logits tf_inkling2_logits_coalesced reverses k_step chunk order",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .bf16_round => .{
                .from = "ink_cap(float(bfloat(acc[rr])), 30.0f)",
                .to = "ink_cap(acc[rr], 30.0f)",
            },
            .pair_assoc => .{
                .from = "acc[rr] = (acc[rr] + w0 * hv.x) + w1 * hv.y;",
                .to = "acc[rr] += w0 * hv.x + w1 * hv.y;",
            },
            .chunk_order => .{
                .from = "const uint d_bf = k_step * 64u;",
                .to = "const uint d_bf = (35u - k_step) * 64u;",
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
    return source(a, src.glue, m);
}

test "every coalesced logits control hits one anchor and refuses missing or duplicated matches" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        const change = m.edit();
        try std.testing.expect(std.mem.indexOf(u8, edited, change.to) != null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 logits"));
        try std.testing.expectError(error.ControlEditCount, source(a, edited, m));
        try std.testing.expectError(error.ControlEditCount, source(a, "", m));
        const doubled = try std.mem.concat(a, u8, &.{ src.glue, src.glue });
        defer a.free(doubled);
        try std.testing.expectError(error.ControlEditCount, source(a, doubled, m));
    }
    const plain = try apply(a, .none);
    defer a.free(plain);
    try std.testing.expectEqualStrings(src.glue, plain);
}

const MockProbe = struct {
    deinits: *usize,
    fn kernel(_: *MockProbe) g.Kernel {
        return .{ .spec = undefined, .ctx = undefined, .vt = undefined };
    }
};

fn addMockProbe(a: std.mem.Allocator, list_a: std.mem.Allocator, probes: *std.ArrayList(*MockProbe), kernels: *std.ArrayList(g.Kernel), p: *MockProbe) !void {
    errdefer {
        p.deinits.* += 1;
        a.destroy(p);
    }
    try contract.register(list_a, probes, kernels, p, p.kernel());
}

test "coalesced logits control apply and probe registration tolerate OOM without leaks" {
    const base = std.testing.allocator;
    var apply_failing = std.testing.FailingAllocator.init(base, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, apply(apply_failing.allocator(), .bf16_round));
    for ([_]usize{ 0, 1, std.math.maxInt(usize) }) |fail_index| {
        var cleaned: usize = 0;
        var probes: std.ArrayList(*MockProbe) = .empty;
        var kernels: std.ArrayList(g.Kernel) = .empty;
        defer probes.deinit(base);
        defer kernels.deinit(base);
        const p = try base.create(MockProbe);
        p.* = .{ .deinits = &cleaned };
        var failing = std.testing.FailingAllocator.init(base, .{ .fail_index = fail_index });
        const result = addMockProbe(base, failing.allocator(), &probes, &kernels, p);
        if (fail_index < 2) try std.testing.expectError(error.OutOfMemory, result) else try result;
        for (probes.items) |owned| {
            owned.deinits.* += 1;
            base.destroy(owned);
        }
        try std.testing.expectEqual(@as(usize, 1), cleaned);
    }
}

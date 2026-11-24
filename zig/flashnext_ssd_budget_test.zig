//! Existing memory and FSR formulas must admit a bounded pool where the requested maximum was refused.
const std = @import("std");
const t = std.testing;
const memory = @import("flashnext_memory");
const layout = @import("src/families/flashnext/fsr_layout.zig");
const Settings = @import("src/families/flashnext/window_config.zig").Settings;
const GiB = memory.GiB;
const Ctx = struct { limit: u64, existing: u64 = 0, extra: u64 = 0, needs_build: bool = false, paired: bool = false };
const inventory: memory.weights_plan.Inventory = .{ .raw = 110 * GiB, .routed = 80 * GiB, .ple = 24 * GiB, .expert_parts = @splat(256 * 1024) };
const settings: Settings = .{ .capacity = 65568, .ssd_experts = 8 * GiB, .ple_on_ssd = true };

fn budget(inv: memory.weights_plan.Inventory, s: Settings, ctx: Ctx) !memory.Budget {
    const p = try layout.of(inv, s);
    var resident = inv;
    resident.raw = p.raw;
    return memory.budgetLiveFor(0, s.capacity, resident, ctx.extra + ctx.existing + p.extra, ctx.existing, s.sibling(ctx.paired));
}

fn fit(inv: memory.weights_plan.Inventory, s: Settings, ctx: Ctx) !Settings {
    if (@hasDecl(layout, "fit")) return layout.fit(inv, s, ctx);
    return s;
}

test "SSD budget: a refused maximum becomes the largest admitted complete-demand pool" {
    const before = try budget(inventory, settings, .{ .limit = 0 });
    const ctx: Ctx = .{ .limit = before.steady - 3 * GiB };
    try t.expectError(error.FlashNextMemoryShortfall, memory.admitBudget(ctx.limit, before, false));
    const next = try fit(inventory, settings, ctx);
    try t.expect(next.ssd_experts.? < settings.ssd_experts.?);
    const p = (try layout.of(inventory, next)).pool.?;
    try t.expect(p.slots >= 512);
    const after = try budget(inventory, next, ctx);
    try t.expectEqual(ctx.limit, try memory.admitBudget(ctx.limit, after, false));
    var wider = next;
    wider.ssd_experts = next.ssd_experts.? + p.stride;
    try t.expectError(error.FlashNextMemoryShortfall, memory.admitBudget(ctx.limit, try budget(inventory, wider, ctx), false));
    try t.expectEqual(settings.capacity, next.capacity);
    try t.expectEqual(settings.factor, next.factor);
    try t.expectEqual(settings.ple_on_ssd, next.ple_on_ssd);
}

test "SSD budget: exact slot boundary counts existing allocations, sessions and requested cache" {
    var smaller = settings;
    const p = (try layout.of(inventory, settings)).pool.?;
    smaller.ssd_experts = 768 * p.stride;
    var ctx: Ctx = .{ .limit = 0, .existing = GiB, .extra = 2 * GiB };
    ctx.limit = (try budget(inventory, smaller, ctx)).steady;
    const next = try fit(inventory, settings, ctx);
    try t.expectEqual(smaller.ssd_experts, next.ssd_experts);
    ctx.limit -= 1;
    const one_less = try fit(inventory, settings, ctx);
    try t.expectEqual(@as(?u64, 767 * p.stride), one_less.ssd_experts);
}

test "SSD budget: admitted settings, no expert request and impossible fits retain their old policy" {
    const before = try budget(inventory, settings, .{ .limit = 0 });
    try t.expectEqualDeep(settings, try fit(inventory, settings, .{ .limit = before.steady }));
    var exact = settings;
    exact.ssd_experts = settings.ssd_experts.? + 17;
    try t.expectEqualDeep(exact, try fit(inventory, exact, .{ .limit = std.math.maxInt(u64) }));
    var in_memory = settings;
    in_memory.ssd_experts = null;
    try t.expectEqualDeep(in_memory, try fit(inventory, in_memory, .{ .limit = 1 }));
    try t.expectEqualDeep(settings, try fit(inventory, settings, .{ .limit = 1 }));
    try t.expectEqualDeep(settings, try fit(inventory, settings, .{ .limit = 1, .paired = true }));
    var cold = inventory;
    cold.source = 80 * GiB;
    cold.decode = GiB;
    cold.prompt = GiB;
    try t.expectEqualDeep(settings, try fit(cold, settings, .{ .limit = before.steady, .needs_build = true }));
}

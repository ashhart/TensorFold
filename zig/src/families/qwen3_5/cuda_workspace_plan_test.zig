//! Typed row budgets and named lifetimes exercise only address arithmetic, never a driver or allocation.
const std = @import("std");
const E = @import("cuda_engine.zig");
const scratch = @import("cuda_scratch.zig");
const workspace = @import("cuda_workspace_plan.zig");
const Plan = workspace.WorkspacePlan;
const t = std.testing;

fn geometry() E.Geometry {
    return .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 131072, .linear = @splat(false) };
}

fn engine(g: E.Geometry) E.Engine {
    var e: E.Engine = undefined;
    e.g = g;
    e.arena_at = 512;
    e.arena = .{ .d = undefined, .ptr = 0x100000000, .len = 1 << 40, .borrowed = true };
    e.routed = .{};
    return e;
}

fn addresses(plan: *const Plan, e: *const E.Engine, base: usize, measured: bool) !void {
    inline for (comptime std.meta.tags(workspace.Name)) |name| if (plan.region(name)) |region| {
        const field = comptime @tagName(name);
        const got = if (comptime std.mem.startsWith(u8, field, "routed_"))
            @field(e.routed, field["routed_".len..])
        else
            @field(e, field);
        try t.expectEqual(if (measured) @as(u64, 0) else e.arena.ptr + base + region.offset, got);
        try t.expectEqual(workspace.Owner.engine_arena, region.owner);
        try t.expectEqual(@as(usize, 256), region.alignment);
        try t.expectEqual(@as(usize, 0), region.offset % region.alignment);
        try t.expect(region.offset + region.bytes <= plan.budget.extent);
    };
}

const Phase = enum { delta, attention, dense_mlp, routed };
fn live(lifetime: workspace.Lifetime, phase: Phase) bool {
    return switch (lifetime) {
        .round => true,
        .delta_attention => phase == .delta or phase == .attention,
        .delta => phase == .delta,
        .attention => phase == .attention,
        .dense_mlp => phase == .dense_mlp,
        .routed => phase == .routed,
    };
}

fn lifetimes(plan: *const Plan) !void {
    for (std.enums.values(Phase)) |phase| for (std.enums.values(workspace.Name), 0..) |a, i| {
        const left = plan.region(a) orelse continue;
        if (!live(left.lifetime, phase)) continue;
        for (std.enums.values(workspace.Name)[0..i]) |b| {
            const right = plan.region(b) orelse continue;
            if (!live(right.lifetime, phase)) continue;
            try t.expect(left.offset + left.bytes <= right.offset or right.offset + right.bytes <= left.offset);
        }
    };
}

test "workspace plan: compiled rows preserve measurement assignments and arena ownership" {
    const plan = try Plan.forRows(geometry(), E.max_rows);
    try t.expectEqual(@as(usize, E.max_rows), plan.rows);
    try t.expectEqual(workspace.Owner.engine_arena, plan.owner);
    try t.expectEqual(@as(usize, 256), plan.alignment);
    var measured = engine(geometry());
    const before = measured.arena_at;
    try t.expectEqual(plan.budget.bytes, try plan.carve(&measured, .measure));
    try t.expectEqual(before, measured.arena_at);
    try addresses(&plan, &measured, before, true);
    var allocated = engine(geometry());
    try t.expectEqual(plan.budget.bytes, try scratch.scratch(&allocated, .allocate));
    try t.expectEqual(before + plan.budget.extent, allocated.arena_at);
    try addresses(&plan, &allocated, before, false);
    try lifetimes(&plan);
}

test "workspace plan: smaller future row limits change real regions without changing verification scratch" {
    const full = try Plan.forRows(geometry(), E.max_rows);
    for ([_]usize{ 1, E.most_verify, E.max_rows / 2, E.max_rows }) |rows| {
        const plan = try Plan.forRows(geometry(), rows);
        try t.expectEqual(rows * geometry().hidden * 2, plan.region(.h).?.bytes);
        try t.expectEqual(full.region(.pm).?.bytes, plan.region(.pm).?.bytes);
        try t.expectEqual(full.region(.logits).?.bytes, plan.region(.logits).?.bytes);
        if (rows < E.max_rows) try t.expect(plan.budget.bytes < full.budget.bytes);
        var e = engine(geometry());
        const before = e.arena_at;
        try t.expectEqual(plan.budget.bytes, try plan.carve(&e, .allocate));
        try t.expectEqual(before + plan.budget.extent, e.arena_at);
        try addresses(&plan, &e, before, false);
        try lifetimes(&plan);
    }
    try t.expectEqual(workspace.Lifetime.delta_attention, full.region(.y).?.lifetime);
    try t.expectEqual(workspace.Lifetime.attention, full.region(.po).?.lifetime);
    try t.expectEqual(workspace.Lifetime.dense_mlp, full.region(.gate).?.lifetime);
    try t.expectEqual(full.region(.y).?.offset, full.region(.gate).?.offset);
}

test "workspace plan: named allocation order includes the old attention aliases and MoE regions" {
    const prefix = [_]workspace.Name{
        .h, .x,   .r,    .xs, .y,    .mix,  .z,  .v,  .q,  .k,  .qkv,  .a,  .b,   .beta,   .g_,
        .y, .mix, .qraw, .qn, .kraw, .vraw, .kn, .pm, .pl, .po, .gate, .up, .act, .logits,
    };
    for ([_]bool{ false, true }) |moe| {
        var g = geometry();
        if (moe) {
            g.experts = 8;
            g.top_k = 2;
        }
        const plan = try Plan.forRows(g, E.max_rows);
        const sequence = plan.allocations();
        try t.expectEqual(@as(usize, if (moe) 41 else 31), sequence.len);
        for (prefix, sequence[0..prefix.len]) |name, allocation| try t.expectEqual(@as(?workspace.Name, name), allocation.name);
        if (moe) {
            const routed = [_]workspace.Name{ .routed_logits, .routed_picks, .routed_wts, .routed_members, .routed_items, .routed_counts, .routed_rank, .routed_hist, .routed_act, .routed_y };
            for (routed, sequence[prefix.len..][0..routed.len]) |name, allocation| try t.expectEqual(@as(?workspace.Name, name), allocation.name);
            try t.expectEqual(@as(usize, 8), plan.region(.routed_counts).?.bytes);
        } else try t.expect(plan.region(.routed_counts) == null);
        try t.expectEqual(@as(?workspace.Name, .amax), sequence[sequence.len - 2].name);
        try t.expectEqual(@as(?workspace.Name, .picks), sequence[sequence.len - 1].name);
        var e = engine(g);
        _ = try plan.carve(&e, .allocate);
        try addresses(&plan, &e, 512, false);
        try lifetimes(&plan);
    }
}

test "workspace plan: alignment and one-byte refusal retain the old partial allocation boundary" {
    const plan = try Plan.forRows(geometry(), E.max_rows / 2);
    var exact = engine(geometry());
    exact.arena_at = 513;
    const bytes = try plan.requiredAt(exact.arena_at);
    exact.arena.len = exact.arena_at + bytes;
    try t.expectEqual(plan.budget.bytes, try plan.carve(&exact, .allocate));
    try t.expectEqual(exact.arena.len, exact.arena_at);
    try addresses(&plan, &exact, 768, false);
    var short = engine(geometry());
    short.arena_at = 513;
    short.arena.len = short.arena_at + bytes - 1;
    short.picks = 77;
    try t.expectError(error.ArenaExhausted, plan.carve(&short, .allocate));
    try t.expectEqual(@as(u64, 77), short.picks);
    try t.expect(short.amax > short.arena.ptr);
    try t.expect(short.arena_at < short.arena.len);
}

test "workspace plan: invalid rows shape and alignment refuse before the engine is changed" {
    try t.expectError(error.InvalidWorkspaceRows, Plan.forRows(geometry(), 0));
    try t.expectError(error.InvalidWorkspaceRows, Plan.forRows(geometry(), E.max_rows + 1));
    var plan = try Plan.forRows(geometry(), E.max_rows);
    var e = engine(geometry());
    e.h = 55;
    e.g.hidden += 256;
    try t.expectError(error.WorkspaceShapeChanged, plan.carve(&e, .allocate));
    try t.expectEqual(@as(usize, 512), e.arena_at);
    try t.expectEqual(@as(u64, 55), e.h);
    e.g = geometry();
    plan.alignment = 128;
    try t.expectError(error.InvalidWorkspaceAlignment, plan.carve(&e, .allocate));
    try t.expectEqual(@as(usize, 512), e.arena_at);
    try t.expectEqual(@as(u64, 55), e.h);
}

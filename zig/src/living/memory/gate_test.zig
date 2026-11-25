//! Tests for the dual-layer neural gate: routing, coverage, bridges, conflicts and ClusterIndex recall.
const std = @import("std");
const whitener = @import("whitener.zig");
const gate = @import("gate.zig");
const UnitGate = gate.UnitGate;
const UnitView = gate.UnitView;
const ClusterIndex = gate.ClusterIndex;
const computeBridgeEdges = gate.computeBridgeEdges;
const findConflictParents = gate.findConflictParents;
const routeUnits = gate.routeUnits;
const routeUnitsIndexed = gate.routeUnitsIndexed;
const normalizeInPlace = @import("cluster_index.zig").normalizeInPlace;

test "gate: hits matching unit, rejects orthogonal query, enforces parent diversity" {
    const gpa = std.testing.allocator;
    const dim = 16;

    var lp0: [dim]f32 = @splat(0);
    lp0[0] = 1.0;
    var lp2: [dim]f32 = @splat(0);
    lp2[1] = 1.0;

    var g0 = UnitGate{
        .dim = dim,
        .lex_protos = try gpa.dupe(f32, &lp0),
        .lex_weights = try gpa.dupe(f32, &.{0.9}),
        .n_lex = 1,
        .n_body_lex = 1,
        .sem_protos = try gpa.dupe(f32, &lp0),
        .sem_weights = try gpa.dupe(f32, &.{0.9}),
        .n_sem = 1,
        .neg_sem_max = 0.1,
        .is_small_fact = true,
    };
    defer g0.deinit(gpa);
    var g2 = UnitGate{
        .dim = dim,
        .lex_protos = try gpa.dupe(f32, &lp2),
        .lex_weights = try gpa.dupe(f32, &.{0.9}),
        .n_lex = 1,
        .n_body_lex = 1,
        .sem_protos = try gpa.dupe(f32, &lp2),
        .sem_weights = try gpa.dupe(f32, &.{0.9}),
        .n_sem = 1,
        .neg_sem_max = 0.1,
        .is_small_fact = false,
    };
    defer g2.deinit(gpa);

    const views = [_]UnitView{
        .{ .index = 0, .parent_id = 10, .n_tokens = 20, .gate = &g0 },
        .{ .index = 1, .parent_id = 20, .n_tokens = 100, .gate = &g2 },
    };

    var q_hit_u: [2 * dim]f32 = @splat(0);
    q_hit_u[0] = 1.0;
    q_hit_u[dim + 0] = 1.0;
    var tr_hit = whitener.Transformed{
        .unit = &q_hit_u,
        .norms = @constCast(&[_]f32{ 2.0, 2.0 }),
        .spec = @constCast(&[_]f32{ 0.9, 0.9 }),
    };
    const active = try routeUnits(gpa, &views, &.{}, &tr_hit, &tr_hit, 1.0, 1.0, .{});
    defer gpa.free(active);
    try std.testing.expectEqualSlices(usize, &.{0}, active);

    var q_miss_u: [2 * dim]f32 = @splat(0);
    q_miss_u[5] = 1.0;
    q_miss_u[dim + 6] = 1.0;
    var tr_miss = whitener.Transformed{
        .unit = &q_miss_u,
        .norms = @constCast(&[_]f32{ 2.0, 2.0 }),
        .spec = @constCast(&[_]f32{ 0.9, 0.9 }),
    };
    const none = try routeUnits(gpa, &views, &.{}, &tr_miss, &tr_miss, 1.0, 1.0, .{});
    defer gpa.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
}

test "gate: submodular coverage picks complementary section over redundant duplicate" {
    const gpa = std.testing.allocator;
    const dim = 16;

    // Query dims: 0 Halcyon-8, 1 Halcyon-7, 2 payload; unit 0 = 0+2, unit 1 = 0+2 (lower), unit 2 = 1+2
    var p0: [2 * dim]f32 = @splat(0);
    p0[0] = 1.0;
    p0[dim + 2] = 1.0;
    var p1: [2 * dim]f32 = @splat(0);
    p1[0] = 0.98;
    p1[3] = @sqrt(1.0 - 0.98 * 0.98);
    p1[dim + 2] = 0.98;
    p1[dim + 4] = @sqrt(1.0 - 0.98 * 0.98);
    var p2: [2 * dim]f32 = @splat(0);
    p2[1] = 0.95;
    p2[5] = @sqrt(1.0 - 0.95 * 0.95);
    p2[dim + 2] = 0.90;
    p2[dim + 6] = @sqrt(1.0 - 0.90 * 0.90);

    var g0 = UnitGate{ .dim = dim, .lex_protos = try gpa.dupe(f32, &p0), .lex_weights = try gpa.dupe(f32, &.{ 0.9, 0.9 }), .n_lex = 2, .n_body_lex = 2, .sem_protos = try gpa.dupe(f32, &p0), .sem_weights = try gpa.dupe(f32, &.{ 0.9, 0.9 }), .n_sem = 2 };
    defer g0.deinit(gpa);
    var g1 = UnitGate{ .dim = dim, .lex_protos = try gpa.dupe(f32, &p1), .lex_weights = try gpa.dupe(f32, &.{ 0.9, 0.9 }), .n_lex = 2, .n_body_lex = 2, .sem_protos = try gpa.dupe(f32, &p1), .sem_weights = try gpa.dupe(f32, &.{ 0.9, 0.9 }), .n_sem = 2 };
    defer g1.deinit(gpa);
    var g2 = UnitGate{ .dim = dim, .lex_protos = try gpa.dupe(f32, &p2), .lex_weights = try gpa.dupe(f32, &.{ 0.85, 0.85 }), .n_lex = 2, .n_body_lex = 2, .sem_protos = try gpa.dupe(f32, &p2), .sem_weights = try gpa.dupe(f32, &.{ 0.85, 0.85 }), .n_sem = 2 };
    defer g2.deinit(gpa);

    const views = [_]UnitView{
        .{ .index = 0, .parent_id = 1, .n_tokens = 120, .gate = &g0 },
        .{ .index = 1, .parent_id = 1, .n_tokens = 110, .gate = &g1 },
        .{ .index = 2, .parent_id = 1, .n_tokens = 100, .gate = &g2 },
    };

    var qu: [3 * dim]f32 = @splat(0);
    qu[0] = 1.0;
    qu[dim + 1] = 1.0;
    qu[2 * dim + 2] = 1.0;
    var tr = whitener.Transformed{
        .unit = &qu,
        .norms = @constCast(&[_]f32{ 2.0, 2.0, 2.0 }),
        .spec = @constCast(&[_]f32{ 0.9, 0.9, 0.9 }),
    };

    // With top_k=2, submodular coverage MUST pick Unit 0 AND Unit 2 (skipping redundant Unit 1)!
    const picked = try routeUnits(gpa, &views, &.{}, &tr, &tr, 1.0, 1.0, .{ .top_k = 2, .max_per_parent = 2 });
    defer gpa.free(picked);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, picked);
}

test "gate: 2-hop associative bridge links rare shared entity without causing drift on unrelated queries" {
    const gpa = std.testing.allocator;
    const dim = 16;

    // Unit 0: dims 0 (Windtunnel-9) + 1 (Rotherham); unit 1: dims 1 + 2 (wing spars); unit 2: dim 3
    var p0: [2 * dim]f32 = @splat(0);
    p0[0] = 1.0;
    p0[dim + 1] = 1.0;
    var p1: [2 * dim]f32 = @splat(0);
    p1[1] = 1.0;
    p1[dim + 2] = 1.0;
    var p2: [dim]f32 = @splat(0);
    p2[3] = 1.0;

    var g0 = UnitGate{ .dim = dim, .lex_protos = try gpa.dupe(f32, &p0), .lex_weights = try gpa.dupe(f32, &.{ 0.9, 0.85 }), .n_lex = 2, .n_body_lex = 2, .sem_protos = try gpa.dupe(f32, &p0), .sem_weights = try gpa.dupe(f32, &.{ 0.9, 0.85 }), .n_sem = 2 };
    defer g0.deinit(gpa);
    var g1 = UnitGate{ .dim = dim, .lex_protos = try gpa.dupe(f32, &p1), .lex_weights = try gpa.dupe(f32, &.{ 0.85, 0.9 }), .n_lex = 2, .n_body_lex = 2, .sem_protos = try gpa.dupe(f32, &p1), .sem_weights = try gpa.dupe(f32, &.{ 0.85, 0.9 }), .n_sem = 2 };
    defer g1.deinit(gpa);
    var g2 = UnitGate{ .dim = dim, .lex_protos = try gpa.dupe(f32, &p2), .lex_weights = try gpa.dupe(f32, &.{0.9}), .n_lex = 1, .n_body_lex = 1, .sem_protos = try gpa.dupe(f32, &p2), .sem_weights = try gpa.dupe(f32, &.{0.9}), .n_sem = 1 };
    defer g2.deinit(gpa);

    const views = [_]UnitView{
        .{ .index = 0, .parent_id = 6, .n_tokens = 200, .gate = &g0 },
        .{ .index = 1, .parent_id = 6, .n_tokens = 180, .gate = &g1 },
        .{ .index = 2, .parent_id = 6, .n_tokens = 150, .gate = &g2 },
    };

    const bridges = try computeBridgeEdges(gpa, &views);
    defer gpa.free(bridges);
    try std.testing.expect(bridges.len >= 2);

    // Query mentions ONLY dim0 (Windtunnel-9) — never mentions Rotherham (dim1) or wing spars (dim2)!
    var q_wt: [2 * dim]f32 = @splat(0);
    q_wt[0] = 1.0;
    q_wt[dim + 0] = 1.0;
    var tr_wt = whitener.Transformed{
        .unit = &q_wt,
        .norms = @constCast(&[_]f32{ 2.0, 2.0 }),
        .spec = @constCast(&[_]f32{ 0.9, 0.9 }),
    };
    const picked = try routeUnits(gpa, &views, bridges, &tr_wt, &tr_wt, 1.0, 1.0, .{});
    defer gpa.free(picked);
    // Hop-1 picks Unit 0 (Windtunnel-9); Hop-2 bridge follows Rotherham -> activates Unit 1 too!
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, picked);

    // Unrelated query (dim 9) -> STILL zero hits (zero drift!)
    var q_unrel: [2 * dim]f32 = @splat(0);
    q_unrel[9] = 1.0;
    q_unrel[dim + 10] = 1.0;
    var tr_unrel = whitener.Transformed{
        .unit = &q_unrel,
        .norms = @constCast(&[_]f32{ 2.0, 2.0 }),
        .spec = @constCast(&[_]f32{ 0.9, 0.9 }),
    };
    const none = try routeUnits(gpa, &views, bridges, &tr_unrel, &tr_unrel, 1.0, 1.0, .{});
    defer gpa.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
}

test "gate(v6): single-token subword veto blocks low-semantic collision and skips superseded units" {
    const gpa = std.testing.allocator;
    const dim = 16;

    // Document chunk with lexical vector on dim 0 ("Loombook"), semantic vector on dim 1 (aerospace software)
    var lp: [dim]f32 = @splat(0);
    lp[0] = 1.0;
    var sp: [dim]f32 = @splat(0);
    sp[1] = 1.0;

    var g_doc = UnitGate{
        .dim = dim,
        .lex_protos = try gpa.dupe(f32, &lp),
        .lex_weights = try gpa.dupe(f32, &.{0.95}),
        .n_lex = 1,
        .n_body_lex = 1,
        .sem_protos = try gpa.dupe(f32, &sp),
        .sem_weights = try gpa.dupe(f32, &.{0.90}),
        .n_sem = 1,
        .neg_sem_max = 0.15,
        .is_small_fact = false,
    };
    defer g_doc.deinit(gpa);

    // Superseded small fact on dim 2 (old belief: "Barnaby") vs active new belief on dim 2 ("Biscuit")
    var p_dog: [dim]f32 = @splat(0);
    p_dog[2] = 1.0;
    var g_old = UnitGate{
        .dim = dim,
        .lex_protos = try gpa.dupe(f32, &p_dog),
        .lex_weights = try gpa.dupe(f32, &.{0.9}),
        .n_lex = 1,
        .n_body_lex = 1,
        .sem_protos = try gpa.dupe(f32, &p_dog),
        .sem_weights = try gpa.dupe(f32, &.{0.9}),
        .n_sem = 1,
        .is_small_fact = true,
        .superseded_by = 99,
    };
    defer g_old.deinit(gpa);
    var g_new = UnitGate{
        .dim = dim,
        .lex_protos = try gpa.dupe(f32, &p_dog),
        .lex_weights = try gpa.dupe(f32, &.{0.9}),
        .n_lex = 1,
        .n_body_lex = 1,
        .sem_protos = try gpa.dupe(f32, &p_dog),
        .sem_weights = try gpa.dupe(f32, &.{0.9}),
        .n_sem = 1,
        .is_small_fact = true,
        .superseded_by = 0,
    };
    defer g_new.deinit(gpa);

    const views = [_]UnitView{
        .{ .index = 0, .parent_id = 6, .n_tokens = 250, .gate = &g_doc },
        .{ .index = 1, .parent_id = 2, .n_tokens = 15, .gate = &g_old },
        .{ .index = 2, .parent_id = 99, .n_tokens = 15, .gate = &g_new },
    };

    // 1. Single-token lexical hit on dim 0 ("loom") with orthogonal semantics (dim 8: weaving) -> MUST NOT FIRE!
    var ql_loom: [2 * dim]f32 = @splat(0);
    ql_loom[0] = 0.96; // "loom" matches "Loombook"
    ql_loom[dim + 7] = 1.0; // "warp" orthogonal
    var qs_loom: [2 * dim]f32 = @splat(0);
    qs_loom[8] = 1.0;
    qs_loom[dim + 9] = 1.0;
    var tr_ql = whitener.Transformed{ .unit = &ql_loom, .norms = @constCast(&[_]f32{ 2.0, 2.0 }), .spec = @constCast(&[_]f32{ 0.9, 0.9 }) };
    var tr_qs = whitener.Transformed{ .unit = &qs_loom, .norms = @constCast(&[_]f32{ 2.0, 2.0 }), .spec = @constCast(&[_]f32{ 0.9, 0.9 }) };
    const hit_loom = try routeUnits(gpa, &views, &.{}, &tr_ql, &tr_qs, 1.0, 1.0, .{});
    defer gpa.free(hit_loom);
    try std.testing.expectEqual(@as(usize, 0), hit_loom.len);

    // 2. Query on dim 2 ("dog") -> MUST pick ONLY active unit 2 (pid=99), NEVER superseded unit 1 (pid=2)!
    var q_dog: [2 * dim]f32 = @splat(0);
    q_dog[2] = 1.0;
    q_dog[dim + 2] = 1.0;
    var tr_dog = whitener.Transformed{ .unit = &q_dog, .norms = @constCast(&[_]f32{ 2.0, 2.0 }), .spec = @constCast(&[_]f32{ 0.9, 0.9 }) };
    const hit_dog = try routeUnits(gpa, &views, &.{}, &tr_dog, &tr_dog, 1.0, 1.0, .{});
    defer gpa.free(hit_dog);
    try std.testing.expectEqualSlices(usize, &.{2}, hit_dog);

    // 3. Entity conflict detector finds active pid=99 (and NOT superseded pid=2)!
    const conflicts = try findConflictParents(gpa, &views, &tr_dog, &tr_dog, 1.0, 1.0, 4);
    defer gpa.free(conflicts);
    try std.testing.expectEqual(@as(usize, 1), conflicts.len);
    try std.testing.expectEqual(@as(u32, 99), conflicts[0].parent_id);
}

test "gate(v6): 2-level ClusterIndex scales to 2,048 units with exact target recall" {
    const gpa = std.testing.allocator;
    const dim = 32;
    const n_units: usize = 2048;

    const gates = try gpa.alloc(UnitGate, n_units);
    defer {
        for (gates) |*g| g.deinit(gpa);
        gpa.free(gates);
    }
    const views = try gpa.alloc(UnitView, n_units);
    defer gpa.free(views);

    var prng = std.Random.DefaultPrng.init(0x5eed1234);
    const rng = prng.random();

    for (0..n_units) |i| {
        const lp = try gpa.alloc(f32, dim);
        const sp = try gpa.alloc(f32, dim);
        for (0..dim) |d| {
            lp[d] = rng.floatNorm(f32);
            sp[d] = rng.floatNorm(f32);
        }
        // Plant a crisp rare needle at unit #1337 on dimensions 0 and 1
        if (i == 1337) {
            @memset(lp, 0.0);
            @memset(sp, 0.0);
            lp[0] = 1.0;
            sp[1] = 1.0;
        } else {
            // Keep distractors away from the exact planted needle axis
            lp[0] *= 0.05;
            sp[1] *= 0.05;
            normalizeInPlace(lp);
            normalizeInPlace(sp);
        }
        gates[i] = .{
            .dim = dim,
            .lex_protos = lp,
            .lex_weights = try gpa.dupe(f32, &.{0.9}),
            .n_lex = 1,
            .n_body_lex = 1,
            .sem_protos = sp,
            .sem_weights = try gpa.dupe(f32, &.{0.9}),
            .n_sem = 1,
            .neg_sem_max = 0.25,
            .is_small_fact = true,
        };
        try gates[i].computeSummaries(gpa);
        views[i] = .{ .index = i, .parent_id = @intCast(i + 1), .n_tokens = 20, .gate = &gates[i] };
    }

    var idx = ClusterIndex{};
    defer idx.deinit(gpa);
    try idx.rebuild(gpa, views, dim);
    try std.testing.expect(idx.isReady());
    try std.testing.expect(idx.n_clusters >= 16);

    var ql_u: [2 * dim]f32 = @splat(0);
    ql_u[0] = 1.0;
    ql_u[dim + 0] = 1.0;
    var qs_u: [2 * dim]f32 = @splat(0);
    qs_u[1] = 1.0;
    qs_u[dim + 1] = 1.0;
    var tr_l = whitener.Transformed{ .unit = &ql_u, .norms = @constCast(&[_]f32{ 2.0, 2.0 }), .spec = @constCast(&[_]f32{ 0.9, 0.9 }) };
    var tr_s = whitener.Transformed{ .unit = &qs_u, .norms = @constCast(&[_]f32{ 2.0, 2.0 }), .spec = @constCast(&[_]f32{ 0.9, 0.9 }) };

    const active = try routeUnitsIndexed(gpa, views, &.{}, &idx, &tr_l, &tr_s, 1.0, 1.0, .{});
    defer gpa.free(active);
    var found = false;
    for (active) |u_idx| if (u_idx == 1337) {
        found = true;
        break;
    };
    try std.testing.expect(found);
}

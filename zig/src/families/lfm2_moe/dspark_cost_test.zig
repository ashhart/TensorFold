//! DSpark cost consumption, repricing boundaries and invalid measurements.
const std = @import("std");
const block = @import("dspark_block.zig");
const ds = @import("dspark.zig");
const lanes = @import("lanes");
const admission = @import("dspark_admission.zig");

test "lfm2 dspark: timing consumes CPU and GPU once without consuming pending EMA metadata" {
    var d: ds.Drafter = undefined;
    d.encode_cpu_ms = 0.25;
    d.hold_spent_ms = 0;
    d.pass_encode_cpu_ms = 0.25;
    d.pass_context_rows = 8;
    try std.testing.expectEqual(@as(f64, 0.25), block.takeHoldMs(&d));
    try std.testing.expectEqual(@as(f64, 0), d.encode_cpu_ms);
    try std.testing.expectEqual(@as(f64, 0.25), d.pass_encode_cpu_ms);
    try std.testing.expectEqual(@as(u32, 8), d.pass_context_rows);
    try std.testing.expectEqual(@as(f64, 0), block.takeHoldMs(&d));
    d.hold_spent_ms = 1.5;
    try std.testing.expectEqual(@as(f64, 1.5), block.takeHoldMs(&d));
    try std.testing.expectEqual(@as(f64, 0.25), d.pass_encode_cpu_ms);
    try std.testing.expectEqual(@as(f64, 0), block.takeHoldMs(&d));
    d.encode_cpu_ms = 0.5;
    d.hold_spent_ms = 2;
    try std.testing.expectEqual(@as(f64, 2.5), block.takeHoldMs(&d));
    try std.testing.expectEqual(@as(f64, 0), block.takeHoldMs(&d));
}

test "lfm2 dspark: a completed hold moves its block counts a tenth of the way to CPU-encode plus GPU ms" {
    var d: ds.Drafter = undefined;
    const grid = [_]lanes.hold_price.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 1 }, .{ .streams = 4, .depth = 1, .ms = 2 }, .{ .streams = 4, .depth = 4, .ms = 4 }, .{ .streams = 8, .depth = 1, .ms = 3 } };
    @memcpy(d.holds[0..grid.len], &grid);
    d.holds_len = grid.len;
    d.contexts_len = 0;
    d.priceHold(4, 1, 0, 4); // priced 2 ms: the 4-block passes rise by a tenth of the doubling
    try std.testing.expectApproxEqAbs(@as(f64, 2.2), d.holds[1].ms, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.4), d.holds[2].ms, 1e-12);
    try std.testing.expectEqual(@as(f64, 1), d.holds[0].ms);
    try std.testing.expectEqual(@as(f64, 3), d.holds[3].ms);
    d.priceHold(6, 1, 0, 0); // between 4 and 8 blocks, no measured time: nothing learned
    try std.testing.expectEqual(@as(f64, 2.2), d.holds[1].ms);
    d.priceHold(6, 1, 0, 2.6 * 2); // both neighbors move
    try std.testing.expect(d.holds[1].ms > 2.2 and d.holds[3].ms > 3 and d.holds[0].ms == 1);
    const before = d.holds[1].ms;
    d.priceHold(0, 1, 0, 5); // a pass of no blocks prices nothing
    try std.testing.expectEqual(before, d.holds[1].ms);
    d.holds_len = 0; // no grid (the load's timing holds): nothing to move
    d.priceHold(4, 1, 0, 100);
    try std.testing.expectEqual(before, d.holds[1].ms);
}

test "lfm2 dspark: hold repricing subtracts context and refuses nonpositive or nonfinite residuals" {
    var d: ds.Drafter = undefined;
    d.holds[0] = .{ .streams = 4, .depth = 1, .ms = 2 };
    d.holds_len = 1;
    d.contexts[0] = .{ .streams = 4, .rows = 8, .ms = 3 };
    d.contexts_len = 1;
    d.priceHold(4, 1, 5, 7); // seven CPU-encode + GPU ms, three estimated context ms: clean moves toward four
    try std.testing.expectApproxEqAbs(@as(f64, 2.2), d.holds[0].ms, 1e-12);
    for ([_]f64{ 3, 2, 0, -1, std.math.inf(f64), std.math.nan(f64) }) |ms| {
        d.priceHold(4, 1, 5, ms);
        try std.testing.expectEqual(@as(f64, 2.2), d.holds[0].ms);
    }
    d.priceHold(4, 1, 9, 7); // debt beyond measured coverage must not teach a fabricated clean time
    try std.testing.expectEqual(@as(f64, 2.2), d.holds[0].ms);
}

const admission_holds = [_]lanes.hold_price.HoldCost{
    .{ .streams = 1, .depth = 1, .ms = 1 },
    .{ .streams = 1, .depth = 9, .ms = 1 },
    .{ .streams = 3, .depth = 1, .ms = 1.5 },
    .{ .streams = 3, .depth = 9, .ms = 1.5 },
};
const admission_contexts = [_]lanes.hold_price.ContextCost{
    .{ .streams = 1, .rows = 16, .ms = 0.25 },
    .{ .streams = 3, .rows = 16, .ms = 0.75 },
    .{ .streams = 1, .rows = 2048, .ms = 3 },
    .{ .streams = 3, .rows = 2048, .ms = 9 },
};

fn admissionPrices(target: *const lanes.table.Table) admission.Prices {
    return .{ .target = target, .holds = &admission_holds, .contexts = &admission_contexts, .streams = 1, .heads = 1, .target_rows = 16, .pass_rows = 32, .block_rows = 9, .depth = 9, .context_rows = 1, .context_chunk = 2048 };
}

test "lfm2 dspark: admission paying prefixes beat bare target rows without discounting the query block" {
    var slots: [17]?f64 = @splat(40);
    slots[1] = 10;
    slots[2] = 11;
    const target: lanes.table.Table = .{ .slots = &slots };
    const prices = admissionPrices(&target);
    const rates = [_]f64{ 0.9, 0, 0, 0, 0, 0, 0, 0, 0 };
    const chosen = admission.quote(prices, &rates).?;
    try std.testing.expect(chosen.pays());
    try std.testing.expectEqual(@as(u32, 1), chosen.best.?.depth);
    try std.testing.expectEqual(@as(u32, 2), chosen.best.?.rows);
    try std.testing.expect(chosen.best.?.gain > 0);
    try std.testing.expectApproxEqAbs(@as(f64, 12.25), chosen.best.?.ms, 1e-12);
    slots[2] = 30; // the same accepted prefix no longer pays for its real target verify rows
    try std.testing.expect(!admission.quote(prices, &rates).?.pays());
}

test "lfm2 dspark: admission full9 splits can reject a fourth prefix or fund two actual passes" {
    var slots: [9]?f64 = @splat(10);
    const target: lanes.table.Table = .{ .slots = &slots };
    var holds = [_]lanes.hold_price.HoldCost{
        .{ .streams = 1, .depth = 1, .ms = 5 },
        .{ .streams = 3, .depth = 1, .ms = 5 },
    };
    const contexts = [_]lanes.hold_price.ContextCost{
        .{ .streams = 1, .rows = 2, .ms = 0.1 },
        .{ .streams = 3, .rows = 2, .ms = 0.3 },
    };
    var prices = admissionPrices(&target);
    prices.holds = &holds;
    prices.contexts = &contexts;
    prices.streams = 4;
    prices.heads = 4;
    prices.depth = 1;
    prices.target_rows = 8;
    const three = admission.quote(prices, &.{0.9}).?;
    try std.testing.expect(three.pays());
    try std.testing.expectEqual(@as(usize, 3), three.best.?.heads);
    try std.testing.expectEqual(@as(usize, 1), three.best.?.passes);
    try std.testing.expectEqual(@as(u32, 7), three.best.?.rows);
    // Four prefixes still run 36 query rows in two passes; lower physical prices make that split pay.
    holds[0].ms = 0.5;
    holds[1].ms = 1;
    const four = admission.quote(prices, &.{0.9}).?;
    try std.testing.expect(four.pays());
    try std.testing.expectEqual(@as(usize, 4), four.best.?.heads);
    try std.testing.expectEqual(@as(usize, 2), four.best.?.passes);
    try std.testing.expectEqual(@as(u32, 8), four.best.?.rows);
    try std.testing.expectApproxEqAbs(@as(f64, 11.9), four.best.?.ms, 1e-12);
    prices.target_rows = 7;
    try std.testing.expectEqual(@as(usize, 3), admission.quote(prices, &.{0.9}).?.best.?.heads);
    prices.target_rows = 4;
    try std.testing.expect(!admission.quote(prices, &.{0.9}).?.pays());
}

test "lfm2 dspark: admission initial prompt overflow is paid rather than amortized away" {
    var slots: [17]?f64 = @splat(11);
    slots[1] = 10;
    const target: lanes.table.Table = .{ .slots = &slots };
    var prices = admissionPrices(&target);
    prices.depth = 1;
    try std.testing.expect(admission.quote(prices, &.{0.9}).?.pays());
    prices.context_rows = 4097; // two individual overflow flushes plus a staged tail
    const expensive = admission.quote(prices, &.{0.9}).?;
    try std.testing.expect(!expensive.pays());
    try std.testing.expectApproxEqAbs(@as(f64, 18.25), expensive.best.?.ms, 1e-12);
    prices.contexts = admission_contexts[0..2]; // no calibrated full-overflow projection
    try std.testing.expect(admission.quote(prices, &.{0.9}) == null);
}

test "lfm2 dspark: serving admission refuses missing and invalid prices without spending a paid trial" {
    var slots: [17]?f64 = @splat(11);
    slots[1] = 10;
    const target: lanes.table.Table = .{ .slots = &slots };
    const good = admissionPrices(&target);
    var recent: admission.Recent = .{};
    var prices = good;
    prices.target = null;
    try std.testing.expect(admission.quote(prices, &recent.rates) == null);
    try std.testing.expect(!recent.enroll(prices, &recent.rates));
    prices = good;
    prices.holds = &.{};
    try std.testing.expect(admission.quote(prices, &recent.rates) == null);
    try std.testing.expect(!recent.enroll(prices, &recent.rates));
    prices = good;
    prices.contexts = &.{};
    try std.testing.expect(admission.quote(prices, &recent.rates) == null);
    try std.testing.expect(!recent.enroll(prices, &recent.rates));
    for ([_]f64{ 0, -1, std.math.inf(f64), std.math.nan(f64) }) |bad| {
        slots[2] = bad;
        try std.testing.expect(admission.quote(good, &recent.rates) == null);
        try std.testing.expect(!recent.enroll(good, &recent.rates));
    }
    slots[2] = 11;
    var bad_holds = admission_holds;
    var bad_contexts = admission_contexts;
    for ([_]f64{ 0, -1, std.math.inf(f64), std.math.nan(f64) }) |bad| {
        prices = good;
        bad_holds[0].ms = bad;
        prices.holds = &bad_holds;
        try std.testing.expect(admission.quote(prices, &recent.rates) == null);
        try std.testing.expect(!recent.enroll(prices, &recent.rates));
        prices = good;
        bad_contexts[0].ms = bad;
        prices.contexts = &bad_contexts;
        try std.testing.expect(admission.quote(prices, &recent.rates) == null);
        try std.testing.expect(!recent.enroll(prices, &recent.rates));
    }
    try std.testing.expect(!recent.observed);
    try std.testing.expect(recent.enroll(good, &recent.rates));
}

test "lfm2 dspark: admission conditional history stops at the first rejection and separates sampled draws" {
    var history: admission.History = .{};
    const greedy = history.recent(false);
    const sampled = history.recent(true);
    const before = greedy.rates;
    greedy.observe(9, 1);
    try std.testing.expect(greedy.rates[0] > before[0]);
    try std.testing.expect(greedy.rates[1] < before[1]);
    try std.testing.expectEqualSlices(f64, before[2..], greedy.rates[2..]);
    try std.testing.expectEqualSlices(f64, &before, &sampled.rates);
    const valid = greedy.rates;
    greedy.observe(0, 0);
    greedy.observe(1, 2);
    try std.testing.expectEqualSlices(f64, &valid, &greedy.rates);
    for (0..96) |_| {
        greedy.observe(9, 9);
        sampled.observe(9, 0);
    }
    var slots: [17]?f64 = @splat(11);
    slots[1] = 10;
    const target: lanes.table.Table = .{ .slots = &slots };
    const prices = admissionPrices(&target);
    try std.testing.expect(admission.quote(prices, &greedy.rates).?.pays());
    try std.testing.expect(!admission.quote(prices, &sampled.rates).?.pays());
    try std.testing.expectEqualSlices(f64, before[1..], sampled.rates[1..]);
    // The finite prior and bounded effective history allow real later data to overturn rejection.
    for (0..128) |_| sampled.observe(9, 9);
    try std.testing.expect(admission.quote(prices, &sampled.rates).?.pays());
}

test "lfm2 dspark: admission zero-hold cold requests consume bounded probes without inventing failures" {
    var slots: [17]?f64 = @splat(11);
    slots[1] = 10;
    const target: lanes.table.Table = .{ .slots = &slots };
    const prices = admissionPrices(&target);
    var recent: admission.Recent = .{};
    const before = recent.rates;
    try std.testing.expect(recent.enroll(prices, &recent.rates));
    var previous: usize = 0;
    for (0..7) |_| {
        var skipped: usize = 0;
        while (skipped < 2048 and !recent.enroll(prices, &recent.rates)) : (skipped += 1) {}
        try std.testing.expect(skipped > 0 and skipped >= previous and skipped <= 1024);
        previous = skipped;
        try std.testing.expect(!recent.observed);
        try std.testing.expectEqualSlices(f64, &before, &recent.rates);
    }
}

test "lfm2 dspark: unpaid requests never spend recovery probes and later paying history can enroll" {
    var slots: [17]?f64 = @splat(11);
    slots[1] = 10;
    const target: lanes.table.Table = .{ .slots = &slots };
    const prices = admissionPrices(&target);
    var recent: admission.Recent = .{};
    try std.testing.expect(recent.enroll(prices, &recent.rates));
    for (0..96) |_| recent.observe(9, 0);
    try std.testing.expect(!recent.enroll(prices, &recent.rates));
    const perfect: [admission.max_depth]f64 = @splat(1);
    try std.testing.expect(admission.quote(prices, &perfect).?.pays());
    for (0..2048) |_| try std.testing.expect(!recent.enroll(prices, &recent.rates));
    for (0..128) |_| recent.observe(9, 9);
    // Paying real outcomes immediately remove the request backoff; no in-request probe state moved.
    try std.testing.expect(recent.enroll(prices, &recent.rates));
    try std.testing.expect(recent.enroll(prices, &recent.rates));
    slots[1] = 0.01; // even perfect drafts cannot repay the physical work: do not schedule blind probes
    for (0..2048) |_| try std.testing.expect(!recent.enroll(prices, &recent.rates));
    slots[1] = 10; // new prices can fund enrollment without waiting for a stale negative latch
    try std.testing.expect(recent.enroll(prices, &recent.rates));
}

test "lfm2 dspark: admission provider costs and head-only history seed fresh consumer state" {
    const gpa = std.testing.allocator;
    const dc = @import("dspark_config.zig");
    var d: ds.Drafter = .{ .gpa = gpa, .io = undefined, .target = undefined, .config = try dc.Config.parse(gpa, dc.stock), .w = undefined, .options = .{ .capacity = 4096 }, .s = undefined };
    @memcpy(d.holds[0..admission_holds.len], &admission_holds);
    d.holds_len = admission_holds.len;
    @memcpy(d.contexts[0..admission_contexts.len], &admission_contexts);
    d.contexts_len = admission_contexts.len;
    var slots: [17]?f64 = @splat(11);
    slots[1] = 10;
    const single: lanes.table.Table = .{ .slots = &slots };
    var shared_slots: [33]?f64 = @splat(1000);
    shared_slots[2] = 0.01;
    const shared: lanes.table.Table = .{ .slots = &shared_slots };
    var old = try lanes.Stream.init(gpa, .{ .id = "observed", .prompt = &.{1}, .max_new = 32 });
    defer old.deinit(gpa);
    var fresh = try lanes.Stream.init(gpa, .{ .id = "fresh", .prompt = &.{1}, .max_new = 32 });
    defer fresh.deinit(gpa);
    var sampled = try lanes.Stream.init(gpa, .{ .id = "sampled", .prompt = &.{1}, .max_new = 32, .sampling = .{ .seed = 1, .temperature = 1 } });
    defer sampled.deinit(gpa);
    const provider = d.drafter();
    const lone: lanes.drafter.Enrollment = .{ .streams = 1, .heads = 1 };
    try std.testing.expect(provider.enroll(&old, lone)); // direct diagnostics have no bound target prices
    d.useTargetCosts(&single, &shared);
    try std.testing.expect(provider.enroll(&old, lone));
    for (0..96) |_| provider.vtable.observe_head.?(provider.ptr, &old, 9, 0);
    try std.testing.expect(!provider.enroll(&fresh, lone));
    try std.testing.expectEqual(@as(usize, 0), fresh.head_prior.len);
    try std.testing.expect(provider.enroll(&sampled, lone)); // greedy misses did not poison sampled priors
    for (0..128) |_| provider.vtable.observe_head.?(provider.ptr, &old, 9, 9);
    try std.testing.expect(provider.enroll(&fresh, lone));
    try std.testing.expect(!provider.enroll(&old, .{ .streams = 2, .heads = 1 })); // bare shared pending-row baseline
    var rule: lanes.depth.Rule = .{ .gpa = gpa, .prior = &admission.prior, .most_drafts = 9, .family_costs = &single, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = true, .granted_depth = true, .batch_rows = 32 };
    defer rule.deinit();
    const who: lanes.depth.Who = .{ .state = &fresh.depth, .draft_room = fresh.draftRoom(), .finished = false, .forced = false, .head_prior = fresh.head_prior };
    const cloned = try rule.rates(who);
    try std.testing.expectEqualSlices(f64, fresh.head_prior, cloned);
    const seeded = cloned[0];
    provider.vtable.observe_head.?(provider.ptr, &old, 9, 0);
    try std.testing.expectEqual(seeded, cloned[0]); // a consumer owns its initial rates, not the provider's mutable array
    try std.testing.expect(fresh.head_prior[0] < cloned[0]);
    try std.testing.expectEqual(@as(u64, 0), fresh.depth.?.rounds);
    try std.testing.expectEqual(@as(usize, 0), fresh.depth.?.tried);
    try std.testing.expect(fresh.depth.?.plain == null and fresh.depth.?.wait == null);
    try std.testing.expectEqual(@as(u32, 0), fresh.hold_context_rows);
}

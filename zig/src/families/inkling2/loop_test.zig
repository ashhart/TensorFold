//! Mini (inkling2_loop) on the shared loop config: the visit plan, its cache slots and the drafter taps by visit.
const std = @import("std");
const config = @import("config.zig");
const visits = @import("visits.zig");

const t = std.testing;

fn load() !config.Config {
    var why: config.Why = .{};
    return config.parse(t.allocator, @import("config_test.zig").mini, &why);
}

test "the visit plan: prelude, three passes with their own slots, coda; messages and restarts on passes 2 and 3" {
    const c = try load();
    const p = visits.build(&c.text);
    try t.expectEqual(@as(u32, 48), p.n);
    try t.expectEqual(@as(u32, 48), p.slots());
    var seen: [48]bool = @splat(false);
    for (p.visits[0..p.n], 0..) |v, k| {
        try t.expect(!seen[v.slot]);
        seen[v.slot] = true;
        const want_layer: u32 = if (k < 6) @intCast(k) else if (k < 42) @intCast(6 + (k - 6) % 12) else @intCast(k - 24);
        const want_pass: u32 = if (k < 6 or k >= 42) 0 else @intCast((k - 6) / 12);
        try t.expectEqual(want_layer, v.layer);
        try t.expectEqual(want_pass, v.pass);
        try t.expectEqual(want_pass > 0, v.messages);
        try t.expectEqual(k == 18 or k == 30, v.restart);
        try t.expectEqual(k == 17 or k == 29, v.saves_bank);
        try t.expectEqual(v.layer >= 6 and v.layer < 18, v.core != null);
    }
    // pass 1 of layer 6 is slot 24, pass 2 of layer 17 is slot 47, the coda keeps its own layer index
    try t.expectEqual(@as(u32, 24), p.at(18).slot);
    try t.expectEqual(@as(u32, 47), p.at(41).slot);
    try t.expectEqual(@as(u32, 18), p.at(42).slot);
    try t.expectEqual(@as(u32, 5), p.at(23).core.?);
    // one block before the core; the bank is layer 12's input and the pass's output: two sources
    try t.expectEqual(@as(u32, 1), p.prelude_blocks);
    try t.expectEqualSlices(u32, &.{12}, p.bank_blocks[0..p.bank_n]);
    try t.expectEqual(@as(u32, 2), p.bankSources());
}

test "drafter taps by visit: 5, 23 and 41 read those visits' inputs (layers 5, 11, 17), 48 the final norm" {
    const c = try load();
    const p = visits.build(&c.text);
    const m = c.mtp.?;
    const want = [_]?u32{ 5, 11, 17, null };
    for (m.taps[0..m.taps_n], want) |tap, layer| {
        const v = try p.tapVisit(tap);
        if (layer) |l| try t.expectEqual(l, p.at(v.?).layer) else try t.expectEqual(@as(?u32, null), v);
    }
    try t.expectEqual(@as(u32, 1), p.at(23).pass);
    try t.expectEqual(@as(u32, 2), p.at(41).pass);
    try t.expectError(error.NoSuchVisit, p.tapVisit(49));
}

test "Small's visit plan is its layers in order, each its own slot, no messages" {
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, @import("config_test.zig").real, &why);
    const p = visits.build(&c.text);
    try t.expectEqual(@as(u32, 48), p.n);
    for (p.visits[0..p.n], 0..) |v, k| {
        try t.expectEqual(@as(u32, @intCast(k)), v.layer);
        try t.expectEqual(v.layer, v.slot);
        try t.expect(!v.messages and !v.restart and !v.saves_bank and v.core == null);
    }
    try t.expectEqual(@as(u32, 0), p.bank_n);
}

test "the cross-pass query rows: q and MLP in queries, k and v in kv_queries, two rows a core layer" {
    try t.expectEqual(visits.QueryRow{ .kv = false, .row = 0 }, visits.queryRow(0, .q));
    try t.expectEqual(visits.QueryRow{ .kv = true, .row = 0 }, visits.queryRow(0, .k));
    try t.expectEqual(visits.QueryRow{ .kv = true, .row = 23 }, visits.queryRow(11, .v));
    try t.expectEqual(visits.QueryRow{ .kv = false, .row = 23 }, visits.queryRow(11, .mlp));
}

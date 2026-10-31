//! Actual family facts flow through the serving engine's info without allocating any device objects.
const tf = @import("tensorfold");
const check = @import("draft_sources_check.zig").check;
const present = @import("draft_sources_check.zig").present;
const std = @import("std");

test "draft sources: Metal Nemotron follows the loaded MTP head" {
    var b: tf.nemotron.backend.Metal = undefined;
    b.costs = .{};
    b.o = .{ .capacity = 32, .streams = 2, .batch_rows = 16 };
    b.head = null;
    try check(b.facts(), null, false, false);
    b.head = present(@TypeOf(b.head.?));
    try check(b.facts(), null, true, true);
}

test "draft sources: Metal Qwen2B and LFM2 MoE follow qualified verification width" {
    var q: tf.qwen35.backend.Metal = undefined;
    var l: tf.lfm2_moe.backend.Metal = undefined;
    q.options.streams = 2;
    l.options.streams = 2;
    l.timed = 0;
    l.shared_timed = 0;
    for ([_]u32{ 1, 16 }) |width| {
        q.width = width;
        l.width = width;
        try check(q.facts(), null, width > 1, false);
        try check(l.facts(), null, width > 1, false);
    }
}

test "draft sources: Metal Qwen27 copies and DFlash2 use actual slot and head facts" {
    const q = @import("qwen27_lanes.zig");
    var runner: tf.qwen27.decode_round.Runner = undefined;
    var b: q.Metal = undefined;
    b.runner = &runner;
    b.draft = null;
    b.timed = 0;
    b.window = 16;
    b.copies = false;
    b.nodes = 15;
    b.draft_slots = 1;
    b.tensor_units = false;
    runner.slots = 1;
    try check(b.facts(), null, false, false);
    runner.slots = 4;
    try check(b.facts(), null, true, false);
    b.draft = present(@TypeOf(b.draft.?));
    try check(b.facts(), .{ .kind = "DFlash2", .name = "example/draft" }, true, false);
    runner.slots = 1;
    try check(b.facts(), .{ .kind = "DFlash2", .name = "example/draft" }, false, false);
}

test "draft sources: Metal Gemma and Spark have copies without a checkpoint head" {
    var g: tf.gemma4.backend.Metal = undefined;
    g.options.streams = 2;
    g.timed = 0;
    g.shared_timed = 0;
    try check(g.facts(), null, true, false);
    var s: tf.spark25_metal.backend.Metal = undefined;
    s.options.streams = 2;
    try check(s.facts(), null, true, false);
}

test "draft sources: Metal Step5 follows the checkpoint MTP layer count" {
    var m: tf.step5.Model = undefined;
    var b: tf.step5.backend.Metal = undefined;
    b.model = &m;
    b.options.streams = 2;
    for ([_]u32{ 0, 1 }) |n| {
        m.config.mtp = n;
        try check(b.facts(), null, true, n > 0);
    }
}

test "draft sources: Metal GLM and GLM DSA follow actual MTP weights" {
    var e: tf.glm.engine.Engine = undefined;
    var weights: @typeInfo(@TypeOf(e.w)).pointer.child = undefined;
    e.w = &weights;
    e.ep = null;
    var slots: tf.glm.slots.Slots = undefined;
    var b: tf.glm.backend.Backend = .{ .gpa = std.testing.allocator, .sl = &slots };
    slots.e = &e;
    slots.slots = &.{};
    e.w.mtp = null;
    try check(b.facts(), null, true, false);
    e.w.mtp = @as(@TypeOf(e.w.mtp.?), undefined);
    try check(b.facts(), null, true, true);
    var d: tf.glm_dsa.engine.Engine = undefined;
    var db: tf.glm_dsa.backend.Backend = undefined;
    db.e = &d;
    db.timed = false;
    db.head_ms = 0;
    d.m = undefined;
    d.slots = &.{};
    d.m.mtp = null;
    try check(db.facts(), null, true, false);
    d.m.mtp = @as(@TypeOf(d.m.mtp.?), undefined);
    try check(db.facts(), null, true, true);
}

test "draft sources: Metal Gemma loaders identify DFlash and the MTP assistant" {
    var d: tf.gemma4.dflash.Drafter = undefined;
    var a: tf.gemma4.assistant.Drafter = undefined;
    try std.testing.expectEqualStrings("DFlash", d.drafter().kind);
    try std.testing.expectEqualStrings("MTP assistant", a.drafter().kind);
}

test "draft sources: Metal Flash Next replay and shared rounds name the checkpoint MTP and copies" {
    const legacy = @import("flashnext_host.zig").draft_sources;
    try std.testing.expect(legacy.mtp and legacy.copies and legacy.drafter == null);
    var b: tf.flashnext_backend.Backend = undefined;
    b.costs = null;
    b.sessions = &.{};
    b.serial_drafts = false;
    try check(b.facts(), null, true, true);
}

test "qwen27 window: native facts expose device width, node budget and hybrid copy policy" {
    const q = @import("qwen27_lanes.zig");
    const policy = @import("qwen27_window.zig");
    var runner: tf.qwen27.decode_round.Runner = undefined;
    var b: q.Metal = undefined;
    b.runner = &runner;
    b.draft = present(@TypeOf(b.draft.?));
    b.timed = 0;
    b.draft_slots = 1;
    b.tensor_units = false;
    runner.slots = 1;
    for ([_]bool{ false, true }) |units| for ([_]bool{ false, true }) |copies| {
        b.tensor_units = units;
        for ([_][]const u8{ "16", "32", "64" }, [_]u32{ 16, 32, 64 }) |text, width| {
            b.window = try policy.requested(units, text);
            b.nodes = policy.nodes(b.window, copies);
            b.copies = copies;
            b.timed = width;
            for (b.costs[0..width], 0..) |*cost, i| cost.* = .{ .width = @intCast(i + 1), .ms = 20 };
            const facts = b.facts();
            try std.testing.expectEqual(width, facts.exact_width);
            try std.testing.expect(facts.allocate_shared_drafts);
            try std.testing.expect(facts.mtp and facts.speculate);
            try std.testing.expectEqual(width - 1, facts.drafts);
            try std.testing.expectEqual(width, facts.batch_rows);
            try std.testing.expectEqual(width, facts.window_costs.len);
            try std.testing.expectEqual(width, facts.window_costs[width - 1].width);
            try std.testing.expectEqual(copies, facts.lone_copies);
            try std.testing.expectEqual(copies, facts.context_copies);
        }
    };
}

test "qwen27 window: measured shared row cliffs retain the lone window's exact prices" {
    const native = @import("qwen27_lanes.zig");
    const lanes = @import("lanes");
    var runner: tf.qwen27.decode_round.Runner = undefined;
    runner.slots = 4;
    var b: native.Metal = undefined;
    b.runner = &runner;
    b.draft = present(@TypeOf(b.draft.?));
    b.window = 16;
    b.nodes = 15;
    b.copies = false;
    b.tensor_units = true;
    b.draft_slots = 8;
    b.timed = 64;
    for (&b.costs, 1..) |*cost, rows| cost.* = .{
        .width = @intCast(rows),
        .ms = if (rows <= 16) 20 else if (rows <= 32) 28 else 44,
    };
    const facts = b.facts();
    try std.testing.expectEqual(@as(usize, 16), facts.window_costs.len);
    try std.testing.expectEqual(@as(usize, 48), facts.shared_costs.len);
    try std.testing.expectEqual(@as(u32, 17), facts.shared_costs[0].width);
    var cfg = try lanes.Config.init(std.testing.allocator, facts, 16, 15);
    defer cfg.deinit(std.testing.allocator);
    for (1..17) |rows| {
        try std.testing.expectEqual(@as(?f64, 20), cfg.family_costs.get(@intCast(rows)));
        try std.testing.expectEqual(@as(?f64, 20), cfg.shared_costs.get(@intCast(rows)));
    }
    try std.testing.expectEqual(@as(?f64, 28), cfg.shared_costs.get(17));
    try std.testing.expectEqual(@as(?f64, 28), cfg.shared_costs.get(32));
    try std.testing.expectEqual(@as(?f64, 44), cfg.shared_costs.get(33));
    try std.testing.expectEqual(@as(u32, 16), facts.exact_width);
    try std.testing.expect(facts.mtp and facts.draft_streams);
}

test "qwen27 window: a copied continuation preserves the matching native tree prefix" {
    const native = @import("qwen27_lanes.zig");
    var copy_stream = try @import("lanes").Stream.init(std.testing.allocator, .{ .id = "copy-prefix", .prompt = &.{1}, .max_new = 32 });
    defer copy_stream.deinit(std.testing.allocator);
    var copy_backend: native.Metal = undefined;
    copy_backend.trees = @splat(null);
    copy_backend.trees[0] = .{ .gpa = std.testing.allocator, .tokens = @constCast(&[_]u32{ 10, 11, 12 }), .parents = @constCast(&[_]i32{ -1, 0, 1 }), .scores = @constCast(&[_]f64{ -0.1, -0.2, -0.3 }) };
    var up = [_]i32{ -1, 0, 1, 2 };
    var window: @import("lanes").backend.Window = .{ .stream = &copy_stream, .pending = 1, .held = 0, .tokens = &.{ 10, 11, 99 }, .parents = &up, .positions = &.{ 0, 1, 2, 3 } };
    try std.testing.expectEqual(@as(usize, 2), copy_backend.treeRows(window));
    up[2] = 0;
    try std.testing.expectEqual(@as(usize, 1), copy_backend.treeRows(window));
    window.tokens = &.{ 99, 11, 12 };
    try std.testing.expectEqual(@as(usize, 0), copy_backend.treeRows(window));
}

test "qwen27 window: an early keep refuses prompts, unrelated rounds and absent verify tails" {
    var early_runner: tf.qwen27.decode_round.Runner = undefined;
    var early_round: tf.qwen27.round_plan.Round = undefined;
    var other_round: tf.qwen27.round_plan.Round = undefined;
    early_runner.failed = false;
    early_runner.accepted_chain = false;
    early_runner.current_round = null;
    early_runner.tail = null;
    early_runner.active = null;
    try std.testing.expectError(error.RoundNotReady, early_runner.checkDeviceKeep(&early_round));
    try std.testing.expectError(error.RoundNotReady, early_runner.prepareDeviceKeep(&early_round, undefined, undefined, undefined));
    early_runner.current_round = &other_round;
    try std.testing.expectError(error.RoundNotReady, early_runner.prepareDeviceKeep(&early_round, undefined, undefined, undefined));
    early_runner.current_round = &early_round;
    try std.testing.expectError(error.RoundNotReady, early_runner.prepareDeviceKeep(&early_round, undefined, undefined, undefined));
    early_runner.accepted_chain = true;
    try std.testing.expectError(error.RoundNotReady, early_runner.prepareDeviceKeep(&early_round, undefined, undefined, undefined));
}

test "qwen27 window: tensor-unit admission publishes the first token before preparing drafts" {
    const native = @import("qwen27_lanes.zig");
    var runner: tf.qwen27.decode_round.Runner = undefined;
    var backend: native.Metal = undefined;
    backend.runner = &runner;
    for ([_]u32{ 1, 16 }) |slots| {
        runner.slots = slots;
        for ([_]bool{ false, true }) |units| {
            backend.tensor_units = units;
            try std.testing.expectEqual(units, backend.backend().first_before_draft);
        }
    }
}

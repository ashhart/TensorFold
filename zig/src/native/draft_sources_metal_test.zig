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
    b.nodes = 15;
    b.draft_slots = 1;
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

//! windows.allocate's shared-round commit of a stream's cached draft chances and their cut marker, per window kind.
const std = @import("std");
const Config = @import("config.zig").Config;
const Model = @import("config.zig").Model;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const be = @import("backend.zig");
const fake = @import("fake.zig");
const win = @import("windows.zig");

const a = std.testing.allocator;

/// The chain chances a three-draft head reports, the two its confidence prefix keeps, and a stream's earlier prices.
const chain = [_]f64{ 0.9, 0.8, 0.7 };
const prefix = [_]f64{ 0.9, 0.8 };
const earlier = [_]f64{ 0.25, 0.2 };

const Hook = *const fn (ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool;

/// A three-draft head's chances, marking the third as the first one a cut left unmeasured.
fn chainChances(_: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
    if (out.len != chain.len) return false;
    @memcpy(out, &chain);
    s.draft_chances_cut = 2;
    return true;
}

/// Five cumulative chances whose mark sits past the confidence prefix, so the commit has to clamp it.
fn deepChances(_: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
    const all = [_]f64{ 0.9, 0.8, 0.4, 0.3, 0.2 };
    if (out.len != all.len) return false;
    @memcpy(out, &all);
    s.draft_chances_cut = 5;
    return true;
}

/// A head that marks its cut and then fails, as a partially read logits buffer does.
fn markThenFail(_: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
    _ = out;
    s.draft_chances_cut = 3;
    return error.NoDraftProbabilities;
}

/// A head with no chances to give, which a confidence floor turns into a refusal.
fn noChances(_: *anyopaque, _: *Stream, _: []f64) anyerror!bool {
    return false;
}

/// One engine on the fake target with this round's chances hook, and the shared commit enabled.
const Round = struct {
    cfg: Config = undefined,
    target: fake.Fake = undefined,
    clock: fake.FixedClock = .{},
    vtable: be.Backend.VTable = undefined,
    engine: Engine = undefined,

    fn start(r: *Round, m: Model, max_draft: u32, hook: ?Hook) !void {
        r.cfg = try Config.init(a, m, 32, max_draft);
        r.target = .{ .gpa = a };
        const plain = r.target.backend();
        r.vtable = plain.vtable.*;
        r.vtable.probabilities = hook;
        r.engine = Engine.init(a, &r.cfg, .{ .ptr = plain.ptr, .vtable = &r.vtable, .chunk_invariant = plain.chunk_invariant }, r.clock.clock());
    }

    fn finish(r: *Round) void {
        r.engine.deinit();
        r.target.deinit();
        r.cfg.deinit(a);
    }
};

/// The shared-round model: the head prices its drafts and their prices are committed per stream.
const shared_model = Model{ .exact_width = 32, .drafts = 5, .batch_rows = 32, .allocate_shared_drafts = true, .draft_probabilities = true };

fn headStream(s: *Stream) !void {
    try s.draft_chances.appendSlice(a, &earlier);
    s.draft_chances_cut = 1;
}

test "cached draft cuts: a head round commits the head's chances and its marked cut" {
    var r: Round = .{};
    try r.start(shared_model, 5, chainChances);
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "head", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
    defer s.deinit(a);
    try headStream(&s);
    const tokens = [_]u32{ 11, 12, 13 };
    var plans = [_]win.Plan{.{ .stream = &s, .position = 3, .kind = .head, .tokens = &tokens }};
    try win.allocate(&r.engine, &plans);
    try std.testing.expectEqualSlices(f64, &chain, s.draft_chances.items);
    try std.testing.expectEqual(@as(?usize, 2), s.draft_chances_cut);
}

test "cached draft cuts: a confidence prefix commits its rows and clamps the marker to them" {
    var r: Round = .{};
    var m = shared_model;
    m.draft_confidence = 0.5; // keeps 0.9 and 0.8, so the head's mark of 5 is past what was committed
    try r.start(m, 5, deepChances);
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "floor", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
    defer s.deinit(a);
    try headStream(&s);
    const tokens = [_]u32{ 11, 12, 13, 14, 15 };
    var plans = [_]win.Plan{.{ .stream = &s, .position = 5, .kind = .head, .tokens = &tokens }};
    try win.allocate(&r.engine, &plans);
    try std.testing.expectEqualSlices(f64, &prefix, s.draft_chances.items);
    try std.testing.expectEqual(@as(?usize, 2), s.draft_chances_cut);
}

test "cached draft cuts: a forced round keeps the previous chances and marker" {
    var r: Round = .{};
    try r.start(shared_model, 5, chainChances);
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "forced", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
    defer s.deinit(a);
    try headStream(&s);
    const tokens = [_]u32{ 11, 12 };
    var plans = [_]win.Plan{.{ .stream = &s, .position = 2, .kind = .forced, .tokens = &tokens, .forced = &tokens }};
    try win.allocate(&r.engine, &plans);
    try std.testing.expectEqualSlices(f64, &earlier, s.draft_chances.items);
    try std.testing.expectEqual(@as(?usize, 1), s.draft_chances_cut);
}

test "cached draft cuts: an empty round keeps the previous chances and marker" {
    var r: Round = .{};
    try r.start(shared_model, 5, chainChances);
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "empty", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
    defer s.deinit(a);
    try headStream(&s);
    var plans = [_]win.Plan{.{ .stream = &s, .position = 1, .kind = .none }};
    try win.allocate(&r.engine, &plans);
    try std.testing.expectEqualSlices(f64, &earlier, s.draft_chances.items);
    try std.testing.expectEqual(@as(?usize, 1), s.draft_chances_cut);
}

test "cached draft cuts: a grafted round is no head, so it keeps the previous chances and marker" {
    var r: Round = .{};
    try r.start(shared_model, 5, chainChances);
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "fill", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
    defer s.deinit(a);
    try headStream(&s);
    const tokens = [_]u32{ 11, 12, 13 };
    var plans = [_]win.Plan{.{ .stream = &s, .position = 3, .kind = .fill, .tokens = &tokens }};
    try win.allocate(&r.engine, &plans);
    try std.testing.expectEqualSlices(f64, &earlier, s.draft_chances.items);
    try std.testing.expectEqual(@as(?usize, 1), s.draft_chances_cut);
}

test "cached draft cuts: a head with no priced rows keeps the previous chances and marker" {
    var r: Round = .{};
    var m = shared_model;
    m.draft_confidence = 0.5;
    try r.start(m, 0, chainChances); // most_drafts 0: the confidence prefix keeps nothing for this head
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "unpriced", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
    defer s.deinit(a);
    try headStream(&s);
    const tokens = [_]u32{ 11, 12, 13 };
    var plans = [_]win.Plan{.{ .stream = &s, .position = 3, .kind = .head, .tokens = &tokens }};
    try win.allocate(&r.engine, &plans);
    try std.testing.expectEqualSlices(f64, &earlier, s.draft_chances.items);
    try std.testing.expectEqual(@as(?usize, 1), s.draft_chances_cut);
}

test "cached draft cuts: a copy round clears both the chances and the marker" {
    var r: Round = .{};
    try r.start(shared_model, 5, chainChances);
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "copy", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
    defer s.deinit(a);
    try headStream(&s);
    const tokens = [_]u32{ 11, 12 };
    var plans = [_]win.Plan{.{ .stream = &s, .position = 2, .kind = .copy, .tokens = &tokens }};
    try win.allocate(&r.engine, &plans);
    try std.testing.expectEqual(@as(usize, 0), s.draft_chances.items.len);
    try std.testing.expectEqual(@as(?usize, null), s.draft_chances_cut);
}

test "cached draft cuts: a context copy clears both the chances and the marker" {
    var r: Round = .{};
    try r.start(shared_model, 5, chainChances);
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "context", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
    defer s.deinit(a);
    try headStream(&s);
    const tokens = [_]u32{ 11, 12, 13 };
    var plans = [_]win.Plan{.{ .stream = &s, .position = 3, .kind = .head, .tokens = &tokens, .context_copy = true }};
    try win.allocate(&r.engine, &plans);
    try std.testing.expectEqual(@as(usize, 0), s.draft_chances.items.len);
    try std.testing.expectEqual(@as(?usize, null), s.draft_chances_cut);
}

test "cached draft cuts: a failing chances hook restores the previous marker" {
    var r: Round = .{};
    try r.start(shared_model, 5, markThenFail);
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "failed", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
    defer s.deinit(a);
    try headStream(&s);
    const tokens = [_]u32{ 11, 12, 13 };
    var plans = [_]win.Plan{.{ .stream = &s, .position = 3, .kind = .head, .tokens = &tokens }};
    try std.testing.expectError(error.NoDraftProbabilities, win.allocate(&r.engine, &plans));
    try std.testing.expectEqualSlices(f64, &earlier, s.draft_chances.items);
    try std.testing.expectEqual(@as(?usize, 1), s.draft_chances_cut);
}

test "cached draft cuts: a head without chances under a floor restores the previous marker" {
    var r: Round = .{};
    var m = shared_model;
    m.draft_confidence = 0.5;
    try r.start(m, 5, noChances);
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "refused", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
    defer s.deinit(a);
    try headStream(&s);
    const tokens = [_]u32{ 11, 12, 13 };
    var plans = [_]win.Plan{.{ .stream = &s, .position = 3, .kind = .head, .tokens = &tokens }};
    try std.testing.expectError(error.NoDraftProbabilities, win.allocate(&r.engine, &plans));
    try std.testing.expectEqualSlices(f64, &earlier, s.draft_chances.items);
    try std.testing.expectEqual(@as(?usize, 1), s.draft_chances_cut);
}

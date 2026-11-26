//! Retained target features through the production Qwen27 Metal wrapper, with host bytes standing in for device rows.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const be = lanes.backend;
const Native = @import("src/native/qwen27_lanes.zig").Metal;
const features = @import("src/native/qwen27_features.zig");
const retained = @import("src/native/sensenova_features.zig");

const row_bytes = 4;
const capacity = 16;
const ids = [_]u32{7};

const Fixture = struct {
    record: mtl.Record,
    model: q.model.Model = undefined,
    runner: q.decode_round.Runner = undefined,
    back: Native = undefined,
    a: lanes.Stream = undefined,
    b: lanes.Stream = undefined,

    fn init(f: *Fixture, external: bool) !void {
        f.record = .{ .gpa = std.testing.allocator };
        f.model.device = f.record.device();
        f.runner.model = &f.model;
        f.runner.slots = 2;
        f.runner.capacity = 1024;
        f.runner.capture_taps = false;
        f.runner.tapped = @splat(.{});
        f.runner.taps = .{
            .buffer = try f.record.device().buffer(capacity * row_bytes, 0),
            .capacity = capacity,
            .width = row_bytes / 2,
            .seen = 1,
            .ids = @splat(7),
            .count = 1,
            .planes = 1,
        };
        f.back.runner = &f.runner;
        f.back.draft = null;
        f.back.external = false;
        f.back.owners = @splat(null);
        f.back.held_taps = null;
        f.back.features_state = .{};
        f.back.round = null;
        f.back.advanced = false;
        if (external) try features.prepare(&f.back, &ids);
        f.back.owners[0] = &f.a;
        f.back.owners[1] = &f.b;
    }

    fn deinit(f: *Fixture) void {
        if (f.back.held_taps) |buffer| buffer.deinit();
        f.runner.taps.buffer.deinit();
        f.record.deinit();
    }

    fn stream(f: *Fixture, slot: u32) *lanes.Stream {
        return f.back.owners[slot].?;
    }

    /// Plant one pass in the shared tap buffer, just as the completed Metal pass leaves it.
    fn stage(f: *Fixture, slot: u32, first: u32, start: u32, staged: []const u8) void {
        std.debug.assert(staged.len % row_bytes == 0);
        const rows: u32 = @intCast(staged.len / row_bytes);
        const at = @as(usize, first) * row_bytes;
        @memcpy(f.runner.taps.buffer.contents()[at..][0..staged.len], staged);
        f.runner.tapped[slot] = .{ .start = start, .first = first, .rows = rows };
        f.runner.taps.rows = @intCast(at / row_bytes + rows);
    }

    fn chunk(f: *Fixture, slot: u32, start: u32, staged: []const u8, last: bool) !void {
        f.stage(slot, 0, start, staged);
        try retained.recordChunk(&f.back, f.stream(slot), slot, start, staged.len / row_bytes, last);
    }

    fn window(f: *Fixture, slot: u32, tokens: []const u32) be.Window {
        return .{
            .stream = f.stream(slot),
            .pending = 0,
            .held = 0,
            .tokens = tokens,
            .parents = null,
            .positions = &.{},
        };
    }

    fn verify(f: *Fixture, slot: u32, start: u32, staged: []const u8, tokens: []const u32) !void {
        var w = [_]be.Window{f.window(slot, tokens[1..])};
        var input = [_]q.round_plan.Input{.{
            .slot = slot,
            .start = start,
            .capacity = f.runner.capacity,
            .ids = tokens,
        }};
        var round = try q.round_plan.Round.init(std.testing.allocator, &input, 4, 2, 1000);
        defer round.deinit();
        f.back.round = round;
        defer f.back.round = null;
        f.stage(slot, round.firsts[0], start, staged);
        try retained.recordVerify(&f.back, &w, &input);
    }

    fn keep(f: *Fixture, slot: u32, tokens: []const u32, path: []const u32) void {
        var w = [_]be.Window{f.window(slot, tokens)};
        retained.recordKeep(&f.back, &w, &.{path});
    }
};

fn materialized(f: be.Features) []u8 {
    const buffer: mtl.Buffer = .{ .id = @ptrFromInt(f.buffer) };
    const at: usize = @intCast(f.offset);
    return buffer.contents()[at..][0 .. @as(usize, f.rows) * f.row_bytes];
}

/// The issue's materialize gate: retained bytes must equal the completed pass's staged bytes.
fn expectMaterialized(f: be.Features, staged: []const u8) !void {
    if (f.space != .device or f.dtype != .bf16) return error.BadMaterializedLayout;
    if (!std.mem.eql(u8, materialized(f), staged)) return error.MaterializedMismatch;
}

fn expectRetainedBuffer(f: *Fixture, got: be.Features) !void {
    try std.testing.expect(got.buffer != @intFromPtr(f.runner.taps.buffer.id));
    try std.testing.expectEqual(@intFromPtr(f.back.held_taps.?.id), got.buffer);
}

const prompt_a = [_]u8{ 0x10, 1, 2, 3, 0x20, 4, 5, 6, 0x30, 7, 8, 9 };
const prompt_b = [_]u8{ 0x40, 10, 11, 12, 0x50, 13, 14, 15, 0x60, 16, 17, 18 };
const verify_a = [_]u8{ 0x71, 19, 20, 21, 0x72, 22, 23, 24, 0x73, 25, 26, 27, 0x74, 28, 29, 30 };
const verify_b = [_]u8{ 0x81, 31, 32, 33, 0x82, 34, 35, 36, 0x83, 37, 38, 39 };

test "qwen27 retained taps: prompt rows survive A/B reuse and materialization has a planted control" {
    var f: Fixture = undefined;
    try f.init(true);
    defer f.deinit();

    try f.chunk(0, 10, &prompt_a, true);
    try f.chunk(1, 40, &prompt_b, true); // overwrites every row that the old prompt alias returned

    const from_a = try features.rows(&f.back, f.stream(0), &ids, 12, 1);
    try expectMaterialized(from_a, prompt_a[8..12]);
    try expectRetainedBuffer(&f, from_a);
    const from_b = try features.rows(&f.back, f.stream(1), &ids, 42, 1);
    try expectMaterialized(from_b, prompt_b[8..12]);
    try expectRetainedBuffer(&f, from_b);

    // Planted corruption must make materialize differ from the deterministic staged reference at the same gate.
    const byte = &materialized(from_a)[0];
    const saved = byte.*;
    byte.* ^= 0xff;
    try std.testing.expectError(error.MaterializedMismatch, expectMaterialized(from_a, prompt_a[8..12]));
    byte.* = saved;
    try expectMaterialized(from_a, prompt_a[8..12]);
}

test "qwen27 retained taps: selected non-prefix verify rows survive an interleaved pass" {
    var f: Fixture = undefined;
    try f.init(true);
    defer f.deinit();
    const tokens_a = [_]u32{ 1, 2, 3, 4 };
    const tokens_b = [_]u32{ 5, 6, 7 };

    try f.verify(0, 30, &verify_a, &tokens_a);
    f.keep(0, tokens_a[1..], &.{ 0, 2, 3 });
    try f.verify(1, 50, &verify_b, &tokens_b);

    const all = try features.rows(&f.back, f.stream(0), &ids, 30, 4);
    try expectMaterialized(all, &verify_a);
    try expectRetainedBuffer(&f, all);
    const partial = try features.rows(&f.back, f.stream(0), &ids, 31, 2);
    try expectMaterialized(partial, verify_a[4..12]);

    // The MTP reader's selected rows address the packed retained range, including its non-prefix tail.
    const bytes = materialized(all);
    var gathered: [2 * row_bytes]u8 = undefined;
    for ([_]u32{ 2, 3 }, 0..) |row, i| {
        @memcpy(gathered[i * row_bytes ..][0..row_bytes], bytes[row * row_bytes ..][0..row_bytes]);
    }
    try std.testing.expectEqualSlices(u8, verify_a[8..16], &gathered);
}

test "qwen27 retained taps: keep limits rows and unavailable ranges never alias the latest pass" {
    var f: Fixture = undefined;
    try f.init(true);
    defer f.deinit();
    const tokens_a = [_]u32{ 1, 2, 3, 4 };

    try f.verify(0, 30, &verify_a, &tokens_a);
    f.keep(0, tokens_a[1..], &.{ 0, 2 });
    try f.chunk(1, 50, &prompt_b, true);

    const held = try features.rows(&f.back, f.stream(0), &ids, 30, 3);
    try expectMaterialized(held, verify_a[0..12]);
    try std.testing.expectError(error.RowsNotHeld, features.rows(&f.back, f.stream(0), &ids, 29, 1));
    try std.testing.expectError(error.RowsNotHeld, features.rows(&f.back, f.stream(0), &ids, 33, 1));
    try std.testing.expectError(error.RowsNotHeld, features.rows(&f.back, f.stream(0), &ids, 30, 4));
}

test "qwen27 retained taps: built-in drafting adds no duplicate retained-tap allocation" {
    var f: Fixture = undefined;
    try f.init(false);
    defer f.deinit();
    const tokens_a = [_]u32{ 1, 2, 3, 4 };
    const before = f.record.allocations.items.len;

    try f.chunk(0, 10, &prompt_a, true);
    var w = [_]be.Window{f.window(0, tokens_a[1..])};
    var input = [_]q.round_plan.Input{.{ .slot = 0, .start = 30, .capacity = 1024, .ids = &tokens_a }};
    try retained.recordVerify(&f.back, &w, &input);

    try std.testing.expect(!f.back.features_state.enabled);
    try std.testing.expect(f.back.held_taps == null);
    try std.testing.expectEqual(before, f.record.allocations.items.len);
}

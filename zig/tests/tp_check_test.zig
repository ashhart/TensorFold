//! Host tests for the mixed-TP gate in tp_check.zig: dump format, rule 1 reference, rule 2 repeat, rule 4 agreement.
const std = @import("std");
const reference = @import("tp").reference;
const check = @import("tp_check.zig");
const Call = check.Call;
const Mode = check.Mode;
const Header = check.Header;
const writeHeader = check.writeHeader;
const writeCollective = check.writeCollective;
const writeLogits = check.writeLogits;
const TokenHash = check.TokenHash;
const parse = check.parse;
const checkReference = check.checkReference;
const checkRepeat = check.checkRepeat;
const repeatPassed = check.repeatPassed;
const Split = check.Split;
const chosenHash = check.chosenHash;
const checkAgree = check.checkAgree;

const testing = std.testing;

fn bf16s(comptime vals: []const f32) [vals.len * 2]u8 {
    var out: [vals.len * 2]u8 = undefined;
    for (vals, 0..) |v, i| std.mem.writeInt(u16, out[2 * i ..][0..2], reference.f32ToBf16(v), .little);
    return out;
}

/// Two ranks' dumps of one bf16 all-reduce, one all-gather and one exchange, with correct results.
fn twoRanks(a: std.mem.Allocator, bad_recv: bool) ![2][]u8 {
    const p0 = comptime bf16s(&.{ 1.0, -2.5, 0.125 });
    const p1 = comptime bf16s(&.{ 0.00390625, 3.0, 0.125 });
    var sum: [6]u8 = undefined;
    reference.reduce(&sum, &.{ &p0, &p1 }, .bf16, .sum);
    var out: [2][]u8 = undefined;
    for (0..2) |r| {
        var w: std.Io.Writer.Allocating = .init(a);
        defer w.deinit();
        const mine: []const u8 = if (r == 0) &p0 else &p1;
        const other: []const u8 = if (r == 0) &p1 else &p0;
        try writeHeader(&w.writer, .{ .rank = @intCast(r), .world = 2, .label = if (r == 0) "metal test" else "cuda test", .provenance = "commit=abc model=test" });
        var recv = sum;
        if (bad_recv and r == 1) recv[2] ^= 1;
        try writeCollective(&w.writer, .{ .seq = 0, .layer = 3, .call = .all_reduce, .dtype = .bf16, .op = .sum, .peer = 0, .send = mine, .recv = &recv });
        const gathered = p0 ++ p1;
        try writeCollective(&w.writer, .{ .seq = 1, .layer = 3, .call = .all_gather, .dtype = .bf16, .op = .sum, .peer = 0, .send = mine, .recv = &gathered });
        try writeCollective(&w.writer, .{ .seq = 2, .layer = 4, .call = .exchange, .dtype = .bf16, .op = .sum, .peer = @intCast(1 - r), .send = mine, .recv = other });
        try writeLogits(&w.writer, 0, 0, 2, &.{ 0.5, 1.0, 4.0 });
        out[r] = try w.toOwnedSlice();
    }
    return out;
}

test "tp check: a dump round-trips" {
    const a = testing.allocator;
    const bytes = try twoRanks(a, false);
    defer for (bytes) |b| a.free(b);
    var d = try parse(a, bytes[1]);
    defer d.deinit(a);
    try testing.expectEqual(@as(u32, 1), d.header.rank);
    try testing.expectEqualStrings("cuda test", d.header.label);
    try testing.expectEqualStrings("commit=abc model=test", d.header.provenance);
    try testing.expectEqual(Mode.greedy, d.header.mode);
    try testing.expectEqual(@as(usize, 3), d.collectives.len);
    try testing.expectEqual(Call.exchange, d.collectives[2].call);
    try testing.expectEqual(@as(u32, 0), d.collectives[2].peer);
    try testing.expectEqual(@as(usize, 1), d.logits.len);
    try testing.expectEqual(@as(f32, 4.0), d.logits[0].at(2));
}

test "tp check: a truncated or foreign dump is refused" {
    const a = testing.allocator;
    const bytes = try twoRanks(a, false);
    defer for (bytes) |b| a.free(b);
    try testing.expectError(error.Truncated, parse(a, bytes[0][0 .. bytes[0].len - 1]));
    var foreign = try a.dupe(u8, bytes[0]);
    defer a.free(foreign);
    foreign[0] = 'X';
    try testing.expectError(error.BadMagic, parse(a, foreign));
}

test "tp check: correct collectives pass rule 1, one flipped bit fails it" {
    const a = testing.allocator;
    for ([_]bool{ false, true }) |bad| {
        const bytes = try twoRanks(a, bad);
        defer for (bytes) |b| a.free(b);
        var d0 = try parse(a, bytes[0]);
        defer d0.deinit(a);
        var d1 = try parse(a, bytes[1]);
        defer d1.deinit(a);
        const r = try checkReference(a, &.{ d0, d1 });
        try testing.expectEqual(@as(usize, 3), r.calls);
        try testing.expectEqual(@as(usize, 6), r.buffers);
        if (!bad) {
            try testing.expectEqual(@as(usize, 0), r.mismatches);
        } else {
            try testing.expectEqual(@as(usize, 1), r.mismatches);
            const m = r.first.?;
            try testing.expectEqual(@as(u32, 1), m.rank);
            try testing.expectEqual(@as(usize, 1), m.element);
            try testing.expectEqualStrings("value", m.why);
        }
    }
}

test "tp check: rule 1 refuses ranks that disagree on the call" {
    const a = testing.allocator;
    const bytes = try twoRanks(a, false);
    defer for (bytes) |b| a.free(b);
    var d0 = try parse(a, bytes[0]);
    defer d0.deinit(a);
    var d1 = try parse(a, bytes[1]);
    defer d1.deinit(a);
    d1.collectives[0].dtype = .f16;
    const r = try checkReference(a, &.{ d0, d1 });
    try testing.expectEqualStrings("ranks disagree on the call", r.first.?.why);
}

test "tp check: rule 2 finds the first record that differs between runs" {
    const a = testing.allocator;
    const good = try twoRanks(a, false);
    defer for (good) |b| a.free(b);
    const bad = try twoRanks(a, true);
    defer for (bad) |b| a.free(b);
    var x = try parse(a, good[1]);
    defer x.deinit(a);
    var y = try parse(a, good[1]);
    defer y.deinit(a);
    try testing.expect(repeatPassed(checkRepeat(x, y)));
    var z = try parse(a, bad[1]);
    defer z.deinit(a);
    const r = checkRepeat(x, z);
    try testing.expect(!repeatPassed(r));
    try testing.expectEqual(@as(?usize, 0), r.first_collective);
}

const Row = struct { prompt: u32, position: u64, chosen: u32, logits: []const f32 };

fn logitsDump(a: std.mem.Allocator, h: Header, rows: []const Row) ![]u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    defer w.deinit();
    try writeHeader(&w.writer, h);
    for (rows) |r| try writeLogits(&w.writer, r.prompt, r.position, r.chosen, r.logits);
    return w.toOwnedSlice();
}

/// Two prompts on one box alone, in order.
const solo_rows = [_]Row{
    .{ .prompt = 0, .position = 0, .chosen = 1, .logits = &.{ 1.0, 2.0, 0.0 } },
    .{ .prompt = 0, .position = 1, .chosen = 0, .logits = &.{ 3.0, 0.0, 0.0 } },
    .{ .prompt = 0, .position = 2, .chosen = 2, .logits = &.{ 0.0, 0.0, 5.0 } },
    .{ .prompt = 1, .position = 0, .chosen = 1, .logits = &.{ 0.0, 4.0, 0.0 } },
    .{ .prompt = 1, .position = 1, .chosen = 0, .logits = &.{ 2.0, 0.0, 0.0 } },
};

/// The mixed run, out of order: prompt 0 splits at 1, and its far-off position 2 must stay out of greedy divergence.
const mixed_rows = [_]Row{
    .{ .prompt = 1, .position = 1, .chosen = 0, .logits = &.{ 2.0, 0.0, 0.0 } },
    .{ .prompt = 0, .position = 2, .chosen = 2, .logits = &.{ 0.0, 0.0, 9.0 } },
    .{ .prompt = 1, .position = 0, .chosen = 1, .logits = &.{ 0.0, 4.0, 0.0 } },
    .{ .prompt = 0, .position = 0, .chosen = 1, .logits = &.{ 1.0, 2.0, 0.0 } },
    .{ .prompt = 0, .position = 1, .chosen = 1, .logits = &.{ 0.0, 3.5, 0.0 } },
};

test "tp check: rule 4 greedy matches by prompt and position and stops each prompt at its first split" {
    const a = testing.allocator;
    const s = try logitsDump(a, .{ .rank = 0, .world = 1, .label = "solo" }, &solo_rows);
    defer a.free(s);
    const m = try logitsDump(a, .{ .rank = 0, .world = 2, .label = "mixed" }, &mixed_rows);
    defer a.free(m);
    var solo = try parse(a, s);
    defer solo.deinit(a);
    var mixed = try parse(a, m);
    defer mixed.deinit(a);
    var r = try checkAgree(a, mixed, solo);
    defer r.deinit(a);
    try testing.expectEqual(@as(usize, 2), r.prompts);
    try testing.expectEqual(@as(usize, 5), r.positions);
    try testing.expectEqual(@as(usize, 3), r.greedy_same);
    try testing.expectEqual(@as(usize, 1), r.splits.len);
    try testing.expectEqual(Split{ .prompt = 0, .position = 1 }, r.splits[0]);
    try testing.expectEqual(@as(usize, 4), r.compared);
    try testing.expectEqual(@as(usize, 3), r.top1_same);
    try testing.expectEqual(@as(f64, 3.5), r.max_abs);
    try testing.expect(r.kl_max > 2.0);
    try testing.expectEqual(@as(?bool, null), r.fed_ok);
    var same = try checkAgree(a, solo, solo);
    defer same.deinit(a);
    try testing.expectEqual(@as(usize, 0), same.splits.len);
    try testing.expectEqual(@as(f64, 0), same.max_abs);
    try testing.expectEqual(@as(f64, 0), same.kl_max);
}

test "tp check: rule 4 greedy stops a prompt at a gap, so a hidden split cannot pair different prefixes" {
    const a = testing.allocator;
    const s = try logitsDump(a, .{ .rank = 0, .world = 1, .label = "solo" }, &solo_rows);
    defer a.free(s);
    // the mixed run lost prompt 0 position 1 (its split); position 2 is far off and must not be compared
    const holed = [_]Row{ mixed_rows[0], mixed_rows[1], mixed_rows[2], mixed_rows[3] };
    const m = try logitsDump(a, .{ .rank = 0, .world = 2, .label = "mixed" }, &holed);
    defer a.free(m);
    var solo = try parse(a, s);
    defer solo.deinit(a);
    var mixed = try parse(a, m);
    defer mixed.deinit(a);
    var r = try checkAgree(a, mixed, solo);
    defer r.deinit(a);
    try testing.expectEqual(@as(usize, 1), r.missing);
    try testing.expectEqual(@as(usize, 1), r.gaps.len);
    try testing.expectEqual(Split{ .prompt = 0, .position = 1 }, r.gaps[0]);
    try testing.expectEqual(@as(usize, 0), r.splits.len);
    try testing.expectEqual(@as(usize, 4), r.positions);
    try testing.expectEqual(@as(usize, 3), r.compared);
    try testing.expectEqual(@as(f64, 0), r.max_abs);
}

test "tp check: rule 4 greedy does not count positions missing after a split" {
    const a = testing.allocator;
    const s = try logitsDump(a, .{ .rank = 0, .world = 1, .label = "solo" }, &solo_rows);
    defer a.free(s);
    // prompt 0 splits at position 1, then the mixed run ends: its position 2 is absent, which is normal
    const short = [_]Row{ mixed_rows[0], mixed_rows[2], mixed_rows[3], mixed_rows[4] };
    const m = try logitsDump(a, .{ .rank = 0, .world = 2, .label = "mixed" }, &short);
    defer a.free(m);
    var solo = try parse(a, s);
    defer solo.deinit(a);
    var mixed = try parse(a, m);
    defer mixed.deinit(a);
    var r = try checkAgree(a, mixed, solo);
    defer r.deinit(a);
    try testing.expectEqual(@as(usize, 0), r.missing);
    try testing.expectEqual(@as(usize, 0), r.gaps.len);
    try testing.expectEqual(@as(usize, 1), r.splits.len);
    try testing.expectEqual(@as(usize, 4), r.compared);
}

test "tp check: rule 4 teacher-forced compares every position, only when fed the solo run's tokens" {
    const a = testing.allocator;
    const s = try logitsDump(a, .{ .rank = 0, .world = 1, .label = "solo" }, &solo_rows);
    defer a.free(s);
    var solo = try parse(a, s);
    defer solo.deinit(a);
    const fed = try chosenHash(a, solo);
    for ([_]u64{ fed, fed ^ 1 }) |h| {
        const m = try logitsDump(a, .{ .rank = 0, .world = 2, .mode = .teacher_forced, .fed_hash = h, .label = "mixed tf" }, &mixed_rows);
        defer a.free(m);
        var mixed = try parse(a, m);
        defer mixed.deinit(a);
        var r = try checkAgree(a, mixed, solo);
        defer r.deinit(a);
        try testing.expectEqual(@as(usize, 5), r.positions);
        try testing.expectEqual(@as(usize, 0), r.splits.len);
        if (h == fed) {
            try testing.expectEqual(@as(?bool, true), r.fed_ok);
            try testing.expectEqual(@as(usize, 5), r.compared);
            try testing.expectEqual(@as(f64, 4.0), r.max_abs);
        } else {
            try testing.expectEqual(@as(?bool, false), r.fed_ok);
            try testing.expectEqual(@as(usize, 0), r.compared);
        }
    }
}

test "tp check: the fed-token hash does not depend on record order" {
    const a = testing.allocator;
    const x = try logitsDump(a, .{ .rank = 0, .world = 1, .label = "x" }, &mixed_rows);
    defer a.free(x);
    const sorted = [_]Row{ mixed_rows[3], mixed_rows[4], mixed_rows[1], mixed_rows[2], mixed_rows[0] };
    const y = try logitsDump(a, .{ .rank = 0, .world = 1, .label = "y" }, &sorted);
    defer a.free(y);
    var dx = try parse(a, x);
    defer dx.deinit(a);
    var dy = try parse(a, y);
    defer dy.deinit(a);
    try testing.expectEqual(try chosenHash(a, dx), try chosenHash(a, dy));
    var t: TokenHash = .{};
    for (sorted) |r| t.add(r.prompt, r.position, r.chosen);
    try testing.expectEqual(t.h, try chosenHash(a, dx));
}

test {
    _ = @import("tp_check_phase_test.zig");
    _ = @import("tp_check_reject_test.zig");
}

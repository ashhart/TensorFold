//! Exact stored bytes and mapped picks, using the common CPU reference selectors.
const std = @import("std");
const core = @import("core");
const r = core.proposal_rows;
const t = std.testing;
const gpa = t.allocator;
const fixture = @embedFile("src/core/testdata/proposal-vocab.json");
fn artifact(affine: bool) !core.proposal_vocab.Artifact {
    var doc = try std.json.parseFromSlice(core.proposal_vocab.Document, gpa, fixture, .{});
    defer doc.deinit();
    doc.value.identity.hidden = 64;
    doc.value.identity.format = if (affine) "affine4" else "bf16";
    doc.value.identity.group = if (affine) 32 else 0;
    doc.value.identity.scale_recipe = if (affine) "mlx-bf16" else "none";
    const bytes = try std.json.Stringify.valueAlloc(gpa, doc.value, .{});
    defer gpa.free(bytes);
    return core.proposal_vocab.load(gpa, bytes, doc.value.identity, &.{0});
}
test "proposal rows: exact stored BF16 and affine codes scales biases survive selection and zero padding" {
    for ([_]bool{ false, true }) |affine| {
        var a = try artifact(affine);
        defer a.deinit();
        var bytes: [11 * 132]u8 = undefined;
        for (&bytes, 0..) |*b, i| b.* = @truncate(i *% 73);
        bytes[0] = 0;
        bytes[1] = 0x80;
        bytes[2] = 0xc1;
        bytes[3] = 0x7f;
        var side: [11 * 6]u8 = undefined;
        for (&side, 0..) |*b, i| b.* = @truncate(i *% 19);
        const original = bytes;
        const row: usize = if (affine) 32 else 128;
        const source: r.Source = .{ .identity = a.identity(), .storage = if (affine) .affine else .bf16, .recipe = if (affine) .affine_mm else .qlinear_decode, .scale_dtype = if (affine) .bf16 else .unknown, .bias_dtype = if (affine) .bf16 else .unknown, .codes = .{ .bytes = &bytes, .stride = 132 }, .scales = if (affine) .{ .bytes = &side, .stride = 6 } else .{}, .biases = if (affine) .{ .bytes = &side, .stride = 6 } else .{}, .tied = true, .queue = 123 };
        const price = try source.price(&a, 128);
        try t.expectError(error.ProposalRowBudget, source.extract(gpa, &a, 128, price - 1));
        var selected = try source.extract(gpa, &a, 128, price);
        defer selected.deinit();
        try t.expectEqual(@as(u32, 128), selected.physical);
        try t.expectEqualSlices(u32, a.ids(), selected.ids);
        for (a.ids(), 0..) |id, i| {
            try t.expectEqualSlices(u8, bytes[id * 132 ..][0..row], selected.codes[i * row ..][0..row]);
            if (affine) {
                try t.expectEqualSlices(u8, side[id * 6 ..][0..4], selected.scales[i * 4 ..][0..4]);
                try t.expectEqualSlices(u8, side[id * 6 ..][0..4], selected.biases[i * 4 ..][0..4]);
            }
        }
        try t.expect(std.mem.allEqual(u8, selected.codes[a.ids().len * row ..], 0));
        try t.expectEqualSlices(u8, &original, &bytes);
        try t.expectEqual(a.fingerprint, selected.fingerprint);
        try t.expect(selected.tied and selected.queue == 123);
    }
}
test "proposal rows: identity opaque basis rank stride and allocation failures preserve the prior rows" {
    var a = try artifact(true);
    defer a.deinit();
    const bytes: [11 * 32]u8 = @splat(0x73);
    const side: [11 * 4]u8 = @splat(0x38);
    var source: r.Source = .{ .identity = a.identity(), .storage = .affine, .recipe = .qlinear_decode, .scale_dtype = .bf16, .bias_dtype = .bf16, .codes = .{ .bytes = &bytes, .stride = 32 }, .scales = .{ .bytes = &side, .stride = 4 }, .biases = .{ .bytes = &side, .stride = 4 } };
    var prior = try source.extract(gpa, &a, 128, 1 << 20);
    defer prior.deinit();
    var wrong = source;
    wrong.identity.hidden = 128;
    try t.expectError(error.ProposalIdentityMismatch, wrong.extract(gpa, &a, 128, 1 << 20));
    wrong = source;
    wrong.storage = .unsupported;
    try t.expectError(error.ProposalBasisUnsupported, wrong.extract(gpa, &a, 128, 1 << 20));
    wrong = source;
    wrong.scale_dtype = .f32;
    try t.expectError(error.ProposalBasisUnsupported, wrong.extract(gpa, &a, 128, 1 << 20));
    wrong = source;
    wrong.ranks = 2;
    try t.expectError(error.ProposalPlacementUnsupported, wrong.extract(gpa, &a, 128, 1 << 20));
    wrong = source;
    wrong.codes.stride = 31;
    try t.expectError(error.ProposalRowShape, wrong.extract(gpa, &a, 128, 1 << 20));
    for (0..4) |failure| {
        var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = failure });
        try t.expectError(error.OutOfMemory, source.extract(failing.allocator(), &a, 128, 1 << 20));
        try t.expect(std.mem.allEqual(u8, prior.codes[0..96], 0x73));
    }
}
test "proposal rows: global ties padding winners mixed pitches and nonfinite rules use shared selectors" {
    const compact: r.Scores = .{ .vocab = 11, .logical = 3, .stride = 8, .ids = &.{ 0, 3, 7 } };
    var words: [8]u16 = @splat(0x7f80);
    words[0] = 0xbf80;
    words[1] = 0xbf80;
    words[2] = 0xc000;
    try t.expectEqual(@as(u32, 0), try compact.pick(&words));
    const best = try compact.best(&words, 2);
    try t.expectEqual(@as(u32, 0), best.entries[0].token);
    try t.expectEqual(@as(u32, 3), best.entries[1].token);
    try t.expectError(error.ProposalPaddingPick, compact.global(3));
    for ([_]u16{ 0x7fc1, 0x7f80, 0xff80 }) |bad| {
        words[1] = bad;
        try t.expectError(error.ProposalNonfinite, compact.pick(&words));
        try t.expectError(error.ProposalNonfinite, compact.best(&words, 2));
    }
    try t.expectError(error.ProposalBasisUnsupported, (r.Scores{ .dtype = .f32, .vocab = 11, .logical = 11, .stride = 11 }).check());
    const full: r.Scores = .{ .vocab = 11, .logical = 11, .stride = 16 };
    var whole: [16]u16 = @splat(0x7f80);
    @memset(whole[0..11], 0xbf80);
    whole[9] = 0x3f80;
    try t.expectEqual(@as(u32, 9), try full.pick(&whole));
    try t.expectError(error.ProposalScoreShape, (r.Scores{ .vocab = 11, .logical = 3, .stride = 8, .ids = &.{ 7, 3, 0 } }).check());
    try t.expectError(error.ProposalScoreShape, (r.Scores{ .vocab = 11, .logical = 3, .stride = 8, .ids = &.{ 0, 3, 11 } }).check());
}
const AffineRecorder = struct {
    var calls: usize = 0;
    pub fn dense(_: u8, _: u8, _: u8, _: u8, q: anytype, _: u8, rows: u32) void {
        std.debug.assert(q.n == 128 and q.k == 64 and rows == 17 and q.w == 1 and q.s == 2 and q.b == 3);
        calls += 1;
    }
};
test "proposal rows: shared affine encoding retains the exact format and source recipe" {
    var a = try artifact(true);
    defer a.deinit();
    const codes: [11 * 32]u8 = @splat(0x98);
    const side: [11 * 4]u8 = @splat(0x38);
    const source: r.Source = .{ .identity = a.identity(), .storage = .affine, .recipe = .affine_mm, .scale_dtype = .bf16, .bias_dtype = .bf16, .codes = .{ .bytes = &codes, .stride = 32 }, .scales = .{ .bytes = &side, .stride = 4 }, .biases = .{ .bytes = &side, .stride = 4 } };
    var selected = try source.extract(gpa, &a, 128, 1 << 20);
    defer selected.deinit();
    AffineRecorder.calls = 0;
    const format = .{ .bits = @as(u8, 4), .group = @as(u16, 32), .out_f32 = false };
    try selected.affine(AffineRecorder, format, @as(u8, 0), @as(u8, 0), .{ .w = @as(u8, 1), .s = @as(u8, 2), .b = @as(u8, 3) }, @as(u8, 0), @as(u8, 0), @as(u8, 0), 17);
    try t.expectEqual(@as(usize, 1), AffineRecorder.calls);
    try t.expectError(error.ProposalBasisUnsupported, selected.affine(AffineRecorder, .{ .bits = @as(u8, 8), .group = @as(u16, 32), .out_f32 = false }, @as(u8, 0), @as(u8, 0), .{ .w = @as(u8, 1), .s = @as(u8, 2), .b = @as(u8, 3) }, @as(u8, 0), @as(u8, 0), @as(u8, 0), 17));
}

test "proposal rows: mixed plans roles and queues partition and scatter by original row index" {
    const base: r.Key = .{ .generation = @splat(1), .binding = 123, .logical = 3, .stride = 128, .hidden = 64, .recipe = .qlinear_decode, .queue = 4 };
    var keys: [6]r.Key = @splat(base);
    keys[1].full = true;
    keys[1].logical = 11;
    keys[2].generation = @splat(2);
    keys[4].recipe = .qlinear_prompt;
    keys[5].queue = 9;
    var order: [6]usize = undefined;
    var groups: [6]r.Group = undefined;
    const p = try r.partition(&keys, &order, &groups);
    try t.expect(p.order.ptr == &order and p.groups.ptr == &groups);
    try t.expectError(error.ProposalRowBudget, r.partition(&keys, order[0..5], &groups));
    try t.expectError(error.ProposalRowBudget, r.partition(&keys, &order, groups[0..5]));
    try t.expectEqualSlices(usize, &.{ 0, 3, 1, 2, 4, 5 }, p.order);
    try t.expectEqual(@as(usize, 5), p.groups.len);
    var out: [6]u32 = undefined;
    try p.scatter(u32, &.{ 7, 3, 9, 0, 3, 7 }, &out);
    try t.expectEqualSlices(u32, &.{ 7, 9, 0, 3, 3, 7 }, &out);
    const before = out;
    try t.expectError(error.ProposalRowAlias, p.scatter(u32, &out, &out));
    try t.expectEqualSlices(u32, &before, &out);
}

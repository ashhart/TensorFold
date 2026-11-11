//! Real row-selection paths with fake CUDA reductions and downloads; no device is opened.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const mtp = @import("cuda_mtp.zig");
const draw = @import("cuda_mtp_draw.zig");
const ops = @import("cuda_mtp_ops.zig");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const t = std.testing;
const gpa = t.allocator;

const fixture = @import("cuda_mtp_nonfinite_fixture.zig");
const Device = fixture.Device;
const reduction = fixture.reduction;
const draft = fixture.draft;
const head = fixture.head;

test "Flash Next nonfinite: single and batched head selections reject sentinels and preserve finite ID mapping" {
    var d = Device.driver();
    defer cuda.download_stage.closeDriver(&d);
    var h = try head(&d);
    defer h.artifact.deinit();
    var host: [3]u16 = undefined;
    var x = draft(&d, &host);
    const st: cuda.Stream = .{ .d = &d, .handle = null };
    var e: prompt.Engine = undefined;
    e.stream = @constCast(&st);
    e.gpa = gpa;
    const o: ops.Ops(struct {}) = .{ .e = &e, .h = &h, .tri = .{} };
    const invalid = [_][4]f32{
        reduction(-std.math.inf(f32), std.math.nan(f32), std.math.maxInt(i32)),
        reduction(std.math.nan(f32), 2, 1),
        reduction(std.math.inf(f32), std.math.nan(f32), 1),
        reduction(1, std.math.nan(f32), 1),
        reduction(1, 2, -1),
        reduction(1, 2, 3),
    };
    for (invalid) |bad| {
        Device.best = bad;
        try t.expectError(error.ProposalNonfinite, o.pick(&x, 12, null));
        for ([_]bool{ false, true }) |batched| {
            try t.expectError(error.ProposalNonfinite, draw.pick(&h, st, gpa, &x, bad, 12, null, batched));
            try t.expectError(error.ProposalNonfinite, draw.pick(&h, st, gpa, &x, bad, 12, .{ .seed = 7, .temperature = 1, .top_k = 2 }, batched));
        }
    }
    for (h.ids, 0..) |id, col| {
        Device.best = reduction(2, 3, @intCast(col));
        const solo = try o.pick(&x, 12, null);
        const batched = try draw.pick(&h, st, gpa, &x, Device.best, 12, null, true);
        try t.expectEqual(id, solo.token);
        try t.expectEqualDeep(solo, batched);
        try t.expectEqual(@exp(@as(f64, -1)), solo.prob);
    }
}

test "Flash Next nonfinite: main-model greedy sampled and masked rows fail before using invalid columns" {
    var d = Device.driver();
    defer cuda.download_stage.closeDriver(&d);
    var h = try head(&d);
    defer h.artifact.deinit();
    const host = try gpa.alloc(u16, wt.vocab);
    defer gpa.free(host);
    var x = draft(&d, host);
    const st: cuda.Stream = .{ .d = &d, .handle = null };
    const sampled: mtp.Sampling = .{ .seed = 7, .temperature = 1, .top_k = 2 };
    for ([_]?mtp.Sampling{ null, sampled }) |sm| {
        Device.best = reduction(-std.math.inf(f32), std.math.nan(f32), std.math.maxInt(i32));
        Device.cols = &.{ -1, 1 };
        var out = [_]u32{42};
        try t.expectError(error.ProposalNonfinite, draw.draws(&h, st, gpa, &x, sm, &.{}, &out));
        try t.expectEqual(@as(u32, 42), out[0]);
    }
    Device.best = reduction(2, 3, 17);
    var out: [1]u32 = undefined;
    try draw.draws(&h, st, gpa, &x, null, &.{}, &out);
    try t.expectEqual(@as(u32, 17), out[0]);
}

test "Flash Next nonfinite: top-k rejects NaNs infinities and invalid columns before mapping or sampling" {
    const s: mtp.Sampling = .{ .seed = 7, .temperature = 1, .top_k = 2 };
    for ([_]u16{ 0x7fc1, 0xffc1, 0x7f80, 0xff80 }) |bad| {
        try t.expectError(error.ProposalNonfinite, draw.drawTopk(gpa, &.{ 0x3f80, bad }, &.{ 0, 1 }, &.{ 0, 3 }, 12, s));
    }
    for ([_]i32{ -1, 2, std.math.maxInt(i32) }) |col| {
        try t.expectError(error.ProposalNonfinite, draw.drawTopk(gpa, &.{ 0x3f80, 0 }, &.{ 0, col }, &.{ 0, 3 }, 12, s));
    }
    try t.expectError(error.ProposalNonfinite, draw.drawTopk(gpa, &.{0x3f80}, &.{wt.vocab}, null, 12, s));
    const row = [_]u16{ 0x3f80, 0, 0xbf80 };
    const ids = [_]u32{ 0, 3, 7 };
    for (0..20) |position| {
        try t.expectEqual(try mtp.drawRow(gpa, &row, &ids, position, s), try draw.drawTopk(gpa, row[0..2], &.{ 0, 1 }, &ids, position, s));
    }
}

test {
    _ = @import("cuda_mtp_nonfinite_more_test.zig");
}

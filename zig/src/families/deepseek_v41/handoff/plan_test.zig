const std = @import("std");
const g = @import("geometry.zig");
const plan = @import("plan.zig");

fn shapeOf(upto: u64, shard: u32, shards: u32) plan.Shape {
    return .{ .upto = upto, .shard = shard, .shards = shards, .split = shards > 1 };
}

test "part names have one spelling and round-trip; anything else is refused" {
    var buf: [40]u8 = undefined;
    const specs = [_]plan.Spec{
        .{ .layer = 0, .kind = .swa, .rows = 1, .cols = 512, .dtype = .bf16 },
        .{ .layer = 2, .kind = .carry, .rows = 2, .cols = 512, .dtype = .f32 },
        .{ .layer = 41, .kind = .ring, .rows = 3, .cols = 512, .dtype = .bf16 },
        .{ .layer = 8, .kind = .comp, .first_page = 160, .rows = 1, .cols = 512, .dtype = .bf16 },
        .{ .layer = 8, .kind = .ixk, .split = true, .first_page = 0, .rows = 1, .cols = 128, .dtype = .bf16 },
    };
    const names = [_][]const u8{ "swa", "carry", "ring", "comp@160", "ixk.split@0" };
    for (specs, names) |sp, n| {
        try std.testing.expectEqualStrings(n, sp.name(&buf));
        const p = plan.parseName(n).?;
        try std.testing.expectEqual(sp.kind, p.kind);
        try std.testing.expectEqual(sp.split, p.split);
        try std.testing.expectEqual(sp.first_page, p.first_page);
    }
    for ([_][]const u8{ "", "comp", "swa@1", "comp@01", "comp@x", "comp@-1", "comp.split.split@1", "ring.split", "swa.split", "foo", "comp@1@2", "COMP@1", "comp @1", "comp@4294967296" }) |bad| {
        try std.testing.expectEqual(@as(?plan.Parsed, null), plan.parseName(bad));
    }
}

test "a shape is refused when its pieces do not fit together" {
    try shapeOf(10, 0, 1).check();
    try shapeOf(10, 1, 2).check();
    try std.testing.expectError(error.BadShape, shapeOf(0, 0, 1).check());
    try std.testing.expectError(error.BadShape, shapeOf((1 << 20) + 1, 0, 1).check());
    try std.testing.expectError(error.BadShape, shapeOf(10, 2, 2).check());
    try std.testing.expectError(error.BadShape, shapeOf(10, 0, 9).check());
    var s = shapeOf(10, 0, 1);
    s.split = true; // split with one producer
    try std.testing.expectError(error.BadShape, s.check());
    s = shapeOf(10, 0, 2);
    s.split = false; // two producers without the residue rule: nothing says who holds what
    try std.testing.expectError(error.BadShape, s.check());
    s = shapeOf(100, 0, 1);
    s.ds_rows = 101;
    try std.testing.expectError(error.BadShape, s.check());
    s = shapeOf(300, 0, 1);
    s.ds_rows = 129;
    try std.testing.expectError(error.BadShape, s.check());
    s.ds_rows = 128;
    try s.check();
}

test "the release plan at 1000 tokens: layers, parts, rows" {
    const gpa = std.testing.allocator;
    const list = try plan.specs(gpa, shapeOf(1000, 0, 1));
    defer gpa.free(list);
    var swa: u32 = 0;
    var carry: u32 = 0;
    var comp: [4]u64 = @splat(0);
    var ixk: [4]u64 = @splat(0);
    var last_layer: u32 = 0;
    for (list) |sp| {
        try std.testing.expect(sp.layer >= last_layer); // model order
        last_layer = sp.layer;
        try std.testing.expect(sp.bytes() <= plan.chunk_bytes);
        const si = g.release.sourceIndex(sp.layer);
        switch (sp.kind) {
            .swa => {
                swa += 1;
                try std.testing.expectEqual(@as(u32, 128), sp.rows);
            },
            .carry => {
                carry += 1;
                try std.testing.expectEqual(@as(u32, 2), sp.rows);
                try std.testing.expectEqual(plan.Dtype.f32, sp.dtype);
            },
            .comp => comp[si.?] += sp.rows,
            .ixk => ixk[si.?] += sp.rows,
            .ring => return error.NoDrafterWasAsked,
        }
    }
    try std.testing.expectEqual(@as(u32, 40), swa);
    try std.testing.expectEqual(@as(u32, 3), carry);
    try std.testing.expectEqualSlices(u64, &.{ 500, 500, 500, 1000 }, &comp);
    try std.testing.expectEqualSlices(u64, &comp, &ixk);
}

test "a short prompt: fewer window rows, no completed groups, the ratio-1 source already holds rows" {
    const gpa = std.testing.allocator;
    const list = try plan.specs(gpa, shapeOf(1, 0, 1));
    defer gpa.free(list);
    for (list) |sp| switch (sp.kind) {
        .swa => try std.testing.expectEqual(@as(u32, 1), sp.rows),
        .comp, .ixk => try std.testing.expectEqual(@as(u32, 20), sp.layer), // 1 / 2 = 0 rows at ratio 2
        else => {},
    };
    // the carry of position 0 still rides at the ratio-2 sources
    var carries: u32 = 0;
    for (list) |sp| carries += @intFromBool(sp.kind == .carry);
    try std.testing.expectEqual(@as(u32, 3), carries);
}

test "drafter rings ride with the first shard only, when rows are given" {
    const gpa = std.testing.allocator;
    var s = shapeOf(1000, 0, 1);
    s.ds_rows = 100;
    const list = try plan.specs(gpa, s);
    defer gpa.free(list);
    var rings: u32 = 0;
    for (list) |sp| if (sp.kind == .ring) {
        try std.testing.expectEqual(@as(u32, 40 + rings), sp.layer);
        try std.testing.expectEqual(@as(u32, 100), sp.rows);
        rings += 1;
    };
    try std.testing.expectEqual(@as(u32, 3), rings);
}

test "split shards partition every source's rows exactly, with a part per chunk window" {
    const gpa = std.testing.allocator;
    for ([_]u32{ 2, 3, 8 }) |world| {
        for ([_]u64{ 1, 129, 700, 4097 }) |upto| {
            var rows: [4]u64 = @splat(0);
            var parts: usize = 0;
            for (0..world) |sh| {
                var s = shapeOf(upto, @intCast(sh), world);
                s.chunk = 70 << 10;
                const list = try plan.specs(gpa, s);
                defer gpa.free(list);
                for (list) |sp| {
                    if (sh > 0) try std.testing.expect(sp.kind == .comp or sp.kind == .ixk);
                    if (sp.kind == .comp) {
                        rows[g.release.sourceIndex(sp.layer).?] += sp.rows;
                        parts += 1;
                        try std.testing.expect(sp.split);
                        try std.testing.expect(sp.bytes() <= @max(s.chunk, @as(u64, 128) * 1024 * 2)); // a page over the chunk is a part alone
                    }
                }
            }
            for (g.release.sources, 0..) |src, i| try std.testing.expectEqual(upto / src.ratio, rows[i]);
        }
    }
}

test "find answers for exactly the parts the plan lists, and for nothing else" {
    const gpa = std.testing.allocator;
    var s = shapeOf(1200, 1, 2);
    s.chunk = 70 << 10;
    const list = try plan.specs(gpa, s);
    defer gpa.free(list);
    var buf: [40]u8 = undefined;
    for (list) |sp| {
        const got = plan.find(s, sp.layer, sp.name(&buf)).?;
        try std.testing.expectEqual(sp, got);
    }
    try std.testing.expectEqual(@as(?plan.Spec, null), plan.find(s, 0, "swa")); // only shard 0 carries the replicated state
    try std.testing.expectEqual(@as(?plan.Spec, null), plan.find(s, 2, "comp@1")); // not a window boundary
    try std.testing.expectEqual(@as(?plan.Spec, null), plan.find(s, 2, "comp@9999"));
    try std.testing.expectEqual(@as(?plan.Spec, null), plan.find(s, 2, "comp@0")); // not .split under a split shape
    try std.testing.expectEqual(@as(?plan.Spec, null), plan.find(s, 3, "comp.split@0")); // layer 3 is no source
    try std.testing.expectEqual(@as(?plan.Spec, null), plan.find(s, 99, "swa"));
    var lead = shapeOf(1200, 0, 1);
    try std.testing.expectEqual(@as(?plan.Spec, null), plan.find(lead, 2, "ring"));
    try std.testing.expectEqual(@as(?plan.Spec, null), plan.find(lead, 20, "carry")); // ratio 1: no carry
    lead.ds_rows = 5;
    try std.testing.expect(plan.find(lead, 41, "ring") != null);
    try std.testing.expectEqual(@as(?plan.Spec, null), plan.find(lead, 39, "ring"));
}

test "at the context limit every part is one frame under the mailbox's 4 MiB segment" {
    const gpa = std.testing.allocator;
    const list = try plan.specs(gpa, shapeOf(1 << 20, 0, 1));
    defer gpa.free(list);
    var most: u64 = 0;
    var bytes: u64 = 0;
    for (list) |sp| {
        most = @max(most, sp.bytes());
        bytes += sp.bytes();
    }
    try std.testing.expect(most <= plan.chunk_bytes and plan.chunk_bytes + 8192 <= 4 << 20);
    // bf16 pool + keys at the limit: (3 x 512K + 1M rows) x (1024 + 256) B, plus 40 windows
    try std.testing.expectEqual(@as(u64, (3 * (1 << 19) + (1 << 20)) * 1280 + 40 * 128 * 1024 + 3 * 4096), bytes);
}

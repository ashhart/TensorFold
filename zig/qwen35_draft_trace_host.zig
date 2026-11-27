//! Exact two-round trace snapshots beside the actual DFlash frame fixtures.
const std = @import("std");
const q = @import("qwen35");
const lanes = @import("lanes");
const a = std.testing.allocator;

pub fn enableTrace(d: *q.draft.Draft, tmp: std.testing.TmpDir, name: []const u8) !*q.draft.trace.Trace {
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, name });
    defer a.free(path);
    d.trace = (try q.draft.trace.openPath(a, std.testing.io, path)).?;
    errdefer q.draft.trace.drop(&d.trace);
    const t = d.trace.?;
    t.steps = try a.alloc(q.draft.tree.GrowNode, q.draft.frame.max_streams * 65 * lanes.draft_tree.branch);
    t.captures = try a.alloc(q.draft.tree.GrowTrace, q.draft.frame.max_streams);
    return t;
}

pub fn run(comptime Record: type, d: *q.draft.Draft, holds: *const [2]lanes.drafter.Hold, tmp: std.testing.TmpDir) !void {
    _ = try enableTrace(d, tmp, "rounds.jsonl");
    defer q.draft.trace.drop(&d.trace);
    Record.fail = false;
    Record.token_bias = 0;
    Record.downloads = 0;
    try d.drafter().hold(holds[0..1]);
    Record.token_bias = 1;
    defer Record.token_bias = 0;
    Record.downloads = 0;
    try d.drafter().hold(holds[1..2]);
    const accepted = [_]u32{41};
    const item: lanes.drafter.Absorb = .{
        .stream = holds[0].stream,
        .features = .{ .buffer = 1, .rows = 0, .row_bytes = 0, .space = .device, .dtype = .bf16 },
        .start = 2048,
        .follow = &accepted,
        .offered = 1,
        .prompt = false,
    };
    try d.drafter().absorb(&.{item});
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/rounds.jsonl", .{tmp.sub_path});
    defer a.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 20));
    defer a.free(bytes);
    const got = try q.draft.trace.parse(a, std.mem.trimEnd(u8, bytes, "\n"));
    defer got.deinit();
    const record = got.value;
    try std.testing.expectEqualStrings("first", record.stream);
    try std.testing.expectEqual(@as(u64, 2049), record.position);
    try std.testing.expectEqual(@as(u32, 0), record.pending);
    const sampling = record.sampling orelse return error.MissingSampling;
    try std.testing.expectEqual(@as(u64, 0x123456789abcdef), sampling.seed);
    try std.testing.expectEqual(@as(u64, 0x3fe8000000000000), sampling.temperature);
    try std.testing.expectEqual(@as(usize, 15), record.candidates.len);
    for (record.candidates) |row| {
        try std.testing.expectEqualSlices(u32, &.{ 17, 128 }, row.ids);
        try std.testing.expectEqualSlices(u32, &.{ 0x3f800000, 0 }, row.unary);
    }
    try std.testing.expectEqual(@as(u32, 32), record.frontier_count orelse return error.MissingFrontierCount);
    try std.testing.expectEqual(@as(u32, 32), record.frontier_expected orelse return error.MissingFrontierExpected);
    const frontier = [_]q.draft.trace.Frontier{
        .{ .sequence = 0, .parent = -1, .token = 17, .row = 0, .own = 13823479343445158052, .cost = 4600107306590382244 },
        .{ .sequence = 1, .parent = -1, .token = 128, .row = 0, .own = 13831467005445467938, .cost = 4608094968590692130 },
        .{ .sequence = 2, .parent = 0, .token = 17, .row = 1, .own = 13822694340203091740, .cost = 4604218404596719584 },
        .{ .sequence = 3, .parent = 0, .token = 128, .row = 1, .own = 13831960193445197766, .cost = 4610197078258541039 },
        .{ .sequence = 4, .parent = 1, .token = 17, .row = 2, .own = 13824935148726914440, .cost = 4607673284686926674 },
        .{ .sequence = 5, .parent = 1, .token = 128, .row = 2, .own = 13830718548323314416, .cost = 4610368103994260192 },
        .{ .sequence = 6, .parent = 2, .token = 17, .row = 3, .own = 13824335296247822401, .cost = 4609496194555711842 },
        .{ .sequence = 7, .parent = 2, .token = 128, .row = 3, .own = 13831004857789825012, .cost = 4612156652438358435 },
        .{ .sequence = 8, .parent = 3, .token = 17, .row = 1, .own = 13822694340203091740, .cost = 4609507639448294633 },
        .{ .sequence = 9, .parent = 3, .token = 128, .row = 1, .own = 13831960193445197766, .cost = 4612845162217927540 },
        .{ .sequence = 10, .parent = 4, .token = 17, .row = 4, .own = 13826092394972818005, .cost = 4611727391134741318 },
        .{ .sequence = 11, .parent = 4, .token = 128, .row = 4, .own = 13829885081559949884, .cost = 4612675562781524288 },
        .{ .sequence = 12, .parent = 5, .token = 17, .row = 2, .own = 13824935148726914440, .cost = 4611480512436852811 },
        .{ .sequence = 13, .parent = 5, .token = 128, .row = 2, .own = 13830718548323314416, .cost = 4612930675085787116 },
        .{ .sequence = 14, .parent = 6, .token = 17, .row = 2, .own = 13824935148726914440, .cost = 4611927984837243560 },
        .{ .sequence = 15, .parent = 6, .token = 128, .row = 2, .own = 13830718548323314416, .cost = 4613275394490910320 },
        .{ .sequence = 16, .parent = 7, .token = 17, .row = 3, .own = 13824335296247822401, .cost = 4611938516145216632 },
        .{ .sequence = 17, .parent = 7, .token = 128, .row = 3, .own = 13831004857789825012, .cost = 4613504062092025194 },
        .{ .sequence = 18, .parent = 8, .token = 17, .row = 3, .own = 13824335296247822401, .cost = 4612494720366512942 },
        .{ .sequence = 19, .parent = 8, .token = 128, .row = 3, .own = 13831004857789825012, .cost = 4614060266313321504 },
        .{ .sequence = 20, .parent = 9, .token = 128, .row = 5, .own = 13826889161666722928, .cost = 4613062867451408994 },
        .{ .sequence = 21, .parent = 9, .token = 17, .row = 5, .own = 13828792325850690483, .cost = 4613538658497400883 },
        .{ .sequence = 22, .parent = 10, .token = 17, .row = 3, .own = 13824335296247822401, .cost = 4612839439771636144 },
        .{ .sequence = 23, .parent = 10, .token = 128, .row = 3, .own = 13831004857789825012, .cost = 4614404985718444706 },
        .{ .sequence = 24, .parent = 11, .token = 17, .row = 4, .own = 13826092394972818005, .cost = 4613074800788408077 },
        .{ .sequence = 25, .parent = 11, .token = 128, .row = 4, .own = 13829885081559949884, .cost = 4614022972435191047 },
        .{ .sequence = 26, .parent = 12, .token = 17, .row = 4, .own = 13826092394972818005, .cost = 4613292937081549880 },
        .{ .sequence = 27, .parent = 12, .token = 128, .row = 4, .own = 13829885081559949884, .cost = 4614241108728332850 },
        .{ .sequence = 28, .parent = 13, .token = 17, .row = 4, .own = 13826092394972818005, .cost = 4613631005009704387 },
        .{ .sequence = 29, .parent = 13, .token = 128, .row = 4, .own = 13829885081559949884, .cost = 4614579176656487357 },
        .{ .sequence = 30, .parent = 14, .token = 128, .row = 5, .own = 13826889161666722928, .cost = 4614011039098191964 },
        .{ .sequence = 31, .parent = 14, .token = 17, .row = 5, .own = 13828792325850690483, .cost = 4614486830144183853 },
    };
    try std.testing.expectEqualSlices(q.draft.trace.Frontier, &frontier, record.frontier);
}

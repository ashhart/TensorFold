//! Host proof for the CUDA DFlash round trace's exact, opt-in JSONL format.
const std = @import("std");
const lanes = @import("lanes");
const trace = @import("cuda_draft_trace.zig");

fn bits(x: f64) u64 {
    return @bitCast(x);
}

test "N15 trace: local scores, ordered frontier and accepted path round-trip exactly" {
    const a = std.testing.allocator;
    var words: [trace.lp_words * 2]u32 = undefined;
    for (&words, 0..) |*x, i| x.* = @intCast(1000 + i * 17);
    var hp: [8]u16 = .{ 0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc0, 0x0001 };
    const nodes = [_]u32{ 41, 42, 43 };
    const parents = [_]i32{ -1, 0, 0 };
    const own = [_]f64{ -0.0, -0.25, -0.5 };
    const frontier = [_]lanes.draft_tree.GrowNode{
        .{ .sequence = 0, .parent = -1, .token = 41, .row = 0, .own = -0.0, .cost = 0.0 },
        .{ .sequence = 1, .parent = -1, .token = 40, .row = 0, .own = -2, .cost = 2 },
        .{ .sequence = 2, .parent = 0, .token = 42, .row = 1, .own = -0.25, .cost = 0.25 },
        .{ .sequence = 3, .parent = 0, .token = 43, .row = 1, .own = -0.5, .cost = 0.5 },
    };
    const accepted_rows = [_]u32{ 1, 2, 5 };
    const accepted_tokens = [_]u32{ 41, 42, 99 };
    const candidates: lanes.draft_tree.Candidates = .{
        .words = std.mem.sliceAsBytes(&words),
        .projected = std.mem.sliceAsBytes(&hp),
        .topk = 3,
        .predecessor = &.{},
        .successor = &.{},
        .rank = 4,
    };
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    try trace.writeRecord(&text.writer, .{
        .stream = "request\\\"one",
        .position = 123,
        .pending = 40,
        .offered = 3,
        .nodes = &nodes,
        .parents = &parents,
        .own = &own,
        .accepted_rows = &accepted_rows,
        .accepted_tokens = &accepted_tokens,
        .candidates = candidates,
        .frontier = &frontier,
        .frontier_expected = frontier.len,
    });
    const got = try trace.parse(a, std.mem.trimEnd(u8, text.written(), "\n"));
    defer got.deinit();
    const r = got.value;
    try std.testing.expectEqual(@as(u8, 2), r.v);
    try std.testing.expectEqualStrings("request\\\"one", r.stream);
    try std.testing.expectEqual(@as(u64, 123), r.position);
    try std.testing.expectEqual(@as(u32, 40), r.pending);
    try std.testing.expectEqual(@as(u32, 3), r.offered);
    try std.testing.expectEqual(@as(u32, 4), r.rank);
    try std.testing.expectEqual(@as(u32, 3), r.topk);
    try std.testing.expectEqualSlices(u32, &nodes, r.nodes);
    try std.testing.expectEqualSlices(i32, &parents, r.parents);
    for (own, r.own) |want, actual| try std.testing.expectEqual(bits(want), actual);
    try std.testing.expectEqual(bits(lanes.draft_tree.scale), r.parameters.scale);
    try std.testing.expectEqual(bits(lanes.draft_tree.edge_weight), r.parameters.edge);
    try std.testing.expectEqual(bits(lanes.draft_tree.noise_weight), r.parameters.noise);
    try std.testing.expectEqual(@as(usize, frontier.len), r.frontier.len);
    try std.testing.expectEqual(@as(u32, @intCast(frontier.len)), r.frontier_expected orelse return error.MissingFrontierExpected);
    try std.testing.expectEqual(@as(u32, @intCast(frontier.len)), r.frontier_count orelse return error.MissingFrontierCount);
    for (frontier, r.frontier) |want, actual| {
        try std.testing.expectEqual(want.sequence, actual.sequence);
        try std.testing.expectEqual(want.parent, actual.parent);
        try std.testing.expectEqual(want.token, actual.token);
        try std.testing.expectEqual(want.row, actual.row);
        try std.testing.expectEqual(bits(want.own), actual.own);
        try std.testing.expectEqual(bits(want.cost), actual.cost);
    }
    try std.testing.expectEqualSlices(u32, &accepted_rows, r.accepted_rows);
    try std.testing.expectEqualSlices(u32, &accepted_tokens, r.accepted_tokens);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, -1 }, r.accepted_nodes);
    try std.testing.expectEqual(@as(usize, 2), r.candidates.len);
    for (r.candidates, 0..) |row, i| {
        const w = words[i * trace.lp_words ..][0..trace.lp_words];
        const ids = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(w)[trace.ids_offset .. trace.ids_offset + 3 * 4]);
        const unary = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(w)[trace.unary_offset .. trace.unary_offset + 3 * 4]);
        try std.testing.expectEqualSlices(u32, ids, row.ids);
        try std.testing.expectEqualSlices(u32, unary, row.unary);
    }
    try std.testing.expectError(error.UnsupportedTraceVersion, trace.parse(a, "{\"v\":1}"));
}

test "N15 trace: sampled replay preserves its seed and exact temperature" {
    const a = std.testing.allocator;
    var words: [trace.lp_words]u32 = @splat(0);
    var hp: [4]u16 = .{ 0x3f80, 0xbf80, 0x0000, 0x8000 };
    const sampling: lanes.Sampling = .{ .seed = 0xfedcba9876543210, .temperature = 0.75 };
    const candidates: lanes.draft_tree.Candidates = .{
        .words = std.mem.sliceAsBytes(&words),
        .projected = std.mem.sliceAsBytes(&hp),
        .topk = 3,
        .rank = 4,
        .predecessor = &.{},
        .successor = &.{},
    };
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    try trace.writeRecord(&text.writer, .{
        .stream = "sampled",
        .position = 77,
        .pending = 9,
        .offered = 0,
        .nodes = &.{},
        .parents = &.{},
        .own = &.{},
        .accepted_rows = &.{},
        .accepted_tokens = &.{},
        .candidates = candidates,
        .sampling = sampling,
    });
    const got = try trace.parse(a, std.mem.trimEnd(u8, text.written(), "\n"));
    defer got.deinit();
    const saved = got.value.sampling orelse return error.MissingSampledReplay;
    try std.testing.expectEqual(sampling.seed, saved.seed);
    try std.testing.expectEqual(bits(sampling.temperature), saved.temperature);
    const replay: lanes.Sampling = .{ .seed = saved.seed, .temperature = @bitCast(saved.temperature) };
    try std.testing.expectEqual(sampling.seed, replay.seed);
    try std.testing.expectEqual(bits(sampling.temperature), bits(replay.temperature));
}

test "N15 trace: all fifteen rows preserve candidate scores without raw projections" {
    const a = std.testing.allocator;
    var words: [15 * trace.lp_words]u32 = undefined;
    for (&words, 0..) |*x, i| x.* = @bitCast(@as(f32, @floatFromInt(i)));
    for (0..15) |row| for (0..16) |j| {
        words[row * trace.lp_words + 2 + j] = @intCast(row * 16 + j);
        words[row * trace.lp_words + 2 + 20 + j] = @bitCast(@as(f32, @floatFromInt(row * 16 + j)));
    };
    var hp: [15 * 256]u16 = undefined;
    for (&hp, 0..) |*x, i| x.* = @truncate(i *% 65521);
    const candidates: lanes.draft_tree.Candidates = .{
        .words = std.mem.sliceAsBytes(&words),
        .projected = std.mem.sliceAsBytes(&hp),
        .topk = 16,
        .rank = 256,
        .predecessor = &.{},
        .successor = &.{},
    };
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    try trace.writeRecord(&text.writer, .{
        .stream = "n15",
        .position = 1,
        .pending = 0,
        .offered = 0,
        .nodes = &.{},
        .parents = &.{},
        .own = &.{},
        .accepted_rows = &.{},
        .accepted_tokens = &.{},
        .candidates = candidates,
    });
    const got = try trace.parse(a, std.mem.trimEnd(u8, text.written(), "\n"));
    defer got.deinit();
    try std.testing.expectEqual(@as(usize, 15), got.value.candidates.len);
    try std.testing.expectEqual(@as(u32, 256), got.value.rank);
    try std.testing.expectEqual(@as(u32, 16), got.value.topk);
    for (got.value.candidates, 0..) |row, i| {
        const w = words[i * trace.lp_words ..][0..trace.lp_words];
        try std.testing.expectEqualSlices(u32, w[2..18], row.ids);
        try std.testing.expectEqualSlices(u32, w[22..38], row.unary);
    }
    try std.testing.expectEqual(@as(usize, 0), got.value.frontier.len);
}

test "N15 trace: writer rejects inconsistent lengths, offered counts and frontier values" {
    var words: [trace.lp_words]u32 = @splat(0);
    const hp: [4]u16 = @splat(0);
    const candidates: lanes.draft_tree.Candidates = .{ .words = std.mem.sliceAsBytes(&words), .projected = std.mem.sliceAsBytes(&hp), .topk = 3, .predecessor = &.{}, .successor = &.{}, .rank = 4 };
    const valid: trace.Round = .{
        .stream = "bad",
        .position = 0,
        .pending = 0,
        .offered = 0,
        .nodes = &.{},
        .parents = &.{},
        .own = &.{},
        .accepted_rows = &.{},
        .accepted_tokens = &.{},
        .candidates = candidates,
    };
    var bad = valid;
    var text: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer text.deinit();
    bad.offered = 1;
    try std.testing.expectError(error.BadDraftTrace, trace.writeRecord(&text.writer, bad));
    bad = valid;
    bad.frontier = &.{.{ .sequence = 0, .parent = -1, .token = 41, .row = 0, .own = 0, .cost = 0 }};
    try std.testing.expectError(error.BadDraftTrace, trace.writeRecord(&text.writer, bad));
    bad = valid;
    bad.nodes = &.{41};
    try std.testing.expectError(error.BadDraftTrace, trace.writeRecord(&text.writer, bad));
    bad = valid;
    bad.own = &.{0};
    try std.testing.expectError(error.BadDraftTrace, trace.writeRecord(&text.writer, bad));
    bad = valid;
    bad.frontier = &.{.{ .sequence = 1, .parent = -1, .token = 41, .row = 0, .own = 0, .cost = 0 }};
    try std.testing.expectError(error.BadDraftTrace, trace.writeRecord(&text.writer, bad));
    bad.frontier = &.{.{ .sequence = 0, .parent = -1, .token = 41, .row = 0, .own = std.math.inf(f64), .cost = 0 }};
    try std.testing.expectError(error.BadDraftTrace, trace.writeRecord(&text.writer, bad));
}

test "N15 trace: actual grow JSONL serializes every insertion and exact priority" {
    const a = std.testing.allocator;
    var words: [2 * trace.lp_words]u32 = @splat(0);
    const unary = [2][4]f32{ .{ 3, 1, 0, -1 }, .{ 2, 2, -5, -5 } };
    for (0..2) |row| for (0..4) |j| {
        words[row * trace.lp_words + 2 + j] = @intCast(10 * row + j);
        words[row * trace.lp_words + 2 + lanes.draft_tree.lp_top + j] = @bitCast(unary[row][j]);
    };
    const zeros: [32]u16 = @splat(0);
    const candidates: lanes.draft_tree.Candidates = .{
        .words = std.mem.sliceAsBytes(&words),
        .projected = std.mem.sliceAsBytes(zeros[0..2]),
        .topk = 4,
        .rank = 1,
        .predecessor = &zeros,
        .successor = &zeros,
    };
    var nodes: [6]u32 = undefined;
    var parents: [6]i32 = undefined;
    var own: [6]f64 = undefined;
    var captured: [lanes.draft_tree.branch * 65]lanes.draft_tree.GrowNode = undefined;
    var capture: lanes.draft_tree.GrowTrace = .{ .nodes = &captured };
    const grown = lanes.draft_tree.growTraced(64, candidates, 5, 2, 6, &nodes, &parents, &own, &capture);
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    try trace.writeRecord(&text.writer, .{
        .stream = "actual-grow",
        .position = 20,
        .pending = 5,
        .offered = grown,
        .nodes = nodes[0..grown],
        .parents = parents[0..grown],
        .own = own[0..grown],
        .accepted_rows = &.{},
        .accepted_tokens = &.{},
        .frontier = capture.nodes[0..capture.len],
        .frontier_expected = capture.expected,
        .candidates = candidates,
    });
    const got = try trace.parse(a, std.mem.trimEnd(u8, text.written(), "\n"));
    defer got.deinit();
    try std.testing.expectEqual(@as(u32, @intCast(capture.len)), got.value.frontier_count orelse return error.MissingFrontierCount);
    try std.testing.expectEqual(@as(u32, @intCast(capture.expected)), got.value.frontier_expected orelse return error.MissingFrontierExpected);
    try std.testing.expectEqual(capture.len, got.value.frontier.len);
    for (capture.nodes[0..capture.len], got.value.frontier) |want, actual| {
        try std.testing.expectEqual(want.sequence, actual.sequence);
        try std.testing.expectEqual(want.parent, actual.parent);
        try std.testing.expectEqual(want.token, actual.token);
        try std.testing.expectEqual(@as(u32, @intCast(want.row)), actual.row);
        try std.testing.expectEqual(bits(want.own), actual.own);
        try std.testing.expectEqual(bits(want.cost), actual.cost);
    }
    std.debug.print("{s}", .{text.written()});
}

test "N15 trace: a held snapshot survives another hold reusing shared staging" {
    const a = std.testing.allocator;
    var words: [2 * trace.lp_words]u32 = undefined;
    for (&words, 0..) |*x, i| x.* = @intCast(1000 + i * 17);
    var hp: [8]u16 = .{ 0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc0, 0x0001 };
    var captured: [2]lanes.draft_tree.GrowNode = .{
        .{ .sequence = 0, .parent = -1, .token = 41, .row = 0, .own = -0.0, .cost = 0.0 },
        .{ .sequence = 1, .parent = -1, .token = 40, .row = 0, .own = -2, .cost = 2 },
    };
    const candidates: lanes.draft_tree.Candidates = .{
        .words = std.mem.sliceAsBytes(&words),
        .projected = std.mem.sliceAsBytes(&hp),
        .topk = 3,
        .rank = 4,
        .predecessor = &.{},
        .successor = &.{},
    };
    var capture: lanes.draft_tree.GrowTrace = .{ .nodes = &captured, .len = captured.len, .expected = captured.len };
    var snapshot: trace.Snapshot(2) = .{};
    defer snapshot.deinit(a);
    try snapshot.save(a, candidates, &capture, 16, 40, .{ .seed = 1234, .temperature = 0.75 });

    const emit = struct {
        fn run(w: anytype, s: anytype) !void {
            try trace.writeRecord(w, .{
                .stream = "first",
                .position = s.position,
                .pending = s.pending,
                .offered = 0,
                .nodes = &.{},
                .parents = &.{},
                .own = &.{},
                .accepted_rows = &.{},
                .accepted_tokens = &.{},
                .candidates = s.candidates(&.{}, &.{}),
                .frontier = s.frontier(),
                .frontier_expected = s.frontier_expected,
                .sampling = s.sampling,
            });
        }
    }.run;
    var before: std.Io.Writer.Allocating = .init(a);
    defer before.deinit();
    try emit(&before.writer, &snapshot);

    // A different stream's hold reuses the staged rows and capture before the first absorb.
    for (&words, 0..) |*x, i| x.* = @intCast(9000 + i * 31);
    @memset(&hp, 0x1234);
    captured[0].token = 999;
    captured[0].cost = 99;
    var after: std.Io.Writer.Allocating = .init(a);
    defer after.deinit();
    try emit(&after.writer, &snapshot);
    try std.testing.expectEqualStrings(before.written(), after.written());
}

test "N15 trace: no environment path means no file and no logger" {
    try std.testing.expectEqual(@as(?*trace.Trace, null), trace.openPath(std.testing.allocator, std.testing.io, null));
    try std.testing.expectEqual(@as(?*trace.Trace, null), trace.openPath(std.testing.allocator, std.testing.io, ""));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/rounds.jsonl", .{tmp.sub_path});
    defer std.testing.allocator.free(path);
    var slot = try trace.openPath(std.testing.allocator, std.testing.io, path);
    try std.testing.expect(slot != null);
    trace.drop(&slot);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, std.testing.allocator, .limited(1));
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqual(@as(usize, 0), bytes.len);
}

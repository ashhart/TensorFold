//! Disjoint request timelines retain overlap and parent self-time instead of summing inclusive spans.
const std = @import("std");
const plan = @import("cuda_prefill_plan.zig");
pub const Axis = enum { gpu, wall };
pub const Bucket = enum { stage, parent_self, overlap, gap };
pub const Segment = struct { begin_ns: u64, end_ns: u64, bucket: Bucket, row: ?usize, class_mask: u64, stream_mask: u64 };
pub const View = struct {
    total_ns: u64,
    accounted_ns: u64 = 0,
    overlap_ns: u64 = 0,
    parent_self_ns: u64 = 0,
    gap_ns: u64 = 0,
    class_ns: [std.meta.tags(plan.Class).len]u64 = @splat(0),
    exclusive_by_row_ns: []u64,
    segments: []Segment,
    pub fn deinit(v: View, a: std.mem.Allocator) void {
        a.free(v.exclusive_by_row_ns);
        a.free(v.segments);
    }
};
const Edge = struct { at: u64, row: usize, begin: bool };
const Key = struct { row: usize, depth: usize, detailed: bool };
fn order(_: void, x: Key, y: Key) std.math.Order {
    if (x.detailed != y.detailed) return if (x.detailed) .lt else .gt;
    if (x.depth != y.depth) return std.math.order(y.depth, x.depth);
    return std.math.order(y.row, x.row);
}
fn aggregate(stage: plan.Stage) bool {
    return switch (stage) {
        .request, .chunk, .ffn, .mtp, .mtp_layer, .joint_ffn, .paired_prefill => true,
        else => false,
    };
}
fn bounds(row: anytype, comptime axis: Axis) ?[2]u64 {
    if (axis == .gpu) return .{ row.cuda_begin_ns orelse return null, row.cuda_end_ns orelse return null };
    const end = row.wall_end_ns orelse return null;
    return .{ std.math.cast(u64, row.wall_begin_ns) orelse return null, std.math.cast(u64, end) orelse return null };
}
/// Literal stream intervals share one CUDA event origin; wall intervals belong to the one issuing host thread.
pub fn build(a: std.mem.Allocator, rows: anytype, comptime axis: Axis) !?View {
    if (rows.len == 0 or rows[0].stage != .request or rows[0].parent != null) return null;
    const root = bounds(rows[0], axis) orelse return null;
    if (root[1] < root[0]) return null;
    const n = rows.len;
    const depths = try a.alloc(usize, n);
    defer a.free(depths);
    const active = try a.alloc(bool, n);
    defer a.free(active);
    @memset(active, false);
    const exclusive = try a.alloc(u64, n);
    errdefer a.free(exclusive);
    @memset(exclusive, 0);
    var edges: std.ArrayList(Edge) = .empty;
    defer edges.deinit(a);
    var stream_count: usize = 1;
    for (rows, 0..) |row, i| {
        const range = bounds(row, axis) orelse {
            a.free(exclusive);
            return null;
        };
        if (range[0] < root[0] or range[1] > root[1] or range[1] < range[0] or row.stream >= 64) {
            a.free(exclusive);
            return null;
        }
        depths[i] = if (row.parent) |p| blk: {
            if (p >= i) {
                a.free(exclusive);
                return null;
            }
            break :blk depths[p] + 1;
        } else 0;
        stream_count = if (axis == .wall) 1 else @max(stream_count, row.stream + 1);
        if (range[0] == range[1]) continue;
        try edges.append(a, .{ .at = range[0] - root[0], .row = i, .begin = true });
        try edges.append(a, .{ .at = range[1] - root[0], .row = i, .begin = false });
    }
    std.mem.sort(Edge, edges.items, {}, struct {
        fn less(_: void, x: Edge, y: Edge) bool {
            return x.at < y.at;
        }
    }.less);
    const Queue = std.PriorityQueue(Key, void, order);
    const queues = try a.alloc(Queue, stream_count);
    defer a.free(queues);
    for (queues) |*queue| queue.* = .initContext({});
    defer for (queues) |*queue| queue.deinit(a);
    var segments: std.ArrayList(Segment) = .empty;
    defer segments.deinit(a);
    var view: View = .{ .total_ns = root[1] - root[0], .exclusive_by_row_ns = exclusive, .segments = &.{} };
    var previous: u64 = 0;
    var index: usize = 0;
    while (index < edges.items.len) {
        const at = edges.items[index].at;
        if (at > previous) {
            var best: ?Key = null;
            var detailed: usize = 0;
            var classes: u64 = 0;
            var streams: u64 = 0;
            for (queues, 0..) |*queue, stream| {
                while (queue.peek()) |key| {
                    if (active[key.row]) break;
                    _ = queue.pop();
                }
                if (queue.peek()) |key| {
                    if (key.detailed) {
                        detailed += 1;
                        classes |= @as(u64, 1) << @intCast(@backingInt(rows[key.row].class));
                        streams |= @as(u64, 1) << @intCast(stream);
                    }
                    if (best == null or order({}, key, best.?) == .lt) best = key;
                }
            }
            const bucket: Bucket = if (detailed > 1) .overlap else if (best) |key| if (key.detailed) .stage else .parent_self else .gap;
            const picked: ?usize = if (bucket == .overlap or best == null) null else best.?.row;
            const ns = at - previous;
            view.accounted_ns += ns;
            switch (bucket) {
                .overlap => view.overlap_ns += ns,
                .gap => view.gap_ns += ns,
                .stage, .parent_self => {
                    exclusive[picked.?] += ns;
                    view.class_ns[@backingInt(rows[picked.?].class)] += ns;
                    if (bucket == .parent_self) view.parent_self_ns += ns;
                },
            }
            try segments.append(a, .{ .begin_ns = previous, .end_ns = at, .bucket = bucket, .row = picked, .class_mask = classes, .stream_mask = streams });
        }
        while (index < edges.items.len and edges.items[index].at == at) : (index += 1) {
            const edge = edges.items[index];
            active[edge.row] = edge.begin;
            if (edge.begin) try queues[if (axis == .wall) 0 else rows[edge.row].stream].push(a, .{ .row = edge.row, .depth = depths[edge.row], .detailed = !aggregate(rows[edge.row].stage) });
        }
        previous = at;
    }
    if (view.accounted_ns != view.total_ns) {
        a.free(exclusive);
        return null;
    }
    view.segments = try segments.toOwnedSlice(a);
    return view;
}

const TestRow = struct {
    stage: plan.Stage,
    class: plan.Class,
    parent: ?usize,
    stream: usize,
    cuda_begin_ns: ?u64,
    cuda_end_ns: ?u64,
    wall_begin_ns: i96,
    wall_end_ns: ?i96,
};
fn testRow(stage: plan.Stage, parent: ?usize, stream: usize, begin: u64, end: u64) TestRow {
    return .{ .stage = stage, .class = plan.class(stage), .parent = parent, .stream = stream, .cuda_begin_ns = begin, .cuda_end_ns = end, .wall_begin_ns = begin, .wall_end_ns = end };
}
test "GLM prefill ledger: additive partition counts parents and concurrent streams exactly once" {
    const t = std.testing;
    const rows = [_]TestRow{
        testRow(.request, null, 0, 0, 100),  testRow(.ffn, 0, 0, 0, 100),
        testRow(.projections, 1, 0, 10, 40), testRow(.partials, 1, 1, 20, 50),
        testRow(.wait_shared, 1, 0, 40, 50), testRow(.head, 1, 0, 60, 80),
    };
    const view = (try build(t.allocator, &rows, .gpu)).?;
    defer view.deinit(t.allocator);
    try t.expectEqual(@as(u64, 100), view.total_ns);
    try t.expectEqual(view.total_ns, view.accounted_ns);
    try t.expectEqual(@as(u64, 30), view.overlap_ns);
    try t.expectEqual(@as(u64, 40), view.parent_self_ns);
    try t.expectEqual(@as(u64, 0), view.gap_ns);
    try t.expectEqualSlices(u64, &.{ 0, 40, 10, 0, 0, 20 }, view.exclusive_by_row_ns);
    var previous: u64 = 0;
    for (view.segments) |segment| {
        try t.expectEqual(previous, segment.begin_ns);
        try t.expect(segment.end_ns > segment.begin_ns);
        previous = segment.end_ns;
        if (segment.bucket == .overlap) try t.expectEqual(@as(u64, 3), segment.stream_mask);
    }
    try t.expectEqual(view.total_ns, previous);
}
test "GLM prefill ledger: unavailable or out of request timings cannot yield an additive view" {
    const t = std.testing;
    var rows = [_]TestRow{ testRow(.request, null, 0, 0, 100), testRow(.norms, 0, 0, 10, 20) };
    rows[1].cuda_end_ns = null;
    try t.expect(try build(t.allocator, &rows, .gpu) == null);
    rows[1].cuda_end_ns = 101;
    try t.expect(try build(t.allocator, &rows, .gpu) == null);
    rows[1].cuda_end_ns = 20;
    rows[1].parent = 1;
    try t.expect(try build(t.allocator, &rows, .gpu) == null);
}

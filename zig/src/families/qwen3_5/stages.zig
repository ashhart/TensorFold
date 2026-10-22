//! Staged references: every engine names a qwen3_5 forward's stages alike and writes their bf16 rows to one layout.
//! Off unless a tool asks for it; tools/zig/qwen36_stage_compare.py reads the directory.
const std = @import("std");

/// The residual stream after the embedding, after each layer's mixer and its MLP, then the final norm and the head.
pub const Point = union(enum) {
    embed,
    mixer: usize,
    block: usize,
    final,
    logits,

    /// The point's plane among `count(layers)`; a layer past the model's is refused.
    pub fn index(p: Point, layers: usize) !usize {
        return switch (p) {
            .embed => 0,
            .mixer => |l| if (l < layers) 1 + 2 * l else error.BadStageLayer,
            .block => |l| if (l < layers) 2 + 2 * l else error.BadStageLayer,
            .final => 1 + 2 * layers,
            .logits => 2 + 2 * layers,
        };
    }

    /// The file stem: embed, l07.mixer, l07.block, final, logits.
    pub fn name(p: Point, buf: []u8) ![]const u8 {
        return switch (p) {
            .embed => "embed",
            .mixer => |l| std.fmt.bufPrint(buf, "l{d:0>2}.mixer", .{l}),
            .block => |l| std.fmt.bufPrint(buf, "l{d:0>2}.block", .{l}),
            .final => "final",
            .logits => "logits",
        };
    }
};

pub fn count(layers: usize) usize {
    return 3 + 2 * layers;
}

/// The point at plane `i`, the inverse of `Point.index`.
pub fn point(i: usize, layers: usize) !Point {
    if (i >= count(layers)) return error.BadStageLayer;
    if (i == 0) return .embed;
    if (i == 1 + 2 * layers) return .final;
    if (i == 2 + 2 * layers) return .logits;
    return if (i % 2 == 1) .{ .mixer = (i - 1) / 2 } else .{ .block = (i - 2) / 2 };
}

/// A device engine's stage hook: `rows` rows of the point's width at device address `ptr`, valid once its stream drains.
pub const DeviceHook = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, at: Point, ptr: u64, rows: usize) anyerror!void,
};

/// One round's planes in an engine's buffer: every stream point `capacity` rows wide, then the logits plane.
pub const Round = struct {
    recorder: *Recorder,
    capacity: u32,
    logit_capacity: u32,
    /// Each plane's rows this round: a stream plane left at 0 when the round ends was never recorded.
    rows: [max_planes]u32 = @splat(0),

    pub const max_planes = 256;

    pub fn init(recorder: *Recorder, capacity: u32, logit_capacity: u32) !Round {
        if (capacity == 0 or logit_capacity == 0 or count(recorder.layers) > max_planes) return error.BadStageShape;
        return .{ .recorder = recorder, .capacity = capacity, .logit_capacity = logit_capacity };
    }

    /// The buffer's bytes: every plane at its capacity.
    pub fn bytes(r: Round) usize {
        return r.offset(.logits) + @as(usize, r.logit_capacity) * r.recorder.vocab * 2;
    }

    pub fn offset(r: Round, at: Point) usize {
        const i = at.index(r.recorder.layers) catch unreachable;
        return i * @as(usize, r.capacity) * r.recorder.hidden * 2;
    }

    pub fn begin(r: *Round) void {
        r.rows = @splat(0);
    }

    /// Claims `at`'s plane for `rows` rows; the engine then copies them to `offset(at)`.
    pub fn mark(r: *Round, at: Point, rows: usize) !void {
        const i = try at.index(r.recorder.layers);
        if (rows == 0 or rows > (if (at == .logits) r.logit_capacity else r.capacity)) return error.BadStageShape;
        if (r.rows[i] != 0) return error.DuplicateStage;
        r.rows[i] = @intCast(rows);
    }

    /// Hands the round in `base` to the recorder; every stream point must hold the round's rows.
    pub fn end(r: *Round, base: []const u8, first: u32) !void {
        const n = count(r.recorder.layers);
        if (base.len < r.bytes()) return error.BadStageShape;
        const rows = r.rows[0];
        for (r.rows[0..n], 0..) |got, i| if ((try point(i, r.recorder.layers)) != .logits and got != rows) return error.IncompleteStages;
        var views: [max_planes][]const u8 = undefined;
        for (views[0..n], 0..) |*view, i| {
            const at = try point(i, r.recorder.layers);
            view.* = base[r.offset(at)..][0 .. @as(usize, r.rows[i]) * r.recorder.width(at) * 2];
        }
        try r.recorder.round(first, rows, r.rows[n - 1], views[0..n]);
    }
};

/// A run's stages on the host: bf16 rows per point and the cache position of every row, in the order rounds ran.
pub const Recorder = struct {
    allocator: std.mem.Allocator,
    layers: usize,
    hidden: usize,
    vocab: usize,
    planes: []std.ArrayList(u8),
    positions: std.ArrayList(u32) = .empty,
    logit_positions: std.ArrayList(u32) = .empty,

    pub fn init(a: std.mem.Allocator, layers: usize, hidden: usize, vocab: usize) !Recorder {
        if (layers == 0 or hidden == 0 or vocab == 0) return error.BadStageShape;
        const planes = try a.alloc(std.ArrayList(u8), count(layers));
        @memset(planes, .empty);
        return .{ .allocator = a, .layers = layers, .hidden = hidden, .vocab = vocab, .planes = planes };
    }

    pub fn deinit(r: *Recorder) void {
        for (r.planes) |*p| p.deinit(r.allocator);
        r.allocator.free(r.planes);
        r.positions.deinit(r.allocator);
        r.logit_positions.deinit(r.allocator);
    }

    pub fn width(r: Recorder, p: Point) usize {
        return if (p == .logits) r.vocab else r.hidden;
    }

    /// One round: `rows` stream rows from cache position `first`, every stream point's rows, and `logits` rows that
    /// end at the round's last row (0: the round had no head, 1: its last row, `rows`: all of them).
    pub fn round(r: *Recorder, first: u32, rows: u32, logits: u32, planes: []const []const u8) !void {
        if (rows == 0 or (logits != 0 and logits != 1 and logits != rows) or planes.len != count(r.layers)) return error.BadStageShape;
        for (planes, 0..) |bytes, i| {
            const p = try point(i, r.layers);
            const n: usize = if (p == .logits) logits else rows;
            if (bytes.len != n * r.width(p) * 2) return error.BadStageShape;
        }
        for (planes, r.planes) |bytes, *plane| try plane.appendSlice(r.allocator, bytes);
        for (0..rows) |row| try r.positions.append(r.allocator, first + @as(u32, @intCast(row)));
        for (rows - logits..rows) |row| try r.logit_positions.append(r.allocator, first + @as(u32, @intCast(row)));
    }

    /// `dir`/<name>.bf16 per point, row-major, and index.json with the shapes and every row's cache position.
    pub fn write(r: Recorder, io: std.Io, dir: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        try cwd.createDirPath(io, dir);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        var name_buf: [32]u8 = undefined;
        const Entry = struct { name: []const u8, width: usize, rows: usize };
        const entries = try r.allocator.alloc(Entry, r.planes.len);
        defer r.allocator.free(entries);
        for (r.planes, entries, 0..) |plane, *entry, i| {
            const p = try point(i, r.layers);
            const stem = try p.name(&name_buf);
            const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}.bf16", .{ dir, stem });
            try cwd.writeFile(io, .{ .sub_path = path, .data = plane.items });
            entry.* = .{ .name = try r.allocator.dupe(u8, stem), .width = r.width(p), .rows = plane.items.len / (r.width(p) * 2) };
        }
        defer for (entries) |e| r.allocator.free(e.name);
        const index = try std.json.Stringify.valueAlloc(r.allocator, .{
            .version = 1,
            .dtype = "bf16",
            .layers = r.layers,
            .hidden = r.hidden,
            .vocab = r.vocab,
            .stages = entries,
            .positions = r.positions.items,
            .logit_positions = r.logit_positions.items,
        }, .{});
        defer r.allocator.free(index);
        try cwd.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&path_buf, "{s}/index.json", .{dir}), .data = index });
    }
};

test "stage planes and names cover every point once, in forward order" {
    var buf: [32]u8 = undefined;
    for ([_]usize{ 1, 2, 40 }) |layers| {
        for (0..count(layers)) |i| try std.testing.expectEqual(i, try (try point(i, layers)).index(layers));
    }
    try std.testing.expectEqualStrings("l07.mixer", try (Point{ .mixer = 7 }).name(&buf));
    try std.testing.expectEqualStrings("l39.block", try (Point{ .block = 39 }).name(&buf));
    try std.testing.expectError(error.BadStageLayer, (Point{ .block = 2 }).index(2));
    try std.testing.expectError(error.BadStageLayer, point(count(2), 2));
}

test "Round: every sequence of four begin, mark and end events matches the model of which rounds land" {
    // One layer: planes embed, mixer, block, final (stream) and logits; capacity 2 rows; marks of 1-3 rows.
    const Event = union(enum) { begin, end, mark: struct { plane: usize, rows: usize } };
    var events: [17]Event = undefined;
    events[0] = .begin;
    events[1] = .end;
    for (0..5) |p| for (1..4) |r| {
        events[2 + p * 3 + r - 1] = .{ .mark = .{ .plane = p, .rows = r } };
    };
    const a = std.testing.allocator;
    var base: [64]u8 = @splat(0);
    var seq: [4]usize = undefined;
    for (0..std.math.pow(usize, events.len + 1, seq.len)) |code| {
        var c = code;
        var len: usize = 0;
        while (len < seq.len) : (len += 1) {
            if (c % (events.len + 1) == events.len) break;
            seq[len] = c % (events.len + 1);
            c /= events.len + 1;
        }
        var rec = try Recorder.init(a, 1, 1, 1);
        defer rec.deinit();
        var round_ = try Round.init(&rec, 2, 2);
        var marked: [5]usize = @splat(0);
        var landed: usize = 0;
        for (seq[0..len]) |e| switch (events[e]) {
            .begin => {
                round_.begin();
                marked = @splat(0);
            },
            .mark => |m| {
                const want: ?anyerror = if (m.rows > 2) error.BadStageShape else if (marked[m.plane] != 0) error.DuplicateStage else null;
                const got = round_.mark(try point(m.plane, 1), m.rows);
                if (want) |err| try std.testing.expectError(err, got) else {
                    try got;
                    marked[m.plane] = m.rows;
                }
            },
            .end => {
                const rows = marked[0];
                const complete = for (marked[0..4]) |x| {
                    if (x != rows) break false;
                } else true;
                const logits = marked[4];
                const want: ?anyerror = if (!complete) error.IncompleteStages else if (rows == 0 or (logits != 0 and logits != 1 and logits != rows)) error.BadStageShape else null;
                const got = round_.end(&base, 9);
                if (want) |err| try std.testing.expectError(err, got) else {
                    try got;
                    landed += rows;
                }
            },
        };
        try std.testing.expectEqual(landed, rec.positions.items.len);
        try std.testing.expectEqual(landed * 2, rec.planes[0].items.len);
    }
}

test "a round's rows land on their positions; a short plane or odd head count is refused" {
    const a = std.testing.allocator;
    var r = try Recorder.init(a, 1, 2, 3);
    defer r.deinit();
    const row: [4]u8 = @splat(0);
    const two: [8]u8 = @splat(0);
    const logit: [6]u8 = @splat(0);
    // A two-row prompt round with its last row's logits, then a one-row step.
    try r.round(5, 2, 1, &.{ &two, &two, &two, &two, &logit });
    try r.round(7, 1, 1, &.{ &row, &row, &row, &row, &logit });
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 7 }, r.positions.items);
    try std.testing.expectEqualSlices(u32, &.{ 6, 7 }, r.logit_positions.items);
    try std.testing.expectEqual(@as(usize, 12), r.planes[0].items.len);
    try std.testing.expectError(error.BadStageShape, r.round(8, 1, 1, &.{ &row, &row, &row, row[0..2], &logit }));
    try std.testing.expectError(error.BadStageShape, r.round(8, 3, 2, &.{ &row, &row, &row, &row, &logit }));
    try std.testing.expectEqual(@as(usize, 3), r.positions.items.len);
}

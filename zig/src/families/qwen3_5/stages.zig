//! Staged references: every engine names a qwen3_5 forward's stages alike and writes their bf16 rows to one layout.
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

/// What an engine may record: one stream's linear chain of rows and its final norm and head; asked before the round.
pub fn admit(streams: usize, chain: bool, head: bool) !void {
    if (streams != 1) return error.StagesNeedOneStream;
    if (!chain) return error.StagesNeedChain;
    if (!head) return error.StagesNeedHead;
}

/// A window whose every row's parent is the row before it (the first's is -1): one linear sequence, no branches.
pub fn isChain(parents: []const i32) bool {
    for (parents, 0..) |parent, row| if (parent != @as(i32, @intCast(row)) - 1) return false;
    return true;
}

/// A device engine's stage hook: `rows` rows of the point's width at device address `ptr`, valid once drained.
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

    /// Hands the round in `base` to the recorder; every stream point must hold its rows. A round ends once.
    pub fn end(r: *Round, base: []const u8, first: u32) !void {
        defer r.rows = @splat(0);
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

    /// One round: `rows` stream rows from cache position `first` at every stream point, and `logits` rows to its last.
    pub fn round(r: *Recorder, first: u32, rows: u32, logits: u32, planes: []const []const u8) !void {
        if (rows == 0 or (logits != 0 and logits != 1 and logits != rows) or planes.len != count(r.layers)) return error.BadStageShape;
        for (planes, 0..) |bytes, i| {
            const p = try point(i, r.layers);
            const n: usize = if (p == .logits) logits else rows;
            if (bytes.len != n * r.width(p) * 2) return error.BadStageShape;
        }
        for (planes, r.planes) |bytes, *plane| try plane.ensureUnusedCapacity(r.allocator, bytes.len);
        try r.positions.ensureUnusedCapacity(r.allocator, rows);
        try r.logit_positions.ensureUnusedCapacity(r.allocator, logits);
        for (planes, r.planes) |bytes, *plane| plane.appendSliceAssumeCapacity(bytes);
        for (0..rows) |row| r.positions.appendAssumeCapacity(first + @as(u32, @intCast(row)));
        for (rows - logits..rows) |row| r.logit_positions.appendAssumeCapacity(first + @as(u32, @intCast(row)));
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

test "isChain and admit: every parent list of up to three rows, stream count and head" {
    const choices = [_]i32{ -2, -1, 0, 1, 2 };
    var parents: [3]i32 = undefined;
    for (0..4) |len| for (0..std.math.pow(usize, choices.len, len)) |code| {
        var c = code;
        for (parents[0..len]) |*p| {
            p.* = choices[c % choices.len];
            c /= choices.len;
        }
        const chain = [_]i32{ -1, 0, 1 };
        try std.testing.expectEqual(std.mem.eql(i32, chain[0..len], parents[0..len]), isChain(parents[0..len]));
    };
    for (0..3) |streams| for ([_]bool{ false, true }) |chain| for ([_]bool{ false, true }) |head| {
        const want: ?anyerror = if (streams != 1) error.StagesNeedOneStream else if (!chain) error.StagesNeedChain else if (!head) error.StagesNeedHead else null;
        if (want) |err| try std.testing.expectError(err, admit(streams, chain, head)) else try admit(streams, chain, head);
    };
}

test "Round: every sequence of four round events lands exactly the rounds a model of the engine's caller lands" {
    // One layer (planes embed, mixer, block, final, logits), one value wide, 2 rows. A fill marks each point's rows.
    const Fill = struct { rows: u32, head: u32 };
    const Event = union(enum) { begin, fill: Fill, partial, end, fail: usize };
    const events = [_]Event{
        .begin,                                 .partial,                               .end,
        .{ .fill = .{ .rows = 1, .head = 0 } }, .{ .fill = .{ .rows = 1, .head = 1 } }, .{ .fill = .{ .rows = 1, .head = 2 } },
        .{ .fill = .{ .rows = 2, .head = 0 } }, .{ .fill = .{ .rows = 2, .head = 1 } }, .{ .fill = .{ .rows = 2, .head = 2 } },
        .{ .fail = 0 },                         .{ .fail = 3 },                         .{ .fail = 6 },
    };
    const a = std.testing.allocator;
    const n = events.len + 1;
    var seq: [4]usize = undefined;
    var landed_rounds: usize = 0;
    var refused_by_oom: usize = 0;
    for (0..std.math.pow(usize, n, seq.len)) |code| {
        var c = code;
        var len: usize = 0;
        while (len < seq.len and c % n != events.len) : (len += 1) {
            seq[len] = c % n;
            c /= n;
        }
        if (len < seq.len and c / n != 0) continue; // events after the end marker would repeat a shorter sequence
        var failing = std.testing.FailingAllocator.init(a, .{});
        var rec = try Recorder.init(failing.allocator(), 1, 1, 1);
        defer rec.deinit();
        var round_ = try Round.init(&rec, 2, 2);
        var base: [64]u8 = @splat(0);
        // The model: what is marked, every plane's bytes and the positions that should have landed.
        var marked: enum { none, partial, fill } = .none;
        var fill: Fill = undefined;
        var tag: u8 = 0;
        var next: u32 = 9;
        var want: [5]std.ArrayList(u8) = @splat(.empty);
        defer for (&want) |*w| w.deinit(a);
        var want_positions: std.ArrayList(u32) = .empty;
        defer want_positions.deinit(a);
        var want_logits: std.ArrayList(u32) = .empty;
        defer want_logits.deinit(a);
        for (seq[0..len]) |e| switch (events[e]) {
            .begin => {
                round_.begin();
                marked = .none;
            },
            .partial => {
                const got = round_.mark(.embed, 1);
                if (marked == .none) {
                    try got;
                    marked = .partial;
                } else try std.testing.expectError(error.DuplicateStage, got);
            },
            .fill => |f| {
                // The engine's order: a refused mark stops the forward, so a fill lands whole or not at all.
                if (marked != .none) {
                    try std.testing.expectError(error.DuplicateStage, round_.mark(.embed, f.rows));
                    continue;
                }
                tag += 1;
                for (0..5) |i| {
                    const at = try point(i, 1);
                    const rows = if (at == .logits) f.head else f.rows;
                    if (rows == 0) continue;
                    try round_.mark(at, rows);
                    @memset(base[round_.offset(at)..][0 .. rows * 2], tag);
                }
                marked = .fill;
                fill = f;
            },
            .end, .fail => {
                const k: ?usize = if (events[e] == .fail) events[e].fail else null;
                if (k) |after| {
                    failing.fail_index = failing.alloc_index + after;
                    failing.resize_fail_index = failing.resize_index + after;
                }
                const got = round_.end(&base, next);
                failing.fail_index = std.math.maxInt(usize);
                failing.resize_fail_index = std.math.maxInt(usize);
                const was = marked;
                marked = .none;
                switch (was) {
                    .none => try std.testing.expectError(error.BadStageShape, got),
                    .partial => try std.testing.expectError(error.IncompleteStages, got),
                    .fill => if (fill.head == 2 and fill.rows == 1) try std.testing.expectError(error.BadStageShape, got) else {
                        got catch |err| {
                            try std.testing.expect(k != null and err == error.OutOfMemory);
                            refused_by_oom += 1;
                            continue;
                        };
                        landed_rounds += 1;
                        for (&want, 0..) |*w, i| try w.appendNTimes(a, tag, 2 * if (i == 4) fill.head else fill.rows);
                        for (0..fill.rows) |row| try want_positions.append(a, next + @as(u32, @intCast(row)));
                        for (fill.rows - fill.head..fill.rows) |row| try want_logits.append(a, next + @as(u32, @intCast(row)));
                        next += fill.rows;
                    },
                }
            },
        };
        for (want, rec.planes) |w, got| try std.testing.expectEqualSlices(u8, w.items, got.items);
        try std.testing.expectEqualSlices(u32, want_positions.items, rec.positions.items);
        try std.testing.expectEqualSlices(u32, want_logits.items, rec.logit_positions.items);
    }
    // The enumeration reaches both a landed round and an allocation failure that leaves the recorder unchanged.
    try std.testing.expect(landed_rounds > 0 and refused_by_oom > 0);
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

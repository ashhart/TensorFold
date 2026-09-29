const std = @import("std");

pub const Markers = struct { openers: []const i32 = &.{}, assistant: []const i32 = &.{} };

pub const Plan = struct {
    step: usize = 2048,
    min_chunk: usize = 256,
    openers: []const i32 = &.{},
    assistant: []const i32 = &.{},

    pub fn validate(p: Plan) !void {
        if (p.step == 0 or p.min_chunk < @max(1, p.assistant.len) or ((p.openers.len > 0 or p.assistant.len > 0) and p.min_chunk > p.step)) return error.InvalidPrefillPlan;
    }

    pub fn points(p: Plan, a: std.mem.Allocator, ids: []const i32) ![]usize {
        var found: std.ArrayList(usize) = .empty;
        errdefer found.deinit(a);
        var messages: usize = 0;
        for (ids, 0..) |id, at| {
            var keep = false;
            if (std.mem.indexOfScalar(i32, p.openers, id) != null) {
                messages += 1;
                keep = messages == 2;
            }
            if (p.assistant.len > 0 and p.assistant.len <= ids.len - at and std.mem.eql(i32, p.assistant, ids[at..][0..p.assistant.len])) keep = true;
            if (keep and at > 0) try found.append(a, at);
        }
        return found.toOwnedSlice(a);
    }

    pub fn chunks(p: Plan, a: std.mem.Allocator, ids: []const i32) !Chunks {
        try p.validate();
        const found = try p.points(a, ids);
        defer a.free(found);
        var starts: std.ArrayList(usize) = .empty;
        errdefer starts.deinit(a);
        try starts.append(a, 0);
        var last: usize = 0;
        var i: usize = 0;
        while (true) {
            while (i < found.len and found[i] < last +| p.min_chunk) : (i += 1) {}
            var next = last +| p.step;
            if (i < found.len and found[i] < next) next = found[i];
            if (next >= ids.len) return .{ .starts = try starts.toOwnedSlice(a), .length = ids.len, .step = p.step };
            try starts.append(a, next);
            last = next;
        }
    }

    pub fn name(p: Plan, a: std.mem.Allocator) std.mem.Allocator.Error![]u8 {
        if (p.openers.len == 0 and p.assistant.len == 0) return std.fmt.allocPrint(a, "grid{d}", .{p.step});
        var out: std.Io.Writer.Allocating = .init(a);
        errdefer out.deinit();
        out.writer.print("grid{d}+msg{d}:", .{ p.step, p.min_chunk }) catch return error.OutOfMemory;
        const sorted = try a.dupe(i32, p.openers);
        defer a.free(sorted);
        std.mem.sort(i32, sorted, {}, std.sort.asc(i32));
        var previous: ?i32 = null;
        for (sorted) |id| {
            if (previous == id) continue;
            if (previous != null) out.writer.writeByte('.') catch return error.OutOfMemory;
            out.writer.print("{d}", .{id}) catch return error.OutOfMemory;
            previous = id;
        }
        out.writer.writeByte(':') catch return error.OutOfMemory;
        for (p.assistant, 0..) |id, i| {
            if (i > 0) out.writer.writeByte('.') catch return error.OutOfMemory;
            out.writer.print("{d}", .{id}) catch return error.OutOfMemory;
        }
        return out.toOwnedSlice();
    }
};

pub const Chunks = struct {
    starts: ?[]const usize,
    length: usize,
    step: usize = 2048,

    pub fn deinit(c: Chunks, a: std.mem.Allocator) void {
        if (c.starts) |starts| a.free(starts);
    }
    pub fn contains(c: Chunks, position: usize) bool {
        if (position == 0 or position >= c.length) return false;
        const starts = c.starts orelse return true;
        const index = after(starts, position);
        return index > 0 and starts[index - 1] == position;
    }
    pub fn floor(c: Chunks, position: usize) usize {
        const limit = @min(position, c.length);
        const starts = c.starts orelse return limit;
        const index = after(starts, limit);
        return if (index == 0) 0 else starts[index - 1];
    }
    pub fn next(c: Chunks, position: usize) usize {
        const starts = c.starts orelse return @min(position +| c.step, c.length);
        const index = after(starts, position);
        return if (index < starts.len) starts[index] else c.length;
    }
    fn after(starts: []const usize, position: usize) usize {
        var low: usize = 0;
        var high = starts.len;
        while (low < high) {
            const middle = low + (high - low) / 2;
            if (starts[middle] <= position) low = middle + 1 else high = middle;
        }
        return low;
    }
};

pub fn check(io: std.Io, path: []const u8) !void {
    const a = std.heap.page_allocator;
    const source = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(32 * 1024 * 1024));
    defer a.free(source);
    const Case = struct {
        plan: Plan,
        tokens: []const i32,
        name: ?[]const u8,
        points: []const usize,
        starts: []const usize,
        positions: []const struct { position: usize, contains: bool, floor: usize },
        spans: []const struct { begin: usize, end: usize, chunks: []const [2]usize },
    };
    const parsed = try std.json.parseFromSlice([]const Case, a, source, .{});
    defer parsed.deinit();
    for (parsed.value) |case| {
        if (case.name == null) {
            try std.testing.expectError(error.InvalidPrefillPlan, case.plan.validate());
            continue;
        }
        const name = try case.plan.name(a);
        defer a.free(name);
        try std.testing.expectEqualStrings(case.name.?, name);
        const points = try case.plan.points(a, case.tokens);
        defer a.free(points);
        try std.testing.expectEqualSlices(usize, case.points, points);
        const chunks = try case.plan.chunks(a, case.tokens);
        defer chunks.deinit(a);
        try std.testing.expectEqualSlices(usize, case.starts, chunks.starts.?);
        for (case.positions) |position| {
            try std.testing.expectEqual(position.contains, chunks.contains(position.position));
            try std.testing.expectEqual(position.floor, chunks.floor(position.position));
        }
        for (case.spans) |span| {
            var at = span.begin;
            for (span.chunks) |pair| {
                const end = @min(chunks.next(at), span.end);
                try std.testing.expectEqual(at, pair[0]);
                try std.testing.expectEqual(end, pair[1]);
                at = end;
            }
            try std.testing.expectEqual(span.end, at);
        }
    }
    std.debug.print("PASS: {d} upstream adaptive prefill plans, resume points and chunk spans\n", .{parsed.value.len});
}

test "empty plans and unbounded chunk positions" {
    const a = std.testing.allocator;
    const empty = try (Plan{}).chunks(a, &.{});
    defer empty.deinit(a);
    try std.testing.expectEqualSlices(usize, &.{0}, empty.starts.?);
    const chunks = Chunks{ .starts = null, .length = 29, .step = 8 };
    try std.testing.expect(chunks.contains(3));
    try std.testing.expect(!chunks.contains(29));
    try std.testing.expectEqual(@as(usize, 29), chunks.floor(31));
    try std.testing.expectEqual(@as(usize, 11), chunks.next(3));
}

fn allocationFailures(a: std.mem.Allocator) !void {
    const plan = Plan{ .step = 8, .min_chunk = 2, .openers = &.{ 7, 6, 7 }, .assistant = &.{ 7, 8 } };
    const chunks = try plan.chunks(a, &.{ 6, 1, 2, 7, 8, 3, 4, 7, 8, 2, 3, 4, 5, 6 });
    defer chunks.deinit(a);
    const name = try plan.name(a);
    defer a.free(name);
    try std.testing.expectEqualStrings("grid8+msg2:6.7:7.8", name);
}

test "plan allocations release owned boundaries and scheme names on failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailures, .{});
}

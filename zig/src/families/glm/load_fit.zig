//! A host-only reduction of priced GLM candidates; it performs no allocation or device operation.
const std = @import("std");
const margin = @import("sys").memory_policy.margin;

/// All readings come from one preflight snapshot. Available is free bytes, not Metal's unused recommendation.
pub const Snapshot = struct {
    ram: ?u64,
    working_set: ?u64,
    footprint: ?u64,
    available: ?u64,
    load_limit: u64,
    read_floor: u64,
};

pub const Room = struct {
    cap: u64,
    baseline: u64,
    additional: u64,

    /// Keep the family's existing parsed load limit; neither free memory nor this helper may raise it.
    pub fn from(s: Snapshot) !Room {
        const ram = s.ram orelse return error.NoMemoryReading;
        const working = s.working_set orelse return error.NoMemoryReading;
        const held = s.footprint orelse return error.NoMemoryReading;
        const free = s.available orelse return error.NoMemoryReading;
        if (ram == 0 or working == 0 or held == 0 or s.load_limit == 0) return error.NoMemoryReading;
        if (free > ram or held > ram) return error.BadMemoryReading;
        const cap = @min(s.load_limit, @min(ram, working));
        return .{
            .cap = cap,
            .baseline = held,
            .additional = @min(cap -| held -| margin, free -| s.read_floor),
        };
    }
};

/// Incremental bytes above Room.baseline, excluding its reserve; disjoint phases are never summed.
pub const Phases = struct {
    loading: u64,
    ready: u64,
    first_request: u64,

    pub fn peak(p: Phases) u64 {
        return @max(p.loading, @max(p.ready, p.first_request));
    }
};

/// Sum only allocations alive together in one phase; overflow is a refusal rather than a wrapped fit.
pub fn phaseBytes(parts: []const u64) !u64 {
    var sum: u64 = 0;
    for (parts) |n| sum = try std.math.add(u64, sum, n);
    return sum;
}

/// One producer-validated geometry/mode; window is the actual advertised capacity after pool reservations.
pub const Candidate = struct { configured_window: u32, window: u32, capacity: u32, streams: u32, phases: Phases };
pub const Request = struct { max_window: u32, exact_window: ?u32 = null, streams: u32, fixed: bool };

/// Fold a bounded producer census without assuming its phase costs are linear or monotone in context.
pub const Choice = struct {
    request: Request,
    room: Room,
    best: ?Candidate = null,

    pub fn init(request: Request, snapshot: Snapshot) !Choice {
        if (request.max_window == 0 or request.streams == 0) return error.BadFitRequest;
        if (request.exact_window) |n| if (n == 0 or n > request.max_window) return error.BadFitRequest;
        return .{ .request = request, .room = try Room.from(snapshot) };
    }

    /// Context first, then streams. Callers must use a separate choice for a different numerical/draft mode.
    pub fn consider(c: *Choice, item: Candidate) !void {
        if (item.window == 0 or item.window > item.configured_window or item.capacity < item.configured_window or
            item.streams == 0 or item.phases.peak() == 0)
            return error.BadFitCandidate;
        if (item.configured_window > c.request.max_window or item.streams > c.request.streams) return;
        if (c.request.exact_window) |n| if (item.configured_window != n) return;
        if (c.request.fixed and item.streams != c.request.streams) return;
        if (item.phases.peak() > c.room.additional) return;
        if (c.best) |old| {
            if (item.window < old.window or (item.window == old.window and item.streams <= old.streams)) return;
        }
        c.best = item;
    }

    pub fn finish(c: *const Choice) !Candidate {
        return c.best orelse error.OverMemoryLimit;
    }
};

const GiB: u64 = 1 << 30;
fn sample() Snapshot {
    return .{ .ram = 100 * GiB, .working_set = 90 * GiB, .footprint = 10 * GiB, .available = 80 * GiB, .load_limit = 70 * GiB, .read_floor = 8 * GiB };
}
fn cell(window: u32, streams: u32, bytes: u64) Candidate {
    return .{ .configured_window = window, .window = window, .capacity = window + 64, .streams = streams, .phases = .{ .loading = bytes, .ready = bytes, .first_request = bytes } };
}

test "fit keeps the family cap and both independent headroom bounds" {
    var s = sample();
    const r = try Room.from(s);
    try std.testing.expectEqual(@as(u64, 58 * GiB), r.additional);
    s.available = 20 * GiB;
    try std.testing.expectEqual(@as(u64, 12 * GiB), (try Room.from(s)).additional);
    s.working_set = 25 * GiB;
    try std.testing.expectEqual(@as(u64, 13 * GiB), (try Room.from(s)).cap - s.footprint.? - margin);
}

test "fit unknown and inconsistent readings never become unlimited room" {
    var s = sample();
    s.available = null;
    try std.testing.expectError(error.NoMemoryReading, Room.from(s));
    s = sample();
    s.available = 101 * GiB;
    try std.testing.expectError(error.BadMemoryReading, Room.from(s));
    s = sample();
    s.available = 0;
    try std.testing.expectEqual(@as(u64, 0), (try Room.from(s)).additional);
}

test "fit exact edge accepts and a one-byte loading or first-use excess refuses" {
    var c = try Choice.init(.{ .max_window = 4096, .streams = 1, .fixed = true }, sample());
    const exact = cell(4096, 1, c.room.additional);
    try c.consider(exact);
    try std.testing.expectEqual(@as(u32, 4096), (try c.finish()).window);
    c.best = null;
    var larger = exact;
    larger.phases.first_request += 1;
    try c.consider(larger);
    try std.testing.expectError(error.OverMemoryLimit, c.finish());
    larger = exact;
    larger.phases.loading += 1;
    try c.consider(larger);
    try std.testing.expectError(error.OverMemoryLimit, c.finish());
}

test "fit maximizes actual window then streams without ordered or monotone prices" {
    var c = try Choice.init(.{ .max_window = 8192, .streams = 8, .fixed = false }, sample());
    try c.consider(cell(4096, 8, 30 * GiB));
    try c.consider(cell(8192, 1, 40 * GiB));
    try c.consider(cell(8192, 4, 45 * GiB));
    try c.consider(cell(8192, 8, 59 * GiB));
    try c.consider(cell(2048, 8, 57 * GiB));
    const chosen = try c.finish();
    try std.testing.expectEqual(@as(u32, 8192), chosen.window);
    try std.testing.expectEqual(@as(u32, 4), chosen.streams);
}

test "fit explicit context and fixed stream count cannot be silently reduced" {
    var c = try Choice.init(.{ .max_window = 8192, .exact_window = 8192, .streams = 4, .fixed = true }, sample());
    try c.consider(cell(4096, 4, 10 * GiB));
    try c.consider(cell(8192, 1, 10 * GiB));
    try std.testing.expectError(error.OverMemoryLimit, c.finish());
}

test "fit keeps configured and advertised pooled capacities separate" {
    var c = try Choice.init(.{ .max_window = 131072, .exact_window = 131072, .streams = 1, .fixed = true }, sample());
    var pooled = cell(131072, 1, GiB);
    pooled.window = 131051;
    try c.consider(pooled);
    const chosen = try c.finish();
    try std.testing.expectEqual(@as(u32, 131072), chosen.configured_window);
    try std.testing.expectEqual(@as(u32, 131051), chosen.window);
}

test "fit sums simultaneous parts with checked arithmetic and takes the maximum of phases" {
    try std.testing.expectEqual(@as(u64, 7), try phaseBytes(&.{ 3, 4 }));
    try std.testing.expectError(error.Overflow, phaseBytes(&.{ std.math.maxInt(u64), 1 }));
    const p = Phases{ .loading = 9, .ready = 4, .first_request = 7 };
    try std.testing.expectEqual(@as(u64, 9), p.peak());
}

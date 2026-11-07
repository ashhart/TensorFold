//! TF_LANE_TRACE=<file>: one JSON line per lane-host action (submit, fill call, decode round, completion, cancel).
const std = @import("std");

/// A line: a request's arrival, a prompt call, a decode round, a readout's completion, a cancel after a prompt began.
pub const Kind = enum { submit, fill, round, complete, cancel };

/// One action on the awake clock: the request (round: 0), its admission number (round: live streams), its prompt rows.
pub const Event = struct {
    t0_ns: i96,
    t1_ns: i96,
    kind: Kind,
    id: u64 = 0,
    stream: u64 = 0,
    rows_from: usize = 0,
    rows_to: usize = 0,
    ok: bool = true, // false: the call, round or completion failed (its rows: what was filled before the failure)

    /// A prompt call of request `id` (admission `stream`) begun now at row `from`.
    pub fn fill(t: *const Trace, id: u64, stream: u64, from: usize) Event {
        return .{ .t0_ns = t.now(), .t1_ns = 0, .kind = .fill, .id = id, .stream = stream, .rows_from = from };
    }

    /// A round begun now over `live` streams (its width read before the step can retire any).
    pub fn round(t: *const Trace, live: usize) Event {
        return .{ .t0_ns = t.now(), .t1_ns = 0, .kind = .round, .stream = live, .rows_to = live };
    }

    /// The call failed with `filled` rows in.
    pub fn failed(e: *Event, filled: usize) void {
        e.ok = false;
        e.rows_to = filled;
    }
};

/// `slot`'s trace from TF_LANE_TRACE, unless one is set already (a test's).
pub fn openInto(slot: *?*Trace, gpa: std.mem.Allocator, io: std.Io) void {
    if (slot.* == null) slot.* = .fromEnv(gpa, io);
}

/// `slot`'s trace closed, if it has one.
pub fn drop(slot: *?*Trace) void {
    if (slot.*) |t| t.close();
    slot.* = null;
}

/// The open trace; the lane host owns it from `fromEnv` to `close`.
pub const Trace = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    broken: std.atomic.Value(bool) = .init(false), // a write failed: no more lines, so no line is ever partial

    /// The file TF_LANE_TRACE names, opened once (truncated), or null when unset, empty or unopenable.
    pub fn fromEnv(gpa: std.mem.Allocator, io: std.Io) ?*Trace {
        const path = std.mem.span(std.c.getenv("TF_LANE_TRACE") orelse return null);
        if (path.len == 0) return null;
        return open(gpa, io, path) catch |e| {
            std.log.warn("TF_LANE_TRACE={s}: cannot open it ({t}); no trace", .{ path, e });
            return null;
        };
    }

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !*Trace {
        const file = try std.Io.Dir.cwd().createFile(io, path, .{});
        errdefer file.close(io);
        const t = try gpa.create(Trace);
        t.* = .{ .gpa = gpa, .io = io, .file = file };
        return t;
    }

    pub fn close(t: *Trace) void {
        t.file.close(t.io);
        t.gpa.destroy(t);
    }

    /// The awake clock every line uses.
    pub fn now(t: *const Trace) i96 {
        return std.Io.Clock.awake.now(t.io).toNanoseconds();
    }

    /// `e` ended now: its line written.
    pub fn end(t: *Trace, e: *Event) void {
        e.t1_ns = t.now();
        t.line(e.*);
    }

    pub fn endFailed(t: *Trace, e: *Event) void {
        e.ok = false;
        t.end(e);
    }

    /// One line in one streaming write (no lock: the OS keeps a shared offset whole); a short write stops the trace.
    pub fn line(t: *Trace, e: Event) void {
        if (t.broken.load(.monotonic)) return;
        var buf: [208]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "{{\"t0_ns\":{d},\"t1_ns\":{d},\"kind\":\"{t}\",\"id\":{d},\"stream\":{d},\"rows_from\":{d},\"rows_to\":{d},\"ok\":{}}}\n", .{ e.t0_ns, e.t1_ns, e.kind, e.id, e.stream, e.rows_from, e.rows_to, e.ok }) catch return;
        const n = t.file.writeStreaming(t.io, &.{}, &.{text}, 1) catch 0;
        if (n == text.len) return;
        if (!t.broken.swap(true, .monotonic)) std.log.warn("TF_LANE_TRACE: a write gave {d} of {d} bytes; the trace stops here", .{ n, text.len });
    }
};

test "lane trace: each event is one JSON line on the file, in the order written" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/trace.jsonl", .{tmp.sub_path});
    defer a.free(path);
    const t = try Trace.open(a, std.testing.io, path);
    t.line(.{ .t0_ns = 1, .t1_ns = 2, .kind = .fill, .id = 7, .stream = 3, .rows_from = 0, .rows_to = 1024 });
    t.line(.{ .t0_ns = 3, .t1_ns = 3, .kind = .submit, .id = 8, .rows_to = 9 });
    t.close();
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 16));
    defer a.free(text);
    try std.testing.expectEqualStrings(
        \\{"t0_ns":1,"t1_ns":2,"kind":"fill","id":7,"stream":3,"rows_from":0,"rows_to":1024,"ok":true}
        \\{"t0_ns":3,"t1_ns":3,"kind":"submit","id":8,"stream":0,"rows_from":0,"rows_to":9,"ok":true}
        \\
    , text);
}
